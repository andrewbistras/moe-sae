"""Strict, target-aware registry for the Gemma scientific results.

The registry deliberately separates five identities which were easy to
conflate in the first five experiments:

``source_id``
    The immutable report or result artifact supplying the evidence.

``score_space_id``
    The model/capture/target/centering/SST basis under which EV values are
    mathematically comparable.

``cell_id``
    One trained condition at one seed and checkpoint milestone.

``cost_space_id``
    The accelerator and timing contract under which measured training costs
    may be paired.

``contrast_id``
    The paired cells, endpoint, estimator, uncertainty, and interpretation
    status of one claim.

Adapters are provided for the Experiment-4 horizon report, the Experiment-5
core report plus its eighteen scientific-result parents, and the Experiment-5
v3 activation-policy report.  A generic fragment interface is the extension
surface for the A512 closure and later experiments.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from moe_sae.contribution_artifacts import canonical_json_bytes, canonical_sha256

REGISTRY_SCHEMA_VERSION = "moe-sae-scientific-result-registry-v1"
FRAGMENT_SCHEMA_VERSION = "moe-sae-scientific-result-registry-fragment-v1"
SOURCE_SCHEMA_VERSION = "moe-sae-scientific-result-source-v1"
SCORE_SPACE_SCHEMA_VERSION = "moe-sae-score-space-v1"
COST_SPACE_SCHEMA_VERSION = "moe-sae-cost-space-v1"
CELL_SCHEMA_VERSION = "moe-sae-result-cell-v1"
CONTRAST_SCHEMA_VERSION = "moe-sae-result-contrast-v1"

MODEL_ID = "google/gemma-4-26B-A4B"
LAYER_ONE_INDEXED = 15
EVAL_TARGET_MANIFEST_SHA256 = (
    "31454e5ce8693521c41a2a95f2ef88d40c5d26c726a0efac5c26d882ed341f5e"
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_RE = re.compile(r"^[0-9a-f]{40}$")
_QUALIFIED_SEPARATOR_RE = re.compile(r"[.:/]")
_BARE_CONDITION_LABELS = frozenset({"A", "B", "F", "H", "R"})
_ID_PREFIXES = {
    "source_id": "source",
    "score_space_id": "score",
    "cost_space_id": "cost",
    "cell_id": "cell",
    "contrast_id": "contrast",
}


class ResultRegistryError(ValueError):
    """A result registry is incomplete, ambiguous, or arithmetically false."""


def _mapping(
    value: Any,
    *,
    name: str,
    fields: set[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ResultRegistryError(f"{name} must be a string-keyed mapping")
    selected = dict(value)
    if fields is not None and set(selected) != fields:
        raise ResultRegistryError(
            f"{name} fields changed: expected={sorted(fields)!r}, "
            f"observed={sorted(selected)!r}"
        )
    return selected


def _sequence(value: Any, *, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ResultRegistryError(f"{name} must be a list")
    return list(value)


def _string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ResultRegistryError(f"{name} must be a nonempty string")
    return value


def _sha256(value: Any, *, name: str, optional: bool = False) -> str | None:
    if optional and value is None:
        return None
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ResultRegistryError(f"{name} must be a lowercase SHA-256")
    return value


def _git(value: Any, *, name: str, optional: bool = False) -> str | None:
    if optional and value is None:
        return None
    if not isinstance(value, str) or _GIT_RE.fullmatch(value) is None:
        raise ResultRegistryError(f"{name} must be a full lowercase Git SHA")
    return value


def _finite(
    value: Any,
    *,
    name: str,
    minimum: float | None = None,
    positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ResultRegistryError(f"{name} must be numeric")
    selected = float(value)
    if not math.isfinite(selected):
        raise ResultRegistryError(f"{name} must be finite")
    if positive and selected <= 0:
        raise ResultRegistryError(f"{name} must be positive")
    if minimum is not None and selected < minimum:
        raise ResultRegistryError(f"{name} must be >= {minimum}")
    return selected


def _integer(
    value: Any,
    *,
    name: str,
    minimum: int = 0,
    optional: bool = False,
) -> int | None:
    if optional and value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ResultRegistryError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _identified(
    body: Mapping[str, Any],
    *,
    id_field: str,
) -> dict[str, Any]:
    prefix = _ID_PREFIXES[id_field]
    selected = copy.deepcopy(dict(body))
    return {
        id_field: f"{prefix}:{canonical_sha256(selected)}",
        **selected,
    }


def _validate_identified(
    value: Any,
    *,
    id_field: str,
    name: str,
) -> tuple[str, dict[str, Any]]:
    selected = _mapping(value, name=name)
    identity = _string(selected.pop(id_field, None), name=f"{name}.{id_field}")
    expected = f"{_ID_PREFIXES[id_field]}:{canonical_sha256(selected)}"
    if identity != expected:
        raise ResultRegistryError(f"{name}.{id_field} is not canonical")
    return identity, selected


def make_source(
    *,
    kind: str,
    schema_version: str,
    scientific_sha256: str,
    file_sha256: str | None = None,
) -> dict[str, Any]:
    """Build one immutable source-artifact identity."""

    body = {
        "schema_version": SOURCE_SCHEMA_VERSION,
        "kind": _string(kind, name="source.kind"),
        "artifact_schema_version": _string(
            schema_version,
            name="source.artifact_schema_version",
        ),
        "scientific_sha256": _sha256(
            scientific_sha256,
            name="source.scientific_sha256",
        ),
        "file_sha256": _sha256(
            file_sha256,
            name="source.file_sha256",
            optional=True,
        ),
    }
    result = _identified(body, id_field="source_id")
    validate_source(result)
    return result


def validate_source(value: Any) -> dict[str, Any]:
    identity, body = _validate_identified(
        value,
        id_field="source_id",
        name="source",
    )
    body = _mapping(
        body,
        name="source",
        fields={
            "schema_version",
            "kind",
            "artifact_schema_version",
            "scientific_sha256",
            "file_sha256",
        },
    )
    if body["schema_version"] != SOURCE_SCHEMA_VERSION:
        raise ResultRegistryError("source schema is incompatible")
    _string(body["kind"], name="source.kind")
    _string(
        body["artifact_schema_version"],
        name="source.artifact_schema_version",
    )
    _sha256(body["scientific_sha256"], name="source.scientific_sha256")
    _sha256(body["file_sha256"], name="source.file_sha256", optional=True)
    return {"source_id": identity, **body}


def make_score_space(
    *,
    model_id: str,
    model_checkpoint_sha256: str | None,
    layer_one_indexed: int,
    target_id: str,
    endpoint: str,
    evaluation_manifest_sha256: str,
    token_identity_sha256: str | None,
    centering_id: str,
    sst_id: str,
    sst_mode: str = "exact",
    sst_rtol: float = 0.0,
    sst_atol: float = 0.0,
) -> dict[str, Any]:
    """Build the identity of one mathematically comparable score lane."""

    body = {
        "schema_version": SCORE_SPACE_SCHEMA_VERSION,
        "model_id": model_id,
        "model_checkpoint_sha256": model_checkpoint_sha256,
        "layer_one_indexed": layer_one_indexed,
        "target_id": target_id,
        "endpoint": endpoint,
        "evaluation_manifest_sha256": evaluation_manifest_sha256,
        "token_identity_sha256": token_identity_sha256,
        "centering_id": centering_id,
        "sst_id": sst_id,
        "sst_policy": {
            "mode": sst_mode,
            "relative_tolerance": sst_rtol,
            "absolute_tolerance": sst_atol,
        },
    }
    result = _identified(body, id_field="score_space_id")
    validate_score_space(result)
    return result


def validate_score_space(value: Any) -> dict[str, Any]:
    identity, body = _validate_identified(
        value,
        id_field="score_space_id",
        name="score space",
    )
    body = _mapping(
        body,
        name="score space",
        fields={
            "schema_version",
            "model_id",
            "model_checkpoint_sha256",
            "layer_one_indexed",
            "target_id",
            "endpoint",
            "evaluation_manifest_sha256",
            "token_identity_sha256",
            "centering_id",
            "sst_id",
            "sst_policy",
        },
    )
    if body["schema_version"] != SCORE_SPACE_SCHEMA_VERSION:
        raise ResultRegistryError("score-space schema is incompatible")
    _string(body["model_id"], name="score_space.model_id")
    _sha256(
        body["model_checkpoint_sha256"],
        name="score_space.model_checkpoint_sha256",
        optional=True,
    )
    _integer(
        body["layer_one_indexed"],
        name="score_space.layer_one_indexed",
        minimum=1,
    )
    for field in ("target_id", "endpoint", "centering_id", "sst_id"):
        _string(body[field], name=f"score_space.{field}")
    _sha256(
        body["evaluation_manifest_sha256"],
        name="score_space.evaluation_manifest_sha256",
    )
    _sha256(
        body["token_identity_sha256"],
        name="score_space.token_identity_sha256",
        optional=True,
    )
    policy = _mapping(
        body["sst_policy"],
        name="score_space.sst_policy",
        fields={"mode", "relative_tolerance", "absolute_tolerance"},
    )
    if policy["mode"] not in {"exact", "numerically_equivalent"}:
        raise ResultRegistryError("score-space SST mode is unsupported")
    rtol = _finite(
        policy["relative_tolerance"],
        name="score_space.sst_policy.relative_tolerance",
        minimum=0,
    )
    atol = _finite(
        policy["absolute_tolerance"],
        name="score_space.sst_policy.absolute_tolerance",
        minimum=0,
    )
    if policy["mode"] == "exact" and (rtol != 0 or atol != 0):
        raise ResultRegistryError("exact SST policy cannot carry tolerances")
    return {"score_space_id": identity, **body}


def make_cost_space(
    *,
    measurement_id: str,
    accelerator_type: str,
    accelerator_generation: str,
    chip_count: int,
    included: Sequence[str],
    excluded: Sequence[str],
    effective_batch_tokens: int | None,
    microbatch_tokens: int | None,
) -> dict[str, Any]:
    """Build one strict measured-cost comparison space."""

    body = {
        "schema_version": COST_SPACE_SCHEMA_VERSION,
        "measurement_id": measurement_id,
        "accelerator_type": accelerator_type,
        "accelerator_generation": accelerator_generation,
        "chip_count": chip_count,
        "included": list(included),
        "excluded": list(excluded),
        "effective_batch_tokens": effective_batch_tokens,
        "microbatch_tokens": microbatch_tokens,
    }
    result = _identified(body, id_field="cost_space_id")
    validate_cost_space(result)
    return result


def validate_cost_space(value: Any) -> dict[str, Any]:
    identity, body = _validate_identified(
        value,
        id_field="cost_space_id",
        name="cost space",
    )
    body = _mapping(
        body,
        name="cost space",
        fields={
            "schema_version",
            "measurement_id",
            "accelerator_type",
            "accelerator_generation",
            "chip_count",
            "included",
            "excluded",
            "effective_batch_tokens",
            "microbatch_tokens",
        },
    )
    if body["schema_version"] != COST_SPACE_SCHEMA_VERSION:
        raise ResultRegistryError("cost-space schema is incompatible")
    for field in (
        "measurement_id",
        "accelerator_type",
        "accelerator_generation",
    ):
        _string(body[field], name=f"cost_space.{field}")
    _integer(body["chip_count"], name="cost_space.chip_count", minimum=1)
    for field in ("included", "excluded"):
        rows = _sequence(body[field], name=f"cost_space.{field}")
        if len(rows) != len(set(rows)):
            raise ResultRegistryError(f"cost_space.{field} contains duplicates")
        for index, row in enumerate(rows):
            _string(row, name=f"cost_space.{field}[{index}]")
    if set(body["included"]) & set(body["excluded"]):
        raise ResultRegistryError("cost-space included/excluded scopes overlap")
    _integer(
        body["effective_batch_tokens"],
        name="cost_space.effective_batch_tokens",
        minimum=1,
        optional=True,
    )
    _integer(
        body["microbatch_tokens"],
        name="cost_space.microbatch_tokens",
        minimum=1,
        optional=True,
    )
    return {"cost_space_id": identity, **body}


def _validate_geometry(value: Any, *, name: str) -> dict[str, Any]:
    selected = _mapping(
        value,
        name=name,
        fields={
            "bank_identity",
            "bank_count",
            "bank_width",
            "total_features",
            "candidates_per_token",
            "selected_slots",
            "positive_l0",
            "trainable_parameters",
        },
    )
    _string(selected["bank_identity"], name=f"{name}.bank_identity")
    _integer(selected["bank_count"], name=f"{name}.bank_count", minimum=1)
    _integer(selected["bank_width"], name=f"{name}.bank_width", minimum=1)
    total = _integer(
        selected["total_features"],
        name=f"{name}.total_features",
        minimum=1,
    )
    candidates = _integer(
        selected["candidates_per_token"],
        name=f"{name}.candidates_per_token",
        minimum=1,
    )
    slots = _integer(
        selected["selected_slots"],
        name=f"{name}.selected_slots",
        minimum=1,
    )
    positive = _finite(
        selected["positive_l0"],
        name=f"{name}.positive_l0",
        minimum=0,
    )
    _integer(
        selected["trainable_parameters"],
        name=f"{name}.trainable_parameters",
        minimum=1,
        optional=True,
    )
    assert total is not None and candidates is not None and slots is not None
    if candidates > total or slots > candidates or positive > slots:
        raise ResultRegistryError(
            f"{name} violates features >= candidates >= slots >= positive L0"
        )
    return selected


def _validate_score(value: Any, *, name: str) -> dict[str, Any]:
    selected = _mapping(
        value,
        name=name,
        fields={
            "score_space_id",
            "sse",
            "sst",
            "nmse",
            "ev",
            "sst_sha256",
        },
    )
    _string(selected["score_space_id"], name=f"{name}.score_space_id")
    sse = _finite(selected["sse"], name=f"{name}.sse", minimum=0)
    sst = _finite(selected["sst"], name=f"{name}.sst", positive=True)
    nmse = _finite(selected["nmse"], name=f"{name}.nmse", minimum=0)
    ev = _finite(selected["ev"], name=f"{name}.ev")
    _sha256(
        selected["sst_sha256"],
        name=f"{name}.sst_sha256",
        optional=True,
    )
    expected_nmse = sse / sst
    if not math.isclose(nmse, expected_nmse, rel_tol=1e-10, abs_tol=1e-10):
        raise ResultRegistryError(f"{name}.nmse differs from SSE/SST")
    if not math.isclose(ev, 1.0 - nmse, rel_tol=1e-10, abs_tol=1e-10):
        raise ResultRegistryError(f"{name}.ev differs from 1-NMSE")
    return selected


def _validate_cost(value: Any, *, name: str) -> dict[str, Any] | None:
    if value is None:
        return None
    selected = _mapping(
        value,
        name=name,
        fields={
            "cost_space_id",
            "chip_seconds",
            "wall_seconds",
            "throughput_tokens",
            "tokens_per_second",
            "useful_mfu",
        },
    )
    _string(selected["cost_space_id"], name=f"{name}.cost_space_id")
    chips = _finite(
        selected["chip_seconds"],
        name=f"{name}.chip_seconds",
        positive=True,
    )
    wall = _finite(
        selected["wall_seconds"],
        name=f"{name}.wall_seconds",
        positive=True,
    )
    _integer(
        selected["throughput_tokens"],
        name=f"{name}.throughput_tokens",
        minimum=1,
    )
    _finite(
        selected["tokens_per_second"],
        name=f"{name}.tokens_per_second",
        positive=True,
    )
    if selected["useful_mfu"] is not None:
        _finite(
            selected["useful_mfu"],
            name=f"{name}.useful_mfu",
            minimum=0,
        )
    # Chip count and actual-token arithmetic are checked with registry context.
    return selected


def make_cell(
    *,
    experiment_id: str,
    protocol_id: str,
    panel: str,
    condition_id: str,
    condition_label: str,
    seed: int,
    checkpoint_tokens: int,
    actual_training_tokens: int,
    encoder_source_id: str,
    activation_policy_id: str,
    geometry: Mapping[str, Any],
    endpoints: Mapping[str, Any],
    cost: Mapping[str, Any] | None,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one result cell and reject ambiguous condition identities."""

    body = {
        "schema_version": CELL_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "protocol_id": protocol_id,
        "panel": panel,
        "condition_id": condition_id,
        "condition_label": condition_label,
        "seed": seed,
        "checkpoint_tokens": checkpoint_tokens,
        "actual_training_tokens": actual_training_tokens,
        "encoder_source_id": encoder_source_id,
        "activation_policy_id": activation_policy_id,
        "geometry": copy.deepcopy(dict(geometry)),
        "endpoints": copy.deepcopy(dict(endpoints)),
        "cost": copy.deepcopy(dict(cost)) if cost is not None else None,
        "provenance": copy.deepcopy(dict(provenance)),
    }
    result = _identified(body, id_field="cell_id")
    validate_cell(result)
    return result


def validate_cell(value: Any) -> dict[str, Any]:
    identity, body = _validate_identified(
        value,
        id_field="cell_id",
        name="cell",
    )
    body = _mapping(
        body,
        name="cell",
        fields={
            "schema_version",
            "experiment_id",
            "protocol_id",
            "panel",
            "condition_id",
            "condition_label",
            "seed",
            "checkpoint_tokens",
            "actual_training_tokens",
            "encoder_source_id",
            "activation_policy_id",
            "geometry",
            "endpoints",
            "cost",
            "provenance",
        },
    )
    if body["schema_version"] != CELL_SCHEMA_VERSION:
        raise ResultRegistryError("cell schema is incompatible")
    for field in (
        "experiment_id",
        "protocol_id",
        "panel",
        "condition_id",
        "condition_label",
        "encoder_source_id",
        "activation_policy_id",
    ):
        _string(body[field], name=f"cell.{field}")
    condition_id = body["condition_id"]
    if (
        condition_id.strip().upper() in _BARE_CONDITION_LABELS
        or _QUALIFIED_SEPARATOR_RE.search(condition_id) is None
    ):
        raise ResultRegistryError(
            "cell.condition_id must be qualified; bare A/B/F/H/R labels "
            "are forbidden"
        )
    _integer(body["seed"], name="cell.seed", minimum=0)
    milestone = _integer(
        body["checkpoint_tokens"],
        name="cell.checkpoint_tokens",
        minimum=1,
    )
    actual = _integer(
        body["actual_training_tokens"],
        name="cell.actual_training_tokens",
        minimum=1,
    )
    assert milestone is not None and actual is not None
    if actual < milestone:
        raise ResultRegistryError("cell actual training tokens precede milestone")
    _validate_geometry(body["geometry"], name="cell.geometry")
    endpoints = _mapping(body["endpoints"], name="cell.endpoints")
    if not endpoints:
        raise ResultRegistryError("cell must contain at least one endpoint")
    for endpoint, score in endpoints.items():
        _string(endpoint, name="cell endpoint")
        _validate_score(score, name=f"cell.endpoints.{endpoint}")
    _validate_cost(body["cost"], name="cell.cost")
    provenance = _mapping(
        body["provenance"],
        name="cell.provenance",
        fields={
            "registry_source_id",
            "result_sha256",
            "checkpoint_sha256",
            "source_revision",
        },
    )
    _string(
        provenance["registry_source_id"],
        name="cell.provenance.registry_source_id",
    )
    _sha256(
        provenance["result_sha256"],
        name="cell.provenance.result_sha256",
    )
    _sha256(
        provenance["checkpoint_sha256"],
        name="cell.provenance.checkpoint_sha256",
        optional=True,
    )
    _git(
        provenance["source_revision"],
        name="cell.provenance.source_revision",
        optional=True,
    )
    return {"cell_id": identity, **body}


def make_contrast(
    *,
    experiment_id: str,
    kind: str,
    label: str,
    left_cell_ids: Sequence[str],
    right_cell_ids: Sequence[str],
    endpoint: str | None,
    aggregation: str,
    estimate: float | None,
    interval: Mapping[str, Any] | None,
    interpreted: bool,
    source_artifact_sha256: str,
) -> dict[str, Any]:
    """Build a quality effect or paired measured-cost ratio."""

    body = {
        "schema_version": CONTRAST_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "kind": kind,
        "label": label,
        "left_cell_ids": list(left_cell_ids),
        "right_cell_ids": list(right_cell_ids),
        "endpoint": endpoint,
        "aggregation": aggregation,
        "estimate": estimate,
        "interval": copy.deepcopy(dict(interval)) if interval is not None else None,
        "interpreted": interpreted,
        "source_artifact_sha256": source_artifact_sha256,
    }
    return _identified(body, id_field="contrast_id")


def _validate_interval(value: Any, *, name: str) -> dict[str, Any] | None:
    if value is None:
        return None
    selected = _mapping(
        value,
        name=name,
        fields={
            "lower",
            "upper",
            "confidence_level",
            "unit",
            "replicates",
        },
    )
    lower = _finite(selected["lower"], name=f"{name}.lower")
    upper = _finite(selected["upper"], name=f"{name}.upper")
    confidence = _finite(
        selected["confidence_level"],
        name=f"{name}.confidence_level",
        positive=True,
    )
    _string(selected["unit"], name=f"{name}.unit")
    _integer(
        selected["replicates"],
        name=f"{name}.replicates",
        minimum=1,
        optional=True,
    )
    if lower > upper or confidence > 1:
        raise ResultRegistryError(f"{name} is invalid")
    return selected


def _validate_contrast_shape(value: Any) -> dict[str, Any]:
    identity, body = _validate_identified(
        value,
        id_field="contrast_id",
        name="contrast",
    )
    body = _mapping(
        body,
        name="contrast",
        fields={
            "schema_version",
            "experiment_id",
            "kind",
            "label",
            "left_cell_ids",
            "right_cell_ids",
            "endpoint",
            "aggregation",
            "estimate",
            "interval",
            "interpreted",
            "source_artifact_sha256",
        },
    )
    if body["schema_version"] != CONTRAST_SCHEMA_VERSION:
        raise ResultRegistryError("contrast schema is incompatible")
    for field in ("experiment_id", "kind", "label", "aggregation"):
        _string(body[field], name=f"contrast.{field}")
    if body["kind"] not in {"quality_difference", "cost_ratio"}:
        raise ResultRegistryError("contrast kind is unsupported")
    left = _sequence(body["left_cell_ids"], name="contrast.left_cell_ids")
    right = _sequence(body["right_cell_ids"], name="contrast.right_cell_ids")
    if not left or len(left) != len(right):
        raise ResultRegistryError("contrast cell pairs are empty or unbalanced")
    for side, rows in (("left", left), ("right", right)):
        if len(rows) != len(set(rows)):
            raise ResultRegistryError(f"contrast {side} cells contain duplicates")
        for index, row in enumerate(rows):
            _string(row, name=f"contrast.{side}_cell_ids[{index}]")
    if body["kind"] == "quality_difference":
        _string(body["endpoint"], name="contrast.endpoint")
        if body["aggregation"] not in {
            "point_ev_difference",
            "mean_ev_difference",
        }:
            raise ResultRegistryError("quality contrast aggregation is unsupported")
        if body["estimate"] is None:
            raise ResultRegistryError("quality contrast estimate is absent")
        _finite(body["estimate"], name="contrast.estimate")
        if body["interpreted"] is not True:
            raise ResultRegistryError("quality contrasts must be interpreted")
    else:
        if body["endpoint"] is not None:
            raise ResultRegistryError("cost contrast cannot name a score endpoint")
        if body["aggregation"] != "paired_median_percent_ratio":
            raise ResultRegistryError("cost contrast aggregation is unsupported")
        if body["interpreted"] is True:
            if body["estimate"] is None:
                raise ResultRegistryError("interpreted cost ratio is absent")
            _finite(body["estimate"], name="contrast.estimate", positive=True)
        elif body["estimate"] is not None:
            raise ResultRegistryError(
                "non-interpreted cost ratio must not publish an estimate"
            )
        elif body["interpreted"] is not False:
            raise ResultRegistryError("contrast.interpreted must be boolean")
    _validate_interval(body["interval"], name="contrast.interval")
    _sha256(
        body["source_artifact_sha256"],
        name="contrast.source_artifact_sha256",
    )
    return {"contrast_id": identity, **body}


def _record_mapping(
    value: Any,
    *,
    name: str,
    id_field: str,
    validator: Any,
) -> dict[str, dict[str, Any]]:
    selected = _mapping(value, name=name)
    output: dict[str, dict[str, Any]] = {}
    for key, row in selected.items():
        checked = validator(row)
        if checked[id_field] != key:
            raise ResultRegistryError(f"{name} key differs from embedded identity")
        output[key] = checked
    return output


def _validate_cross_references(
    *,
    sources: Mapping[str, Mapping[str, Any]],
    score_spaces: Mapping[str, Mapping[str, Any]],
    cost_spaces: Mapping[str, Mapping[str, Any]],
    cells: Mapping[str, Mapping[str, Any]],
    contrasts: Mapping[str, Mapping[str, Any]],
) -> None:
    source_scientific_hashes = {
        source["scientific_sha256"] for source in sources.values()
    }
    score_values: dict[str, list[float]] = {}
    for cell_id, cell in cells.items():
        provenance = cell["provenance"]
        if provenance["registry_source_id"] not in sources:
            raise ResultRegistryError(f"{cell_id} references an absent source")
        for endpoint, score in cell["endpoints"].items():
            score_id = score["score_space_id"]
            space = score_spaces.get(score_id)
            if space is None:
                raise ResultRegistryError(
                    f"{cell_id}.{endpoint} references an absent score space"
                )
            if space["endpoint"] != endpoint:
                raise ResultRegistryError(
                    f"{cell_id}.{endpoint} differs from score-space endpoint"
                )
            score_values.setdefault(score_id, []).append(float(score["sst"]))
        cost = cell["cost"]
        if cost is not None:
            space = cost_spaces.get(cost["cost_space_id"])
            if space is None:
                raise ResultRegistryError(f"{cell_id} references an absent cost space")
            chips = float(cost["chip_seconds"])
            wall = float(cost["wall_seconds"])
            chip_count = int(space["chip_count"])
            if not math.isclose(
                chips,
                wall * chip_count,
                rel_tol=1e-9,
                abs_tol=1e-6,
            ):
                raise ResultRegistryError(
                    f"{cell_id} chip-seconds differ from wall × chips"
                )
            throughput_tokens = int(cost["throughput_tokens"])
            if not (
                int(cell["checkpoint_tokens"])
                <= throughput_tokens
                <= int(cell["actual_training_tokens"])
            ):
                raise ResultRegistryError(
                    f"{cell_id} throughput token scope lies outside its "
                    "checkpoint/actual token interval"
                )
            expected_rate = throughput_tokens / wall
            if not math.isclose(
                float(cost["tokens_per_second"]),
                expected_rate,
                rel_tol=1e-9,
                abs_tol=1e-6,
            ):
                raise ResultRegistryError(
                    f"{cell_id} tokens/s differs from tokens/wall"
                )

    for score_id, values in score_values.items():
        policy = score_spaces[score_id]["sst_policy"]
        reference = values[0]
        for value in values[1:]:
            if not math.isclose(
                value,
                reference,
                rel_tol=float(policy["relative_tolerance"]),
                abs_tol=float(policy["absolute_tolerance"]),
            ):
                raise ResultRegistryError(
                    f"{score_id} contains incompatible aggregate SST values"
                )

    for contrast_id, contrast in contrasts.items():
        if contrast["source_artifact_sha256"] not in source_scientific_hashes:
            raise ResultRegistryError(
                f"{contrast_id} references an absent source-artifact hash"
            )
        left = [cells.get(identity) for identity in contrast["left_cell_ids"]]
        right = [cells.get(identity) for identity in contrast["right_cell_ids"]]
        if any(row is None for row in (*left, *right)):
            raise ResultRegistryError(f"{contrast_id} references an absent cell")
        checked_left = [row for row in left if row is not None]
        checked_right = [row for row in right if row is not None]
        if any(
            row["experiment_id"] != contrast["experiment_id"]
            for row in (*checked_left, *checked_right)
        ):
            raise ResultRegistryError(f"{contrast_id} crosses experiment identities")
        for left_cell, right_cell in zip(
            checked_left,
            checked_right,
            strict=True,
        ):
            left_pair = (
                left_cell["panel"],
                left_cell["seed"],
                left_cell["checkpoint_tokens"],
            )
            right_pair = (
                right_cell["panel"],
                right_cell["seed"],
                right_cell["checkpoint_tokens"],
            )
            if left_pair != right_pair:
                raise ResultRegistryError(
                    f"{contrast_id} contains an unpaired "
                    "panel/seed/checkpoint comparison"
                )
        if contrast["kind"] == "quality_difference":
            endpoint = contrast["endpoint"]
            scores = [
                row["endpoints"].get(endpoint)
                for row in (*checked_left, *checked_right)
            ]
            if any(score is None for score in scores):
                raise ResultRegistryError(
                    f"{contrast_id} endpoint is absent from one or more cells"
                )
            score_ids = {
                score["score_space_id"] for score in scores if score is not None
            }
            if len(score_ids) != 1:
                raise ResultRegistryError(f"{contrast_id} crosses score spaces")
            estimate = statistics.fmean(
                row["endpoints"][endpoint]["ev"] for row in checked_left
            ) - statistics.fmean(
                row["endpoints"][endpoint]["ev"] for row in checked_right
            )
            if not math.isclose(
                float(contrast["estimate"]),
                estimate,
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                raise ResultRegistryError(
                    f"{contrast_id} estimate differs from registered cells"
                )
        elif contrast["interpreted"]:
            ratios: list[float] = []
            for left_cell, right_cell in zip(checked_left, checked_right, strict=True):
                left_cost = left_cell["cost"]
                right_cost = right_cell["cost"]
                if left_cost is None or right_cost is None:
                    raise ResultRegistryError(
                        f"{contrast_id} interpreted cost pair lacks measured cost"
                    )
                if left_cost["cost_space_id"] != right_cost["cost_space_id"]:
                    raise ResultRegistryError(
                        f"{contrast_id} interprets a cross-hardware/cost-space pair"
                    )
                ratios.append(
                    100.0
                    * float(left_cost["chip_seconds"])
                    / float(right_cost["chip_seconds"])
                )
            estimate = statistics.median(ratios)
            if not math.isclose(
                float(contrast["estimate"]),
                estimate,
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                raise ResultRegistryError(
                    f"{contrast_id} cost ratio differs from registered cells"
                )


def _validate_contents(value: Mapping[str, Any]) -> dict[str, Any]:
    sources = _record_mapping(
        value["sources"],
        name="registry.sources",
        id_field="source_id",
        validator=validate_source,
    )
    score_spaces = _record_mapping(
        value["score_spaces"],
        name="registry.score_spaces",
        id_field="score_space_id",
        validator=validate_score_space,
    )
    cost_spaces = _record_mapping(
        value["cost_spaces"],
        name="registry.cost_spaces",
        id_field="cost_space_id",
        validator=validate_cost_space,
    )
    cells = _record_mapping(
        value["cells"],
        name="registry.cells",
        id_field="cell_id",
        validator=validate_cell,
    )
    contrasts = _record_mapping(
        value["contrasts"],
        name="registry.contrasts",
        id_field="contrast_id",
        validator=_validate_contrast_shape,
    )
    _validate_cross_references(
        sources=sources,
        score_spaces=score_spaces,
        cost_spaces=cost_spaces,
        cells=cells,
        contrasts=contrasts,
    )
    return {
        "sources": sources,
        "score_spaces": score_spaces,
        "cost_spaces": cost_spaces,
        "cells": cells,
        "contrasts": contrasts,
    }


def make_fragment(
    *,
    sources: Sequence[Mapping[str, Any]],
    score_spaces: Sequence[Mapping[str, Any]],
    cost_spaces: Sequence[Mapping[str, Any]],
    cells: Sequence[Mapping[str, Any]],
    contrasts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the generic ingestion unit used by future closure reports."""

    body = {
        "schema_version": FRAGMENT_SCHEMA_VERSION,
        "sources": {row["source_id"]: dict(row) for row in sources},
        "score_spaces": {row["score_space_id"]: dict(row) for row in score_spaces},
        "cost_spaces": {row["cost_space_id"]: dict(row) for row in cost_spaces},
        "cells": {row["cell_id"]: dict(row) for row in cells},
        "contrasts": {row["contrast_id"]: dict(row) for row in contrasts},
    }
    return validate_fragment(body)


def validate_fragment(value: Any) -> dict[str, Any]:
    selected = _mapping(
        value,
        name="fragment",
        fields={
            "schema_version",
            "sources",
            "score_spaces",
            "cost_spaces",
            "cells",
            "contrasts",
        },
    )
    if selected["schema_version"] != FRAGMENT_SCHEMA_VERSION:
        raise ResultRegistryError("registry fragment schema is incompatible")
    contents = _validate_contents(selected)
    return {"schema_version": FRAGMENT_SCHEMA_VERSION, **contents}


def build_registry(fragments: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Merge validated fragments, allowing only byte-identical duplicates."""

    merged: dict[str, dict[str, Any]] = {
        "sources": {},
        "score_spaces": {},
        "cost_spaces": {},
        "cells": {},
        "contrasts": {},
    }
    for index, fragment in enumerate(fragments):
        selected = validate_fragment(fragment)
        for kind in merged:
            for identity, row in selected[kind].items():
                previous = merged[kind].get(identity)
                if previous is not None and previous != row:
                    raise ResultRegistryError(
                        f"fragment {index} changes existing {kind} {identity}"
                    )
                merged[kind][identity] = row
    body = {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        **{kind: dict(sorted(rows.items())) for kind, rows in merged.items()},
    }
    registry = {**body, "registry_sha256": canonical_sha256(body)}
    return validate_registry(registry)


def validate_registry(value: Any) -> dict[str, Any]:
    selected = _mapping(
        value,
        name="registry",
        fields={
            "schema_version",
            "sources",
            "score_spaces",
            "cost_spaces",
            "cells",
            "contrasts",
            "registry_sha256",
        },
    )
    claimed = _sha256(
        selected.pop("registry_sha256"),
        name="registry.registry_sha256",
    )
    if selected["schema_version"] != REGISTRY_SCHEMA_VERSION:
        raise ResultRegistryError("registry schema is incompatible")
    if canonical_sha256(selected) != claimed:
        raise ResultRegistryError("registry self-hash is invalid")
    contents = _validate_contents(selected)
    return {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        **contents,
        "registry_sha256": claimed,
    }


def load_registry(path: str | Path) -> dict[str, Any]:
    selected = Path(path).expanduser().resolve()
    try:
        value = json.loads(selected.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ResultRegistryError(f"cannot read registry: {selected}") from error
    return validate_registry(value)


def write_registry(path: str | Path, value: Mapping[str, Any]) -> Path:
    """Write a registry create-only; an exact replay is harmless."""

    selected = validate_registry(value)
    output = Path(path).expanduser().resolve()
    payload = canonical_json_bytes(selected) + b"\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    except FileExistsError:
        try:
            observed = output.read_bytes()
        except OSError as error:
            raise ResultRegistryError(f"cannot replay registry: {output}") from error
        if observed != payload:
            raise ResultRegistryError(
                f"refusing to replace a different registry: {output}"
            )
        return output
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    return output


def _source_provenance(
    source_id: str,
    *,
    result_sha256: str,
    checkpoint_sha256: str | None,
    source_revision: str | None,
) -> dict[str, Any]:
    return {
        "registry_source_id": source_id,
        "result_sha256": result_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "source_revision": source_revision,
    }


def _score_row(
    score: Mapping[str, Any],
    *,
    score_space_id: str,
    sst_sha256: str | None = None,
) -> dict[str, Any]:
    return {
        "score_space_id": score_space_id,
        "sse": float(score["sse"]),
        "sst": float(score["sst"]),
        "nmse": float(score["nmse"]),
        "ev": float(score["ev"]),
        "sst_sha256": sst_sha256,
    }


def _hardware_generation(accelerator_type: str) -> str:
    if accelerator_type.startswith("v4"):
        return "v4"
    if accelerator_type.startswith("v5"):
        return "v5e"
    if accelerator_type.startswith("v6"):
        return "v6e"
    return accelerator_type


def _exp4_geometry(panel: str, condition: str) -> tuple[str, dict[str, Any]]:
    activation = (
        "relu-strict-positive-topk2"
        if panel == "sae"
        else "softplus-exact-positive-topk2"
    )
    if condition in {"A", "R"}:
        identity = (
            "true-expert-id"
            if condition == "A"
            else "per-sequence-randomized-expert-id"
        )
        banks, width = 128, 256
    elif condition == "B":
        identity, banks, width = "router-rank", 8, 256
    else:
        identity, banks, width = "router-rank", 8, 4_096
    geometry = {
        "bank_identity": identity,
        "bank_count": banks,
        "bank_width": width,
        "total_features": banks * width,
        "candidates_per_token": 8 * width,
        "selected_slots": 16,
        "positive_l0": 16.0,
        "trainable_parameters": None,
    }
    return activation, geometry


def _qualified_exp4_condition(panel: str, condition: str) -> str:
    activation, geometry = _exp4_geometry(panel, condition)
    return (
        f"experiment4.{panel}.{condition.lower()}."
        f"{geometry['bank_identity']}.{geometry['bank_count']}x"
        f"{geometry['bank_width']}.{activation}"
    )


def fragment_from_experiment4_report(
    value: Mapping[str, Any],
    *,
    file_sha256: str | None = None,
) -> dict[str, Any]:
    """Ingest the sealed Experiment-4 six-point learning-horizon report."""

    from moe_sae.horizon_curve_report import render_horizon_curve_report

    report = copy.deepcopy(dict(value))
    render_horizon_curve_report(report)
    source = make_source(
        kind="experiment4_learning_horizon_report",
        schema_version=report["schema_version"],
        scientific_sha256=report["report_sha256"],
        file_sha256=file_sha256,
    )
    score_spaces: dict[tuple[str, str], dict[str, Any]] = {}
    cost_spaces: dict[str, dict[str, Any]] = {}
    cells: list[dict[str, Any]] = []
    by_role: dict[tuple[str, str, int], str] = {}
    for panel, panel_row in report["scientific_models"].items():
        for endpoint, target_id in {
            "component": "gemma4.block15.selected-expert-component",
            "routed": "gemma4.block15.routed-branch-recomposition",
        }.items():
            score_spaces[(panel, endpoint)] = make_score_space(
                model_id=MODEL_ID,
                model_checkpoint_sha256=None,
                layer_one_indexed=LAYER_ONE_INDEXED,
                target_id=target_id,
                endpoint=endpoint,
                evaluation_manifest_sha256=EVAL_TARGET_MANIFEST_SHA256,
                token_identity_sha256=None,
                centering_id=f"{report['protocol_id']}:{panel}:{endpoint}:centering",
                sst_id=f"{report['protocol_id']}:{panel}:{endpoint}:common-sst",
            )
        for point in panel_row["curve"]:
            milestone = int(point["target_training_tokens"])
            for condition, row in point["conditions"].items():
                hardware = row["hardware"]
                training = row["training"]
                accelerator = hardware["accelerator_type"]
                cost_space = make_cost_space(
                    measurement_id="marginal-training-chip-seconds-v1",
                    accelerator_type=accelerator,
                    accelerator_generation=hardware["hardware_generation"],
                    chip_count=int(hardware["chip_count"]),
                    included=[
                        "preprocessing",
                        "synchronized-optimizer-steps",
                        "input-stalls",
                    ],
                    excluded=[
                        "compilation",
                        "checkpoint-serialization",
                        "evaluation",
                        "shared-harvest",
                    ],
                    effective_batch_tokens=16_384,
                    microbatch_tokens=4_096,
                )
                cost_spaces[cost_space["cost_space_id"]] = cost_space
                activation, geometry = _exp4_geometry(panel, condition)
                actual = int(row["actual_training_tokens"])
                cell = make_cell(
                    experiment_id="experiment4-learning-horizon",
                    protocol_id=report["protocol_id"],
                    panel=panel,
                    condition_id=_qualified_exp4_condition(panel, condition),
                    condition_label=condition,
                    seed=0,
                    checkpoint_tokens=milestone,
                    actual_training_tokens=actual,
                    encoder_source_id=(
                        "selected-expert-output"
                        if panel == "sae"
                        else "common-expert-input"
                    ),
                    activation_policy_id=activation,
                    geometry=geometry,
                    endpoints={
                        endpoint: _score_row(
                            row["held_out_evaluation"][endpoint],
                            score_space_id=score_spaces[(panel, endpoint)][
                                "score_space_id"
                            ],
                        )
                        for endpoint in ("component", "routed")
                    },
                    cost={
                        "cost_space_id": cost_space["cost_space_id"],
                        "chip_seconds": float(training["marginal_chip_seconds"]),
                        "wall_seconds": float(training["marginal_wall_seconds"]),
                        "throughput_tokens": min(
                            (milestone, actual),
                            key=lambda tokens: abs(
                                float(training["tokens_per_second"])
                                * float(training["marginal_wall_seconds"])
                                - tokens
                            ),
                        ),
                        "tokens_per_second": float(training["tokens_per_second"]),
                        "useful_mfu": training["useful_mfu"],
                    },
                    provenance=_source_provenance(
                        source["source_id"],
                        result_sha256=point["source_result_sha256"],
                        checkpoint_sha256=None,
                        source_revision=None,
                    ),
                )
                cells.append(cell)
                by_role[(panel, condition, milestone)] = cell["cell_id"]
    contrasts: list[dict[str, Any]] = []
    for panel in report["scientific_models"]:
        for milestone in (2_000_000, 5_000_000, 10_000_000):
            for left, right in (("A", "R"), ("A", "B"), ("A", "F"), ("F", "B")):
                left_id = by_role[(panel, left, milestone)]
                right_id = by_role[(panel, right, milestone)]
                cell_index = {cell["cell_id"]: cell for cell in cells}
                estimate = (
                    cell_index[left_id]["endpoints"]["component"]["ev"]
                    - cell_index[right_id]["endpoints"]["component"]["ev"]
                )
                contrasts.append(
                    make_contrast(
                        experiment_id="experiment4-learning-horizon",
                        kind="quality_difference",
                        label=f"{panel}:{milestone}:{left}-{right}:component",
                        left_cell_ids=[left_id],
                        right_cell_ids=[right_id],
                        endpoint="component",
                        aggregation="point_ev_difference",
                        estimate=estimate,
                        interval=None,
                        interpreted=True,
                        source_artifact_sha256=report["report_sha256"],
                    )
                )
            left_id = by_role[(panel, "A", milestone)]
            right_id = by_role[(panel, "F", milestone)]
            cell_index = {cell["cell_id"]: cell for cell in cells}
            comparable = (
                cell_index[left_id]["cost"]["cost_space_id"]
                == cell_index[right_id]["cost"]["cost_space_id"]
            )
            ratio = (
                100.0
                * cell_index[left_id]["cost"]["chip_seconds"]
                / cell_index[right_id]["cost"]["chip_seconds"]
                if comparable
                else None
            )
            contrasts.append(
                make_contrast(
                    experiment_id="experiment4-learning-horizon",
                    kind="cost_ratio",
                    label=f"{panel}:{milestone}:A/F:marginal-cost",
                    left_cell_ids=[left_id],
                    right_cell_ids=[right_id],
                    endpoint=None,
                    aggregation="paired_median_percent_ratio",
                    estimate=ratio,
                    interval=None,
                    interpreted=comparable,
                    source_artifact_sha256=report["report_sha256"],
                )
            )
    return make_fragment(
        sources=[source],
        score_spaces=list(score_spaces.values()),
        cost_spaces=list(cost_spaces.values()),
        cells=cells,
        contrasts=contrasts,
    )


def _core_geometry(condition: str, row: Mapping[str, Any]) -> dict[str, Any]:
    if condition in {"H", "R"}:
        identity, banks, width = (
            ("true-expert-plus-shared", 129, 512)
            if condition == "H"
            else ("randomized-expert-plus-shared", 129, 512)
        )
    else:
        identity, banks, width = "global", 1, int(row["total_features"])
    return {
        "bank_identity": identity,
        "bank_count": banks,
        "bank_width": width,
        "total_features": int(row["total_features"]),
        "candidates_per_token": int(row["candidates_per_token"]),
        "selected_slots": int(row["exact_l0"]),
        "positive_l0": float(row["exact_l0"]),
        "trainable_parameters": (46_145_536 if condition == "B" else 392_237_056),
    }


def fragment_from_experiment5_core(
    report_value: Mapping[str, Any],
    result_values: Sequence[Mapping[str, Any]],
    *,
    file_sha256: str | None = None,
) -> dict[str, Any]:
    """Ingest the core H/R/B/F report and all eighteen parent results."""

    from moe_sae.experiment5_report import (
        CROSS_HARDWARE_SST_RTOL,
        CROSS_HARDWARE_SST_SUM_RTOL,
        validate_experiment5_report,
        validate_experiment5_result,
    )

    report = validate_experiment5_report(report_value)
    validated = [validate_experiment5_result(value) for value in result_values]
    expected_hashes = set(report["parents"]["sae_result_sha256s"]) | set(
        report["parents"]["transcoder_result_sha256s"]
    )
    observed_hashes = {row.result_sha256 for row in validated}
    if len(validated) != 18 or observed_hashes != expected_hashes:
        raise ResultRegistryError(
            "Experiment-5 core results differ from the report's 18 parents"
        )
    source = make_source(
        kind="experiment5_core_report",
        schema_version=report["schema_version"],
        scientific_sha256=report["report_sha256"],
        file_sha256=file_sha256,
    )
    score_spaces: dict[tuple[str, str], dict[str, Any]] = {}
    endpoint_targets = {
        "full_ffn": "gemma4.block15.full-ffn-output-y",
        "routed": "gemma4.block15.routed-branch",
        "shared": "gemma4.block15.shared-branch",
        "oracle_shared_full_ffn": "gemma4.block15.full-ffn-output-y",
        "oracle_routed_full_ffn": "gemma4.block15.full-ffn-output-y",
    }
    for panel in ("sae", "transcoder"):
        for endpoint, target_id in endpoint_targets.items():
            score_spaces[(panel, endpoint)] = make_score_space(
                model_id=MODEL_ID,
                model_checkpoint_sha256=None,
                layer_one_indexed=LAYER_ONE_INDEXED,
                target_id=target_id,
                endpoint=endpoint,
                evaluation_manifest_sha256=EVAL_TARGET_MANIFEST_SHA256,
                token_identity_sha256=None,
                centering_id=f"{report['protocol_id']}:{panel}:{endpoint}:centering",
                sst_id=f"{report['protocol_id']}:{panel}:{endpoint}:common-sst",
                sst_mode="numerically_equivalent",
                sst_rtol=(
                    CROSS_HARDWARE_SST_RTOL
                    if endpoint in {"routed", "shared"}
                    else CROSS_HARDWARE_SST_SUM_RTOL
                ),
            )
    cost_spaces: dict[str, dict[str, Any]] = {}
    cells: list[dict[str, Any]] = []
    by_role: dict[tuple[str, str, int, int], str] = {}
    for checked in validated:
        body = checked.body
        panel = checked.panel
        seed = checked.seed
        milestone = checked.milestone
        for condition, condition_row in body["conditions"].items():
            training = condition_row["training"]
            cost_space = make_cost_space(
                measurement_id="marginal-training-chip-seconds-v1",
                accelerator_type=training["accelerator_type"],
                accelerator_generation=training["accelerator_generation"],
                chip_count=int(training["chip_count"]),
                included=[
                    "preprocessing",
                    "synchronized-optimizer-steps",
                    "input-stalls",
                ],
                excluded=[
                    "compilation",
                    "checkpoint-serialization",
                    "evaluation",
                    "shared-harvest",
                ],
                effective_batch_tokens=16_384,
                microbatch_tokens=4_096,
            )
            cost_spaces[cost_space["cost_space_id"]] = cost_space
            endpoint_rows = {}
            for endpoint in endpoint_targets:
                score = condition_row[endpoint]
                if score is not None:
                    endpoint_rows[endpoint] = _score_row(
                        score,
                        score_space_id=score_spaces[(panel, endpoint)][
                            "score_space_id"
                        ],
                    )
            condition_id = (
                f"experiment5-core.{panel}.{condition.lower()}."
                f"{_core_geometry(condition, condition_row)['bank_identity']}."
                "softplus-exact-positive"
            )
            cell = make_cell(
                experiment_id="experiment5-core-full-ffn",
                protocol_id=report["protocol_id"],
                panel=panel,
                condition_id=condition_id,
                condition_label=condition,
                seed=seed,
                checkpoint_tokens=milestone,
                actual_training_tokens=int(body["checkpoint_actual_tokens"][condition]),
                encoder_source_id=(
                    "routed-and-shared-branch-operands"
                    if panel == "sae" and condition in {"H", "R"}
                    else (
                        "full-ffn-output-y" if panel == "sae" else "block15-ffn-input"
                    )
                ),
                activation_policy_id="softplus-exact-positive-topk",
                geometry=_core_geometry(condition, condition_row),
                endpoints=endpoint_rows,
                cost={
                    "cost_space_id": cost_space["cost_space_id"],
                    "chip_seconds": float(training["marginal_chip_seconds"]),
                    "wall_seconds": float(training["wall_seconds"]),
                    "throughput_tokens": int(
                        body["checkpoint_actual_tokens"][condition]
                    ),
                    "tokens_per_second": float(training["tokens_per_second"]),
                    "useful_mfu": training["useful_mfu"],
                },
                provenance=_source_provenance(
                    source["source_id"],
                    result_sha256=checked.result_sha256,
                    checkpoint_sha256=None,
                    source_revision=body["source_revision"],
                ),
            )
            cells.append(cell)
            by_role[(panel, condition, seed, milestone)] = cell["cell_id"]
    contrasts: list[dict[str, Any]] = []
    for panel, panel_report in report["panels"].items():
        for label, effect in panel_report["ten_million_contrasts"].items():
            left, right = label.split("-", maxsplit=1)
            contrasts.append(
                make_contrast(
                    experiment_id="experiment5-core-full-ffn",
                    kind="quality_difference",
                    label=f"{panel}:10000000:{label}:full_ffn",
                    left_cell_ids=[
                        by_role[(panel, left, seed, 10_000_000)] for seed in (0, 1, 2)
                    ],
                    right_cell_ids=[
                        by_role[(panel, right, seed, 10_000_000)] for seed in (0, 1, 2)
                    ],
                    endpoint="full_ffn",
                    aggregation="mean_ev_difference",
                    estimate=float(effect["mean"]),
                    interval={
                        "lower": float(effect["lower"]),
                        "upper": float(effect["upper"]),
                        "confidence_level": float(effect["confidence_level"]),
                        "unit": "crossed_seed_and_packed_sequence",
                        "replicates": int(effect["bootstrap_replicates"]),
                    },
                    interpreted=True,
                    source_artifact_sha256=report["report_sha256"],
                )
            )
        for cost_row in panel_report["equal_token_cost"]:
            milestone = int(cost_row["checkpoint_tokens"])
            comparable = bool(cost_row["all_same_generation"])
            contrasts.append(
                make_contrast(
                    experiment_id="experiment5-core-full-ffn",
                    kind="cost_ratio",
                    label=f"{panel}:{milestone}:H/F:marginal-cost",
                    left_cell_ids=[
                        by_role[(panel, "H", seed, milestone)] for seed in (0, 1, 2)
                    ],
                    right_cell_ids=[
                        by_role[(panel, "F", seed, milestone)] for seed in (0, 1, 2)
                    ],
                    endpoint=None,
                    aggregation="paired_median_percent_ratio",
                    estimate=(
                        float(cost_row["median_h_over_f_percent"])
                        if comparable
                        else None
                    ),
                    interval=None,
                    interpreted=comparable,
                    source_artifact_sha256=report["report_sha256"],
                )
            )
    return make_fragment(
        sources=[source],
        score_spaces=list(score_spaces.values()),
        cost_spaces=list(cost_spaces.values()),
        cells=cells,
        contrasts=contrasts,
    )


def fragment_from_v3_report(
    value: Mapping[str, Any],
    *,
    file_sha256: str | None = None,
) -> dict[str, Any]:
    """Ingest the sealed Experiment-5 v3 activation-policy report."""

    from moe_sae.experiment5_routed_ladder_v3_report import (
        RELU_CONDITION,
        SOFTPLUS_CONDITION,
        validate_v3_primary_report,
    )

    report = validate_v3_primary_report(value)
    source = make_source(
        kind="experiment5_v3_activation_policy_report",
        schema_version=report["schema_version"],
        scientific_sha256=report["report_sha256"],
        file_sha256=file_sha256,
    )
    common = report["common_evaluation_identity"]
    score_spaces: dict[tuple[str, str], dict[str, Any]] = {}
    for panel in ("sae", "transcoder"):
        for endpoint, target in {
            "component": "gemma4.block15.selected-expert-component",
            "routed": "gemma4.block15.routed-branch-recomposition",
            "oracle_shared": "gemma4.block15.full-ffn-oracle-shared",
        }.items():
            endpoint_sst = common["endpoint_sst"][endpoint]
            score_spaces[(panel, endpoint)] = make_score_space(
                model_id=MODEL_ID,
                model_checkpoint_sha256=None,
                layer_one_indexed=LAYER_ONE_INDEXED,
                target_id=target,
                endpoint=endpoint,
                evaluation_manifest_sha256=common["capture_manifest_sha256"],
                token_identity_sha256=common["token_identity_sha256"],
                centering_id=(f"{report['protocol_id']}:{panel}:{endpoint}:centering"),
                sst_id=endpoint_sst["reference_sst_sha256"],
                sst_mode="numerically_equivalent",
                sst_rtol=float(endpoint_sst["aggregate_relative_tolerance"]),
            )
    costs_by_role = {
        (row["panel"], row["condition"], int(row["seed"])): row
        for row in report["training_cost_comparison"]["trajectory_costs"]
    }
    cost_spaces: dict[str, dict[str, Any]] = {}
    cells: list[dict[str, Any]] = []
    by_role: dict[tuple[str, str, int, int], str] = {}
    for row in report["cells"]:
        panel = row["panel"]
        condition = row["condition"]
        seed = int(row["seed"])
        milestone = int(row["checkpoint_tokens"])
        sparsity = row["sparsity"]
        is_relu = condition == RELU_CONDITION
        cost = None
        if milestone == 10_000_000:
            cost_row = costs_by_role[(panel, condition, seed)]
            accelerator = cost_row["accelerator_type"]
            cost_space = make_cost_space(
                measurement_id=cost_row["chip_seconds_scope"],
                accelerator_type=accelerator,
                accelerator_generation=_hardware_generation(accelerator),
                chip_count=int(cost_row["chip_count"]),
                included=[
                    "preprocessing",
                    "synchronized-optimizer-steps",
                    "input-stalls",
                ],
                excluded=list(cost_row["chip_seconds_excludes"]),
                effective_batch_tokens=16_384,
                microbatch_tokens=4_096,
            )
            cost_spaces[cost_space["cost_space_id"]] = cost_space
            cost = {
                "cost_space_id": cost_space["cost_space_id"],
                "chip_seconds": float(cost_row["chip_seconds"]),
                "wall_seconds": float(cost_row["marginal_wall_seconds"]),
                "throughput_tokens": int(cost_row["tokens"]),
                "tokens_per_second": float(cost_row["tokens_per_second"]),
                "useful_mfu": None,
            }
            actual_tokens = int(cost_row["tokens"])
        else:
            actual_tokens = milestone
        condition_id = f"experiment5-v3.{panel}.{condition}." + (
            "relu-inclusive-zero-topk4" if is_relu else "softplus-exact-positive-topk4"
        )
        cell = make_cell(
            experiment_id="experiment5-v3-activation-policy",
            protocol_id=report["protocol_id"],
            panel=panel,
            condition_id=condition_id,
            condition_label=condition,
            seed=seed,
            checkpoint_tokens=milestone,
            actual_training_tokens=actual_tokens,
            encoder_source_id=(
                "selected-expert-output" if panel == "sae" else "common-expert-input"
            ),
            activation_policy_id=(
                "relu-inclusive-zero-topk"
                if is_relu
                else "softplus-exact-positive-topk"
            ),
            geometry={
                "bank_identity": "true-expert-id",
                "bank_count": 128,
                "bank_width": 512,
                "total_features": int(sparsity["total_features"]),
                "candidates_per_token": 4_096,
                "selected_slots": int(sparsity["selected_slot_count"]),
                "positive_l0": float(sparsity["positive_l0"]),
                "trainable_parameters": None,
            },
            endpoints={
                endpoint: _score_row(
                    score,
                    score_space_id=score_spaces[(panel, endpoint)]["score_space_id"],
                    sst_sha256=score["sst_sha256"],
                )
                for endpoint, score in row["endpoints"].items()
            },
            cost=cost,
            provenance=_source_provenance(
                source["source_id"],
                result_sha256=row["result_sha256"],
                checkpoint_sha256=row["parent_checkpoint_pointer_sha256"],
                source_revision=row["training_source_revision"],
            ),
        )
        cells.append(cell)
        by_role[(panel, condition, seed, milestone)] = cell["cell_id"]
    contrasts: list[dict[str, Any]] = []
    for row in report["contrasts"]:
        panel = row["panel"]
        milestone = int(row["checkpoint_tokens"])
        endpoint = row["endpoint"]
        across = row["across_seed"]
        contrasts.append(
            make_contrast(
                experiment_id="experiment5-v3-activation-policy",
                kind="quality_difference",
                label=(
                    f"{panel}:{milestone}:{SOFTPLUS_CONDITION}-"
                    f"{RELU_CONDITION}:{endpoint}"
                ),
                left_cell_ids=[
                    by_role[(panel, SOFTPLUS_CONDITION, seed, milestone)]
                    for seed in (0, 1, 2)
                ],
                right_cell_ids=[
                    by_role[(panel, RELU_CONDITION, seed, milestone)]
                    for seed in (0, 1, 2)
                ],
                endpoint=endpoint,
                aggregation="mean_ev_difference",
                estimate=float(across["estimate_mean_ev_difference"]),
                interval={
                    "lower": float(across["ci95_lower"]),
                    "upper": float(across["ci95_upper"]),
                    "confidence_level": 0.95,
                    "unit": "crossed_seed_and_packed_sequence",
                    "replicates": 20_000,
                },
                interpreted=True,
                source_artifact_sha256=report["report_sha256"],
            )
        )
    panel_costs = {
        row["panel"]: row
        for row in report["training_cost_comparison"]["panel_summaries"]
    }
    for panel, row in panel_costs.items():
        comparable = (
            row["comparison_status"] == "comparable_all_seeds_same_accelerator_type"
        )
        contrasts.append(
            make_contrast(
                experiment_id="experiment5-v3-activation-policy",
                kind="cost_ratio",
                label=f"{panel}:10000000:Softplus/ReLU:marginal-cost",
                left_cell_ids=[
                    by_role[(panel, SOFTPLUS_CONDITION, seed, 10_000_000)]
                    for seed in (0, 1, 2)
                ],
                right_cell_ids=[
                    by_role[(panel, RELU_CONDITION, seed, 10_000_000)]
                    for seed in (0, 1, 2)
                ],
                endpoint=None,
                aggregation="paired_median_percent_ratio",
                estimate=(
                    float(row["median_paired_softplus_to_relu_ratio_percent"])
                    if comparable
                    else None
                ),
                interval=None,
                interpreted=comparable,
                source_artifact_sha256=report["report_sha256"],
            )
        )
    return make_fragment(
        sources=[source],
        score_spaces=list(score_spaces.values()),
        cost_spaces=list(cost_spaces.values()),
        cells=cells,
        contrasts=contrasts,
    )


def read_json(path: str | Path) -> dict[str, Any]:
    selected = Path(path).expanduser().resolve()
    try:
        value = json.loads(selected.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ResultRegistryError(f"cannot read JSON: {selected}") from error
    return _mapping(value, name=str(selected))


def file_sha256(path: str | Path) -> str:
    selected = Path(path).expanduser().resolve()
    try:
        payload = selected.read_bytes()
    except OSError as error:
        raise ResultRegistryError(f"cannot hash file: {selected}") from error
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "CELL_SCHEMA_VERSION",
    "CONTRAST_SCHEMA_VERSION",
    "COST_SPACE_SCHEMA_VERSION",
    "FRAGMENT_SCHEMA_VERSION",
    "REGISTRY_SCHEMA_VERSION",
    "ResultRegistryError",
    "SCORE_SPACE_SCHEMA_VERSION",
    "SOURCE_SCHEMA_VERSION",
    "build_registry",
    "file_sha256",
    "fragment_from_experiment4_report",
    "fragment_from_experiment5_core",
    "fragment_from_v3_report",
    "load_registry",
    "make_cell",
    "make_contrast",
    "make_cost_space",
    "make_fragment",
    "make_score_space",
    "make_source",
    "read_json",
    "validate_cell",
    "validate_cost_space",
    "validate_fragment",
    "validate_registry",
    "validate_score_space",
    "validate_source",
    "write_registry",
]
