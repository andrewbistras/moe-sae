"""Immutable panel-native A256 score context for Experiment-5 v3.

The six source score rows already exist in the completed Experiment-4 direct
horizon: SAE/transcoder at 2M, 5M, and 10M training tokens.  This module does
not evaluate or reinterpret those checkpoints.  It validates their complete
lineage against the v3 A256 provenance addendum and seals only the historical
``A`` rows as descriptive ``A256Native`` context.

The two panels intentionally retain their native activation semantics:

* SAE: ``A256ReLU`` with strict-positive TopK;
* transcoder: ``A256Softplus`` with exact-positive TopK.

They are therefore never pooled, never treated as an activation-policy
contrast, and never admitted to the primary A512 inference.
"""

from __future__ import annotations

import copy
import hashlib
import math
import os
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from moe_sae.contribution_artifacts import canonical_sha256
from moe_sae.contribution_transcoder_report import (
    validate_transcoder_scientific_result,
)
from moe_sae.direct_horizon_campaign import (
    PROTOCOL_ID as EXPERIMENT4_PROTOCOL_ID,
)
from moe_sae.direct_horizon_campaign import (
    SAE_EVALUATION_SCHEMA_VERSION,
    TRANSCODER_EVALUATION_SCHEMA_VERSION,
    DirectCampaignPlan,
    inspect_evaluation_completion,
)
from moe_sae.experiment5_campaign import _read_self_hashed_json
from moe_sae.experiment5_routed_ladder_v3_config import (
    A256_NATIVE_ACTIVATION,
    A256_NATIVE_LABEL,
    A256_NATIVE_SELECTION_POLICY,
    PROTOCOL_ID,
)
from moe_sae.experiment5_routed_ladder_v3_provenance import (
    A256_PROVENANCE_ADDENDUM_ID,
    MILESTONES,
    A256AddendumPanel,
    parse_a256_provenance_addendum,
    validate_a256_addendum_panel,
)
from moe_sae.horizon_curve_report import (
    HorizonCurveReportError,
    _validate_sae_result,
)

A256_CONTEXT_SCORE_MANIFEST_SCHEMA_VERSION = (
    "gemma4-experiment5-routed-activation-ladder-v3-"
    "a256native-context-score-manifest-v1"
)
A256_CONTEXT_SCORE_MANIFEST_ID = (
    "gemma4-experiment5-v3-a256native-context-scores-20260729"
)
PANELS = ("sae", "transcoder")
ENDPOINTS = ("component", "routed")
EVALUATION_TOKENS = 250_000

_SCOPE = {
    "condition": "A256Native",
    "evidentiary_role": "panel_native_descriptive_score_context_only",
    "included_in_primary_a512_inference": False,
    "included_in_primary_a512_estimand": False,
    "cross_panel_pooling_allowed": False,
    "across_seed_inference_allowed": False,
    "cross_panel_activation_contrast_allowed": False,
    "allowed_use": "within_panel_historical_learning_curve_context",
    "prohibited_uses": [
        "matched_a512_activation_policy_contrast",
        "cross_panel_activation_policy_contrast",
        "three_seed_inference",
        "pooled_a256_estimate",
        "replacement_for_any_a512_cell",
    ],
}
_TOKEN_IDENTITY_POLICY = (
    "The target capture SHA-256 and retained token count are common across "
    "both panels. Ordered token and SST-content hashes are panel-local "
    "representations because the historical evaluators serialized and "
    "aggregated the same capture in different row orders; no cross-panel "
    "byte-hash equality or pooling is asserted."
)


class A256ContextScoreError(RuntimeError):
    """The historical A256 score evidence is incomplete or inconsistent."""


def _exact(value: Any, fields: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        observed = sorted(value) if isinstance(value, Mapping) else type(value)
        raise A256ContextScoreError(
            f"{name} fields changed: expected={sorted(fields)}, "
            f"observed={observed}"
        )
    return value


def _sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise A256ContextScoreError(f"{name} must be a lowercase SHA-256")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise A256ContextScoreError(f"{name} must be a positive integer")
    return int(value)


def _finite(value: Any, name: str, *, nonnegative: bool = False) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or (nonnegative and float(value) < 0)
    ):
        qualifier = " nonnegative" if nonnegative else ""
        raise A256ContextScoreError(
            f"{name} must be a finite{qualifier} number"
        )
    return float(value)


def _score(value: Any, name: str) -> dict[str, float]:
    row = _exact(value, {"sse", "sst", "nmse", "ev"}, name)
    sse = _finite(row["sse"], f"{name}.sse", nonnegative=True)
    sst = _finite(row["sst"], f"{name}.sst")
    nmse = _finite(row["nmse"], f"{name}.nmse")
    ev = _finite(row["ev"], f"{name}.ev")
    if sst <= 0:
        raise A256ContextScoreError(f"{name}.sst must be positive")
    expected_nmse = sse / sst
    if (
        not math.isclose(nmse, expected_nmse, rel_tol=1e-12, abs_tol=1e-12)
        or not math.isclose(ev, 1.0 - nmse, rel_tol=1e-12, abs_tol=1e-12)
    ):
        raise A256ContextScoreError(
            f"{name} does not satisfy NMSE=SSE/SST and EV=1-NMSE"
        )
    return {"sse": sse, "sst": sst, "nmse": nmse, "ev": ev}


def _optional_sha256(value: Any, name: str) -> str | None:
    return None if value is None else _sha256(value, name)


def _source_result_evidence(value: Any, name: str) -> dict[str, str]:
    row = _exact(
        value,
        {
            "controller_path",
            "remote_path",
            "file_sha256",
            "result_sha256",
        },
        name,
    )
    controller = row["controller_path"]
    remote = row["remote_path"]
    if (
        not isinstance(controller, str)
        or not Path(controller).is_absolute()
        or not isinstance(remote, str)
        or not PurePosixPath(remote).is_absolute()
    ):
        raise A256ContextScoreError(
            f"{name} paths must be absolute controller/worker paths"
        )
    return {
        "controller_path": controller,
        "remote_path": remote,
        "file_sha256": _sha256(
            row["file_sha256"], f"{name}.file_sha256"
        ),
        "result_sha256": _sha256(
            row["result_sha256"], f"{name}.result_sha256"
        ),
    }


def _parent_hashes(value: Any, name: str) -> dict[str, str | None]:
    fields = {
        "target_capture_manifest_sha256",
        "source_capture_manifest_sha256",
        "training_target_manifest_sha256",
        "training_source_manifest_sha256",
        "recomposition_sidecar_sha256",
        "run_contract_sha256",
        "run_contract_file_sha256",
        "training_result_sha256",
        "training_result_file_sha256",
        "checkpoint_pointer_sha256",
        "checkpoint_pointer_file_sha256",
        "evaluation_receipt_sha256",
        "common_sst_receipt_sha256",
        "common_sst_receipt_file_sha256",
        "score_artifact_sha256",
        "ordered_token_identity_sha256",
        "component_sst_content_sha256",
        "routed_sst_content_sha256",
    }
    row = _exact(value, fields, name)
    nullable = {
        "source_capture_manifest_sha256",
        "training_source_manifest_sha256",
        "recomposition_sidecar_sha256",
        "common_sst_receipt_file_sha256",
    }
    result: dict[str, str | None] = {}
    for field in fields:
        raw = row[field]
        result[field] = (
            _optional_sha256(raw, f"{name}.{field}")
            if field in nullable
            else _sha256(raw, f"{name}.{field}")
        )
    return result


def _parse_cell(value: Any, index: int) -> dict[str, Any]:
    name = f"a256_context_score_manifest.cells[{index}]"
    row = _exact(
        value,
        {
            "panel",
            "condition",
            "source_condition",
            "source_label",
            "seed",
            "checkpoint_tokens",
            "checkpoint_actual_tokens",
            "encoder_activation",
            "selection_policy",
            "source_config_semantic_sha256",
            "source_config_file_sha256",
            "source_result",
            "parent_hashes",
            "endpoints",
            "sparsity",
        },
        name,
    )
    panel = row["panel"]
    milestone = row["checkpoint_tokens"]
    if (
        panel not in PANELS
        or row["condition"] != "A256Native"
        or row["source_condition"] != "A"
        or row["source_label"] != A256_NATIVE_LABEL.get(str(panel))
        or row["seed"] != 0
        or milestone not in MILESTONES
        or row["encoder_activation"]
        != A256_NATIVE_ACTIVATION.get(str(panel))
        or row["selection_policy"]
        != A256_NATIVE_SELECTION_POLICY.get(str(panel))
    ):
        raise A256ContextScoreError(
            f"{name} is not panel-native A256 seed-zero context"
        )
    actual = _positive_int(
        row["checkpoint_actual_tokens"],
        f"{name}.checkpoint_actual_tokens",
    )
    if actual < int(milestone):
        raise A256ContextScoreError(
            f"{name} checkpoint precedes its nominal milestone"
        )
    endpoints = _exact(row["endpoints"], set(ENDPOINTS), f"{name}.endpoints")
    checked_endpoints = {
        endpoint: _score(
            endpoints[endpoint],
            f"{name}.endpoints.{endpoint}",
        )
        for endpoint in ENDPOINTS
    }
    sparsity = _exact(
        row["sparsity"],
        {
            "l0_per_contribution",
            "total_l0",
            "dead_rate",
            "fired_feature_fraction",
        },
        f"{name}.sparsity",
    )
    l0_per = _positive_int(
        sparsity["l0_per_contribution"],
        f"{name}.sparsity.l0_per_contribution",
    )
    total_l0 = _positive_int(
        sparsity["total_l0"],
        f"{name}.sparsity.total_l0",
    )
    dead = _finite(
        sparsity["dead_rate"],
        f"{name}.sparsity.dead_rate",
        nonnegative=True,
    )
    fired = _finite(
        sparsity["fired_feature_fraction"],
        f"{name}.sparsity.fired_feature_fraction",
        nonnegative=True,
    )
    if (
        l0_per != 2
        or total_l0 != 16
        or dead > 1
        or fired > 1
        or not math.isclose(dead + fired, 1.0, abs_tol=1e-10)
    ):
        raise A256ContextScoreError(
            f"{name} does not preserve native A256 sparsity"
        )
    return {
        **copy.deepcopy(dict(row)),
        "checkpoint_tokens": int(milestone),
        "checkpoint_actual_tokens": actual,
        "source_config_semantic_sha256": _sha256(
            row["source_config_semantic_sha256"],
            f"{name}.source_config_semantic_sha256",
        ),
        "source_config_file_sha256": _sha256(
            row["source_config_file_sha256"],
            f"{name}.source_config_file_sha256",
        ),
        "source_result": _source_result_evidence(
            row["source_result"], f"{name}.source_result"
        ),
        "parent_hashes": _parent_hashes(
            row["parent_hashes"], f"{name}.parent_hashes"
        ),
        "endpoints": checked_endpoints,
        "sparsity": {
            "l0_per_contribution": l0_per,
            "total_l0": total_l0,
            "dead_rate": dead,
            "fired_feature_fraction": fired,
        },
    }


def parse_a256_context_score_manifest(
    value: Any,
) -> Mapping[str, Any]:
    """Validate one self-hashed six-cell context-score manifest."""

    selected = _exact(
        value,
        {
            "schema_version",
            "manifest_id",
            "source_protocol_id",
            "target_protocol_id",
            "source_plan_sha256",
            "provenance_addendum",
            "scientific_scope",
            "shared_held_out_identity",
            "cells",
            "manifest_sha256",
        },
        "a256_context_score_manifest",
    )
    if (
        selected["schema_version"]
        != A256_CONTEXT_SCORE_MANIFEST_SCHEMA_VERSION
        or selected["manifest_id"] != A256_CONTEXT_SCORE_MANIFEST_ID
        or selected["source_protocol_id"] != EXPERIMENT4_PROTOCOL_ID
        or selected["target_protocol_id"] != PROTOCOL_ID
        or selected["scientific_scope"] != _SCOPE
    ):
        raise A256ContextScoreError(
            "A256 context-score manifest identity or scope changed"
        )
    source_plan_sha = _sha256(
        selected["source_plan_sha256"],
        "a256_context_score_manifest.source_plan_sha256",
    )
    addendum = _exact(
        selected["provenance_addendum"],
        {
            "addendum_id",
            "controller_path",
            "file_sha256",
            "manifest_sha256",
        },
        "a256_context_score_manifest.provenance_addendum",
    )
    if (
        addendum["addendum_id"] != A256_PROVENANCE_ADDENDUM_ID
        or not isinstance(addendum["controller_path"], str)
        or not Path(addendum["controller_path"]).is_absolute()
    ):
        raise A256ContextScoreError(
            "A256 context-score provenance addendum identity changed"
        )
    _sha256(
        addendum["file_sha256"],
        "a256_context_score_manifest.provenance_addendum.file_sha256",
    )
    _sha256(
        addendum["manifest_sha256"],
        "a256_context_score_manifest.provenance_addendum.manifest_sha256",
    )
    shared = _exact(
        selected["shared_held_out_identity"],
        {
            "evaluation_tokens",
            "target_capture_manifest_sha256",
            "component_sst",
            "routed_sst",
            "panel_representations",
            "token_identity_policy",
        },
        "a256_context_score_manifest.shared_held_out_identity",
    )
    if (
        shared["evaluation_tokens"] != EVALUATION_TOKENS
        or shared["token_identity_policy"] != _TOKEN_IDENTITY_POLICY
    ):
        raise A256ContextScoreError(
            "A256 shared held-out token identity changed"
        )
    target_capture = _sha256(
        shared["target_capture_manifest_sha256"],
        "shared_held_out_identity.target_capture_manifest_sha256",
    )
    shared_sst = {
        endpoint: _finite(
            shared[f"{endpoint}_sst"],
            f"shared_held_out_identity.{endpoint}_sst",
        )
        for endpoint in ENDPOINTS
    }
    if any(value <= 0 for value in shared_sst.values()):
        raise A256ContextScoreError("shared held-out SST must be positive")
    representations = _exact(
        shared["panel_representations"],
        set(PANELS),
        "shared_held_out_identity.panel_representations",
    )
    checked_representations: dict[str, Mapping[str, Any]] = {}
    for panel in PANELS:
        representation = _exact(
            representations[panel],
            {
                "ordered_token_identity_sha256",
                "component_sst_content_sha256",
                "routed_sst_content_sha256",
            },
            f"shared_held_out_identity.panel_representations.{panel}",
        )
        for field in representation:
            _sha256(
                representation[field],
                f"panel_representations.{panel}.{field}",
            )
        checked_representations[panel] = representation
    raw_cells = selected["cells"]
    if not isinstance(raw_cells, list):
        raise A256ContextScoreError("A256 context-score cells must be a list")
    cells = [_parse_cell(row, index) for index, row in enumerate(raw_cells)]
    expected = {
        (panel, 0, milestone)
        for panel in PANELS
        for milestone in MILESTONES
    }
    identities = {
        (str(cell["panel"]), int(cell["seed"]), int(cell["checkpoint_tokens"]))
        for cell in cells
    }
    if len(cells) != len(expected) or identities != expected:
        raise A256ContextScoreError(
            "A256 context-score manifest requires SAE/transcoder at "
            "2M/5M/10M"
        )
    for cell in cells:
        panel = str(cell["panel"])
        parents = cell["parent_hashes"]
        if (
            parents["target_capture_manifest_sha256"] != target_capture
            or parents["ordered_token_identity_sha256"]
            != checked_representations[panel][
                "ordered_token_identity_sha256"
            ]
            or parents["component_sst_content_sha256"]
            != checked_representations[panel][
                "component_sst_content_sha256"
            ]
            or parents["routed_sst_content_sha256"]
            != checked_representations[panel][
                "routed_sst_content_sha256"
            ]
            or any(
                cell["endpoints"][endpoint]["sst"] != shared_sst[endpoint]
                for endpoint in ENDPOINTS
            )
        ):
            raise A256ContextScoreError(
                f"A256 {panel} score differs from shared held-out identity"
            )
    body = {
        key: copy.deepcopy(selected[key])
        for key in selected
        if key != "manifest_sha256"
    }
    claimed = _sha256(
        selected["manifest_sha256"],
        "a256_context_score_manifest.manifest_sha256",
    )
    if canonical_sha256(body) != claimed:
        raise A256ContextScoreError(
            "A256 context-score manifest self-hash changed"
        )
    # This assignment makes the source-plan check explicit to static readers;
    # it is also used by the builder to bind the addendum and direct plan.
    body["source_plan_sha256"] = source_plan_sha
    return copy.deepcopy(selected)


def _read_json_result(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        return _read_self_hashed_json(path, hash_field="result_sha256")
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        raise A256ContextScoreError(
            f"cannot read self-hashed scientific result: {path}"
        ) from error


def _checked_embedded_sae_sst(
    result: Mapping[str, Any],
) -> tuple[str, dict[str, str]]:
    raw = result.get("common_sst_canonicalization")
    if not isinstance(raw, Mapping):
        raise A256ContextScoreError("SAE result lacks common SST evidence")
    receipt_sha = _sha256(
        raw.get("receipt_sha256"),
        "SAE common SST receipt",
    )
    unsigned = {key: raw[key] for key in raw if key != "receipt_sha256"}
    if canonical_sha256(unsigned) != receipt_sha:
        raise A256ContextScoreError("SAE common SST receipt self-hash changed")
    hashes = {
        "ordered_token_identity_sha256": _sha256(
            raw.get("token_identity_sha256"),
            "SAE common SST token identity",
        ),
        "component_sst_content_sha256": _sha256(
            raw.get("component_sst_sha256"),
            "SAE component SST content hash",
        ),
        "routed_sst_content_sha256": _sha256(
            raw.get("routed_sst_sha256"),
            "SAE routed SST content hash",
        ),
    }
    return receipt_sha, hashes


def _checked_transcoder_sst_receipt(
    task_result_path: Path,
    task_result_root_remote: str,
    result: Mapping[str, Any],
) -> tuple[str, str, dict[str, str]]:
    path = task_result_path.parent / "common-sst-receipt.json"
    try:
        receipt, payload = _read_self_hashed_json(
            path,
            hash_field="receipt_sha256",
        )
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        raise A256ContextScoreError(
            f"cannot read transcoder common SST receipt: {path}"
        ) from error
    row = _exact(
        receipt,
        {
            "schema_version",
            "score_artifact_sha256",
            "target_manifest_sha256",
            "token_identity_sha256",
            "endpoint_sst_sha256",
            "receipt_sha256",
        },
        "transcoder common SST receipt",
    )
    parents = result.get("parents")
    paired = result.get("paired_scores")
    if (
        row["schema_version"]
        != "gemma4-contribution-transcoder-common-sst-v1"
        or not isinstance(parents, Mapping)
        or not isinstance(paired, Mapping)
        or row["receipt_sha256"] != parents.get("common_sst_receipt_sha256")
        or row["score_artifact_sha256"]
        != paired.get("score_artifact_sha256")
        or row["target_manifest_sha256"]
        != parents.get("eval_target_manifest_sha256")
        or row["token_identity_sha256"]
        != paired.get("token_identity_sha256")
    ):
        raise A256ContextScoreError(
            "transcoder common SST receipt differs from result parents"
        )
    endpoint_hashes = _exact(
        row["endpoint_sst_sha256"],
        {"component", "routed", "oracle_shared"},
        "transcoder common SST endpoint hashes",
    )
    hashes = {
        "ordered_token_identity_sha256": _sha256(
            row["token_identity_sha256"],
            "transcoder common SST token identity",
        ),
        "component_sst_content_sha256": _sha256(
            endpoint_hashes["component"],
            "transcoder component SST content hash",
        ),
        "routed_sst_content_sha256": _sha256(
            endpoint_hashes["routed"],
            "transcoder routed SST content hash",
        ),
    }
    # Validate the remote filename now even though only the exact file digest
    # enters the compact parent matrix.
    remote_path = str(
        PurePosixPath(task_result_root_remote) / "common-sst-receipt.json"
    )
    if not PurePosixPath(remote_path).is_absolute():
        raise A256ContextScoreError(
            "transcoder common SST remote path is not absolute"
        )
    return (
        _sha256(row["receipt_sha256"], "transcoder common SST receipt"),
        hashlib.sha256(payload).hexdigest(),
        hashes,
    )


def _checkpoint_actual_tokens(
    *,
    result: Mapping[str, Any],
    panel: str,
) -> int:
    if panel == "sae":
        actual = result.get("checkpoint_actual_tokens")
    else:
        checkpoint = result.get("learning_curve_checkpoint")
        actual = (
            checkpoint.get("actual_tokens_by_condition")
            if isinstance(checkpoint, Mapping)
            else None
        )
    if not isinstance(actual, Mapping):
        raise A256ContextScoreError(
            f"{panel} result lacks checkpoint actual tokens"
        )
    return _positive_int(actual.get("A"), f"{panel} A actual tokens")


def _validate_pointer_binding(
    panel: A256AddendumPanel,
    milestone: int,
    *,
    checkpoint_sha256: str,
    actual_tokens: int,
) -> tuple[str, str]:
    matches = [
        checkpoint
        for checkpoint in panel.checkpoints
        if checkpoint.checkpoint_tokens == milestone
    ]
    if len(matches) != 1:
        raise A256ContextScoreError(
            f"A256 {panel.panel} lacks one {milestone:,}-token pointer"
        )
    checkpoint = matches[0]
    try:
        pointer, payload = _read_self_hashed_json(
            checkpoint.checkpoint_pointer.controller_path,
            hash_field="pointer_sha256",
        )
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        raise A256ContextScoreError(
            f"cannot read A256 {panel.panel} milestone pointer"
        ) from error
    if (
        pointer.get("pointer_sha256")
        != checkpoint.checkpoint_pointer_sha256
        or checkpoint_sha256 != checkpoint.checkpoint_pointer_sha256
        or pointer.get("tokens_seen") != actual_tokens
        or hashlib.sha256(payload).hexdigest()
        != checkpoint.checkpoint_pointer.sha256
    ):
        raise A256ContextScoreError(
            f"A256 {panel.panel} score is not bound to its {milestone:,} "
            "checkpoint"
        )
    return (
        checkpoint.checkpoint_pointer_sha256,
        checkpoint.checkpoint_pointer.sha256,
    )


def _sparsity(
    value: Mapping[str, Any],
    *,
    name: str,
) -> dict[str, Any]:
    l0_per = _positive_int(
        value.get("l0_per_contribution"),
        f"{name}.l0_per_contribution",
    )
    total_l0 = _positive_int(value.get("total_l0"), f"{name}.total_l0")
    dead = _finite(
        value.get("dead_rate"),
        f"{name}.dead_rate",
        nonnegative=True,
    )
    fired = _finite(
        value.get("fired_feature_fraction"),
        f"{name}.fired_feature_fraction",
        nonnegative=True,
    )
    if (
        l0_per != 2
        or total_l0 != 16
        or dead > 1
        or fired > 1
        or not math.isclose(dead + fired, 1.0, abs_tol=1e-10)
    ):
        raise A256ContextScoreError(f"{name} sparsity changed")
    return {
        "l0_per_contribution": l0_per,
        "total_l0": total_l0,
        "dead_rate": dead,
        "fired_feature_fraction": fired,
    }


def _build_sae_cell(
    *,
    task: Any,
    panel: A256AddendumPanel,
    result: Mapping[str, Any],
    payload: bytes,
    milestone: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        checked = _validate_sae_result(result)
    except HorizonCurveReportError as error:
        raise A256ContextScoreError(
            f"invalid SAE scientific result at {milestone:,} tokens"
        ) from error
    if (
        result.get("schema_version") != SAE_EVALUATION_SCHEMA_VERSION
        or result.get("checkpoint_target_tokens") != milestone
        or result.get("seed") != 0
        or result.get("evaluation_tokens") != EVALUATION_TOKENS
    ):
        raise A256ContextScoreError("SAE scientific result role changed")
    condition_evaluations = result.get("condition_evaluations")
    checkpoint_sha256s = result.get("checkpoint_sha256s")
    endpoints = result.get("endpoints")
    score_references = result.get("score_manifest_references")
    if not all(
        isinstance(value, Mapping)
        for value in (
            condition_evaluations,
            checkpoint_sha256s,
            endpoints,
            score_references,
        )
    ):
        raise A256ContextScoreError("SAE scientific result lacks A evidence")
    condition = condition_evaluations.get("A")
    if not isinstance(condition, Mapping):
        raise A256ContextScoreError("SAE result lacks condition A")
    actual_tokens = _checkpoint_actual_tokens(result=result, panel="sae")
    checkpoint_sha = _sha256(
        checkpoint_sha256s.get("A"), "SAE A checkpoint hash"
    )
    pointer_sha, pointer_file_sha = _validate_pointer_binding(
        panel,
        milestone,
        checkpoint_sha256=checkpoint_sha,
        actual_tokens=actual_tokens,
    )
    if condition.get("checkpoint_sha256") != checkpoint_sha:
        raise A256ContextScoreError(
            "SAE A condition names another checkpoint"
        )
    receipt_sha, representation = _checked_embedded_sae_sst(result)
    checked_endpoints = checked["_checked_endpoints"]
    endpoint_rows = {
        endpoint: _score(
            checked_endpoints[endpoint]["A"],
            f"SAE A {endpoint}",
        )
        for endpoint in ENDPOINTS
    }
    evaluation_capture = _sha256(
        result.get("evaluation_capture_manifest_sha256"),
        "SAE evaluation capture",
    )
    if condition.get("capture_manifest_sha256") != evaluation_capture:
        raise A256ContextScoreError(
            "SAE A condition uses another evaluation capture"
        )
    score_reference = score_references.get("A")
    if not isinstance(score_reference, Mapping):
        raise A256ContextScoreError("SAE A score reference is absent")
    score_sha = _sha256(
        score_reference.get("sha256"), "SAE A score artifact"
    )
    common = result["common_sst_canonicalization"]
    if common.get("canonical_manifest_sha256") != score_sha:
        raise A256ContextScoreError(
            "SAE A score reference differs from common SST parent"
        )
    cell = {
        "panel": "sae",
        "condition": "A256Native",
        "source_condition": "A",
        "source_label": panel.source_label,
        "seed": 0,
        "checkpoint_tokens": milestone,
        "checkpoint_actual_tokens": actual_tokens,
        "encoder_activation": panel.encoder_activation,
        "selection_policy": panel.selection_policy,
        "source_config_semantic_sha256": (
            panel.observed_source_config_semantic_sha256
        ),
        "source_config_file_sha256": (
            panel.observed_source_config_file_sha256
        ),
        "source_result": {
            "controller_path": str(task.completion_path_controller),
            "remote_path": str(
                PurePosixPath(task.result_root_remote)
                / "scientific-result.json"
            ),
            "file_sha256": hashlib.sha256(payload).hexdigest(),
            "result_sha256": result["result_sha256"],
        },
        "parent_hashes": {
            "target_capture_manifest_sha256": evaluation_capture,
            "source_capture_manifest_sha256": None,
            "training_target_manifest_sha256": _sha256(
                result.get("training_capture_manifest_sha256"),
                "SAE training capture",
            ),
            "training_source_manifest_sha256": None,
            "recomposition_sidecar_sha256": None,
            "run_contract_sha256": panel.run_contract_sha256,
            "run_contract_file_sha256": panel.run_contract.sha256,
            "training_result_sha256": panel.training_result_sha256,
            "training_result_file_sha256": panel.training_result.sha256,
            "checkpoint_pointer_sha256": pointer_sha,
            "checkpoint_pointer_file_sha256": pointer_file_sha,
            "evaluation_receipt_sha256": _sha256(
                condition.get("receipt_sha256"),
                "SAE A evaluation receipt",
            ),
            "common_sst_receipt_sha256": receipt_sha,
            "common_sst_receipt_file_sha256": None,
            "score_artifact_sha256": score_sha,
            **representation,
        },
        "endpoints": endpoint_rows,
        "sparsity": _sparsity(condition, name="SAE A"),
    }
    return cell, representation


def _build_transcoder_cell(
    *,
    task: Any,
    panel: A256AddendumPanel,
    result: Mapping[str, Any],
    payload: bytes,
    milestone: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        checked = validate_transcoder_scientific_result(result)
    except (TypeError, ValueError) as error:
        raise A256ContextScoreError(
            f"invalid transcoder scientific result at {milestone:,} tokens"
        ) from error
    checkpoint = checked.get("learning_curve_checkpoint")
    conditions = checked.get("conditions")
    parents = checked.get("parents")
    paired = checked.get("paired_scores")
    if (
        checked.get("schema_version")
        != TRANSCODER_EVALUATION_SCHEMA_VERSION
        or not isinstance(checkpoint, Mapping)
        or checkpoint.get("target_tokens") != milestone
        or checked.get("seed") != 0
        or not isinstance(conditions, Mapping)
        or not isinstance(parents, Mapping)
        or not isinstance(paired, Mapping)
    ):
        raise A256ContextScoreError(
            "transcoder scientific result role changed"
        )
    condition = conditions.get("A")
    run_contracts = parents.get("run_contract_sha256")
    if not isinstance(condition, Mapping) or not isinstance(
        run_contracts, Mapping
    ):
        raise A256ContextScoreError(
            "transcoder result lacks condition A parents"
        )
    if (
        checked.get("config_sha256")
        != panel.observed_source_config_semantic_sha256
        or run_contracts.get("A") != panel.run_contract_sha256
    ):
        raise A256ContextScoreError(
            "transcoder A result differs from corrected provenance"
        )
    training = condition.get("training")
    endpoints = condition.get("endpoints")
    if not isinstance(training, Mapping) or not isinstance(endpoints, Mapping):
        raise A256ContextScoreError(
            "transcoder A result lacks training/endpoints"
        )
    if training.get("training_result_sha256") != panel.training_result_sha256:
        raise A256ContextScoreError(
            "transcoder A result names another training result"
        )
    actual_tokens = _checkpoint_actual_tokens(
        result=checked, panel="transcoder"
    )
    checkpoint_sha = _sha256(
        condition.get("checkpoint_sha256"),
        "transcoder A checkpoint hash",
    )
    pointer_sha, pointer_file_sha = _validate_pointer_binding(
        panel,
        milestone,
        checkpoint_sha256=checkpoint_sha,
        actual_tokens=actual_tokens,
    )
    receipt_sha, receipt_file_sha, representation = (
        _checked_transcoder_sst_receipt(
            task.completion_path_controller,
            task.result_root_remote,
            checked,
        )
    )
    endpoint_rows = {
        endpoint: _score(
            endpoints.get(endpoint),
            f"transcoder A {endpoint}",
        )
        for endpoint in ENDPOINTS
    }
    target_capture = _sha256(
        parents.get("eval_target_manifest_sha256"),
        "transcoder evaluation target capture",
    )
    source_capture = _sha256(
        parents.get("eval_source_manifest_sha256"),
        "transcoder evaluation source capture",
    )
    if (
        condition.get("target_manifest_sha256") != target_capture
        or condition.get("source_manifest_sha256") != source_capture
        or condition.get("tokens") != EVALUATION_TOKENS
        or paired.get("tokens") != EVALUATION_TOKENS
    ):
        raise A256ContextScoreError(
            "transcoder A condition differs from paired held-out identity"
        )
    cell = {
        "panel": "transcoder",
        "condition": "A256Native",
        "source_condition": "A",
        "source_label": panel.source_label,
        "seed": 0,
        "checkpoint_tokens": milestone,
        "checkpoint_actual_tokens": actual_tokens,
        "encoder_activation": panel.encoder_activation,
        "selection_policy": panel.selection_policy,
        "source_config_semantic_sha256": (
            panel.observed_source_config_semantic_sha256
        ),
        "source_config_file_sha256": (
            panel.observed_source_config_file_sha256
        ),
        "source_result": {
            "controller_path": str(task.completion_path_controller),
            "remote_path": str(
                PurePosixPath(task.result_root_remote)
                / "scientific-result.json"
            ),
            "file_sha256": hashlib.sha256(payload).hexdigest(),
            "result_sha256": checked["result_sha256"],
        },
        "parent_hashes": {
            "target_capture_manifest_sha256": target_capture,
            "source_capture_manifest_sha256": source_capture,
            "training_target_manifest_sha256": _sha256(
                parents.get("train_target_manifest_sha256"),
                "transcoder training target capture",
            ),
            "training_source_manifest_sha256": _sha256(
                parents.get("train_source_manifest_sha256"),
                "transcoder training source capture",
            ),
            "recomposition_sidecar_sha256": _sha256(
                parents.get("recomposition_sidecar_sha256"),
                "transcoder recomposition sidecar",
            ),
            "run_contract_sha256": panel.run_contract_sha256,
            "run_contract_file_sha256": panel.run_contract.sha256,
            "training_result_sha256": panel.training_result_sha256,
            "training_result_file_sha256": panel.training_result.sha256,
            "checkpoint_pointer_sha256": pointer_sha,
            "checkpoint_pointer_file_sha256": pointer_file_sha,
            "evaluation_receipt_sha256": _sha256(
                condition.get("evaluation_receipt_sha256"),
                "transcoder A evaluation receipt",
            ),
            "common_sst_receipt_sha256": receipt_sha,
            "common_sst_receipt_file_sha256": receipt_file_sha,
            "score_artifact_sha256": _sha256(
                paired.get("score_artifact_sha256"),
                "transcoder paired score artifact",
            ),
            **representation,
        },
        "endpoints": endpoint_rows,
        "sparsity": _sparsity(condition, name="transcoder A"),
    }
    return cell, representation


def build_a256_context_score_manifest(
    horizon_plan: DirectCampaignPlan,
    provenance_addendum_path: str | Path,
) -> dict[str, Any]:
    """Validate the six historical scores and build one immutable manifest."""

    addendum_path = Path(provenance_addendum_path).expanduser().resolve()
    try:
        addendum_payload = addendum_path.read_bytes()
        addendum_value = yaml.safe_load(addendum_payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise A256ContextScoreError(
            f"cannot read A256 provenance addendum: {addendum_path}"
        ) from error
    try:
        addendum, panels = parse_a256_provenance_addendum(addendum_value)
        for panel in panels:
            validate_a256_addendum_panel(panel)
    except (TypeError, ValueError, RuntimeError) as error:
        raise A256ContextScoreError(
            "A256 provenance addendum is invalid"
        ) from error
    if addendum["source_plan_sha256"] != horizon_plan.plan_sha256:
        raise A256ContextScoreError(
            "A256 provenance addendum names another direct-horizon plan"
        )
    by_panel = {panel.panel: panel for panel in panels}
    expected_tasks = {
        (panel, milestone)
        for panel in PANELS
        for milestone in MILESTONES
    }
    tasks = {
        (task.scientific_model, task.milestone_tokens): task
        for task in horizon_plan.evaluations
        if task.seed == 0
    }
    if set(tasks) != expected_tasks or len(tasks) != len(expected_tasks):
        raise A256ContextScoreError(
            "direct horizon does not contain the exact six A256 score parents"
        )
    cells: list[dict[str, Any]] = []
    representations: dict[str, dict[str, str]] = {}
    for panel_name in PANELS:
        for milestone in MILESTONES:
            task = tasks[(panel_name, milestone)]
            completion = inspect_evaluation_completion(task)
            if completion is None:
                raise A256ContextScoreError(
                    f"{panel_name} {milestone:,}-token evaluation is incomplete"
                )
            result, payload = _read_json_result(
                task.completion_path_controller
            )
            if (
                completion.sha256 != hashlib.sha256(payload).hexdigest()
                or completion.scientific_sha256 != result.get("result_sha256")
            ):
                raise A256ContextScoreError(
                    f"{panel_name} {milestone:,}-token completion changed"
                )
            if panel_name == "sae":
                cell, representation = _build_sae_cell(
                    task=task,
                    panel=by_panel[panel_name],
                    result=result,
                    payload=payload,
                    milestone=milestone,
                )
            else:
                cell, representation = _build_transcoder_cell(
                    task=task,
                    panel=by_panel[panel_name],
                    result=result,
                    payload=payload,
                    milestone=milestone,
                )
            previous = representations.setdefault(panel_name, representation)
            if previous != representation:
                raise A256ContextScoreError(
                    f"{panel_name} held-out byte identities changed across "
                    "milestones"
                )
            cells.append(cell)
    target_captures = {
        cell["parent_hashes"]["target_capture_manifest_sha256"]
        for cell in cells
    }
    component_sst = {cell["endpoints"]["component"]["sst"] for cell in cells}
    routed_sst = {cell["endpoints"]["routed"]["sst"] for cell in cells}
    if (
        len(target_captures) != 1
        or len(component_sst) != 1
        or len(routed_sst) != 1
    ):
        raise A256ContextScoreError(
            "A256 panels do not share one target capture and endpoint SST"
        )
    body = {
        "schema_version": A256_CONTEXT_SCORE_MANIFEST_SCHEMA_VERSION,
        "manifest_id": A256_CONTEXT_SCORE_MANIFEST_ID,
        "source_protocol_id": EXPERIMENT4_PROTOCOL_ID,
        "target_protocol_id": PROTOCOL_ID,
        "source_plan_sha256": horizon_plan.plan_sha256,
        "provenance_addendum": {
            "addendum_id": addendum["addendum_id"],
            "controller_path": str(addendum_path),
            "file_sha256": hashlib.sha256(addendum_payload).hexdigest(),
            "manifest_sha256": addendum["manifest_sha256"],
        },
        "scientific_scope": copy.deepcopy(_SCOPE),
        "shared_held_out_identity": {
            "evaluation_tokens": EVALUATION_TOKENS,
            "target_capture_manifest_sha256": next(iter(target_captures)),
            "component_sst": next(iter(component_sst)),
            "routed_sst": next(iter(routed_sst)),
            "panel_representations": {
                panel: representations[panel] for panel in PANELS
            },
            "token_identity_policy": _TOKEN_IDENTITY_POLICY,
        },
        "cells": sorted(
            cells,
            key=lambda row: (str(row["panel"]), int(row["checkpoint_tokens"])),
        ),
    }
    manifest = {**body, "manifest_sha256": canonical_sha256(body)}
    parse_a256_context_score_manifest(manifest)
    return manifest


def load_a256_context_score_manifest(path: str | Path) -> Mapping[str, Any]:
    selected = Path(path).expanduser().resolve()
    try:
        value = yaml.safe_load(selected.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise A256ContextScoreError(
            f"cannot read A256 context-score manifest: {selected}"
        ) from error
    return parse_a256_context_score_manifest(value)


def write_a256_context_score_manifest(
    path: str | Path,
    manifest: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Write a validated manifest exactly once."""

    parsed = parse_a256_context_score_manifest(manifest)
    selected = Path(path).expanduser().resolve()
    selected.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    payload = yaml.safe_dump(
        dict(parsed),
        sort_keys=False,
        default_flow_style=False,
    ).encode("utf-8")
    try:
        descriptor = os.open(
            selected,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o640,
        )
    except FileExistsError as error:
        raise A256ContextScoreError(
            f"refusing to overwrite A256 context-score manifest: {selected}"
        ) from error
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        selected.unlink(missing_ok=True)
        raise
    return parsed


__all__ = [
    "A256_CONTEXT_SCORE_MANIFEST_ID",
    "A256_CONTEXT_SCORE_MANIFEST_SCHEMA_VERSION",
    "EVALUATION_TOKENS",
    "A256ContextScoreError",
    "build_a256_context_score_manifest",
    "load_a256_context_score_manifest",
    "parse_a256_context_score_manifest",
    "write_a256_context_score_manifest",
]
