"""Immutable six-point report for the Experiment 4 learning horizon.

The report consumes the three SAE and three transcoder scientific results at
2M, 5M, and 10M training tokens.  Training run contracts are separate inputs:
they bind every curve to the accelerator generation that produced it.  Raw
chip-seconds are always retained, but a ratio is interpreted only when its two
conditions ran on the same accelerator generation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

from moe_sae.contribution_artifacts import canonical_json_bytes, canonical_sha256
from moe_sae.contribution_transcoder_report import (
    LEARNING_HORIZON_RESULT_SCHEMA_VERSION as TRANSCODER_RESULT_SCHEMA_VERSION,
)
from moe_sae.contribution_transcoder_report import (
    validate_transcoder_scientific_result,
)
from moe_sae.learning_horizon_config import (
    CONDITIONS,
    PHASE_NAME,
    PROTOCOL_ID,
)
from moe_sae.runtime import SUPPORTED_TRAINING_ACCELERATOR_TYPES

REPORT_SCHEMA_VERSION = "gemma4-experiment4-learning-horizon-report-v1"
SAE_RESULT_SCHEMA_VERSION = "gemma4-contribution-learning-horizon-curve-point-v1"
RUN_CONTRACT_SCHEMA_VERSION = "gemma4-contribution-sae-operator-run-v1"
MILESTONES = (2_000_000, 5_000_000, 10_000_000)
MODELS = ("sae", "transcoder")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class HorizonCurveReportError(ValueError):
    """The Experiment 4 curve evidence is incomplete or inconsistent."""


def _read_json(path: str | Path, *, name: str) -> dict[str, Any]:
    selected = Path(path).expanduser().resolve()
    try:
        value = json.loads(selected.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HorizonCurveReportError(f"cannot read {name}: {selected}") from error
    if not isinstance(value, dict):
        raise HorizonCurveReportError(f"{name} must be a JSON object")
    return value


def _sha256(name: str, value: Any) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise HorizonCurveReportError(f"{name} must be a lowercase SHA-256")
    return value


def _finite(name: str, value: Any, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HorizonCurveReportError(f"{name} must be numeric")
    selected = float(value)
    if not math.isfinite(selected) or (positive and selected <= 0):
        qualifier = "finite and positive" if positive else "finite"
        raise HorizonCurveReportError(f"{name} must be {qualifier}")
    return selected


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise HorizonCurveReportError(f"{name} must be a positive integer")
    return int(value)


def _condition_matrix(name: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(CONDITIONS):
        raise HorizonCurveReportError(f"{name} must contain exactly A/R/B/F")
    return value


def _self_hashed(
    value: Mapping[str, Any],
    *,
    hash_field: str,
    name: str,
) -> tuple[dict[str, Any], str]:
    body = dict(value)
    claimed = _sha256(f"{name}.{hash_field}", body.pop(hash_field, None))
    if canonical_sha256(body) != claimed:
        raise HorizonCurveReportError(f"{name} self-hash is invalid")
    return body, claimed


def _score(name: str, value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise HorizonCurveReportError(f"{name} must be a score mapping")
    sse = _finite(f"{name}.sse", value.get("sse"))
    sst = _finite(f"{name}.sst", value.get("sst"), positive=True)
    nmse = _finite(f"{name}.nmse", value.get("nmse"))
    ev = _finite(f"{name}.ev", value.get("ev"))
    expected_nmse = sse / sst
    if not math.isclose(nmse, expected_nmse, rel_tol=1e-10, abs_tol=1e-10):
        raise HorizonCurveReportError(f"{name}.nmse differs from SSE/SST")
    if not math.isclose(ev, 1.0 - nmse, rel_tol=1e-10, abs_tol=1e-10):
        raise HorizonCurveReportError(f"{name}.ev differs from 1-NMSE")
    return {"sse": sse, "sst": sst, "nmse": nmse, "ev": ev}


def _checkpoint(
    *,
    target: Any,
    actual: Any,
    owner: str,
) -> tuple[int, dict[str, int]]:
    selected_target = _positive_int(f"{owner}.target_tokens", target)
    if selected_target not in MILESTONES:
        raise HorizonCurveReportError(
            f"{owner}.target_tokens is not an Experiment 4 milestone"
        )
    rows = _condition_matrix(f"{owner}.actual_tokens", actual)
    selected_actual = {
        condition: _positive_int(
            f"{owner}.actual_tokens.{condition}",
            rows[condition],
        )
        for condition in CONDITIONS
    }
    if any(tokens < selected_target for tokens in selected_actual.values()):
        raise HorizonCurveReportError(
            f"{owner} checkpoint precedes its nominal milestone"
        )
    return selected_target, selected_actual


def _validate_sae_result(value: Mapping[str, Any]) -> dict[str, Any]:
    body, result_sha = _self_hashed(
        value,
        hash_field="result_sha256",
        name="SAE scientific result",
    )
    if (
        body.get("schema_version") != SAE_RESULT_SCHEMA_VERSION
        or body.get("protocol_id") != PROTOCOL_ID
        or body.get("status") != "curve_point_measured"
        or body.get("phase") != PHASE_NAME
    ):
        raise HorizonCurveReportError("SAE scientific result identity changed")
    target, actual = _checkpoint(
        target=body.get("checkpoint_target_tokens"),
        actual=body.get("checkpoint_actual_tokens"),
        owner="SAE checkpoint",
    )
    endpoints = body.get("endpoints")
    if not isinstance(endpoints, Mapping) or not {"component", "routed"} <= set(
        endpoints
    ):
        raise HorizonCurveReportError("SAE result lacks component/routed endpoints")
    checked_endpoints = {
        endpoint: {
            condition: _score(
                f"SAE.{endpoint}.{condition}",
                _condition_matrix(
                    f"SAE endpoints.{endpoint}",
                    endpoints[endpoint],
                )[condition],
            )
            for condition in CONDITIONS
        }
        for endpoint in ("component", "routed")
    }
    costs = {
        condition: _finite(
            f"SAE training_chip_seconds.{condition}",
            _condition_matrix(
                "SAE training_chip_seconds",
                body.get("training_chip_seconds"),
            )[condition],
            positive=True,
        )
        for condition in CONDITIONS
    }
    primary = body.get("primary")
    if not isinstance(primary, Mapping):
        raise HorizonCurveReportError("SAE result lacks primary metrics")
    c2 = _finite("SAE primary.c2_percent", primary.get("c2_percent"))
    expected_c2 = 100.0 * costs["A"] / costs["F"]
    if not math.isclose(c2, expected_c2, rel_tol=1e-10, abs_tol=1e-10):
        raise HorizonCurveReportError("SAE C2 differs from its cost ledger")
    _sha256(
        "SAE training capture manifest",
        body.get("training_capture_manifest_sha256"),
    )
    _sha256(
        "SAE evaluation capture manifest",
        body.get("evaluation_capture_manifest_sha256"),
    )
    return {
        **dict(value),
        "_target_tokens": target,
        "_actual_tokens": actual,
        "_checked_endpoints": checked_endpoints,
        "_costs": costs,
        "_result_sha256": result_sha,
    }


def _validate_transcoder_result(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        selected = validate_transcoder_scientific_result(value)
    except (TypeError, ValueError) as error:
        raise HorizonCurveReportError(
            "transcoder scientific result is invalid"
        ) from error
    if (
        selected.get("schema_version") != TRANSCODER_RESULT_SCHEMA_VERSION
        or selected.get("protocol_id") != PROTOCOL_ID
        or selected.get("status") != "curve_point_measured"
    ):
        raise HorizonCurveReportError(
            "transcoder scientific result identity changed"
        )
    parents = selected.get("parents")
    if not isinstance(parents, Mapping):
        raise HorizonCurveReportError("transcoder result lacks parents")
    for name in (
        "train_source_manifest_sha256",
        "train_target_manifest_sha256",
        "eval_source_manifest_sha256",
        "eval_target_manifest_sha256",
        "recomposition_sidecar_sha256",
    ):
        _sha256(f"transcoder parents.{name}", parents.get(name))
    checkpoint = selected.get("learning_curve_checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise HorizonCurveReportError("transcoder result lacks its curve checkpoint")
    target, actual = _checkpoint(
        target=checkpoint.get("target_tokens"),
        actual=checkpoint.get("actual_tokens_by_condition"),
        owner="transcoder checkpoint",
    )
    conditions = _condition_matrix(
        "transcoder conditions",
        selected.get("conditions"),
    )
    checked_endpoints: dict[str, dict[str, dict[str, float]]] = {
        endpoint: {} for endpoint in ("component", "routed")
    }
    costs: dict[str, float] = {}
    training: dict[str, dict[str, Any]] = {}
    for condition in CONDITIONS:
        row = conditions[condition]
        if not isinstance(row, Mapping):
            raise HorizonCurveReportError(
                f"transcoder condition {condition} must be a mapping"
            )
        endpoints = row.get("endpoints")
        if not isinstance(endpoints, Mapping):
            raise HorizonCurveReportError(
                f"transcoder condition {condition} lacks endpoints"
            )
        for endpoint in ("component", "routed"):
            checked_endpoints[endpoint][condition] = _score(
                f"transcoder.{endpoint}.{condition}",
                endpoints.get(endpoint),
            )
        ledger = row.get("training")
        if not isinstance(ledger, Mapping):
            raise HorizonCurveReportError(
                f"transcoder condition {condition} lacks a training ledger"
            )
        costs[condition] = _finite(
            f"transcoder training.{condition}.marginal_chip_seconds",
            ledger.get("marginal_chip_seconds"),
            positive=True,
        )
        training[condition] = dict(ledger)
    primary = selected.get("primary")
    if not isinstance(primary, Mapping):
        raise HorizonCurveReportError("transcoder result lacks primary metrics")
    c3 = _finite("transcoder primary.c3_percent", primary.get("c3_percent"))
    expected_c3 = 100.0 * costs["A"] / costs["F"]
    if not math.isclose(c3, expected_c3, rel_tol=1e-10, abs_tol=1e-10):
        raise HorizonCurveReportError("transcoder C3 differs from its cost ledger")
    return {
        **selected,
        "_target_tokens": target,
        "_actual_tokens": actual,
        "_checked_endpoints": checked_endpoints,
        "_costs": costs,
        "_training": training,
        "_result_sha256": selected["result_sha256"],
    }


def _load_curve(
    paths: Sequence[str | Path],
    *,
    model: str,
) -> dict[int, dict[str, Any]]:
    if len(paths) != len(MILESTONES):
        raise HorizonCurveReportError(
            f"{model} requires exactly three scientific results"
        )
    validator = (
        _validate_sae_result if model == "sae" else _validate_transcoder_result
    )
    rows: dict[int, dict[str, Any]] = {}
    for path in paths:
        value = validator(_read_json(path, name=f"{model} scientific result"))
        target = value["_target_tokens"]
        if target in rows:
            raise HorizonCurveReportError(
                f"{model} contains duplicate {target:,}-token results"
            )
        rows[target] = value
    if set(rows) != set(MILESTONES):
        raise HorizonCurveReportError(
            f"{model} must contain exactly the 2M/5M/10M curve"
        )
    ordered = [rows[target] for target in MILESTONES]
    for condition in CONDITIONS:
        actual = [row["_actual_tokens"][condition] for row in ordered]
        costs = [row["_costs"][condition] for row in ordered]
        if actual != sorted(actual) or len(set(actual)) != len(actual):
            raise HorizonCurveReportError(
                f"{model}.{condition} checkpoint tokens are not strictly increasing"
            )
        if any(right < left for left, right in pairwise(costs)):
            raise HorizonCurveReportError(
                f"{model}.{condition} cumulative chip-seconds decreased"
            )
    if model == "sae":
        stable = (
            "seed",
            "training_capture_manifest_sha256",
            "evaluation_capture_manifest_sha256",
            "evaluation_tokens",
            "endpoint_centering",
        )
    else:
        stable = (
            "seed",
            "source_revision",
            "training_source_revision",
            "config_sha256",
        )
    for name in stable:
        first = ordered[0].get(name)
        if any(row.get(name) != first for row in ordered[1:]):
            raise HorizonCurveReportError(
                f"{model} curve changed stable field {name}"
            )
    if model == "transcoder":
        parent_fields = (
            "train_source_manifest_sha256",
            "train_target_manifest_sha256",
            "eval_source_manifest_sha256",
            "eval_target_manifest_sha256",
            "recomposition_sidecar_sha256",
            "run_contract_sha256",
        )
        first_parents = ordered[0].get("parents")
        if not isinstance(first_parents, Mapping):
            raise HorizonCurveReportError("transcoder curve lacks parents")
        for name in parent_fields:
            first = first_parents.get(name)
            if any(
                not isinstance(row.get("parents"), Mapping)
                or row["parents"].get(name) != first
                for row in ordered[1:]
            ):
                raise HorizonCurveReportError(
                    f"transcoder curve changed parent field {name}"
                )
    return rows


def _hardware_generation(accelerator_type: str) -> str:
    if accelerator_type == "v5litepod-16":
        return "v5e"
    if accelerator_type.startswith("v6e-"):
        return "v6e"
    if accelerator_type.startswith("v4-"):
        return "v4"
    raise HorizonCurveReportError(
        f"unsupported accelerator type {accelerator_type!r}"
    )


def _chip_count(accelerator_type: str) -> int:
    # A v4-32 TPU VM is 32 cores / 16 dual-core chips.  The v5e/v6e names
    # already count chips, so deriving this from the final numeric suffix would
    # silently double every v4 wall-time and throughput.
    return {
        "v4-32": 16,
        "v5litepod-16": 16,
        "v6e-16": 16,
    }[accelerator_type]


def _validate_run_contract(
    value: Mapping[str, Any],
    *,
    model: str,
    condition: str,
) -> dict[str, Any]:
    body, contract_sha = _self_hashed(
        value,
        hash_field="run_contract_sha256",
        name=f"{model}.{condition} run contract",
    )
    accelerator_type = body.get("accelerator_type")
    if (
        body.get("schema_version") != RUN_CONTRACT_SCHEMA_VERSION
        or body.get("phase") != PHASE_NAME
        or body.get("condition") != condition
        or not isinstance(accelerator_type, str)
        or accelerator_type not in SUPPORTED_TRAINING_ACCELERATOR_TYPES
    ):
        raise HorizonCurveReportError(
            f"{model}.{condition} run contract identity changed"
        )
    declared_model = body.get("scientific_model")
    if declared_model is not None and declared_model != model:
        raise HorizonCurveReportError(
            f"{model}.{condition} run contract names another scientific model"
        )
    return {
        "run_contract_sha256": contract_sha,
        "accelerator_type": accelerator_type,
        "hardware_generation": _hardware_generation(accelerator_type),
        "chip_count": _chip_count(accelerator_type),
        "project_commit": body.get("project_commit"),
        "run_id": body.get("run_id"),
        "seed": body.get("seed"),
        "capture_manifest_sha256": body.get("capture_manifest_sha256"),
        "source_overlay_manifest_sha256": body.get(
            "source_overlay_manifest_sha256"
        ),
    }


def _parse_contract_specs(
    specifications: Sequence[str],
) -> dict[str, dict[str, Path]]:
    selected: dict[str, dict[str, Path]] = {
        model: {} for model in MODELS
    }
    for specification in specifications:
        try:
            identity, raw_path = specification.split("=", 1)
            model, condition = identity.split(":", 1)
        except ValueError as error:
            raise HorizonCurveReportError(
                "--run-contract must be MODEL:CONDITION=PATH"
            ) from error
        model = model.lower()
        condition = condition.upper()
        if model not in MODELS or condition not in CONDITIONS or not raw_path:
            raise HorizonCurveReportError(
                "--run-contract identity must be sae/transcoder:A/R/B/F"
            )
        if condition in selected[model]:
            raise HorizonCurveReportError(
                f"duplicate run contract for {model}:{condition}"
            )
        selected[model][condition] = Path(raw_path)
    if any(set(selected[model]) != set(CONDITIONS) for model in MODELS):
        raise HorizonCurveReportError(
            "exactly eight run contracts are required: SAE/transcoder A/R/B/F"
        )
    return selected


def _load_hardware(
    specifications: Sequence[str],
    curves: Mapping[str, Mapping[int, Mapping[str, Any]]],
) -> dict[str, dict[str, dict[str, Any]]]:
    paths = _parse_contract_specs(specifications)
    hardware = {
        model: {
            condition: _validate_run_contract(
                _read_json(
                    paths[model][condition],
                    name=f"{model}.{condition} run contract",
                ),
                model=model,
                condition=condition,
            )
            for condition in CONDITIONS
        }
        for model in MODELS
    }
    for model in MODELS:
        seed = next(iter(curves[model].values())).get("seed")
        for condition in CONDITIONS:
            row = hardware[model][condition]
            if row["seed"] != seed:
                raise HorizonCurveReportError(
                    f"{model}.{condition} run contract seed differs from results"
                )
        if model == "sae":
            manifest = next(iter(curves[model].values()))[
                "training_capture_manifest_sha256"
            ]
            if any(
                row["capture_manifest_sha256"] != manifest
                for row in hardware[model].values()
            ):
                raise HorizonCurveReportError(
                    "SAE run contracts name another training capture"
                )
        else:
            first = next(iter(curves[model].values()))
            parents = first["parents"]
            expected_contracts = parents["run_contract_sha256"]
            for condition in CONDITIONS:
                row = hardware[model][condition]
                if (
                    row["run_contract_sha256"] != expected_contracts[condition]
                    or row["capture_manifest_sha256"]
                    != parents["train_target_manifest_sha256"]
                    or row["source_overlay_manifest_sha256"]
                    != parents["train_source_manifest_sha256"]
                ):
                    raise HorizonCurveReportError(
                        f"transcoder.{condition} run contract provenance differs"
                    )
    return hardware


def _cost_comparability(
    hardware: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    generations = {
        condition: str(hardware[condition]["hardware_generation"])
        for condition in CONDITIONS
    }
    groups: dict[str, list[str]] = {}
    for condition, generation in generations.items():
        groups.setdefault(generation, []).append(condition)
    pairs: dict[str, dict[str, Any]] = {}
    for left_index, left in enumerate(CONDITIONS):
        for right in CONDITIONS[left_index + 1 :]:
            comparable = generations[left] == generations[right]
            pairs[f"{left}/{right}"] = {
                "comparable": comparable,
                "reason": (
                    "same accelerator generation"
                    if comparable
                    else "mixed accelerator generations"
                ),
            }
    return {
        "policy": (
            "Raw measured chip-seconds are preserved. Ratios are interpreted "
            "only within one accelerator generation; no v4/v5e/v6e "
            "normalization is inferred."
        ),
        "accelerator_generation_by_condition": generations,
        "same_generation_groups": {
            generation: sorted(conditions)
            for generation, conditions in sorted(groups.items())
        },
        "all_conditions_same_generation": len(groups) == 1,
        "pairs": pairs,
    }


def _source_gain(model: str, primary: Mapping[str, Any]) -> Any:
    if model == "sae":
        return primary.get("g2_percent")
    gain = primary.get("g3")
    return gain.get("gain_recovery_percent") if isinstance(gain, Mapping) else None


def _source_cost_ratio(model: str, primary: Mapping[str, Any]) -> float:
    field = "c2_percent" if model == "sae" else "c3_percent"
    return _finite(f"{model} primary.{field}", primary.get(field))


def build_horizon_curve_report(
    *,
    sae_results: Sequence[str | Path],
    transcoder_results: Sequence[str | Path],
    run_contracts: Sequence[str],
) -> dict[str, Any]:
    """Validate and combine the six immutable scientific curve points."""

    curves = {
        "sae": _load_curve(sae_results, model="sae"),
        "transcoder": _load_curve(transcoder_results, model="transcoder"),
    }
    hardware = _load_hardware(run_contracts, curves)
    model_rows: dict[str, Any] = {}
    for model in MODELS:
        comparability = _cost_comparability(hardware[model])
        a_f_comparable = comparability["pairs"]["A/F"]["comparable"]
        points = []
        for target in MILESTONES:
            source = curves[model][target]
            primary = dict(source["primary"])
            source_ratio = _source_cost_ratio(model, primary)
            conditions: dict[str, Any] = {}
            for condition in CONDITIONS:
                actual = source["_actual_tokens"][condition]
                if model == "sae":
                    chip_seconds = source["_costs"][condition]
                    chips = hardware[model][condition]["chip_count"]
                    wall_seconds = chip_seconds / chips
                    training = {
                        "marginal_chip_seconds": chip_seconds,
                        "marginal_wall_seconds": wall_seconds,
                        "tokens_per_second": actual / wall_seconds,
                        "useful_mfu": None,
                        "useful_mfu_availability": (
                            "not embedded in the SAE scientific result"
                        ),
                    }
                else:
                    training = source["_training"][condition]
                conditions[condition] = {
                    "hardware": hardware[model][condition],
                    "actual_training_tokens": actual,
                    "training": training,
                    "held_out_evaluation": {
                        endpoint: source["_checked_endpoints"][endpoint][condition]
                        for endpoint in ("component", "routed")
                    },
                }
            points.append(
                {
                    "target_training_tokens": target,
                    "source_result_sha256": source["_result_sha256"],
                    "conditions": conditions,
                    "source_primary": primary,
                    "cost_interpretation": {
                        "a_over_f_comparable": a_f_comparable,
                        "source_measured_a_over_f_percent": source_ratio,
                        "interpretable_a_over_f_percent": (
                            source_ratio if a_f_comparable else None
                        ),
                        "reason": (
                            "A and F used the same accelerator generation"
                            if a_f_comparable
                            else (
                                "A and F used different accelerator generations; "
                                "the source ratio is retained but not interpreted"
                            )
                        ),
                    },
                    "gain_recovery_percent": _source_gain(model, primary),
                }
            )
        model_rows[model] = {
            "hardware": hardware[model],
            "cost_comparability": comparability,
            "curve": points,
        }
    body = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": "complete",
        "experiment": "Experiment 4: pilot-width SAE/transcoder learning horizon",
        "milestone_training_tokens": list(MILESTONES),
        "scientific_models": model_rows,
        "cross_model_cost_policy": (
            "SAE and transcoder quality curves may be compared at matched token "
            "milestones. Their raw chip-seconds remain hardware-labelled and "
            "are not treated as generation-normalized costs."
        ),
    }
    return {**body, "report_sha256": canonical_sha256(body)}


def render_horizon_curve_report(value: Mapping[str, Any]) -> str:
    """Render the immutable JSON report as a concise scientific handoff."""

    body, claimed = _self_hashed(
        value,
        hash_field="report_sha256",
        name="Experiment 4 report",
    )
    if (
        body.get("schema_version") != REPORT_SCHEMA_VERSION
        or body.get("protocol_id") != PROTOCOL_ID
        or body.get("status") != "complete"
    ):
        raise HorizonCurveReportError("Experiment 4 report identity changed")
    lines = [
        "# Experiment 4: SAE/transcoder learning horizon",
        "",
        (
            "Held-out reconstruction is measured jointly at 2M, 5M, and 10M "
            "training tokens. Raw training chip-seconds retain their hardware "
            "labels; mixed v4/v5e/v6e ratios are not interpreted as equivalent."
        ),
        "",
    ]
    models = body["scientific_models"]
    for model in MODELS:
        selected = models[model]
        lines.extend(
            [
                f"## {model.upper()}",
                "",
                "| condition | accelerator | generation | run contract |",
                "|---|---|---|---|",
            ]
        )
        for condition in CONDITIONS:
            hardware = selected["hardware"][condition]
            lines.append(
                f"| {condition} | {hardware['accelerator_type']} | "
                f"{hardware['hardware_generation']} | "
                f"`{hardware['run_contract_sha256']}` |"
            )
        lines.append("")
        if not selected["cost_comparability"]["all_conditions_same_generation"]:
            lines.extend(
                [
                    (
                        "**Cost warning:** this model used mixed accelerator "
                        "generations. Raw chip-seconds are shown, but affected "
                        "cross-condition ratios are not scientific comparisons."
                    ),
                    "",
                ]
            )
        for point in selected["curve"]:
            lines.extend(
                [
                    f"### {point['target_training_tokens']:,} training tokens",
                    "",
                    (
                        "| condition | actual tokens | component EV | routed EV "
                        "| marginal chip-s | train tok/s | useful MFU |"
                    ),
                    "|---|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for condition in CONDITIONS:
                row = point["conditions"][condition]
                training = row["training"]
                evaluation = row["held_out_evaluation"]
                mfu = training.get("useful_mfu")
                lines.append(
                    f"| {condition} | {row['actual_training_tokens']:,} | "
                    f"{evaluation['component']['ev']:.6f} | "
                    f"{evaluation['routed']['ev']:.6f} | "
                    f"{training['marginal_chip_seconds']:.1f} | "
                    f"{training['tokens_per_second']:.1f} | "
                    f"{'—' if mfu is None else f'{100.0 * float(mfu):.2f}%'} |"
                )
            cost = point["cost_interpretation"]
            interpreted = (
                f"{cost['interpretable_a_over_f_percent']:.2f}%"
                if cost["interpretable_a_over_f_percent"] is not None
                else "not comparable across hardware generations"
            )
            gain = point["gain_recovery_percent"]
            gain_text = "undefined" if gain is None else f"{float(gain):.2f}%"
            lines.extend(
                [
                    "",
                    (
                        f"Gain recovery: {gain_text}; "
                        f"A/F marginal cost: {interpreted}."
                    ),
                    f"Source result: `{point['source_result_sha256']}`",
                    "",
                ]
            )
    lines.extend(
        [
            "## Identity",
            "",
            f"Report SHA-256: `{claimed}`",
            "",
        ]
    )
    return "\n".join(lines)


def _create_only(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o640)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def write_horizon_curve_report(
    output_root: str | Path,
    value: Mapping[str, Any],
) -> tuple[Path, Path]:
    """Create one immutable directory containing JSON and Markdown."""

    report = dict(value)
    render = render_horizon_curve_report(report)
    root = Path(output_root).expanduser().resolve()
    try:
        root.mkdir(parents=True, exist_ok=False, mode=0o700)
    except FileExistsError as error:
        raise HorizonCurveReportError(
            f"immutable output root already exists: {root}"
        ) from error
    result_path = root / "experiment4-result.json"
    markdown_path = root / "EXPERIMENT_4.md"
    _create_only(
        result_path,
        canonical_json_bytes(report) + b"\n",
    )
    _create_only(markdown_path, render.encode("utf-8"))
    return result_path, markdown_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sae-result",
        action="append",
        required=True,
        help="SAE scientific-result.json; provide exactly three, in any order",
    )
    parser.add_argument(
        "--transcoder-result",
        action="append",
        required=True,
        help=(
            "transcoder scientific-result.json; provide exactly three, in any order"
        ),
    )
    parser.add_argument(
        "--run-contract",
        action="append",
        required=True,
        help=(
            "training contract as MODEL:CONDITION=PATH; provide SAE/transcoder "
            "A/R/B/F"
        ),
    )
    parser.add_argument("--output-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = build_horizon_curve_report(
        sae_results=args.sae_result,
        transcoder_results=args.transcoder_result,
        run_contracts=args.run_contract,
    )
    result_path, markdown_path = write_horizon_curve_report(
        args.output_root,
        result,
    )
    print(
        json.dumps(
            {
                "markdown": str(markdown_path),
                "report": str(result_path),
                "report_sha256": result["report_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
