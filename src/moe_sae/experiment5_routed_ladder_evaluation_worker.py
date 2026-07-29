"""Single-cell multi-host evaluator for the routed-activation ladder.

One invocation evaluates exactly one of the 54 ladder evidence cells:

``panel in {sae,transcoder} x
 condition in {A512ReLU,A512Softplus,H} x seed in {0,1,2} x
 checkpoint in {2M,5M,10M}``.

The A cells restore the existing contribution SAE/transcoder checkpoint layout
and score the component, routed, and exact-oracle-shared endpoints.  The H
cells restore the frozen Experiment-5 H checkpoint read-only, rescore the same
three routed endpoints, and add its learned full-FFN endpoint.  Every cell uses
the complete common held-out capture and seals:

* one self-hashed, content-addressed evidence object per JAX process; and
* one process-zero ``RoutedLadderTokenScores`` artifact bound to the config,
  source revisions, run contract, checkpoint pointer, capture, sidecar, and
  exact scientific token identities.

The public A512ReLU/A512Softplus names never enter the grouped kernel.  Both
map to the existing condition-A implementation.  Their identical geometry
and distinct encoder activations are carried by the immutable ladder config
and checkpoint identity.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import itertools
import json
import math
import os
import re
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Literal

import numpy as np
from safetensors.numpy import load as load_safetensors
from safetensors.numpy import save as save_safetensors

from moe_sae.contribution_artifacts import (
    HIDDEN_SIZE,
    TOP_K_EXPERTS,
    bf16_bits_to_float32,
    canonical_json_bytes,
    canonical_sha256,
    token_identity_sha256,
)
from moe_sae.experiment5_routed_ladder import (
    ENDPOINTS,
    H_ENDPOINTS,
    SCORE_SCHEMA_VERSION,
    RoutedLadderHTokenScores,
    RoutedLadderTokenScores,
)
from moe_sae.experiment5_routed_ladder_config import (
    CONDITIONS,
    PHASE_NAME,
    PROTOCOL_ID,
    RoutedLadderConfig,
    load_routed_ladder_config,
)

PROCESS_EVIDENCE_SCHEMA_VERSION = (
    "gemma4-experiment5-routed-activation-ladder-process-evidence-v1"
)
RESULT_SCHEMA_VERSION = (
    "gemma4-experiment5-routed-activation-ladder-cell-evaluation-v1"
)
CENTERING_SCHEMA_VERSION = (
    "gemma4-experiment5-routed-activation-ladder-centering-v1"
)
ENDPOINT_CENTERING_CACHE_SCHEMA_VERSION = (
    "gemma4-experiment5-routed-activation-ladder-endpoint-centering-cache-v1"
)
ENDPOINT_CENTERING_ALGORITHM_SCHEMA_VERSION = (
    "gemma4-contribution-transcoder-endpoint-centering-v1"
)
PANELS = ("sae", "transcoder")
EVALUATED_CONDITIONS = (*CONDITIONS, "H")
CHECKPOINT_TOKENS = (2_000_000, 5_000_000, 10_000_000)
_GIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ROUTED_TENSORS = {
    "sequence_id",
    "position",
    "component_sse",
    "component_sst",
    "routed_sse",
    "routed_sst",
    "oracle_shared_sse",
    "oracle_shared_sst",
}


def _tensor_names(condition: str) -> set[str]:
    if condition in CONDITIONS:
        return set(_ROUTED_TENSORS)
    if condition == "H":
        return {
            *_ROUTED_TENSORS,
            "learned_full_ffn_sse",
            "learned_full_ffn_sst",
        }
    raise RoutedLadderEvaluationError("evaluation condition is invalid")


class RoutedLadderEvaluationError(RuntimeError):
    """One activation-ladder cell cannot produce valid evidence."""


def _read_json(path: str | Path, *, name: str) -> dict[str, Any]:
    selected = Path(path)
    try:
        value = json.loads(selected.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RoutedLadderEvaluationError(
            f"cannot read {name}: {selected}"
        ) from error
    if not isinstance(value, dict):
        raise RoutedLadderEvaluationError(f"{name} must be a JSON object")
    return value


def _create_only(path: str | Path, payload: bytes) -> Path:
    """Create one immutable file, allowing only a byte-identical replay."""

    selected = Path(path)
    selected.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        descriptor = os.open(
            selected,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o640,
        )
    except FileExistsError:
        try:
            existing = selected.read_bytes()
        except OSError as error:
            raise RoutedLadderEvaluationError(
                f"cannot replay immutable artifact: {selected}"
            ) from error
        if existing != payload:
            raise RoutedLadderEvaluationError(
                f"immutable artifact differs on replay: {selected}"
            )
        return selected
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        selected.unlink(missing_ok=True)
        raise
    return selected


def _write_json(path: str | Path, value: Mapping[str, Any]) -> Path:
    return _create_only(
        path,
        canonical_json_bytes(dict(value)) + b"\n",
    )


def _self_hash(
    value: Mapping[str, Any],
    *,
    field: str,
) -> dict[str, Any]:
    unsigned = dict(value)
    unsigned.pop(field, None)
    return {**unsigned, field: canonical_sha256(unsigned)}


def _require_sha256(name: str, value: Any) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise RoutedLadderEvaluationError(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _require_git_sha(name: str, value: Any) -> str:
    if not isinstance(value, str) or _GIT_RE.fullmatch(value) is None:
        raise RoutedLadderEvaluationError(
            f"{name} must be a lowercase Git SHA"
        )
    return value


def _artifact_file(
    root: Path,
    tensors: Mapping[str, np.ndarray],
) -> tuple[str, int, str]:
    blob = save_safetensors(
        {
            name: np.ascontiguousarray(value)
            for name, value in tensors.items()
        }
    )
    digest = hashlib.sha256(blob).hexdigest()
    relative = f"objects/{digest}.safetensors"
    _create_only(root / relative, blob)
    return digest, len(blob), relative


def _load_artifact_file(
    root: Path,
    *,
    relative: Any,
    digest: Any,
    size_bytes: Any,
) -> dict[str, np.ndarray]:
    expected = _require_sha256("data_sha256", digest)
    if (
        not isinstance(relative, str)
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
    ):
        raise RoutedLadderEvaluationError(
            "evidence data object must be a relative child path"
        )
    if (
        isinstance(size_bytes, bool)
        or not isinstance(size_bytes, int)
        or size_bytes <= 0
    ):
        raise RoutedLadderEvaluationError("evidence data size is invalid")
    path = root / relative
    try:
        blob = path.read_bytes()
    except OSError as error:
        raise RoutedLadderEvaluationError(
            f"cannot read evidence tensors: {path}"
        ) from error
    if len(blob) != size_bytes or hashlib.sha256(blob).hexdigest() != expected:
        raise RoutedLadderEvaluationError(
            "evidence tensor object differs from its content identity"
        )
    try:
        return {
            name: np.ascontiguousarray(value)
            for name, value in load_safetensors(blob).items()
        }
    except Exception as error:
        raise RoutedLadderEvaluationError(
            "evidence tensor object cannot be decoded"
        ) from error


def _score_arrays(scores: Any, *, condition: str) -> dict[str, np.ndarray]:
    valid = np.asarray(scores.valid_mask)
    if valid.ndim != 1 or valid.dtype != np.bool_ or not np.any(valid):
        raise RoutedLadderEvaluationError(
            "scientific score mask must be nonempty bool [rows]"
        )
    result = {
        "component_sse": np.asarray(scores.component_sse)[valid],
        "component_sst": np.asarray(scores.component_sst)[valid],
        "routed_sse": np.asarray(scores.routed_sse)[valid],
        "routed_sst": np.asarray(scores.routed_sst)[valid],
        "oracle_shared_sse": np.asarray(scores.oracle_shared_sse)[valid],
        "oracle_shared_sst": np.asarray(scores.oracle_shared_sst)[valid],
    }
    if condition == "H":
        result.update(
            {
                "learned_full_ffn_sse": np.asarray(
                    scores.learned_full_ffn_sse
                )[valid],
                "learned_full_ffn_sst": np.asarray(
                    scores.learned_full_ffn_sst
                )[valid],
            }
        )
    return result


def seal_process_evidence(
    root: str | Path,
    *,
    panel: str,
    condition: str,
    seed: int,
    checkpoint_tokens: int,
    checkpoint_actual_tokens: int,
    process_index: int,
    process_count: int,
    source_revision: str,
    training_source_revision: str,
    config_sha256: str,
    config_file_sha256: str,
    h_config_sha256: str | None,
    h_config_file_sha256: str | None,
    training_result_sha256: str,
    training_contract_sha256: str,
    checkpoint_sha256: str,
    target_manifest_sha256: str,
    source_manifest_sha256: str | None,
    recomposition_sidecar_sha256: str,
    centering_receipt_sha256: str,
    input_receipt_sha256: str,
    learned_input_receipt_sha256: str | None,
    evaluation_wall_seconds: float,
    sequence_id: Any,
    position: Any,
    scores: Any,
) -> dict[str, Any]:
    """Seal one process-local, token-level sufficient-score artifact."""

    destination = Path(root).expanduser().resolve()
    sequence = np.asarray(sequence_id)
    positions = np.asarray(position)
    valid = np.asarray(scores.valid_mask)
    if (
        sequence.dtype != np.int64
        or positions.dtype != np.uint16
        or sequence.shape != valid.shape
        or positions.shape != valid.shape
    ):
        raise RoutedLadderEvaluationError(
            "process token keys do not align with score rows"
        )
    scientific_sequence = np.ascontiguousarray(sequence[valid], dtype=np.int64)
    scientific_position = np.ascontiguousarray(positions[valid], dtype=np.uint16)
    if (
        scientific_sequence.size == 0
        or np.any(scientific_sequence < 0)
        or np.any(scientific_position >= 1_024)
    ):
        raise RoutedLadderEvaluationError(
            "process evidence contains invalid scientific token identities"
        )
    keys = np.rec.fromarrays(
        (scientific_sequence, scientific_position),
        names=("sequence_id", "position"),
    )
    if np.unique(keys).size != keys.size:
        raise RoutedLadderEvaluationError(
            "process evidence contains duplicate token identities"
        )
    score_arrays = _score_arrays(scores, condition=condition)
    tensors = {
        "sequence_id": scientific_sequence,
        "position": scientific_position,
        **{
            name: np.asarray(value, dtype=np.float64)
            for name, value in score_arrays.items()
        },
    }
    data_sha, data_bytes, data_object = _artifact_file(destination, tensors)
    unsigned = {
        "schema_version": PROCESS_EVIDENCE_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "score_schema_version": SCORE_SCHEMA_VERSION,
        "panel": panel,
        "condition": condition,
        "seed": seed,
        "checkpoint_tokens": checkpoint_tokens,
        "checkpoint_actual_tokens": checkpoint_actual_tokens,
        "process_index": process_index,
        "process_count": process_count,
        "source_revision": source_revision,
        "training_source_revision": training_source_revision,
        "config_sha256": config_sha256,
        "config_file_sha256": config_file_sha256,
        "h_config_sha256": h_config_sha256,
        "h_config_file_sha256": h_config_file_sha256,
        "training_result_sha256": training_result_sha256,
        "training_contract_sha256": training_contract_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "target_manifest_sha256": target_manifest_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "recomposition_sidecar_sha256": recomposition_sidecar_sha256,
        "centering_receipt_sha256": centering_receipt_sha256,
        "input_receipt_sha256": input_receipt_sha256,
        "learned_input_receipt_sha256": learned_input_receipt_sha256,
        "evaluation_wall_seconds": float(evaluation_wall_seconds),
        "tokens": int(scientific_sequence.size),
        "token_identity_sha256": token_identity_sha256(
            scientific_sequence,
            scientific_position,
        ),
        "data_object": data_object,
        "data_sha256": data_sha,
        "data_bytes": data_bytes,
    }
    evidence = _self_hash(unsigned, field="evidence_sha256")
    _write_json(destination / "process-evidence.json", evidence)
    return evidence


def load_process_evidence(
    path: str | Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Validate one process evidence envelope and its content-addressed rows."""

    selected = Path(path).expanduser().resolve()
    metadata = _read_json(selected, name="process evidence")
    claimed = _require_sha256(
        "evidence_sha256",
        metadata.get("evidence_sha256"),
    )
    unsigned = {
        name: value
        for name, value in metadata.items()
        if name != "evidence_sha256"
    }
    if (
        canonical_sha256(unsigned) != claimed
        or metadata.get("schema_version") != PROCESS_EVIDENCE_SCHEMA_VERSION
        or metadata.get("protocol_id") != PROTOCOL_ID
        or metadata.get("score_schema_version") != SCORE_SCHEMA_VERSION
    ):
        raise RoutedLadderEvaluationError(
            "process evidence identity or self-hash is invalid"
        )
    tensors = _load_artifact_file(
        selected.parent,
        relative=metadata.get("data_object"),
        digest=metadata.get("data_sha256"),
        size_bytes=metadata.get("data_bytes"),
    )
    expected_tensors = _tensor_names(str(metadata.get("condition")))
    if set(tensors) != expected_tensors:
        raise RoutedLadderEvaluationError(
            "process evidence tensor inventory changed"
        )
    tokens = metadata.get("tokens")
    sequence = tensors["sequence_id"]
    position = tensors["position"]
    if (
        isinstance(tokens, bool)
        or not isinstance(tokens, int)
        or tokens <= 0
        or sequence.dtype != np.int64
        or sequence.shape != (tokens,)
        or position.dtype != np.uint16
        or position.shape != (tokens,)
        or token_identity_sha256(sequence, position)
        != metadata.get("token_identity_sha256")
    ):
        raise RoutedLadderEvaluationError(
            "process evidence token identity proof is invalid"
        )
    keys = np.rec.fromarrays(
        (sequence, position),
        names=("sequence_id", "position"),
    )
    if (
        np.unique(keys).size != tokens
        or np.any(sequence < 0)
        or np.any(position >= 1_024)
    ):
        raise RoutedLadderEvaluationError(
            "process evidence scientific keys are invalid"
        )
    for name in expected_tensors - {"sequence_id", "position"}:
        value = tensors[name]
        if (
            value.dtype != np.float64
            or value.shape != (tokens,)
            or not np.isfinite(value).all()
            or np.any(value < 0)
        ):
            raise RoutedLadderEvaluationError(
                f"process evidence tensor {name} is invalid"
            )
    for name in (
        "source_revision",
        "training_source_revision",
    ):
        _require_git_sha(name, metadata.get(name))
    for name in (
        "config_sha256",
        "config_file_sha256",
        "training_result_sha256",
        "training_contract_sha256",
        "checkpoint_sha256",
        "target_manifest_sha256",
        "recomposition_sidecar_sha256",
        "centering_receipt_sha256",
        "input_receipt_sha256",
        "token_identity_sha256",
        "data_sha256",
    ):
        _require_sha256(name, metadata.get(name))
    source_manifest = metadata.get("source_manifest_sha256")
    if source_manifest is not None:
        _require_sha256("source_manifest_sha256", source_manifest)
    for name in ("h_config_sha256", "h_config_file_sha256"):
        value = metadata.get(name)
        if metadata.get("condition") == "H":
            _require_sha256(name, value)
        elif value is not None:
            raise RoutedLadderEvaluationError(
                f"{name} is valid only for H evidence"
            )
    learned_receipt = metadata.get("learned_input_receipt_sha256")
    if metadata.get("condition") == "H":
        _require_sha256(
            "learned_input_receipt_sha256",
            learned_receipt,
        )
    elif learned_receipt is not None:
        raise RoutedLadderEvaluationError(
            "learned input receipt is valid only for H evidence"
        )
    return metadata, tensors


def _fixed_process_fields() -> tuple[str, ...]:
    return (
        "schema_version",
        "protocol_id",
        "score_schema_version",
        "panel",
        "condition",
        "seed",
        "checkpoint_tokens",
        "checkpoint_actual_tokens",
        "process_count",
        "source_revision",
        "training_source_revision",
        "config_sha256",
        "config_file_sha256",
        "h_config_sha256",
        "h_config_file_sha256",
        "training_result_sha256",
        "training_contract_sha256",
        "checkpoint_sha256",
        "target_manifest_sha256",
        "source_manifest_sha256",
        "recomposition_sidecar_sha256",
        "centering_receipt_sha256",
    )


def merge_process_evidence(
    rows: Sequence[
        tuple[Mapping[str, Any], Mapping[str, np.ndarray]]
    ],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Join complete process-contiguous lanes and reject gaps/substitution."""

    if not rows:
        raise RoutedLadderEvaluationError("no process evidence was supplied")
    ordered = sorted(rows, key=lambda item: int(item[0]["process_index"]))
    count = int(ordered[0][0]["process_count"])
    if (
        count <= 0
        or len(ordered) != count
        or [int(item[0]["process_index"]) for item in ordered]
        != list(range(count))
    ):
        raise RoutedLadderEvaluationError(
            "process evidence lanes are incomplete or duplicated"
        )
    for name in _fixed_process_fields():
        if len({item[0].get(name) for item in ordered}) != 1:
            raise RoutedLadderEvaluationError(
                f"process evidence field {name} differs across lanes"
            )
    expected_tensors = _tensor_names(str(ordered[0][0]["condition"]))
    combined = {
        name: np.ascontiguousarray(
            np.concatenate([item[1][name] for item in ordered], axis=0)
        )
        for name in expected_tensors
    }
    sequence = combined["sequence_id"]
    position = combined["position"]
    keys = np.rec.fromarrays(
        (sequence, position),
        names=("sequence_id", "position"),
    )
    if np.unique(keys).size != sequence.size:
        raise RoutedLadderEvaluationError(
            "merged process evidence has duplicate scientific token keys"
        )
    summary = {
        **{
            name: ordered[0][0][name]
            for name in _fixed_process_fields()
        },
        "tokens": int(sequence.size),
        "packed_sequences": int(np.unique(sequence).size),
        "token_identity_sha256": token_identity_sha256(sequence, position),
        "process_evidence_sha256s": [
            item[0]["evidence_sha256"] for item in ordered
        ],
        "evaluation_wall_seconds": max(
            float(item[0]["evaluation_wall_seconds"]) for item in ordered
        ),
        "evaluation_input_receipt_sha256s": [
            item[0]["input_receipt_sha256"] for item in ordered
        ],
        "learned_evaluation_input_receipt_sha256s": (
            [
                item[0]["learned_input_receipt_sha256"]
                for item in ordered
            ]
            if ordered[0][0]["condition"] == "H"
            else None
        ),
    }
    return summary, combined


def seal_merged_result(
    root: str | Path,
    *,
    summary: Mapping[str, Any],
    tensors: Mapping[str, np.ndarray],
    expected_tokens: int,
    expected_token_identity_sha256: str | None = None,
) -> dict[str, Any]:
    """Seal the process-zero score artifact consumed by the ladder report."""

    destination = Path(root).expanduser().resolve()
    if int(summary["tokens"]) != expected_tokens:
        raise RoutedLadderEvaluationError(
            "merged evidence does not contain the complete held-out capture"
        )
    if (
        expected_token_identity_sha256 is not None
        and _require_sha256(
            "expected_token_identity_sha256",
            expected_token_identity_sha256,
        )
        != summary["token_identity_sha256"]
    ):
        raise RoutedLadderEvaluationError(
            "held-out token identity differs from the common cohort proof"
        )
    condition = str(summary["condition"])
    selected_endpoints = H_ENDPOINTS if condition == "H" else ENDPOINTS
    endpoint_sse = {
        endpoint: np.asarray(tensors[f"{endpoint}_sse"], dtype=np.float64)
        for endpoint in selected_endpoints
    }
    endpoint_sst = {
        endpoint: np.asarray(tensors[f"{endpoint}_sst"], dtype=np.float64)
        for endpoint in selected_endpoints
    }
    score_arguments = {
        "panel": summary["panel"],
        "seed": int(summary["seed"]),
        "checkpoint_tokens": int(summary["checkpoint_tokens"]),
        "source_revision": summary["source_revision"],
        "capture_manifest_sha256": summary["target_manifest_sha256"],
        "recomposition_sidecar_sha256": summary[
            "recomposition_sidecar_sha256"
        ],
        "sequence_id": tensors["sequence_id"],
        "position": tensors["position"],
        "endpoint_sse": endpoint_sse,
        "endpoint_sst": endpoint_sst,
    }
    scores: RoutedLadderTokenScores | RoutedLadderHTokenScores
    if condition == "H":
        scores = RoutedLadderHTokenScores(**score_arguments)
    else:
        scores = RoutedLadderTokenScores(
            condition=condition,
            **score_arguments,
        )
    if (
        scores.tokens != expected_tokens
        or scores.token_identity_sha256 != summary["token_identity_sha256"]
    ):
        raise RoutedLadderEvaluationError(
            "activation-ladder scoring core rejected merged token identity"
        )
    score_tensors = {
        "sequence_id": scores.sequence_id,
        "position": scores.position,
        **{
            f"{endpoint}_sse": scores.endpoint_sse[endpoint]
            for endpoint in selected_endpoints
        },
        **{
            f"{endpoint}_sst": scores.endpoint_sst[endpoint]
            for endpoint in selected_endpoints
        },
    }
    data_sha, data_bytes, data_object = _artifact_file(
        destination,
        score_tensors,
    )
    body = {
        **dict(summary),
        "schema_version": RESULT_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "score_schema_version": SCORE_SCHEMA_VERSION,
        "status": "cell_evaluated",
        "milestone_name": (
            f"{int(summary['checkpoint_tokens']) // 1_000_000}m"
        ),
        "cell_id": (
            f"{summary['panel']}-{summary['condition']}-"
            f"seed{summary['seed']}-"
            f"{int(summary['checkpoint_tokens']) // 1_000_000}m"
        ),
        "score_artifact_sha256": scores.score_artifact_sha256,
        "score_data_object": data_object,
        "score_data_sha256": data_sha,
        "score_data_bytes": data_bytes,
        "endpoints": {
            endpoint: scores.aggregate(endpoint)
            for endpoint in selected_endpoints
        },
        "endpoint_sst_sha256": {
            endpoint: scores.endpoint_sst_sha256(endpoint)
            for endpoint in selected_endpoints
        },
    }
    result = _self_hash(body, field="result_sha256")
    _write_json(destination / "evaluation-result.json", result)
    return result


def load_merged_scores(
    result_path: str | Path,
) -> RoutedLadderTokenScores | RoutedLadderHTokenScores:
    """Load and validate one final cell artifact for report assembly."""

    selected = Path(result_path).expanduser().resolve()
    result = _read_json(selected, name="activation-ladder evaluation result")
    claimed = _require_sha256(
        "result_sha256",
        result.get("result_sha256"),
    )
    unsigned = {
        name: value for name, value in result.items() if name != "result_sha256"
    }
    if (
        canonical_sha256(unsigned) != claimed
        or result.get("schema_version") != RESULT_SCHEMA_VERSION
        or result.get("protocol_id") != PROTOCOL_ID
        or result.get("score_schema_version") != SCORE_SCHEMA_VERSION
        or result.get("status") != "cell_evaluated"
    ):
        raise RoutedLadderEvaluationError(
            "activation-ladder evaluation result is invalid"
        )
    tensors = _load_artifact_file(
        selected.parent,
        relative=result.get("score_data_object"),
        digest=result.get("score_data_sha256"),
        size_bytes=result.get("score_data_bytes"),
    )
    condition = str(result.get("condition"))
    expected_tensors = _tensor_names(condition)
    if set(tensors) != expected_tensors:
        raise RoutedLadderEvaluationError(
            "merged score tensor inventory changed"
        )
    selected_endpoints = H_ENDPOINTS if condition == "H" else ENDPOINTS
    score_arguments = {
        "panel": result["panel"],
        "seed": result["seed"],
        "checkpoint_tokens": result["checkpoint_tokens"],
        "source_revision": result["source_revision"],
        "capture_manifest_sha256": result["target_manifest_sha256"],
        "recomposition_sidecar_sha256": result[
            "recomposition_sidecar_sha256"
        ],
        "sequence_id": tensors["sequence_id"],
        "position": tensors["position"],
        "endpoint_sse": {
            endpoint: tensors[f"{endpoint}_sse"]
            for endpoint in selected_endpoints
        },
        "endpoint_sst": {
            endpoint: tensors[f"{endpoint}_sst"]
            for endpoint in selected_endpoints
        },
    }
    if condition == "H":
        scores: RoutedLadderTokenScores | RoutedLadderHTokenScores = (
            RoutedLadderHTokenScores(**score_arguments)
        )
    else:
        scores = RoutedLadderTokenScores(
            condition=condition,
            **score_arguments,
        )
    if (
        scores.score_artifact_sha256 != result.get("score_artifact_sha256")
        or scores.token_identity_sha256 != result.get("token_identity_sha256")
        or scores.tokens != result.get("tokens")
        or {
            endpoint: scores.aggregate(endpoint)
            for endpoint in selected_endpoints
        }
        != result.get("endpoints")
        or {
            endpoint: scores.endpoint_sst_sha256(endpoint)
            for endpoint in selected_endpoints
        }
        != result.get("endpoint_sst_sha256")
    ):
        raise RoutedLadderEvaluationError(
            "merged score rows differ from the result envelope"
        )
    return scores


def _layout(config: RoutedLadderConfig, condition: str, seed: int) -> Any:
    from moe_sae.contribution_sae import ContributionSAEConfig

    selected_condition = "A512Softplus" if condition == "H" else condition
    matches = [
        item
        for item in config.layouts_for_phase(PHASE_NAME)
        if item.condition == selected_condition
    ]
    if len(matches) != 1:
        raise RoutedLadderEvaluationError(
            "condition has no unique activation-ladder layout"
        )
    selected = matches[0]
    return ContributionSAEConfig(
        condition="A",
        hidden_dim=config.model.hidden_size,
        num_experts=config.model.num_routed_experts,
        top_routed=config.model.experts_per_token,
        bank_width=selected.bank_width,
        k_per_contribution=selected.topk_per_component,
        randomization_key=seed,
    )


def _encoder_activation(
    config: RoutedLadderConfig,
    condition: str,
) -> str:
    """Return the checkpoint's frozen encoder nonlinearity."""

    selected_condition = "A512Softplus" if condition == "H" else condition
    matches = [
        item
        for item in config.layouts_for_phase(PHASE_NAME)
        if item.condition == selected_condition
    ]
    if len(matches) != 1:
        raise RoutedLadderEvaluationError(
            "condition has no unique encoder activation"
        )
    activation = matches[0].encoder_activation
    if activation not in {"relu", "softplus"}:
        raise RoutedLadderEvaluationError(
            "condition encoder activation is invalid"
        )
    return activation


def _optimizer(config: RoutedLadderConfig, plan: Any) -> Any:
    from moe_sae.contribution_training import ContributionOptimizerConfig

    return ContributionOptimizerConfig(
        parameter_dtype=config.optimizer.parameter_dtype,
        adam_beta1=config.optimizer.adam_beta1,
        adam_beta2=config.optimizer.adam_beta2,
        adam_eps=config.optimizer.adam_epsilon,
        grad_clip=config.optimizer.global_gradient_clip,
        peak_lr=config.optimizer.peak_learning_rate,
        min_lr=config.optimizer.minimum_learning_rate,
        warmup_steps=plan.warmup_updates,
        total_steps=plan.planned_updates,
    )


def _training_config(
    plan: Any,
    sae: Any,
    *,
    bank_capacity: int,
) -> Any:
    from moe_sae.pipeline import TrainingConfig

    return TrainingConfig(
        sae=sae,
        seed=plan.seed,
        total_tokens=plan.total_tokens,
        effective_batch_tokens=plan.effective_batch_tokens,
        microbatch_tokens=plan.global_microbatch_tokens,
        checkpoint_every_tokens=plan.total_tokens,
        health_gate_tokens=plan.total_tokens,
        a_capacity=bank_capacity,
    )


def _validate_run_contract(
    run_root: Path,
    *,
    config_path: Path,
    config: RoutedLadderConfig,
    condition: str,
    seed: int,
    training_source_revision: str,
    target_manifest_sha256: str,
    source_manifest_sha256: str | None,
) -> dict[str, Any]:
    contract = _read_json(run_root / "run-contract.json", name="run contract")
    claimed = _require_sha256(
        "run_contract_sha256",
        contract.get("run_contract_sha256"),
    )
    unsigned = {
        name: value
        for name, value in contract.items()
        if name != "run_contract_sha256"
    }
    if canonical_sha256(unsigned) != claimed:
        raise RoutedLadderEvaluationError(
            "training run contract self-hash is invalid"
        )
    expected_file_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
    if (
        contract.get("project_commit") != training_source_revision
        or contract.get("config_file_sha256") != expected_file_sha
        or contract.get("phase") != PHASE_NAME
        or contract.get("condition") != condition
        or contract.get("seed") != seed
        or contract.get("capture_manifest_sha256")
        != target_manifest_sha256
    ):
        raise RoutedLadderEvaluationError(
            "training run contract differs from the requested ladder cell"
        )
    if (
        config.panel == "transcoder"
        and (
            contract.get("scientific_model") != "transcoder"
            or contract.get("source_overlay_manifest_sha256")
            != source_manifest_sha256
        )
    ):
        raise RoutedLadderEvaluationError(
            "transcoder run contract has a different source overlay"
        )
    return contract


def _shared_output_for_update(
    dataset: Any,
    source: Any,
    *,
    update_index: int,
    microbatches: Sequence[Any],
) -> np.ndarray:
    """Read the exact shared-output bits aligned to one process update."""

    shared = np.zeros(
        (len(microbatches), source.local_rows, HIDDEN_SIZE),
        dtype=np.uint16,
    )
    for local_index, microbatch in enumerate(microbatches):
        valid = np.asarray(microbatch.valid_mask)
        count = int(np.count_nonzero(valid))
        if count == 0:
            continue
        microbatch_index = (
            update_index * source.microbatches_per_update + local_index
        )
        start = source.process_range_start + microbatch_index * source.local_rows
        end = start + count
        pieces = tuple(dataset.iter_shared_token_range(start, end))
        selected = np.ascontiguousarray(
            np.concatenate(
                [np.asarray(piece.shared_output) for piece in pieces],
                axis=0,
            )
        )
        sequence = np.ascontiguousarray(
            np.concatenate(
                [np.asarray(piece.sequence_id) for piece in pieces],
                axis=0,
            )
        )
        position = np.ascontiguousarray(
            np.concatenate(
                [np.asarray(piece.position) for piece in pieces],
                axis=0,
            )
        )
        if (
            selected.dtype != np.uint16
            or selected.shape != (count, HIDDEN_SIZE)
            or not np.array_equal(sequence, microbatch.sequence_id[:count])
            or not np.array_equal(position, microbatch.position[:count])
        ):
            raise RoutedLadderEvaluationError(
                "shared-output rows differ from evaluator token identities"
            )
        shared[local_index, :count] = selected
    return shared


def _endpoint_centering_cache_identity(
    *,
    target_manifest_sha256: str,
    recomposition_sidecar_sha256: str,
) -> dict[str, str]:
    """Return the immutable input/algorithm identity for endpoint centering."""

    body = {
        "cache_schema_version": ENDPOINT_CENTERING_CACHE_SCHEMA_VERSION,
        "algorithm_schema_version": (
            ENDPOINT_CENTERING_ALGORITHM_SCHEMA_VERSION
        ),
        "training_target_manifest_sha256": _require_sha256(
            "training_target_manifest_sha256",
            target_manifest_sha256,
        ),
        "recomposition_sidecar_sha256": _require_sha256(
            "recomposition_sidecar_sha256",
            recomposition_sidecar_sha256,
        ),
    }
    return {**body, "cache_key_sha256": canonical_sha256(body)}


def _endpoint_mean_sha256(domain: bytes, value: np.ndarray) -> str:
    selected = np.asarray(value)
    if (
        selected.dtype != np.float32
        or selected.shape != (HIDDEN_SIZE,)
        or not np.isfinite(selected).all()
    ):
        raise RoutedLadderEvaluationError(
            "cached endpoint mean must be finite float32 [2816]"
        )
    digest = hashlib.sha256()
    digest.update(domain + b"\0")
    digest.update(
        np.ascontiguousarray(selected, dtype="<f4").tobytes(order="C")
    )
    return digest.hexdigest()


def _validate_endpoint_centering_values(
    values: tuple[np.ndarray, str, np.ndarray, str, Mapping[str, Any]],
    *,
    identity: Mapping[str, str],
    expected_tokens: int,
) -> tuple[np.ndarray, str, np.ndarray, str, dict[str, Any]]:
    """Validate one computed or cached endpoint-centering value bundle."""

    if not isinstance(values, tuple) or len(values) != 5:
        raise RoutedLadderEvaluationError(
            "endpoint centering must contain five exact values"
        )
    routed_mean = np.ascontiguousarray(values[0])
    routed_sha = _require_sha256("routed_mean_sha256", values[1])
    oracle_mean = np.ascontiguousarray(values[2])
    oracle_sha = _require_sha256("oracle_shared_mean_sha256", values[3])
    receipt = dict(values[4])
    if (
        routed_sha
        != _endpoint_mean_sha256(
            b"gemma4-transcoder-routed-endpoint-mean-v1",
            routed_mean,
        )
        or oracle_sha
        != _endpoint_mean_sha256(
            b"gemma4-transcoder-oracle-shared-endpoint-mean-v1",
            oracle_mean,
        )
    ):
        raise RoutedLadderEvaluationError(
            "endpoint mean arrays differ from their scientific hashes"
        )
    receipt_unsigned = {
        name: value
        for name, value in receipt.items()
        if name != "receipt_sha256"
    }
    receipt_sha = _require_sha256(
        "endpoint receipt_sha256",
        receipt.get("receipt_sha256"),
    )
    if (
        canonical_sha256(receipt_unsigned) != receipt_sha
        or receipt.get("schema_version")
        != ENDPOINT_CENTERING_ALGORITHM_SCHEMA_VERSION
        or receipt.get("training_target_manifest_sha256")
        != identity["training_target_manifest_sha256"]
        or receipt.get("recomposition_sidecar_sha256")
        != identity["recomposition_sidecar_sha256"]
        or receipt.get("tokens") != expected_tokens
        or receipt.get("routed_mean_sha256") != routed_sha
        or receipt.get("oracle_shared_mean_sha256") != oracle_sha
        or receipt.get("reduction_dtype")
        != "float64_high_residual_low_float32_collective"
    ):
        raise RoutedLadderEvaluationError(
            "endpoint centering receipt differs from its immutable inputs"
        )
    return routed_mean, routed_sha, oracle_mean, oracle_sha, receipt


def _endpoint_centering_cache_root(
    cache_root: str | Path,
    identity: Mapping[str, str],
) -> Path:
    return (
        Path(cache_root).expanduser().resolve()
        / "endpoint-centering"
        / identity["cache_key_sha256"]
    )


def _load_endpoint_centering_cache(
    cache_root: str | Path,
    *,
    target_manifest_sha256: str,
    recomposition_sidecar_sha256: str,
    expected_tokens: int,
) -> (
    tuple[np.ndarray, str, np.ndarray, str, dict[str, Any]] | None
):
    """Load one process-private immutable endpoint-centering cache entry."""

    identity = _endpoint_centering_cache_identity(
        target_manifest_sha256=target_manifest_sha256,
        recomposition_sidecar_sha256=recomposition_sidecar_sha256,
    )
    root = _endpoint_centering_cache_root(cache_root, identity)
    metadata_path = root / "centering-cache.json"
    if not metadata_path.exists():
        return None
    if not metadata_path.is_file() or metadata_path.is_symlink():
        raise RoutedLadderEvaluationError(
            "endpoint centering cache metadata is not one regular file"
        )
    metadata = _read_json(
        metadata_path,
        name="endpoint centering cache",
    )
    claimed = _require_sha256(
        "endpoint cache_sha256",
        metadata.get("cache_sha256"),
    )
    unsigned = {
        name: value
        for name, value in metadata.items()
        if name != "cache_sha256"
    }
    if (
        canonical_sha256(unsigned) != claimed
        or metadata.get("schema_version")
        != ENDPOINT_CENTERING_CACHE_SCHEMA_VERSION
        or metadata.get("algorithm_schema_version")
        != ENDPOINT_CENTERING_ALGORITHM_SCHEMA_VERSION
        or metadata.get("cache_key_sha256")
        != identity["cache_key_sha256"]
        or metadata.get("training_target_manifest_sha256")
        != identity["training_target_manifest_sha256"]
        or metadata.get("recomposition_sidecar_sha256")
        != identity["recomposition_sidecar_sha256"]
        or metadata.get("tokens") != expected_tokens
    ):
        raise RoutedLadderEvaluationError(
            "endpoint centering cache identity is invalid"
        )
    tensors = _load_artifact_file(
        root,
        relative=metadata.get("data_object"),
        digest=metadata.get("data_sha256"),
        size_bytes=metadata.get("data_bytes"),
    )
    if set(tensors) != {"routed_mean", "oracle_shared_mean"}:
        raise RoutedLadderEvaluationError(
            "endpoint centering cache tensor inventory changed"
        )
    if metadata.get("data_object") != (
        f"objects/{metadata.get('data_sha256')}.safetensors"
    ):
        raise RoutedLadderEvaluationError(
            "endpoint centering cache object is not content-addressed"
        )
    receipt = metadata.get("endpoint_receipt")
    if not isinstance(receipt, Mapping):
        raise RoutedLadderEvaluationError(
            "endpoint centering cache lacks its scientific receipt"
        )
    return _validate_endpoint_centering_values(
        (
            tensors["routed_mean"],
            metadata.get("routed_mean_sha256"),
            tensors["oracle_shared_mean"],
            metadata.get("oracle_shared_mean_sha256"),
            receipt,
        ),
        identity=identity,
        expected_tokens=expected_tokens,
    )


def _seal_endpoint_centering_cache(
    cache_root: str | Path,
    *,
    target_manifest_sha256: str,
    recomposition_sidecar_sha256: str,
    expected_tokens: int,
    values: tuple[np.ndarray, str, np.ndarray, str, Mapping[str, Any]],
) -> tuple[np.ndarray, str, np.ndarray, str, dict[str, Any]]:
    """Create one cache entry, publishing metadata only after its data object."""

    identity = _endpoint_centering_cache_identity(
        target_manifest_sha256=target_manifest_sha256,
        recomposition_sidecar_sha256=recomposition_sidecar_sha256,
    )
    checked = _validate_endpoint_centering_values(
        values,
        identity=identity,
        expected_tokens=expected_tokens,
    )
    root = _endpoint_centering_cache_root(cache_root, identity)
    data_sha, data_bytes, data_object = _artifact_file(
        root,
        {
            "routed_mean": checked[0],
            "oracle_shared_mean": checked[2],
        },
    )
    unsigned = {
        "schema_version": ENDPOINT_CENTERING_CACHE_SCHEMA_VERSION,
        "algorithm_schema_version": (
            ENDPOINT_CENTERING_ALGORITHM_SCHEMA_VERSION
        ),
        "cache_key_sha256": identity["cache_key_sha256"],
        "training_target_manifest_sha256": (
            identity["training_target_manifest_sha256"]
        ),
        "recomposition_sidecar_sha256": (
            identity["recomposition_sidecar_sha256"]
        ),
        "tokens": expected_tokens,
        "routed_mean_sha256": checked[1],
        "oracle_shared_mean_sha256": checked[3],
        "endpoint_receipt": checked[4],
        "data_object": data_object,
        "data_sha256": data_sha,
        "data_bytes": data_bytes,
    }
    _write_json(
        root / "centering-cache.json",
        _self_hash(unsigned, field="cache_sha256"),
    )
    loaded = _load_endpoint_centering_cache(
        cache_root,
        target_manifest_sha256=target_manifest_sha256,
        recomposition_sidecar_sha256=recomposition_sidecar_sha256,
        expected_tokens=expected_tokens,
    )
    if loaded is None:
        raise RoutedLadderEvaluationError(
            "sealed endpoint centering cache could not be reloaded"
        )
    return loaded


def _cached_endpoint_training_means(
    cache_root: str | Path,
    *,
    target_manifest_sha256: str,
    recomposition_sidecar_sha256: str,
    expected_tokens: int,
    compute: Callable[
        [],
        tuple[
            np.ndarray,
            str,
            np.ndarray,
            str,
            Mapping[str, Any],
        ],
    ],
    force_recompute: bool = False,
) -> tuple[
    tuple[np.ndarray, str, np.ndarray, str, dict[str, Any]],
    Literal["hit", "miss"],
]:
    """Use a valid entry or compute and immutably publish one exact value."""

    if type(force_recompute) is not bool:
        raise RoutedLadderEvaluationError(
            "force_recompute must be boolean"
        )
    cached = _load_endpoint_centering_cache(
        cache_root,
        target_manifest_sha256=target_manifest_sha256,
        recomposition_sidecar_sha256=recomposition_sidecar_sha256,
        expected_tokens=expected_tokens,
    )
    if cached is not None and not force_recompute:
        return cached, "hit"
    computed = compute()
    sealed = _seal_endpoint_centering_cache(
        cache_root,
        target_manifest_sha256=target_manifest_sha256,
        recomposition_sidecar_sha256=recomposition_sidecar_sha256,
        expected_tokens=expected_tokens,
        values=computed,
    )
    return sealed, "miss"


def _centering_receipt(
    *,
    config: RoutedLadderConfig,
    target_manifest_sha256: str,
    source_manifest_sha256: str | None,
    recomposition_sidecar_sha256: str,
    target_mean_sha256: str,
    routed_mean_sha256: str,
    oracle_mean_sha256: str,
    endpoint_receipt_sha256: str,
    endpoint_cache_key_sha256: str,
    source_mean_sha256: str | None,
) -> dict[str, Any]:
    unsigned = {
        "schema_version": CENTERING_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "panel": config.panel,
        "config_sha256": config.config_hash,
        "target_manifest_sha256": target_manifest_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "recomposition_sidecar_sha256": recomposition_sidecar_sha256,
        "target_mean_sha256": target_mean_sha256,
        "source_mean_sha256": source_mean_sha256,
        "routed_mean_sha256": routed_mean_sha256,
        "oracle_shared_mean_sha256": oracle_mean_sha256,
        "endpoint_receipt_sha256": endpoint_receipt_sha256,
        "endpoint_cache_key_sha256": endpoint_cache_key_sha256,
    }
    return _self_hash(unsigned, field="receipt_sha256")


def _extract_h_routed_params(
    params: Any,
    *,
    feature_axis_size: int,
) -> Any:
    """Remove each physical shard's shared slice from a frozen H checkpoint."""

    import jax.numpy as jnp

    from moe_sae.sae import SAEParams

    if (
        isinstance(feature_axis_size, bool)
        or not isinstance(feature_axis_size, int)
        or feature_axis_size <= 0
    ):
        raise RoutedLadderEvaluationError(
            "H routed extraction requires a positive feature axis"
        )
    routed_features = 128 * 512
    total_features = routed_features + 4_096
    if (
        tuple(params.encoder.shape) != (total_features, HIDDEN_SIZE)
        or tuple(params.decoder.shape) != (HIDDEN_SIZE, total_features)
        or tuple(params.bias.shape) != (total_features,)
        or total_features % feature_axis_size
        or routed_features % feature_axis_size
    ):
        raise RoutedLadderEvaluationError(
            "H checkpoint parameter geometry differs from the frozen hybrid"
        )
    local_total = total_features // feature_axis_size
    local_routed = routed_features // feature_axis_size
    encoder = jnp.reshape(
        jnp.reshape(
            params.encoder,
            (feature_axis_size, local_total, HIDDEN_SIZE),
        )[:, :local_routed, :],
        (routed_features, HIDDEN_SIZE),
    )
    decoder = jnp.reshape(
        jnp.reshape(
            params.decoder,
            (HIDDEN_SIZE, feature_axis_size, local_total),
        )[:, :, :local_routed],
        (HIDDEN_SIZE, routed_features),
    )
    bias = jnp.reshape(
        jnp.reshape(
            params.bias,
            (feature_axis_size, local_total),
        )[:, :local_routed],
        (routed_features,),
    )
    return SAEParams(encoder=encoder, decoder=decoder, bias=bias)


def _h_training_binding(
    *,
    h_config_sha256: str,
    h_config_file_sha256: str,
    training_result_sha256: str,
    training_source_revision: str,
    checkpoint_sha256: str,
    target_manifest_sha256: str,
    source_manifest_sha256: str | None,
) -> dict[str, Any]:
    """Seal the immutable H lineage in lieu of an A-worker run contract."""

    unsigned = {
        "schema_version": (
            "gemma4-experiment5-routed-activation-ladder-h-training-binding-v1"
        ),
        "source_protocol_id": (
            "gemma4-routed-shared-hybrid-v1-20260728"
        ),
        "h_config_sha256": h_config_sha256,
        "h_config_file_sha256": h_config_file_sha256,
        "training_result_sha256": training_result_sha256,
        "training_source_revision": training_source_revision,
        "checkpoint_sha256": checkpoint_sha256,
        "target_manifest_sha256": target_manifest_sha256,
        "source_manifest_sha256": source_manifest_sha256,
    }
    return _self_hash(unsigned, field="training_contract_sha256")


def _validate_args(args: argparse.Namespace) -> RoutedLadderConfig:
    config = load_routed_ladder_config(
        args.config,
        expected_panel=args.panel,
    )
    phase = config.phase(PHASE_NAME)
    if args.condition not in EVALUATED_CONDITIONS:
        raise RoutedLadderEvaluationError(
            "condition must be A512ReLU, A512Softplus, or read-only H"
        )
    if args.seed not in phase.seeds:
        raise RoutedLadderEvaluationError("seed must be 0, 1, or 2")
    if args.checkpoint_tokens not in phase.evaluation_checkpoint_tokens:
        raise RoutedLadderEvaluationError(
            "checkpoint must be 2M, 5M, or 10M tokens"
        )
    _require_git_sha("source_revision", args.source_revision)
    _require_git_sha(
        "training_source_revision",
        args.training_source_revision,
    )
    if args.expected_token_identity_sha256 is not None:
        _require_sha256(
            "expected_token_identity_sha256",
            args.expected_token_identity_sha256,
        )
    if args.panel == "transcoder":
        for name in (
            "train_source_root",
            "eval_source_root",
            "train_source_object_prefix",
            "eval_source_object_prefix",
        ):
            if getattr(args, name) is None:
                raise RoutedLadderEvaluationError(
                    f"transcoder panel requires --{name.replace('_', '-')}"
                )
    if args.condition == "H" and args.h_config is None:
        raise RoutedLadderEvaluationError(
            "read-only H rescoring requires --h-config"
        )
    if args.condition != "H" and args.h_config is not None:
        raise RoutedLadderEvaluationError(
            "--h-config is valid only for read-only H rescoring"
        )
    return config


def run(args: argparse.Namespace) -> int:
    """Execute one activation-ladder cell on an initialized multi-host TPU."""

    import jax

    from moe_sae.contribution_evaluation import (
        ContributionDistributedEvaluator,
        _addressable_feature_mask,
    )
    from moe_sae.contribution_evaluation_worker import (
        _checkpoint_a_capacity,
        _load_run,
        _selected_checkpoint_path,
    )
    from moe_sae.contribution_input import (
        CapacityReceipt,
        ContributionProcessInput,
        merge_capacity_receipts,
    )
    from moe_sae.contribution_recomposition import decode_recomposition_sidecar
    from moe_sae.contribution_storage import ContributionDataset
    from moe_sae.contribution_training import optimizer_sae_config
    from moe_sae.contribution_training_worker import (
        TrainingWorkerPlan,
        _collective_barrier,
        _collective_records,
        _pooled_routed_mean,
        _verify_repository_revision,
    )
    from moe_sae.contribution_transcoder_evaluation import (
        ContributionTranscoderDistributedEvaluator,
        score_transcoder_sufficient_rows,
    )
    from moe_sae.contribution_transcoder_evaluation_worker import (
        _endpoint_training_means,
        _evaluation_capacity_roles,
        _target_auxiliary,
        _validate_phase_datasets,
    )
    from moe_sae.contribution_transcoder_input import (
        ContributionTranscoderProcessInput,
        routed_input_mean,
    )
    from moe_sae.contribution_transcoder_storage import RoutedInputDataset
    from moe_sae.checkpointing import milestone_pointer_path
    from moe_sae.contribution_distributed import (
        init_contribution_distributed_state,
    )
    from moe_sae.contribution_transcoder_distributed import (
        init_contribution_transcoder_distributed_state,
    )
    from moe_sae.distributed import (
        DistributedConfig,
        create_mesh,
        restore_distributed_checkpoint,
    )
    from moe_sae.object_store import GCSObjectStore, ShardCache
    from moe_sae.runtime import (
        initialize_jax_distributed,
        require_accelerator_runtime,
    )
    from moe_sae.experiment5_config import load_experiment5_config
    from moe_sae.experiment5_device_evaluation import (
        Experiment5DistributedEvaluator,
        process_local_experiment5_device_scores,
    )
    from moe_sae.experiment5_distributed import (
        init_experiment5_distributed_state,
    )
    from moe_sae.experiment5_evaluation_worker import (
        _checkpoint_training_config as _h_checkpoint_training_config,
        _validate_training_context as _validate_h_training_context,
    )
    from moe_sae.experiment5_hybrid import (
        Experiment5Recomposition,
        production_experiment5_layout,
    )
    from moe_sae.experiment5_input import Experiment5ProcessInput

    config = _validate_args(args)
    config_path = Path(args.config).expanduser().resolve()
    _verify_repository_revision(args.source_revision)
    initialize_jax_distributed()
    require_accelerator_runtime(
        args.accelerator_type,
        peak_flops_per_second=args.peak_flops_per_second,
    )
    if int(jax.process_index()) != args.process_index:
        raise RoutedLadderEvaluationError(
            "physical controller/JAX process mapping changed"
        )
    distributed = DistributedConfig(
        data_axis_size=int(jax.process_count()),
        feature_axis_size=int(jax.local_device_count()),
        microbatches_per_update=(
            config.runtime.effective_batch_tokens
            // config.runtime.microbatch_tokens
        ),
    )
    mesh = create_mesh(distributed)
    sidecar_path = Path(args.recomposition_sidecar).expanduser().resolve()
    sidecar = _read_json(sidecar_path, name="recomposition sidecar")
    decoded_sidecar = decode_recomposition_sidecar(sidecar)
    expected_sidecar_file_sha = config.raw["artifacts"][
        "recomposition_sidecar_file_sha256"
    ]
    if (
        decoded_sidecar.sidecar_sha256 != config.recomposition_sidecar_sha256
        or hashlib.sha256(sidecar_path.read_bytes()).hexdigest()
        != expected_sidecar_file_sha
    ):
        raise RoutedLadderEvaluationError(
            "recomposition sidecar differs from the frozen ladder config"
        )

    object_store: Any = GCSObjectStore()
    if args.regional_mirror_seal is not None:
        from moe_sae.regional_execution import RegionalMirrorStore

        object_store = RegionalMirrorStore(
            _read_json(
                args.regional_mirror_seal,
                name="regional mirror seal",
            ),
            object_store,
        )
    cache = ShardCache(
        Path(args.cache_root),
        object_store,
        max_bytes=args.cache_max_bytes,
    )
    train_target = ContributionDataset(
        args.train_target_root,
        object_prefix=args.train_target_object_prefix,
        cache=cache,
        recomposition_sidecar=sidecar,
    )
    eval_target = ContributionDataset(
        args.eval_target_root,
        object_prefix=args.eval_target_object_prefix,
        cache=cache,
        recomposition_sidecar=sidecar,
    )
    phase = config.phase(PHASE_NAME)
    if (
        train_target.total_tokens != phase.train_tokens
        or eval_target.total_tokens != phase.eval_tokens
        or train_target.manifest_sha256
        != config.parents["train_target_manifest_sha256"]
        or eval_target.manifest_sha256
        != config.parents["eval_target_manifest_sha256"]
    ):
        raise RoutedLadderEvaluationError(
            "target captures differ from the frozen ladder config"
        )
    train_source: Any = None
    eval_source: Any = None
    if config.panel == "transcoder":
        train_source = RoutedInputDataset(
            args.train_source_root,
            object_prefix=args.train_source_object_prefix,
            cache=cache,
            target_manifest=train_target.manifest,
        )
        eval_source = RoutedInputDataset(
            args.eval_source_root,
            object_prefix=args.eval_source_object_prefix,
            cache=cache,
            target_manifest=eval_target.manifest,
        )
        _validate_phase_datasets(
            config,
            phase,
            train_target=train_target,
            train_source=train_source,
            eval_target=eval_target,
            eval_source=eval_source,
        )

    layout = _layout(config, args.condition, args.seed)
    run_root = Path(args.run_root).expanduser().resolve()
    source_manifest_sha = (
        train_source.manifest_sha256 if train_source is not None else None
    )
    h_config: Any = None
    h_layout: Any = None
    h_context: Mapping[str, Any] | None = None
    h_centering: Any = None
    h_config_sha: str | None = None
    h_config_file_sha: str | None = None
    if args.condition == "H":
        h_config_path = Path(args.h_config).expanduser().resolve()
        h_config = load_experiment5_config(
            h_config_path,
            expected_panel=config.panel,
        )
        h_config_sha = h_config.config_hash
        h_config_file_sha = hashlib.sha256(
            h_config_path.read_bytes()
        ).hexdigest()
        if (
            h_config_sha
            != config.existing_h_reference.source_config_sha256
            or h_config.protocol_id
            != config.existing_h_reference.source_protocol_id
        ):
            raise RoutedLadderEvaluationError(
                "H parent config differs from the ladder reference"
            )
        plan = TrainingWorkerPlan.from_config(
            h_config,
            phase=config.existing_h_reference.source_phase_name,
            condition="H",
            seed=args.seed,
        )
        # Experiment5Config itself has no self-reference; the literal phase is
        # already pinned by the ladder's ExistingHReference.
        if plan.phase_name != config.existing_h_reference.source_phase_name:
            raise RoutedLadderEvaluationError(
                "H parent phase differs from the ladder reference"
            )
        result, pointer = _load_run(
            run_root,
            plan,
            args.checkpoint_payload_root,
            checkpoint_target_tokens=args.checkpoint_tokens,
        )
        if result.get("source_revision") != args.training_source_revision:
            raise RoutedLadderEvaluationError(
                "H training result source revision differs from the request"
            )
        h_layout = production_experiment5_layout(
            "H",
            randomization_key=plan.randomization_key,
        )
        (
            h_context,
            _h_execution,
            h_centering,
            _h_centering_identity,
        ) = _validate_h_training_context(
            result,
            config=h_config,
            plan=plan,
            layout=h_layout,
            feature_axis_size=int(jax.local_device_count()),
            global_chip_count=int(
                np.asarray(mesh.devices, dtype=object).size
            ),
            accelerator_type=args.accelerator_type,
            training_target=train_target,
            training_source=train_source,
            sidecar=sidecar,
        )
        binding = _h_training_binding(
            h_config_sha256=h_config_sha,
            h_config_file_sha256=h_config_file_sha,
            training_result_sha256=result["result_sha256"],
            training_source_revision=args.training_source_revision,
            checkpoint_sha256=pointer["pointer_sha256"],
            target_manifest_sha256=train_target.manifest_sha256,
            source_manifest_sha256=source_manifest_sha,
        )
        training_contract_sha = binding["training_contract_sha256"]
    else:
        plan = TrainingWorkerPlan.from_config(
            config,
            phase=PHASE_NAME,
            condition=args.condition,
            seed=args.seed,
        )
        result, pointer = _load_run(
            run_root,
            plan,
            args.checkpoint_payload_root,
            checkpoint_target_tokens=args.checkpoint_tokens,
        )
        run_contract = _validate_run_contract(
            run_root,
            config_path=config_path,
            config=config,
            condition=args.condition,
            seed=args.seed,
            training_source_revision=args.training_source_revision,
            target_manifest_sha256=train_target.manifest_sha256,
            source_manifest_sha256=source_manifest_sha,
        )
        training_contract_sha = run_contract["run_contract_sha256"]
    optimizer = _optimizer(config, plan)

    target_mean, target_mean_sha = _pooled_routed_mean(train_target)
    endpoint_cache_identity = _endpoint_centering_cache_identity(
        target_manifest_sha256=train_target.manifest_sha256,
        recomposition_sidecar_sha256=decoded_sidecar.sidecar_sha256,
    )
    cached_endpoint: (
        tuple[np.ndarray, str, np.ndarray, str, dict[str, Any]] | None
    ) = None
    cache_probe_error: str | None = None
    try:
        cached_endpoint = _load_endpoint_centering_cache(
            args.cache_root,
            target_manifest_sha256=train_target.manifest_sha256,
            recomposition_sidecar_sha256=decoded_sidecar.sidecar_sha256,
            expected_tokens=int(train_target.total_tokens),
        )
    except RoutedLadderEvaluationError as error:
        cache_probe_error = str(error)
    cache_probe_rows = _collective_records(
        {
            "process_index": int(jax.process_index()),
            "cache_key_sha256": endpoint_cache_identity[
                "cache_key_sha256"
            ],
            "hit": cached_endpoint is not None,
            "error": cache_probe_error,
            "routed_mean_sha256": (
                cached_endpoint[1] if cached_endpoint is not None else None
            ),
            "oracle_shared_mean_sha256": (
                cached_endpoint[3] if cached_endpoint is not None else None
            ),
            "endpoint_receipt_sha256": (
                cached_endpoint[4]["receipt_sha256"]
                if cached_endpoint is not None
                else None
            ),
        }
    )
    cache_errors = [
        str(row["error"])
        for row in cache_probe_rows
        if row.get("error") is not None
    ]
    if cache_errors:
        raise RoutedLadderEvaluationError(
            "endpoint centering cache validation failed on at least one "
            f"process: {cache_errors}"
        )
    all_cache_hits = all(bool(row["hit"]) for row in cache_probe_rows)
    if all_cache_hits:
        cache_signatures = {
            (
                row["cache_key_sha256"],
                row["routed_mean_sha256"],
                row["oracle_shared_mean_sha256"],
                row["endpoint_receipt_sha256"],
            )
            for row in cache_probe_rows
        }
        if len(cache_signatures) != 1 or cached_endpoint is None:
            raise RoutedLadderEvaluationError(
                "process-local endpoint caches disagree on one scientific "
                "centering value"
            )
        endpoint_values = cached_endpoint
        endpoint_cache_status = "hit"
    else:
        endpoint_values, endpoint_cache_status = (
            _cached_endpoint_training_means(
                args.cache_root,
                target_manifest_sha256=train_target.manifest_sha256,
                recomposition_sidecar_sha256=(
                    decoded_sidecar.sidecar_sha256
                ),
                expected_tokens=int(train_target.total_tokens),
                compute=lambda: _endpoint_training_means(
                    train_target,
                    sidecar,
                ),
                # If one process missed, every process must enter the
                # multihost centering collectives. Existing entries are
                # revalidated by their create-only replay.
                force_recompute=True,
            )
        )
    (
        routed_mean,
        routed_mean_sha,
        oracle_mean,
        oracle_mean_sha,
        endpoint_receipt,
    ) = endpoint_values
    endpoint_rows = _collective_records(
        {
            "process_index": int(jax.process_index()),
            "cache_key_sha256": endpoint_cache_identity[
                "cache_key_sha256"
            ],
            "routed_mean_sha256": routed_mean_sha,
            "oracle_shared_mean_sha256": oracle_mean_sha,
            "endpoint_receipt_sha256": endpoint_receipt[
                "receipt_sha256"
            ],
        }
    )
    if len(
        {
            (
                row["cache_key_sha256"],
                row["routed_mean_sha256"],
                row["oracle_shared_mean_sha256"],
                row["endpoint_receipt_sha256"],
            )
            for row in endpoint_rows
        }
    ) != 1:
        raise RoutedLadderEvaluationError(
            "endpoint centering differs across process-private caches"
        )
    if int(jax.process_index()) == 0:
        print(
            json.dumps(
                {
                    "action": "endpoint-centering-cache",
                    "cache_key_sha256": endpoint_cache_identity[
                        "cache_key_sha256"
                    ],
                    "status": endpoint_cache_status,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    source_mean: np.ndarray | None = None
    source_mean_sha: str | None = None
    if config.panel == "transcoder":
        source_mean, source_mean_sha = routed_input_mean(train_source)
    if args.condition == "H":
        if h_centering is None:
            raise RoutedLadderEvaluationError("H centering was not restored")
        if not np.array_equal(
            np.asarray(h_centering.routed_target_mean, dtype=np.float32),
            np.asarray(target_mean, dtype=np.float32),
        ) or not np.array_equal(
            np.asarray(h_centering.full_target_mean, dtype=np.float32),
            np.asarray(oracle_mean, dtype=np.float32),
        ):
            raise RoutedLadderEvaluationError(
                "H and activation-ladder target centering arrays differ"
            )
        if config.panel == "transcoder" and not np.array_equal(
            np.asarray(h_centering.source_mean, dtype=np.float32),
            np.asarray(source_mean, dtype=np.float32),
        ):
            raise RoutedLadderEvaluationError(
                "H and activation-ladder transcoder source centering differs"
            )
    centering = _centering_receipt(
        config=config,
        target_manifest_sha256=train_target.manifest_sha256,
        source_manifest_sha256=source_manifest_sha,
        recomposition_sidecar_sha256=decoded_sidecar.sidecar_sha256,
        target_mean_sha256=target_mean_sha,
        routed_mean_sha256=routed_mean_sha,
        oracle_mean_sha256=oracle_mean_sha,
        endpoint_receipt_sha256=endpoint_receipt["receipt_sha256"],
        endpoint_cache_key_sha256=endpoint_cache_identity[
            "cache_key_sha256"
        ],
        source_mean_sha256=source_mean_sha,
    )
    output = Path(args.result_root).expanduser().resolve()
    if int(jax.process_index()) == 0:
        _write_json(output / "centering-receipt.json", centering)
    _collective_barrier(
        f"routed-activation-ladder-centering-{config.panel}-{args.condition}-"
        f"{args.seed}-{args.checkpoint_tokens}"
    )

    if config.panel == "sae":
        source = ContributionProcessInput(
            eval_target,
            layout,
            total_tokens=phase.eval_tokens,
            global_microbatch_tokens=plan.global_microbatch_tokens,
            microbatches_per_update=plan.microbatches_per_update,
            process_count=int(jax.process_count()),
            process_index=int(jax.process_index()),
        )
    else:
        source = ContributionTranscoderProcessInput(
            eval_source,
            eval_target,
            layout,
            total_tokens=phase.eval_tokens,
            global_microbatch_tokens=plan.global_microbatch_tokens,
            microbatches_per_update=plan.microbatches_per_update,
            process_count=int(jax.process_count()),
            process_index=int(jax.process_index()),
        )
    local_capacity = source.scan_local_capacity(
        margin_fraction=config.runtime.capacity_margin_fraction
    )
    capacity_receipts = tuple(
        CapacityReceipt(**row)
        for row in _collective_records(dataclasses.asdict(local_capacity))
    )
    evaluation_capacity = merge_capacity_receipts(capacity_receipts)
    if args.condition == "H":
        if h_context is None:
            raise RoutedLadderEvaluationError(
                "H training context is unavailable"
            )
        training_capacity = h_context.get("bank_capacity")
        if (
            isinstance(training_capacity, bool)
            or not isinstance(training_capacity, int)
            or training_capacity <= 0
        ):
            raise RoutedLadderEvaluationError(
                "H training context has no valid routed capacity"
            )
    else:
        training_capacity = _checkpoint_a_capacity(pointer)
    try:
        capacity_roles = _evaluation_capacity_roles(
            "A",
            training_capacity=training_capacity,
            evaluation_capacity=int(evaluation_capacity.capacity),
        )
    except ValueError as error:
        raise RoutedLadderEvaluationError(
            "checkpoint or held-out grouped capacity is invalid"
        ) from error

    h_state: Any = None
    if args.condition == "H":
        assert h_config is not None
        assert h_layout is not None
        h_state = init_experiment5_distributed_state(
            jax.random.key(args.seed),
            mesh,
            h_layout,
            _optimizer(h_config, plan),
            distributed,
        )
        h_state, checkpoint_metadata = restore_distributed_checkpoint(
            milestone_pointer_path(
                run_root / "checkpoints",
                args.checkpoint_tokens,
            ),
            h_state,
            mesh,
            _h_checkpoint_training_config(
                h_config,
                plan,
                h_layout,
                int(training_capacity),
            ),
            checkpoint_payload_root=args.checkpoint_payload_root,
            checkpoint_identity=result["run_identity"],
            checkpoint_metadata_identity={
                "run_identity_sha256": result["run_identity"][
                    "run_identity_sha256"
                ]
            },
        )
        evaluation_params = _extract_h_routed_params(
            h_state.params,
            feature_axis_size=int(jax.local_device_count()),
        )
    elif config.panel == "sae":
        state = init_contribution_distributed_state(
            jax.random.key(args.seed),
            mesh,
            layout,
            optimizer,
            distributed,
        )
        state, checkpoint_metadata = restore_distributed_checkpoint(
            _selected_checkpoint_path(run_root, args.checkpoint_tokens),
            state,
            mesh,
            _training_config(
                plan,
                optimizer_sae_config(layout, optimizer),
                bank_capacity=int(capacity_roles.checkpoint),
            ),
            checkpoint_payload_root=args.checkpoint_payload_root,
            checkpoint_identity=result["run_identity"],
            checkpoint_metadata_identity={
                "run_identity_sha256": result["run_identity"][
                    "run_identity_sha256"
                ]
            },
        )
        evaluation_params = state.params
    else:
        state = init_contribution_transcoder_distributed_state(
            jax.random.key(args.seed),
            mesh,
            layout,
            optimizer,
            distributed,
        )
        state, checkpoint_metadata = restore_distributed_checkpoint(
            _selected_checkpoint_path(run_root, args.checkpoint_tokens),
            state,
            mesh,
            _training_config(
                plan,
                optimizer_sae_config(layout, optimizer),
                bank_capacity=int(capacity_roles.checkpoint),
            ),
            checkpoint_payload_root=args.checkpoint_payload_root,
            checkpoint_identity=result["run_identity"],
            checkpoint_metadata_identity={
                "run_identity_sha256": result["run_identity"][
                    "run_identity_sha256"
                ]
            },
        )
        evaluation_params = state.params
    if not isinstance(checkpoint_metadata, Mapping):
        raise RoutedLadderEvaluationError(
            "restored checkpoint lacks immutable metadata"
        )
    evaluator: Any
    encoder_activation = _encoder_activation(config, args.condition)
    if config.panel == "sae":
        evaluator = ContributionDistributedEvaluator(
            mesh,
            layout,
            distributed,
            bank_capacity=int(capacity_roles.execution),
            encoder_activation=encoder_activation,
        )
    else:
        evaluator = ContributionTranscoderDistributedEvaluator(
            mesh,
            layout,
            distributed,
            bank_capacity=int(capacity_roles.execution),
            encoder_activation=encoder_activation,
        )

    pieces: dict[str, list[np.ndarray]] = {
        name: [] for name in _ROUTED_TENSORS
    }
    fired = np.zeros((layout.total_features,), dtype=np.bool_)
    started = time.perf_counter()
    for update_index, microbatches in enumerate(source.iter_updates()):
        if config.panel == "sae":
            aux, chunk = evaluator.step(
                evaluation_params,
                microbatches,
                target_mean,
                process_local=int(jax.process_count()) > 1,
            )
            shared_bits = _shared_output_for_update(
                eval_target,
                source,
                update_index=update_index,
                microbatches=microbatches,
            )
        else:
            assert source_mean is not None
            aux, chunk = evaluator.step(
                evaluation_params,
                microbatches,
                source_mean,
                target_mean,
                process_local=int(jax.process_count()) > 1,
            )
            shared_bits, _expert_ids = _target_auxiliary(
                source,
                update_index=update_index,
                microbatches=microbatches,
            )
        fired |= _addressable_feature_mask(aux.fired_mask)
        flat_valid = np.asarray(chunk.valid_mask).reshape(-1)
        scored = score_transcoder_sufficient_rows(
            component_sse=np.asarray(
                chunk.component_token_sse
            ).reshape(-1),
            component_sst=np.asarray(
                chunk.component_token_sst
            ).reshape(-1),
            route_component_sse=np.asarray(
                chunk.component_route_sse
            ).reshape(-1, TOP_K_EXPERTS),
            route_component_sst=np.asarray(
                chunk.component_route_sst
            ).reshape(-1, TOP_K_EXPERTS),
            raw_routed_target=np.asarray(
                chunk.raw_routed_target
            ).reshape(-1, HIDDEN_SIZE),
            raw_routed_reconstruction=np.asarray(
                chunk.raw_routed_reconstruction
            ).reshape(-1, HIDDEN_SIZE),
            shared_output_bf16_bits=shared_bits.reshape(-1, HIDDEN_SIZE),
            routed_mean=routed_mean,
            oracle_shared_mean=oracle_mean,
            valid_mask=flat_valid,
            recomposition_sidecar=sidecar,
        )
        sequence = np.stack(
            [np.asarray(item.sequence_id) for item in microbatches],
            axis=0,
        ).reshape(-1)
        position = np.stack(
            [np.asarray(item.position) for item in microbatches],
            axis=0,
        ).reshape(-1)
        pieces["sequence_id"].append(sequence[flat_valid])
        pieces["position"].append(position[flat_valid])
        for endpoint in ENDPOINTS:
            pieces[f"{endpoint}_sse"].append(
                np.asarray(getattr(scored, f"{endpoint}_sse"))[flat_valid]
            )
            pieces[f"{endpoint}_sst"].append(
                np.asarray(getattr(scored, f"{endpoint}_sst"))[flat_valid]
            )
    combined = {
        name: np.ascontiguousarray(np.concatenate(values, axis=0))
        for name, values in pieces.items()
    }
    expected_local = source.process_range_end - source.process_range_start
    if combined["sequence_id"].size != expected_local:
        raise RoutedLadderEvaluationError(
            "evaluator did not retain its complete process token lane"
        )
    learned_input_receipt_sha: str | None = None
    if args.condition == "H":
        if (
            h_state is None
            or h_layout is None
            or h_context is None
            or h_centering is None
        ):
            raise RoutedLadderEvaluationError(
                "H learned-full evaluator lacks its frozen checkpoint context"
            )
        h_source = Experiment5ProcessInput(
            eval_target,
            sidecar,
            h_layout,
            panel=config.panel,
            total_tokens=phase.eval_tokens,
            source_dataset=eval_source,
            global_microbatch_tokens=plan.global_microbatch_tokens,
            microbatches_per_update=plan.microbatches_per_update,
            process_count=int(jax.process_count()),
            process_index=int(jax.process_index()),
        )
        learned_input_receipt_sha = h_source.receipt.receipt_sha256
        recomposition = Experiment5Recomposition(
            routed_scale=bf16_bits_to_float32(
                decoded_sidecar.routed_rmsnorm_scale_bf16_bits
            ),
            outer_scale=bf16_bits_to_float32(
                decoded_sidecar.outer_ffn_rmsnorm_scale_bf16_bits
            ),
            epsilon=float(decoded_sidecar.epsilon),
        )
        h_evaluator = Experiment5DistributedEvaluator(
            mesh,
            h_layout,
            distributed,
            encoder_kind=config.panel,
            bank_capacity=int(capacity_roles.execution),
            rmsnorm_epsilon=float(recomposition.epsilon),
            expected_physical_layout_sha256=h_context[
                "physical_feature_layout_identity"
            ]["identity_sha256"],
        )
        iterator = iter(h_source.iter_updates())
        try:
            first = next(iterator)
        except StopIteration as error:
            raise RoutedLadderEvaluationError(
                "H learned-full held-out input produced no updates"
            ) from error
        h_evaluator.compile(
            h_state,
            first,
            h_centering,
            recomposition,
            process_local=int(jax.process_count()) > 1,
        )
        learned_sequence: list[np.ndarray] = []
        learned_position: list[np.ndarray] = []
        learned_sse: list[np.ndarray] = []
        learned_device_sst: list[np.ndarray] = []
        for microbatches in itertools.chain((first,), iterator):
            device_scores = h_evaluator.score(
                h_state,
                microbatches,
                h_centering,
                recomposition,
                process_local=int(jax.process_count()) > 1,
            )
            host_scores = process_local_experiment5_device_scores(
                device_scores
            )
            valid_mask = np.stack(
                [np.asarray(item.valid_mask) for item in microbatches],
                axis=0,
            ).reshape(-1)
            sequence = np.stack(
                [np.asarray(item.sequence_id) for item in microbatches],
                axis=0,
            ).reshape(-1)
            position = np.stack(
                [np.asarray(item.position) for item in microbatches],
                axis=0,
            ).reshape(-1)
            learned_sequence.append(sequence[valid_mask])
            learned_position.append(position[valid_mask])
            learned_sse.append(
                np.asarray(host_scores.full_sse).reshape(-1)[valid_mask]
            )
            learned_device_sst.append(
                np.asarray(host_scores.full_sst).reshape(-1)[valid_mask]
            )
        selected_sequence = np.ascontiguousarray(
            np.concatenate(learned_sequence, axis=0)
        )
        selected_position = np.ascontiguousarray(
            np.concatenate(learned_position, axis=0)
        )
        selected_sse = np.ascontiguousarray(
            np.concatenate(learned_sse, axis=0),
            dtype=np.float64,
        )
        selected_device_sst = np.ascontiguousarray(
            np.concatenate(learned_device_sst, axis=0),
            dtype=np.float64,
        )
        canonical_full_sst = np.asarray(
            combined["oracle_shared_sst"],
            dtype=np.float64,
        )
        if (
            not np.array_equal(selected_sequence, combined["sequence_id"])
            or not np.array_equal(selected_position, combined["position"])
            or selected_sse.shape != (expected_local,)
            or selected_device_sst.shape != (expected_local,)
            or not np.allclose(
                selected_device_sst,
                canonical_full_sst,
                rtol=5e-4,
                atol=1e-5
                * max(1.0, float(np.max(canonical_full_sst))),
            )
        ):
            raise RoutedLadderEvaluationError(
                "H learned-full rows differ from the routed common token/SST "
                "cohort"
            )
        # The host scorer supplies the canonical FP64 target-only SST used by
        # both estimators. Device full_sst is a checked FP32 reduction, not a
        # second denominator.
        combined["learned_full_ffn_sse"] = selected_sse
        combined["learned_full_ffn_sst"] = canonical_full_sst.copy()
        del h_evaluator
    valid = np.ones((expected_local,), dtype=np.bool_)
    local_scores = SimpleNamespace(
        component_sse=combined["component_sse"],
        component_sst=combined["component_sst"],
        routed_sse=combined["routed_sse"],
        routed_sst=combined["routed_sst"],
        oracle_shared_sse=combined["oracle_shared_sse"],
        oracle_shared_sst=combined["oracle_shared_sst"],
        learned_full_ffn_sse=combined.get("learned_full_ffn_sse"),
        learned_full_ffn_sst=combined.get("learned_full_ffn_sst"),
        valid_mask=valid,
    )
    elapsed = time.perf_counter() - started
    timing_rows = _collective_records(
        {
            "process_index": int(jax.process_index()),
            "evaluation_wall_seconds": float(elapsed),
        }
    )
    evaluation_seconds = max(
        float(row["evaluation_wall_seconds"]) for row in timing_rows
    )
    lane_root = output / f"process-p{int(jax.process_index()):05d}"
    seal_process_evidence(
        lane_root,
        panel=config.panel,
        condition=args.condition,
        seed=args.seed,
        checkpoint_tokens=args.checkpoint_tokens,
        checkpoint_actual_tokens=int(pointer["tokens_seen"]),
        process_index=int(jax.process_index()),
        process_count=int(jax.process_count()),
        source_revision=args.source_revision,
        training_source_revision=args.training_source_revision,
        config_sha256=config.config_hash,
        config_file_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        h_config_sha256=h_config_sha,
        h_config_file_sha256=h_config_file_sha,
        training_result_sha256=result["result_sha256"],
        training_contract_sha256=training_contract_sha,
        checkpoint_sha256=pointer["pointer_sha256"],
        target_manifest_sha256=eval_target.manifest_sha256,
        source_manifest_sha256=(
            eval_source.manifest_sha256 if eval_source is not None else None
        ),
        recomposition_sidecar_sha256=decoded_sidecar.sidecar_sha256,
        centering_receipt_sha256=centering["receipt_sha256"],
        input_receipt_sha256=source.receipt.receipt_sha256,
        learned_input_receipt_sha256=learned_input_receipt_sha,
        evaluation_wall_seconds=evaluation_seconds,
        sequence_id=combined["sequence_id"],
        position=combined["position"],
        scores=local_scores,
    )
    if args.condition == "H":
        del h_state
    else:
        del state
    jax.clear_caches()
    _collective_barrier(
        f"routed-activation-ladder-evidence-{config.panel}-{args.condition}-"
        f"{args.seed}-{args.checkpoint_tokens}"
    )

    if int(jax.process_index()) == 0:
        rows = [
            load_process_evidence(
                output
                / f"process-p{process_index:05d}"
                / "process-evidence.json"
            )
            for process_index in range(int(jax.process_count()))
        ]
        summary, merged = merge_process_evidence(rows)
        summary = {
            **summary,
            "dead_rate": float(1.0 - np.mean(fired)),
            "dead_rate_scope": (
                "heldout_routed_features_only"
                if args.condition == "H"
                else "heldout_all_features"
            ),
            "total_features": int(
                h_layout.total_features
                if args.condition == "H"
                else layout.total_features
            ),
            "exact_l0": int(
                h_layout.total_l0
                if args.condition == "H"
                else layout.total_l0
            ),
            "capacity_contract": {
                "training_capacity": int(capacity_roles.checkpoint),
                "evaluation_capacity": int(evaluation_capacity.capacity),
                "selected_capacity": int(capacity_roles.execution),
                "evaluation_capacity_receipt_sha256": (
                    evaluation_capacity.receipt_sha256
                ),
            },
        }
        final = seal_merged_result(
            output,
            summary=summary,
            tensors=merged,
            expected_tokens=phase.eval_tokens,
            expected_token_identity_sha256=(
                args.expected_token_identity_sha256
            ),
        )
        print(json.dumps(final, indent=2, sort_keys=True), flush=True)
    _collective_barrier(
        f"routed-activation-ladder-final-{config.panel}-{args.condition}-"
        f"{args.seed}-{args.checkpoint_tokens}"
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--panel", choices=PANELS, required=True)
    parser.add_argument(
        "--condition",
        choices=EVALUATED_CONDITIONS,
        required=True,
    )
    parser.add_argument("--h-config")
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument(
        "--checkpoint-tokens",
        type=int,
        choices=CHECKPOINT_TOKENS,
        required=True,
    )
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--training-source-revision", required=True)
    parser.add_argument("--train-target-root", required=True)
    parser.add_argument("--eval-target-root", required=True)
    parser.add_argument("--train-target-object-prefix", required=True)
    parser.add_argument("--eval-target-object-prefix", required=True)
    parser.add_argument("--train-source-root")
    parser.add_argument("--eval-source-root")
    parser.add_argument("--train-source-object-prefix")
    parser.add_argument("--eval-source-object-prefix")
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--cache-max-bytes", type=int, required=True)
    parser.add_argument("--regional-mirror-seal")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--checkpoint-payload-root", required=True)
    parser.add_argument("--recomposition-sidecar", required=True)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--process-index", type=int, required=True)
    parser.add_argument("--expected-token-identity-sha256")
    parser.add_argument(
        "--accelerator-type",
        choices=("v4-32", "v6e-16", "v5litepod-16"),
        default="v4-32",
    )
    parser.add_argument(
        "--peak-flops-per-second",
        type=int,
        default=4_400_000_000_000_000,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return run(_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CENTERING_SCHEMA_VERSION",
    "CHECKPOINT_TOKENS",
    "PROCESS_EVIDENCE_SCHEMA_VERSION",
    "RESULT_SCHEMA_VERSION",
    "RoutedLadderEvaluationError",
    "load_merged_scores",
    "load_process_evidence",
    "main",
    "merge_process_evidence",
    "run",
    "seal_merged_result",
    "seal_process_evidence",
]
