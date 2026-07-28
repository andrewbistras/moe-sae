"""Strict run contract for the 10M pilot-width learning-horizon study.

The completed Experiment-2 and Experiment-3 configurations are immutable
historical protocols.  This module defines a small successor profile which
reuses their artifact schemas and training operators without changing either
completed protocol.

Checked-in documents are templates because the extended 10M training
manifests do not exist until harvesting seals them.  Training and evaluation
loaders reject templates.  :func:`finalize_learning_horizon_config` is the
only supported transition: it inserts explicit sealed manifest SHA-256s,
writes a new file atomically, and reloads it under the strict execution
contract.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import yaml

from moe_sae.contribution_artifacts import canonical_sha256

SCHEMA_VERSION = "gemma4-pilot-learning-horizon-run-config-v1"
PROTOCOL_ID = "gemma4-pilot-learning-horizon-v1-20260728"
PHASE_NAME = "learning_horizon_10m"
CONDITIONS = ("A", "R", "B", "F")
SCIENTIFIC_MODELS = ("sae", "transcoder")

PENDING_TRAIN_TARGET = "PENDING_SEALED_10M_TARGET_MANIFEST"
PENDING_TRAIN_SOURCE = "PENDING_SEALED_10M_SOURCE_OVERLAY_MANIFEST"
PENDING_SOURCE_CAPTURE_CONTRACT = (
    "PENDING_SEALED_10M_SOURCE_OVERLAY_CAPTURE_CONTRACT"
)

EVAL_TARGET_MANIFEST_SHA256 = (
    "31454e5ce8693521c41a2a95f2ef88d40c5d26c726a0efac5c26d882ed341f5e"
)
EVAL_SOURCE_OVERLAY_MANIFEST_SHA256 = (
    "f68e2fe6e4cf568e694925697dd5d65fe10160a52d357c641892de85ee5c3da7"
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_RE = re.compile(r"^[0-9a-f]{40}$")
_TRAIN_TOKENS = 10_000_000
_EVAL_TOKENS = 250_000
_EFFECTIVE_BATCH_TOKENS = 16_384
_MICROBATCH_TOKENS = 4_096
_PLANNED_UPDATES = 611
_WARMUP_UPDATES = 61
_TAIL_MASK_TOKENS = 10_624
_DIRECT_RUNTIME_MAX_SECONDS = 43_200
_RESTART_TOKENS = tuple(range(1_000_000, 10_000_001, 1_000_000))
_EVALUATION_TOKENS = (2_000_000, 5_000_000, 10_000_000)


class LearningHorizonConfigError(ValueError):
    """A learning-horizon document is unsafe or scientifically incompatible."""


def _mapping(
    value: Any,
    *,
    owner: str,
    keys: set[str],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LearningHorizonConfigError(f"{owner} must be a mapping")
    if set(value) != keys:
        raise LearningHorizonConfigError(
            f"{owner} fields differ: expected {sorted(keys)}, got {sorted(value)}"
        )
    return value


def _exact(value: Any, expected: Any, *, name: str) -> None:
    if value != expected:
        raise LearningHorizonConfigError(
            f"{name} is pinned to {expected!r}, got {value!r}"
        )


def _sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise LearningHorizonConfigError(
            f"{name} must be a lowercase 64-character SHA-256"
        )
    return value


@dataclass(frozen=True, slots=True)
class HorizonProvenance:
    maxtext_fork_commit: str


@dataclass(frozen=True, slots=True)
class HorizonModel:
    hidden_size: int
    num_routed_experts: int
    experts_per_token: int


@dataclass(frozen=True, slots=True)
class HorizonRuntime:
    effective_batch_tokens: int
    microbatch_tokens: int
    capacity_margin_fraction: float
    accelerator_min_useful_mfu: float
    accelerator_qualification_max_seconds: int
    direct_runtime_max_seconds: int


@dataclass(frozen=True, slots=True)
class HorizonOptimizer:
    parameter_dtype: str
    adam_beta1: float
    adam_beta2: float
    adam_epsilon: float
    global_gradient_clip: float
    peak_learning_rate: float
    minimum_learning_rate: float


@dataclass(frozen=True, slots=True)
class HorizonLayout:
    condition: str
    target: str
    bank_identity: str
    bank_count: int
    bank_width: int
    components_per_token: int
    topk_per_component: int

    @property
    def total_features(self) -> int:
        return self.bank_count * self.bank_width

    @property
    def candidates_per_token(self) -> int:
        return self.components_per_token * self.bank_width

    @property
    def total_l0(self) -> int:
        return self.components_per_token * self.topk_per_component


@dataclass(frozen=True, slots=True)
class HorizonPhase:
    name: str
    layout_set: str
    seeds: tuple[int, ...]
    train_tokens: int
    eval_tokens: int
    planned_updates: int
    warmup_updates: int
    tail_mask_tokens: int
    restart_checkpoint_tokens: tuple[int, ...]
    evaluation_checkpoint_tokens: tuple[int, ...]
    initialization: str
    escalation_gates: tuple[str, ...]
    blind_relative_quality_until_health: bool

    @property
    def checkpoint_tokens(self) -> tuple[int, ...]:
        """Compatibility surface consumed by the existing training worker.

        These are rolling crash-recovery saves.  Scientific evaluation points
        remain separately named by ``evaluation_checkpoint_tokens``.
        """

        return self.restart_checkpoint_tokens


@dataclass(frozen=True, slots=True)
class LearningHorizonConfig:
    """Worker-compatible view of one finalized SAE or transcoder contract."""

    raw: Mapping[str, Any]
    config_hash: str
    protocol_id: str
    scientific_model: Literal["sae", "transcoder"]
    encoder_activation: str
    provenance: HorizonProvenance
    model: HorizonModel
    runtime: HorizonRuntime
    optimizer: HorizonOptimizer
    layouts: tuple[HorizonLayout, ...]
    parents: Mapping[str, Any]
    horizon_phase: HorizonPhase

    def validate(self) -> None:
        if canonical_sha256(self.raw) != self.config_hash:
            raise LearningHorizonConfigError(
                "learning-horizon config canonical identity changed"
            )
        if self.protocol_id != PROTOCOL_ID:
            raise LearningHorizonConfigError("learning-horizon protocol drifted")
        if self.scientific_model not in SCIENTIFIC_MODELS:
            raise LearningHorizonConfigError("scientific model is unsupported")
        if tuple(item.condition for item in self.layouts) != CONDITIONS:
            raise LearningHorizonConfigError(
                "learning horizon must contain exactly A/R/B/F"
            )

    def phase(self, name: str) -> HorizonPhase:
        if name != PHASE_NAME:
            raise LearningHorizonConfigError(
                f"learning horizon has one phase named {PHASE_NAME!r}"
            )
        return self.horizon_phase

    def layouts_for_phase(self, name: str) -> tuple[HorizonLayout, ...]:
        self.phase(name)
        return self.layouts

    def as_dict(self) -> dict[str, Any]:
        return json.loads(
            json.dumps(
                self.raw,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )


def _validate_document(
    value: Any,
    *,
    expected_scientific_model: str | None,
    require_finalized: bool,
) -> LearningHorizonConfig:
    top = _mapping(
        value,
        owner="config",
        keys={
            "schema_version",
            "protocol_id",
            "scientific_model",
            "parent_protocol_id",
            "provenance",
            "model",
            "artifacts",
            "training",
            "execution",
            "optimizer",
            "layouts",
            "evaluation",
        },
    )
    _exact(top["schema_version"], SCHEMA_VERSION, name="schema_version")
    _exact(top["protocol_id"], PROTOCOL_ID, name="protocol_id")
    scientific_model = top["scientific_model"]
    if scientific_model not in SCIENTIFIC_MODELS:
        raise LearningHorizonConfigError(
            f"scientific_model must be one of {SCIENTIFIC_MODELS}"
        )
    if (
        expected_scientific_model is not None
        and scientific_model != expected_scientific_model
    ):
        raise LearningHorizonConfigError(
            f"expected a {expected_scientific_model} learning-horizon config, "
            f"got {scientific_model}"
        )
    _exact(
        top["parent_protocol_id"],
        (
            "gemma4-contribution-v1-20260725"
            if scientific_model == "sae"
            else "gemma4-contribution-transcoder-v2-20260727"
        ),
        name="parent_protocol_id",
    )

    provenance = _mapping(
        top["provenance"],
        owner="provenance",
        keys={
            "model_id",
            "model_revision",
            "maxtext_base_commit",
            "maxtext_fork_commit",
            "dataset_name",
            "dataset_config",
            "dataset_revision",
            "train_split",
            "eval_split",
        },
    )
    expected_provenance = {
        "model_id": "google/gemma-4-26B-A4B",
        "model_revision": "24548b62aa021d562695c04aaf7758a1ea47990b",
        "maxtext_base_commit": "48c7da3e28e8f57fe36ddf09d00eec53072a2ea0",
        "maxtext_fork_commit": "73bafe53569eb17dc18542f2568dd591a8db9e28",
        "dataset_name": "allenai/c4",
        "dataset_config": "en",
        "dataset_revision": "1588ec454efa1a09f29cd18ddd04fe05fc8653a2",
        "train_split": "train",
        "eval_split": "validation",
    }
    for name, expected in expected_provenance.items():
        _exact(provenance[name], expected, name=f"provenance.{name}")
    if _GIT_RE.fullmatch(str(provenance["maxtext_fork_commit"])) is None:
        raise LearningHorizonConfigError("MaxText fork commit is invalid")

    model = _mapping(
        top["model"],
        owner="model",
        keys={
            "target_layer_one_indexed",
            "hidden_size",
            "num_routed_experts",
            "experts_per_token",
        },
    )
    for name, expected in {
        "target_layer_one_indexed": 15,
        "hidden_size": 2_816,
        "num_routed_experts": 128,
        "experts_per_token": 8,
    }.items():
        _exact(model[name], expected, name=f"model.{name}")

    artifacts = _mapping(
        top["artifacts"],
        owner="artifacts",
        keys={
            "target_schema_version",
            "source_overlay_schema_version",
            "train_target_manifest_sha256",
            "eval_target_manifest_sha256",
            "train_source_overlay_manifest_sha256",
            "eval_source_overlay_manifest_sha256",
            "source_overlay_capture_contract_sha256",
            "train_tokens",
            "eval_tokens",
            "recomposition_sidecar_schema_version",
        },
    )
    _exact(
        artifacts["target_schema_version"],
        "gemma4-expert-contribution-v2",
        name="artifacts.target_schema_version",
    )
    _exact(
        artifacts["train_tokens"],
        _TRAIN_TOKENS,
        name="artifacts.train_tokens",
    )
    _exact(
        artifacts["eval_tokens"],
        _EVAL_TOKENS,
        name="artifacts.eval_tokens",
    )
    _exact(
        artifacts["eval_target_manifest_sha256"],
        EVAL_TARGET_MANIFEST_SHA256,
        name="artifacts.eval_target_manifest_sha256",
    )
    _exact(
        artifacts["recomposition_sidecar_schema_version"],
        "gemma4-contribution-recomposition-v1",
        name="artifacts.recomposition_sidecar_schema_version",
    )
    if scientific_model == "sae":
        _exact(
            artifacts["source_overlay_schema_version"],
            None,
            name="artifacts.source_overlay_schema_version",
        )
        _exact(
            artifacts["train_source_overlay_manifest_sha256"],
            None,
            name="artifacts.train_source_overlay_manifest_sha256",
        )
        _exact(
            artifacts["eval_source_overlay_manifest_sha256"],
            None,
            name="artifacts.eval_source_overlay_manifest_sha256",
        )
        _exact(
            artifacts["source_overlay_capture_contract_sha256"],
            None,
            name="artifacts.source_overlay_capture_contract_sha256",
        )
    else:
        _exact(
            artifacts["source_overlay_schema_version"],
            "gemma4-routed-input-overlay-v1",
            name="artifacts.source_overlay_schema_version",
        )
        _exact(
            artifacts["eval_source_overlay_manifest_sha256"],
            EVAL_SOURCE_OVERLAY_MANIFEST_SHA256,
            name="artifacts.eval_source_overlay_manifest_sha256",
        )

    pending: list[str] = []
    train_target = artifacts["train_target_manifest_sha256"]
    if train_target == PENDING_TRAIN_TARGET:
        pending.append("train_target_manifest_sha256")
    else:
        _sha256(train_target, name="artifacts.train_target_manifest_sha256")
    train_source = artifacts["train_source_overlay_manifest_sha256"]
    if scientific_model == "transcoder":
        if train_source == PENDING_TRAIN_SOURCE:
            pending.append("train_source_overlay_manifest_sha256")
        else:
            _sha256(
                train_source,
                name="artifacts.train_source_overlay_manifest_sha256",
            )
        source_capture_contract = artifacts[
            "source_overlay_capture_contract_sha256"
        ]
        if source_capture_contract == PENDING_SOURCE_CAPTURE_CONTRACT:
            pending.append("source_overlay_capture_contract_sha256")
        else:
            _sha256(
                source_capture_contract,
                name="artifacts.source_overlay_capture_contract_sha256",
            )
    else:
        source_capture_contract = None
    if require_finalized and pending:
        raise LearningHorizonConfigError(
            "learning-horizon template is not executable; finalize sealed "
            f"manifest bindings first: {', '.join(pending)}"
        )

    training = _mapping(
        top["training"],
        owner="training",
        keys={
            "phase_name",
            "layout_set",
            "seed",
            "initialization",
            "encoder_activation",
            "train_tokens",
            "eval_tokens",
            "effective_batch_tokens",
            "microbatch_tokens",
            "planned_updates",
            "warmup_updates",
            "tail_mask_tokens",
            "source_centering",
            "target_centering",
            "output_bias",
            "contribution_loss",
            "router_weight_math_dtype",
            "topk_requires_strictly_positive",
            "topk_tie_break",
            "capacity_margin_fraction",
            "capacity_overflow_policy",
        },
    )
    expected_training = {
        "phase_name": PHASE_NAME,
        "layout_set": "pilot",
        "seed": 0,
        "initialization": "from_scratch",
        "encoder_activation": "relu" if scientific_model == "sae" else "softplus",
        "train_tokens": _TRAIN_TOKENS,
        "eval_tokens": _EVAL_TOKENS,
        "effective_batch_tokens": _EFFECTIVE_BATCH_TOKENS,
        "microbatch_tokens": _MICROBATCH_TOKENS,
        "planned_updates": _PLANNED_UPDATES,
        "warmup_updates": _WARMUP_UPDATES,
        "tail_mask_tokens": _TAIL_MASK_TOKENS,
        "source_centering": (
            "raw_expert_output"
            if scientific_model == "sae"
            else "one_global_routed_input_mean"
        ),
        "target_centering": "one_pooled_raw_expert_output_mean",
        "output_bias": "none",
        "contribution_loss": "normalized_router_weighted_sse",
        "router_weight_math_dtype": "float32",
        "topk_requires_strictly_positive": True,
        "topk_tie_break": "global_latent_id_ascending",
        "capacity_margin_fraction": 0.10,
        "capacity_overflow_policy": "hard_error",
    }
    for name, expected in expected_training.items():
        _exact(training[name], expected, name=f"training.{name}")
    if (
        math.ceil(_TRAIN_TOKENS / _EFFECTIVE_BATCH_TOKENS)
        != _PLANNED_UPDATES
        or _PLANNED_UPDATES * _EFFECTIVE_BATCH_TOKENS - _TRAIN_TOKENS
        != _TAIL_MASK_TOKENS
    ):
        raise AssertionError("internal learning-horizon schedule arithmetic drifted")

    execution = _mapping(
        top["execution"],
        owner="execution",
        keys={
            "accelerator_min_useful_mfu",
            "accelerator_qualification_max_seconds",
            "direct_runtime_max_seconds",
            "restart_checkpoint_tokens",
        },
    )
    _exact(
        execution["accelerator_min_useful_mfu"],
        0.10,
        name="execution.accelerator_min_useful_mfu",
    )
    _exact(
        execution["accelerator_qualification_max_seconds"],
        600,
        name="execution.accelerator_qualification_max_seconds",
    )
    _exact(
        execution["direct_runtime_max_seconds"],
        _DIRECT_RUNTIME_MAX_SECONDS,
        name="execution.direct_runtime_max_seconds",
    )
    _exact(
        execution["restart_checkpoint_tokens"],
        list(_RESTART_TOKENS),
        name="execution.restart_checkpoint_tokens",
    )

    optimizer = _mapping(
        top["optimizer"],
        owner="optimizer",
        keys={
            "parameter_dtype",
            "matmul_dtype",
            "moment_dtype",
            "reduction_dtype",
            "adam_beta1",
            "adam_beta2",
            "adam_epsilon",
            "weight_decay",
            "global_gradient_clip",
            "peak_learning_rate",
            "minimum_learning_rate",
            "schedule",
            "decoder_init",
            "encoder_init",
            "encoder_bias_init",
            "tangent_decoder_gradient_projection",
            "decoder_column_renormalization",
        },
    )
    expected_optimizer = {
        "parameter_dtype": "bfloat16",
        "matmul_dtype": "bfloat16",
        "moment_dtype": "float32",
        "reduction_dtype": "float32",
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_epsilon": 1e-8,
        "weight_decay": 0.0,
        "global_gradient_clip": 1.0,
        "peak_learning_rate": 3e-4,
        "minimum_learning_rate": 3e-5,
        "schedule": "linear_warmup_cosine_decay",
        "decoder_init": "unit_norm_columns",
        "encoder_init": "decoder_transpose",
        "encoder_bias_init": "zeros",
        "tangent_decoder_gradient_projection": True,
        "decoder_column_renormalization": True,
    }
    for name, expected in expected_optimizer.items():
        _exact(optimizer[name], expected, name=f"optimizer.{name}")

    raw_layouts = top["layouts"]
    if not isinstance(raw_layouts, list) or len(raw_layouts) != len(CONDITIONS):
        raise LearningHorizonConfigError("layouts must contain exactly A/R/B/F")
    expected_layouts = {
        "A": ("true_expert_id", 128, 256),
        "R": ("per_sequence_randomized_expert_id", 128, 256),
        "B": ("router_rank", 8, 256),
        "F": ("router_rank", 8, 4_096),
    }
    layouts: list[HorizonLayout] = []
    for index, condition in enumerate(CONDITIONS):
        item = _mapping(
            raw_layouts[index],
            owner=f"layouts[{index}]",
            keys={
                "condition",
                "target",
                "bank_identity",
                "bank_count",
                "bank_width",
                "components_per_token",
                "topk_per_component",
            },
        )
        identity, banks, width = expected_layouts[condition]
        expected_item = {
            "condition": condition,
            "target": "raw_selected_expert_output",
            "bank_identity": identity,
            "bank_count": banks,
            "bank_width": width,
            "components_per_token": 8,
            "topk_per_component": 2,
        }
        for name, expected in expected_item.items():
            _exact(item[name], expected, name=f"layouts[{index}].{name}")
        layouts.append(
            HorizonLayout(
                condition=condition,
                target=str(item["target"]),
                bank_identity=str(item["bank_identity"]),
                bank_count=int(item["bank_count"]),
                bank_width=int(item["bank_width"]),
                components_per_token=int(item["components_per_token"]),
                topk_per_component=int(item["topk_per_component"]),
            )
        )

    evaluation = _mapping(
        top["evaluation"],
        owner="evaluation",
        keys={
            "checkpoint_tokens",
            "relative_quality_gates",
            "common_target_sst",
            "bootstrap_unit",
            "bootstrap_replicates",
            "paired_confidence_level",
            "primary_contrasts",
        },
    )
    expected_evaluation = {
        "checkpoint_tokens": list(_EVALUATION_TOKENS),
        "relative_quality_gates": [],
        "common_target_sst": True,
        "bootstrap_unit": "packed_sequence",
        "bootstrap_replicates": 20_000,
        "paired_confidence_level": 0.95,
        "primary_contrasts": ["A-R", "A-B", "A-F", "F-B"],
    }
    for name, expected in expected_evaluation.items():
        _exact(evaluation[name], expected, name=f"evaluation.{name}")
    if not set(_EVALUATION_TOKENS).issubset(_RESTART_TOKENS):
        raise AssertionError("curve points must have durable restart checkpoints")

    canonical = json.loads(
        json.dumps(
            dict(top),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    parents = {
        "target_schema_version": artifacts["target_schema_version"],
        "train_target_manifest_sha256": train_target,
        "eval_target_manifest_sha256": artifacts["eval_target_manifest_sha256"],
        "train_source_overlay_manifest_sha256": train_source,
        "eval_source_overlay_manifest_sha256": artifacts[
            "eval_source_overlay_manifest_sha256"
        ],
        "source_overlay_capture_contract_sha256": source_capture_contract,
        "train_tokens": _TRAIN_TOKENS,
        "eval_tokens": _EVAL_TOKENS,
        "recomposition_sidecar_schema_version": artifacts[
            "recomposition_sidecar_schema_version"
        ],
    }
    result = LearningHorizonConfig(
        raw=canonical,
        config_hash=canonical_sha256(canonical),
        protocol_id=PROTOCOL_ID,
        scientific_model=scientific_model,
        encoder_activation=str(training["encoder_activation"]),
        provenance=HorizonProvenance(
            maxtext_fork_commit=str(provenance["maxtext_fork_commit"])
        ),
        model=HorizonModel(
            hidden_size=int(model["hidden_size"]),
            num_routed_experts=int(model["num_routed_experts"]),
            experts_per_token=int(model["experts_per_token"]),
        ),
        runtime=HorizonRuntime(
            effective_batch_tokens=int(training["effective_batch_tokens"]),
            microbatch_tokens=int(training["microbatch_tokens"]),
            capacity_margin_fraction=float(training["capacity_margin_fraction"]),
            accelerator_min_useful_mfu=float(
                execution["accelerator_min_useful_mfu"]
            ),
            accelerator_qualification_max_seconds=int(
                execution["accelerator_qualification_max_seconds"]
            ),
            direct_runtime_max_seconds=int(
                execution["direct_runtime_max_seconds"]
            ),
        ),
        optimizer=HorizonOptimizer(
            parameter_dtype=str(optimizer["parameter_dtype"]),
            adam_beta1=float(optimizer["adam_beta1"]),
            adam_beta2=float(optimizer["adam_beta2"]),
            adam_epsilon=float(optimizer["adam_epsilon"]),
            global_gradient_clip=float(optimizer["global_gradient_clip"]),
            peak_learning_rate=float(optimizer["peak_learning_rate"]),
            minimum_learning_rate=float(optimizer["minimum_learning_rate"]),
        ),
        layouts=tuple(layouts),
        parents=parents,
        horizon_phase=HorizonPhase(
            name=PHASE_NAME,
            layout_set="pilot",
            seeds=(0,),
            train_tokens=_TRAIN_TOKENS,
            eval_tokens=_EVAL_TOKENS,
            planned_updates=_PLANNED_UPDATES,
            warmup_updates=_WARMUP_UPDATES,
            tail_mask_tokens=_TAIL_MASK_TOKENS,
            restart_checkpoint_tokens=_RESTART_TOKENS,
            evaluation_checkpoint_tokens=_EVALUATION_TOKENS,
            initialization="from_scratch",
            escalation_gates=(),
            blind_relative_quality_until_health=False,
        ),
    )
    result.validate()
    return result


def load_learning_horizon_config(
    path: str | Path,
    *,
    expected_scientific_model: str | None = None,
    require_finalized: bool = True,
) -> LearningHorizonConfig:
    """Load one strict horizon document.

    ``require_finalized=False`` exists only for inspection and the deterministic
    finalizer.  Accelerator-facing callers must retain the default.
    """

    selected = Path(path).expanduser().resolve()
    try:
        raw = yaml.safe_load(selected.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise LearningHorizonConfigError(
            f"cannot load learning-horizon config {selected}"
        ) from error
    return learning_horizon_config_from_mapping(
        raw,
        expected_scientific_model=expected_scientific_model,
        require_finalized=require_finalized,
    )


def learning_horizon_config_from_mapping(
    value: Any,
    *,
    expected_scientific_model: str | None = None,
    require_finalized: bool = True,
) -> LearningHorizonConfig:
    return _validate_document(
        value,
        expected_scientific_model=expected_scientific_model,
        require_finalized=require_finalized,
    )


def finalize_learning_horizon_config(
    template_path: str | Path,
    destination_path: str | Path,
    *,
    train_target_manifest_sha256: str,
    train_source_overlay_manifest_sha256: str | None = None,
    source_overlay_capture_contract_sha256: str | None = None,
) -> LearningHorizonConfig:
    """Bind sealed 10M artifact identities and atomically write a run config."""

    template = load_learning_horizon_config(
        template_path,
        require_finalized=False,
    )
    target_sha = _sha256(
        train_target_manifest_sha256,
        name="train_target_manifest_sha256",
    )
    if template.scientific_model == "transcoder":
        source_sha = _sha256(
            train_source_overlay_manifest_sha256,
            name="train_source_overlay_manifest_sha256",
        )
        source_capture_contract_sha = _sha256(
            source_overlay_capture_contract_sha256,
            name="source_overlay_capture_contract_sha256",
        )
    elif train_source_overlay_manifest_sha256 is not None:
        raise LearningHorizonConfigError(
            "SAE horizon config cannot bind a routed-input overlay"
        )
    else:
        source_sha = None
        if source_overlay_capture_contract_sha256 is not None:
            raise LearningHorizonConfigError(
                "SAE horizon config cannot bind an overlay capture contract"
            )
        source_capture_contract_sha = None

    raw = template.as_dict()
    raw["artifacts"]["train_target_manifest_sha256"] = target_sha
    raw["artifacts"]["train_source_overlay_manifest_sha256"] = source_sha
    raw["artifacts"][
        "source_overlay_capture_contract_sha256"
    ] = source_capture_contract_sha
    finalized = learning_horizon_config_from_mapping(
        raw,
        expected_scientific_model=template.scientific_model,
        require_finalized=True,
    )

    destination = Path(destination_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        existing = load_learning_horizon_config(
            destination,
            expected_scientific_model=template.scientific_model,
        )
        if existing.config_hash != finalized.config_hash:
            raise LearningHorizonConfigError(
                "finalized destination already exists with different content"
            )
        return existing
    temporary = destination.with_name(
        f".{destination.name}.tmp-{os.getpid()}"
    )
    payload = yaml.safe_dump(finalized.as_dict(), sort_keys=False)
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            existing = load_learning_horizon_config(
                destination,
                expected_scientific_model=template.scientific_model,
            )
            if existing.config_hash != finalized.config_hash:
                raise LearningHorizonConfigError(
                    "concurrent finalization published different content"
                ) from None
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return load_learning_horizon_config(
        destination,
        expected_scientific_model=template.scientific_model,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    finalize = commands.add_parser("finalize")
    finalize.add_argument("--template", required=True)
    finalize.add_argument("--output", required=True)
    finalize.add_argument("--train-target-manifest-sha256", required=True)
    finalize.add_argument("--train-source-overlay-manifest-sha256")
    finalize.add_argument("--source-overlay-capture-contract-sha256")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command != "finalize":  # pragma: no cover - argparse is exhaustive.
        raise AssertionError("unknown command")
    config = finalize_learning_horizon_config(
        args.template,
        args.output,
        train_target_manifest_sha256=args.train_target_manifest_sha256,
        train_source_overlay_manifest_sha256=(
            args.train_source_overlay_manifest_sha256
        ),
        source_overlay_capture_contract_sha256=(
            args.source_overlay_capture_contract_sha256
        ),
    )
    print(
        json.dumps(
            {
                "config": str(Path(args.output).expanduser().resolve()),
                "config_sha256": config.config_hash,
                "scientific_model": config.scientific_model,
                "phase": config.horizon_phase.name,
            },
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "CONDITIONS",
    "EVAL_SOURCE_OVERLAY_MANIFEST_SHA256",
    "EVAL_TARGET_MANIFEST_SHA256",
    "LearningHorizonConfig",
    "LearningHorizonConfigError",
    "PHASE_NAME",
    "PROTOCOL_ID",
    "SCHEMA_VERSION",
    "finalize_learning_horizon_config",
    "learning_horizon_config_from_mapping",
    "load_learning_horizon_config",
]


if __name__ == "__main__":
    raise SystemExit(main())
