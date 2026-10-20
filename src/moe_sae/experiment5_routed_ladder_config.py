"""Immutable configuration boundary for the Experiment-5 activation ladder.

The activation ladder answers the ablation left open by Experiment 5 without
changing either completed parent protocol:

* ``A512ReLU`` is the full-sized true-ID routed condition with ReLU;
* ``A512Softplus`` changes only the encoder activation to Softplus;
* ``H`` is referenced from Experiment 5 and is never trained by this protocol.

Only the two full-sized A conditions are executable layouts.  Their public
condition names keep result identities distinct, while
:attr:`RoutedLadderLayout.worker_condition` maps both layouts onto the
existing condition-A grouped kernel.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from moe_sae.contribution_artifacts import canonical_sha256

SCHEMA_VERSION = "gemma4-experiment5-routed-activation-ladder-run-config-v1"
PROTOCOL_ID = "gemma4-experiment5-routed-activation-ladder-v1-20260729"
PHASE_NAME = "experiment5_routed_activation_ladder_10m"
CONDITIONS = ("A512ReLU", "A512Softplus")
PANELS = ("sae", "transcoder")

EXPERIMENT4_PROTOCOL_ID = "gemma4-pilot-learning-horizon-v1-20260728"
EXPERIMENT5_PROTOCOL_ID = "gemma4-routed-shared-hybrid-v1-20260728"

TRAIN_TARGET_SHA256 = "f29aaf45d9ff93be08cc3d175cdcea9fa4aecc37fc0e256d09720e88e1ae838c"
EVAL_TARGET_SHA256 = "31454e5ce8693521c41a2a95f2ef88d40c5d26c726a0efac5c26d882ed341f5e"
TRAIN_SOURCE_SHA256 = "8f27dcf9c1dc0c7f3ffa2e2941daec13d8a72c7204077f09420e15aacb02ffe0"
EVAL_SOURCE_SHA256 = "f68e2fe6e4cf568e694925697dd5d65fe10160a52d357c641892de85ee5c3da7"
SOURCE_CAPTURE_CONTRACT_SHA256 = (
    "fe0b004ca6e48c9eb8163237a5ea20f45b33acdbd9a3a267a9fb4e670ed2c954"
)
RECOMPOSITION_SIDECAR_SHA256 = (
    "37c7b8c012f8fc0c1f63210986c0b3071a261261c88ee4bc1f502220ce3b8526"
)
RECOMPOSITION_SIDECAR_FILE_SHA256 = (
    "6486288677e81665b21815879729c424c6acb8c7d0e45be264b961a9d652fe3d"
)

H_CONFIG_SHA256 = {
    "sae": "e43b0b3f8a6a5ceac65e955dfde2161959db2266697b265a01426d8498c661fe",
    "transcoder": ("bee06b96362f2a973e7b5570b62bd8a64eefb7c0309fd51b3d07c2035f736545"),
}

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
_RESTART_CHECKPOINTS = tuple(range(1_000_000, 10_000_001, 1_000_000))
_EVALUATION_CHECKPOINTS = (2_000_000, 5_000_000, 10_000_000)


class RoutedLadderConfigError(ValueError):
    """A routed-ladder document differs from its immutable protocol."""


def _mapping(
    value: Any,
    *,
    name: str,
    keys: set[str] | None = None,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RoutedLadderConfigError(f"{name} must be a mapping")
    if keys is not None and set(value) != keys:
        raise RoutedLadderConfigError(
            f"{name} fields differ: expected {sorted(keys)}, got {sorted(value)}"
        )
    return value


def _exact(value: Any, expected: Any, *, name: str) -> None:
    if value != expected:
        raise RoutedLadderConfigError(
            f"{name} is frozen to {expected!r}, got {value!r}"
        )


def _sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise RoutedLadderConfigError(
            f"{name} must be a lowercase 64-character SHA-256"
        )
    return value


def _finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RoutedLadderConfigError(f"{name} must be numeric")
    selected = float(value)
    if not math.isfinite(selected):
        raise RoutedLadderConfigError(f"{name} must be finite")
    return selected


@dataclass(frozen=True, slots=True)
class RoutedLadderProvenance:
    maxtext_fork_commit: str


@dataclass(frozen=True, slots=True)
class RoutedLadderModel:
    hidden_size: int
    num_routed_experts: int
    experts_per_token: int


@dataclass(frozen=True, slots=True)
class RoutedLadderRuntime:
    effective_batch_tokens: int
    microbatch_tokens: int
    capacity_margin_fraction: float
    accelerator_min_useful_mfu: float
    accelerator_qualification_max_seconds: int
    direct_runtime_max_seconds: int


@dataclass(frozen=True, slots=True)
class RoutedLadderOptimizer:
    parameter_dtype: str
    adam_beta1: float
    adam_beta2: float
    adam_epsilon: float
    global_gradient_clip: float
    peak_learning_rate: float
    minimum_learning_rate: float


@dataclass(frozen=True, slots=True)
class RoutedLadderLayout:
    condition: str
    base_condition: str
    target: str
    bank_identity: str
    bank_count: int
    bank_width: int
    components_per_token: int
    topk_per_component: int
    total_features: int
    candidates_per_token: int
    total_l0: int
    trainable_parameters: int
    encoder_activation: str
    softplus_positive_floor: str | None

    @property
    def worker_condition(self) -> str:
        """Condition understood by the existing grouped contribution kernel."""

        return self.base_condition


@dataclass(frozen=True, slots=True)
class RoutedLadderPhase:
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

    @property
    def checkpoint_tokens(self) -> tuple[int, ...]:
        """Rolling recovery checkpoints consumed by ``TrainingWorkerPlan``."""

        return self.restart_checkpoint_tokens


@dataclass(frozen=True, slots=True)
class ExistingHReference:
    condition: str
    source_protocol_id: str
    source_config_sha256: str
    source_phase_name: str
    training_policy: str
    encoder_activation: str
    objective: str
    checkpoint_tokens: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RoutedLadderConfig:
    """Worker-compatible view of one immutable routed-ladder panel."""

    raw: Mapping[str, Any]
    config_hash: str
    protocol_id: str
    panel: Literal["sae", "transcoder"]
    scientific_model: Literal["sae", "transcoder"]
    provenance: RoutedLadderProvenance
    model: RoutedLadderModel
    runtime: RoutedLadderRuntime
    optimizer: RoutedLadderOptimizer
    layouts: tuple[RoutedLadderLayout, ...]
    parents: Mapping[str, Any]
    parent_protocol_ids: Mapping[str, str]
    ladder_phase: RoutedLadderPhase
    existing_h_reference: ExistingHReference
    recomposition_sidecar_sha256: str

    def validate(self) -> None:
        if canonical_sha256(self.raw) != self.config_hash:
            raise RoutedLadderConfigError("routed-ladder config identity changed")
        if self.protocol_id != PROTOCOL_ID or self.panel not in PANELS:
            raise RoutedLadderConfigError("routed-ladder protocol identity drifted")
        if tuple(item.condition for item in self.layouts) != CONDITIONS:
            raise RoutedLadderConfigError(
                "layouts must be exactly A512ReLU/A512Softplus"
            )
        if any(item.worker_condition != "A" for item in self.layouts):
            raise RoutedLadderConfigError(
                "routed-ladder layouts must reuse condition-A worker semantics"
            )
        if (
            self.existing_h_reference.condition != "H"
            or self.existing_h_reference.training_policy
            != "reuse_existing_artifacts_only"
        ):
            raise RoutedLadderConfigError("H must remain a read-only reference")

    def phase(self, name: str) -> RoutedLadderPhase:
        if name != PHASE_NAME:
            raise RoutedLadderConfigError(
                f"routed ladder has one phase named {PHASE_NAME!r}"
            )
        return self.ladder_phase

    def layouts_for_phase(self, name: str) -> tuple[RoutedLadderLayout, ...]:
        self.phase(name)
        return self.layouts

    def layout_for_condition(self, condition: str) -> RoutedLadderLayout:
        matches = tuple(item for item in self.layouts if item.condition == condition)
        if len(matches) != 1:
            raise RoutedLadderConfigError(
                f"condition {condition!r} does not name one trainable layout"
            )
        return matches[0]

    def as_dict(self) -> dict[str, Any]:
        return json.loads(
            json.dumps(
                self.raw,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )


def _validate_layout(value: Any, *, index: int) -> RoutedLadderLayout:
    raw = _mapping(
        value,
        name=f"layouts[{index}]",
        keys={
            "condition",
            "base_condition",
            "target",
            "bank_identity",
            "bank_count",
            "bank_width",
            "components_per_token",
            "topk_per_component",
            "total_features",
            "candidates_per_token",
            "exact_total_l0",
            "trainable_parameters",
            "encoder_activation",
            "softplus_positive_floor",
        },
    )
    condition = CONDITIONS[index]
    activation = ("relu", "softplus")[index]
    width, topk = 512, 4
    features = 128 * width
    candidates = 8 * width
    l0 = 8 * topk
    parameters = features * (2 * 2_816 + 1)
    expected = {
        "condition": condition,
        "base_condition": "A",
        "target": "raw_selected_expert_output",
        "bank_identity": "true_expert_id",
        "bank_count": 128,
        "bank_width": width,
        "components_per_token": 8,
        "topk_per_component": topk,
        "total_features": features,
        "candidates_per_token": candidates,
        "exact_total_l0": l0,
        "trainable_parameters": parameters,
        "encoder_activation": activation,
        "softplus_positive_floor": (
            None if activation == "relu" else "smallest_normal_float32"
        ),
    }
    for key, expected_value in expected.items():
        _exact(raw[key], expected_value, name=f"layouts[{index}].{key}")
    return RoutedLadderLayout(
        condition=condition,
        base_condition="A",
        target="raw_selected_expert_output",
        bank_identity="true_expert_id",
        bank_count=128,
        bank_width=width,
        components_per_token=8,
        topk_per_component=topk,
        total_features=features,
        candidates_per_token=candidates,
        total_l0=l0,
        trainable_parameters=parameters,
        encoder_activation=activation,
        softplus_positive_floor=(
            None if activation == "relu" else "smallest_normal_float32"
        ),
    )


def _validate_document(
    value: Any,
    *,
    expected_panel: str | None,
) -> RoutedLadderConfig:
    top = _mapping(
        value,
        name="config",
        keys={
            "schema_version",
            "protocol_id",
            "panel",
            "parent_protocol_ids",
            "provenance",
            "model",
            "artifacts",
            "training",
            "optimizer",
            "layouts",
            "existing_h_reference",
            "execution",
            "evaluation",
        },
    )
    _exact(top["schema_version"], SCHEMA_VERSION, name="schema_version")
    _exact(top["protocol_id"], PROTOCOL_ID, name="protocol_id")
    panel = top["panel"]
    if panel not in PANELS:
        raise RoutedLadderConfigError(f"panel must be one of {PANELS}")
    if expected_panel is not None and panel != expected_panel:
        raise RoutedLadderConfigError(
            f"expected a {expected_panel} routed-ladder config, got {panel}"
        )

    parent_protocols = _mapping(
        top["parent_protocol_ids"],
        name="parent_protocol_ids",
        keys={"experiment_4", "experiment_5"},
    )
    _exact(
        parent_protocols["experiment_4"],
        EXPERIMENT4_PROTOCOL_ID,
        name="parent_protocol_ids.experiment_4",
    )
    _exact(
        parent_protocols["experiment_5"],
        EXPERIMENT5_PROTOCOL_ID,
        name="parent_protocol_ids.experiment_5",
    )

    provenance = _mapping(
        top["provenance"],
        name="provenance",
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
    for key, expected in expected_provenance.items():
        _exact(provenance[key], expected, name=f"provenance.{key}")
    fork = str(provenance["maxtext_fork_commit"])
    if _GIT_RE.fullmatch(fork) is None:
        raise RoutedLadderConfigError("maxtext_fork_commit must be a Git SHA")

    model = _mapping(
        top["model"],
        name="model",
        keys={
            "target_layer_one_indexed",
            "hidden_size",
            "num_routed_experts",
            "experts_per_token",
            "shared_experts",
        },
    )
    expected_model = {
        "target_layer_one_indexed": 15,
        "hidden_size": 2_816,
        "num_routed_experts": 128,
        "experts_per_token": 8,
        "shared_experts": 1,
    }
    for key, expected in expected_model.items():
        _exact(model[key], expected, name=f"model.{key}")

    artifacts = _mapping(
        top["artifacts"],
        name="artifacts",
        keys={
            "new_harvest_allowed",
            "target_schema_version",
            "source_overlay_schema_version",
            "train_target_manifest_sha256",
            "eval_target_manifest_sha256",
            "train_source_overlay_manifest_sha256",
            "eval_source_overlay_manifest_sha256",
            "source_overlay_capture_contract_sha256",
            "recomposition_sidecar_content_sha256",
            "recomposition_sidecar_file_sha256",
            "train_tokens",
            "eval_tokens",
        },
    )
    fixed_artifacts = {
        "new_harvest_allowed": False,
        "target_schema_version": "gemma4-expert-contribution-v2",
        "source_overlay_schema_version": (
            None if panel == "sae" else "gemma4-routed-input-overlay-v1"
        ),
        "train_target_manifest_sha256": TRAIN_TARGET_SHA256,
        "eval_target_manifest_sha256": EVAL_TARGET_SHA256,
        "train_source_overlay_manifest_sha256": (
            None if panel == "sae" else TRAIN_SOURCE_SHA256
        ),
        "eval_source_overlay_manifest_sha256": (
            None if panel == "sae" else EVAL_SOURCE_SHA256
        ),
        "source_overlay_capture_contract_sha256": (
            None if panel == "sae" else SOURCE_CAPTURE_CONTRACT_SHA256
        ),
        "recomposition_sidecar_content_sha256": RECOMPOSITION_SIDECAR_SHA256,
        "recomposition_sidecar_file_sha256": RECOMPOSITION_SIDECAR_FILE_SHA256,
        "train_tokens": _TRAIN_TOKENS,
        "eval_tokens": _EVAL_TOKENS,
    }
    for key, expected in fixed_artifacts.items():
        if key.endswith("_sha256") and expected is not None:
            _sha256(artifacts[key], name=f"artifacts.{key}")
        _exact(artifacts[key], expected, name=f"artifacts.{key}")

    training = _mapping(
        top["training"],
        name="training",
        keys={
            "phase_name",
            "layout_set",
            "seeds",
            "initialization",
            "train_tokens",
            "eval_tokens",
            "effective_batch_tokens",
            "microbatch_tokens",
            "planned_updates",
            "warmup_updates",
            "tail_mask_tokens",
            "source",
            "source_centering",
            "target",
            "target_centering",
            "output_bias",
            "objective",
            "router_weight_math_dtype",
            "topk_requires_strictly_positive",
            "topk_tie_break",
            "capacity_margin_fraction",
            "capacity_overflow_policy",
            "token_drop_policy",
        },
    )
    expected_training = {
        "phase_name": PHASE_NAME,
        "layout_set": "true_id_routed_activation_ladder",
        "seeds": [0, 1, 2],
        "initialization": "from_scratch",
        "train_tokens": _TRAIN_TOKENS,
        "eval_tokens": _EVAL_TOKENS,
        "effective_batch_tokens": _EFFECTIVE_BATCH_TOKENS,
        "microbatch_tokens": _MICROBATCH_TOKENS,
        "planned_updates": _PLANNED_UPDATES,
        "warmup_updates": _WARMUP_UPDATES,
        "tail_mask_tokens": _TAIL_MASK_TOKENS,
        "source": (
            "raw_selected_expert_output"
            if panel == "sae"
            else "block15_post_attention_routed_input"
        ),
        "source_centering": (
            "raw_expert_output" if panel == "sae" else "one_global_routed_input_mean"
        ),
        "target": "raw_selected_expert_output",
        "target_centering": "one_pooled_raw_expert_output_mean",
        "output_bias": "none",
        "objective": "normalized_router_weighted_contribution_sse",
        "router_weight_math_dtype": "float32",
        "topk_requires_strictly_positive": True,
        "topk_tie_break": "global_latent_id_ascending",
        "capacity_margin_fraction": 0.10,
        "capacity_overflow_policy": "hard_error",
        "token_drop_policy": "hard_error",
    }
    for key, expected in expected_training.items():
        _exact(training[key], expected, name=f"training.{key}")
    if (
        math.ceil(_TRAIN_TOKENS / _EFFECTIVE_BATCH_TOKENS) != _PLANNED_UPDATES
        or _PLANNED_UPDATES * _EFFECTIVE_BATCH_TOKENS - _TRAIN_TOKENS
        != _TAIL_MASK_TOKENS
    ):
        raise AssertionError("internal routed-ladder schedule arithmetic drifted")

    optimizer = _mapping(
        top["optimizer"],
        name="optimizer",
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
        "adam_epsilon": 1.0e-8,
        "weight_decay": 0.0,
        "global_gradient_clip": 1.0,
        "peak_learning_rate": 3.0e-4,
        "minimum_learning_rate": 3.0e-5,
        "schedule": "linear_warmup_cosine_decay",
        "decoder_init": "unit_norm_columns",
        "encoder_init": "decoder_transpose",
        "encoder_bias_init": "zeros",
        "tangent_decoder_gradient_projection": True,
        "decoder_column_renormalization": True,
    }
    for key, expected in expected_optimizer.items():
        _exact(optimizer[key], expected, name=f"optimizer.{key}")

    raw_layouts = top["layouts"]
    if not isinstance(raw_layouts, list) or len(raw_layouts) != len(CONDITIONS):
        raise RoutedLadderConfigError(
            "layouts must contain exactly A512ReLU/A512Softplus"
        )
    layouts = tuple(
        _validate_layout(item, index=index) for index, item in enumerate(raw_layouts)
    )

    h_reference = _mapping(
        top["existing_h_reference"],
        name="existing_h_reference",
        keys={
            "condition",
            "source_protocol_id",
            "source_config_sha256",
            "source_phase_name",
            "training_policy",
            "encoder_activation",
            "objective",
            "checkpoint_tokens",
        },
    )
    expected_h_reference = {
        "condition": "H",
        "source_protocol_id": EXPERIMENT5_PROTOCOL_ID,
        "source_config_sha256": H_CONFIG_SHA256[panel],
        "source_phase_name": "routed_shared_hybrid_10m",
        "training_policy": "reuse_existing_artifacts_only",
        "encoder_activation": "softplus",
        "objective": "normalized_full_ffn_sse",
        "checkpoint_tokens": list(_EVALUATION_CHECKPOINTS),
    }
    for key, expected in expected_h_reference.items():
        if key == "source_config_sha256":
            _sha256(h_reference[key], name="existing_h_reference.source_config_sha256")
        _exact(h_reference[key], expected, name=f"existing_h_reference.{key}")

    execution = _mapping(
        top["execution"],
        name="execution",
        keys={
            "accelerator_mfu_gate_enforced",
            "accelerator_min_useful_mfu",
            "accelerator_qualification_max_seconds",
            "direct_runtime_max_seconds",
            "restart_checkpoint_tokens",
            "new_bulk_capture_allowed",
            "cross_region_bulk_copy_allowed",
            "require_region_local_artifact_access",
        },
    )
    expected_execution = {
        "accelerator_mfu_gate_enforced": False,
        "accelerator_min_useful_mfu": 0.10,
        "accelerator_qualification_max_seconds": 600,
        "direct_runtime_max_seconds": _DIRECT_RUNTIME_MAX_SECONDS,
        "restart_checkpoint_tokens": list(_RESTART_CHECKPOINTS),
        "new_bulk_capture_allowed": False,
        "cross_region_bulk_copy_allowed": False,
        "require_region_local_artifact_access": True,
    }
    for key, expected in expected_execution.items():
        _exact(execution[key], expected, name=f"execution.{key}")

    evaluation = _mapping(
        top["evaluation"],
        name="evaluation",
        keys={
            "checkpoint_tokens",
            "primary_endpoint",
            "objective",
            "router_weighted",
            "common_target_sst",
            "bootstrap_unit",
            "bootstrap_replicates",
            "paired_confidence_level",
            "ladder_conditions",
            "h_reference_training_allowed",
        },
    )
    expected_evaluation = {
        "checkpoint_tokens": list(_EVALUATION_CHECKPOINTS),
        "primary_endpoint": "raw_selected_expert_output",
        "objective": "normalized_router_weighted_contribution_sse",
        "router_weighted": True,
        "common_target_sst": True,
        "bootstrap_unit": "packed_sequence",
        "bootstrap_replicates": 20_000,
        "paired_confidence_level": 0.95,
        "ladder_conditions": ["A512ReLU", "A512Softplus", "H"],
        "h_reference_training_allowed": False,
    }
    for key, expected in expected_evaluation.items():
        _exact(evaluation[key], expected, name=f"evaluation.{key}")

    runtime = RoutedLadderRuntime(
        effective_batch_tokens=_EFFECTIVE_BATCH_TOKENS,
        microbatch_tokens=_MICROBATCH_TOKENS,
        capacity_margin_fraction=_finite(
            training["capacity_margin_fraction"],
            name="training.capacity_margin_fraction",
        ),
        accelerator_min_useful_mfu=_finite(
            execution["accelerator_min_useful_mfu"],
            name="execution.accelerator_min_useful_mfu",
        ),
        accelerator_qualification_max_seconds=int(
            execution["accelerator_qualification_max_seconds"]
        ),
        direct_runtime_max_seconds=int(execution["direct_runtime_max_seconds"]),
    )
    parents = {
        "train_target_manifest_sha256": TRAIN_TARGET_SHA256,
        "eval_target_manifest_sha256": EVAL_TARGET_SHA256,
        "train_source_overlay_manifest_sha256": (
            TRAIN_SOURCE_SHA256 if panel == "transcoder" else None
        ),
        "eval_source_overlay_manifest_sha256": (
            EVAL_SOURCE_SHA256 if panel == "transcoder" else None
        ),
        "source_overlay_capture_contract_sha256": (
            SOURCE_CAPTURE_CONTRACT_SHA256 if panel == "transcoder" else None
        ),
    }
    raw_copy = json.loads(
        json.dumps(top, sort_keys=True, separators=(",", ":"), allow_nan=False)
    )
    result = RoutedLadderConfig(
        raw=raw_copy,
        config_hash=canonical_sha256(raw_copy),
        protocol_id=PROTOCOL_ID,
        panel=panel,
        scientific_model=panel,
        provenance=RoutedLadderProvenance(maxtext_fork_commit=fork),
        model=RoutedLadderModel(
            hidden_size=2_816,
            num_routed_experts=128,
            experts_per_token=8,
        ),
        runtime=runtime,
        optimizer=RoutedLadderOptimizer(
            parameter_dtype="bfloat16",
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1.0e-8,
            global_gradient_clip=1.0,
            peak_learning_rate=3.0e-4,
            minimum_learning_rate=3.0e-5,
        ),
        layouts=layouts,
        parents=parents,
        parent_protocol_ids={
            "experiment_4": EXPERIMENT4_PROTOCOL_ID,
            "experiment_5": EXPERIMENT5_PROTOCOL_ID,
        },
        ladder_phase=RoutedLadderPhase(
            name=PHASE_NAME,
            layout_set="true_id_routed_activation_ladder",
            seeds=(0, 1, 2),
            train_tokens=_TRAIN_TOKENS,
            eval_tokens=_EVAL_TOKENS,
            planned_updates=_PLANNED_UPDATES,
            warmup_updates=_WARMUP_UPDATES,
            tail_mask_tokens=_TAIL_MASK_TOKENS,
            restart_checkpoint_tokens=_RESTART_CHECKPOINTS,
            evaluation_checkpoint_tokens=_EVALUATION_CHECKPOINTS,
            initialization="from_scratch",
        ),
        existing_h_reference=ExistingHReference(
            condition="H",
            source_protocol_id=EXPERIMENT5_PROTOCOL_ID,
            source_config_sha256=H_CONFIG_SHA256[panel],
            source_phase_name="routed_shared_hybrid_10m",
            training_policy="reuse_existing_artifacts_only",
            encoder_activation="softplus",
            objective="normalized_full_ffn_sse",
            checkpoint_tokens=_EVALUATION_CHECKPOINTS,
        ),
        recomposition_sidecar_sha256=RECOMPOSITION_SIDECAR_SHA256,
    )
    result.validate()
    return result


def load_routed_ladder_config(
    path: str | Path,
    *,
    expected_panel: Literal["sae", "transcoder"] | None = None,
) -> RoutedLadderConfig:
    """Load and exhaustively validate one routed-ladder YAML contract."""

    source = Path(path).expanduser().resolve()
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise RoutedLadderConfigError(
            f"cannot read routed-ladder config {source}"
        ) from error
    return _validate_document(raw, expected_panel=expected_panel)


__all__ = [
    "CONDITIONS",
    "EXPERIMENT4_PROTOCOL_ID",
    "EXPERIMENT5_PROTOCOL_ID",
    "H_CONFIG_SHA256",
    "PHASE_NAME",
    "PROTOCOL_ID",
    "SCHEMA_VERSION",
    "ExistingHReference",
    "RoutedLadderConfig",
    "RoutedLadderConfigError",
    "RoutedLadderLayout",
    "load_routed_ladder_config",
]
