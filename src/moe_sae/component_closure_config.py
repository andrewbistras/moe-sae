"""Strict A512 ReLU-inclusive component-closure protocol.

This is a successor contract, not an edit to Experiment 5 v3.  The completed
v3 A512 ReLU-inclusive trajectories are read-only ``A`` parents.  Only
``R``/``B``/``F`` are trainable here, using the already-general contribution
kernels and the same optimizer, captures, token schedule, and activation
policy as A.
"""

from __future__ import annotations

import copy
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from moe_sae.contribution_artifacts import canonical_sha256

SCHEMA_VERSION = "gemma4-a512-component-closure-run-config-v1"
PROTOCOL_ID = "gemma4-a512-component-closure-v1-20260729"
PHASE_NAME = "a512_component_closure_10m"
PANELS = ("sae", "transcoder")
CONDITIONS = ("A", "R", "B", "F")
TRAINED_CONDITIONS = ("R", "B", "F")
SEEDS = (0, 1, 2)
TRAIN_TOKENS = 10_000_000
EVAL_TOKENS = 250_000
CHECKPOINT_TOKENS = (2_000_000, 5_000_000, 10_000_000)
RESTART_CHECKPOINT_TOKENS = tuple(range(1_000_000, 10_000_001, 1_000_000))
PLANNED_UPDATES = 611
WARMUP_UPDATES = 61
TAIL_MASK_TOKENS = 10_624
EFFECTIVE_BATCH_TOKENS = 16_384
MICROBATCH_TOKENS = 4_096
SOURCE_PROTOCOL_ID = "gemma4-experiment5-routed-activation-ladder-v3-20260729"
SOURCE_PHASE_NAME = "experiment5_routed_activation_ladder_v3_10m"
SOURCE_CONDITION = "A512ReLUInclusiveTopK"
SOURCE_CONFIG_SHA256 = {
    "sae": "725c1d846bc2189096f5742a758e6845db4d7cad1fdc203d950e50de152bafc5",
    "transcoder": "059d845bbb7b1140ac50f6866efc7727fd2ee88fcc4e788d147ddcfa86c9798f",
}
SOURCE_CONFIG_FILE_SHA256 = {
    "sae": "8a919c25d9b00e9c238e704cd5127688a26f8751e9dbfeb48dfc6e3149d48c6f",
    "transcoder": "052dcc73e6cac691ee01de0e3a2d49357bdd7a11d35b20ed70153fddf2dc2c78",
}
PRIMARY_CONTRASTS = ("A-R", "A-B", "A-F", "F-B")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ComponentClosureConfigError(ValueError):
    """A component-closure document differs from the frozen protocol."""


def _mapping(value: Any, *, name: str, keys: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ComponentClosureConfigError(f"{name} must be a mapping")
    if set(value) != keys:
        raise ComponentClosureConfigError(
            f"{name} fields differ: expected {sorted(keys)}, got {sorted(value)}"
        )
    return value


def _exact(value: Any, expected: Any, *, name: str) -> None:
    if value != expected:
        raise ComponentClosureConfigError(
            f"{name} is frozen to {expected!r}, got {value!r}"
        )


def _sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ComponentClosureConfigError(f"{name} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class ClosureProvenance:
    maxtext_fork_commit: str


@dataclass(frozen=True, slots=True)
class ClosureModel:
    hidden_size: int
    num_routed_experts: int
    experts_per_token: int


@dataclass(frozen=True, slots=True)
class ClosureRuntime:
    effective_batch_tokens: int
    microbatch_tokens: int
    capacity_margin_fraction: float
    accelerator_min_useful_mfu: float
    accelerator_qualification_max_seconds: int
    direct_runtime_max_seconds: int
    topk_requires_strictly_positive: bool


@dataclass(frozen=True, slots=True)
class ClosureOptimizer:
    parameter_dtype: str
    adam_beta1: float
    adam_beta2: float
    adam_epsilon: float
    global_gradient_clip: float
    peak_learning_rate: float
    minimum_learning_rate: float


@dataclass(frozen=True, slots=True)
class ClosureLayout:
    condition: str
    target: str
    bank_identity: str
    bank_count: int
    bank_width: int
    components_per_token: int
    topk_per_component: int
    encoder_activation: str
    selection_policy: str
    topk_requires_strictly_positive: bool
    training_seeds: tuple[int, ...]

    @property
    def total_features(self) -> int:
        return self.bank_count * self.bank_width

    @property
    def candidates_per_token(self) -> int:
        return self.components_per_token * self.bank_width

    @property
    def selected_slots_per_token(self) -> int:
        return self.components_per_token * self.topk_per_component

    @property
    def total_l0(self) -> int:
        """Kernel compatibility alias; this is a slot ceiling, not positive L0."""

        return self.selected_slots_per_token


@dataclass(frozen=True, slots=True)
class ClosurePhase:
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
        return self.restart_checkpoint_tokens


@dataclass(frozen=True, slots=True)
class ExistingAReference:
    source_protocol_id: str
    source_schema_version: str
    source_config_sha256: str
    source_config_file_sha256: str
    source_phase_name: str
    source_condition: str
    import_policy: str
    seeds: tuple[int, ...]
    checkpoint_tokens: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ComponentClosureConfig:
    raw: Mapping[str, Any]
    config_hash: str
    protocol_id: str
    panel: Literal["sae", "transcoder"]
    scientific_model: Literal["sae", "transcoder"]
    encoder_activation: str
    provenance: ClosureProvenance
    model: ClosureModel
    runtime: ClosureRuntime
    optimizer: ClosureOptimizer
    layouts: tuple[ClosureLayout, ...]
    parents: Mapping[str, Any]
    closure_phase: ClosurePhase
    existing_a_reference: ExistingAReference
    report_contract: Mapping[str, Any]
    recomposition_sidecar_sha256: str

    def validate(self) -> None:
        if canonical_sha256(self.raw) != self.config_hash:
            raise ComponentClosureConfigError("config canonical identity changed")
        if self.protocol_id != PROTOCOL_ID or self.panel not in PANELS:
            raise ComponentClosureConfigError("protocol identity changed")
        if tuple(row.condition for row in self.layouts) != CONDITIONS:
            raise ComponentClosureConfigError("layouts must be exactly A/R/B/F")
        a, r, b, f = self.layouts
        if (
            a.training_seeds
            or r.training_seeds != SEEDS
            or b.training_seeds != SEEDS
            or f.training_seeds != SEEDS
        ):
            raise ComponentClosureConfigError("A must be imported and R/B/F fresh")
        if not (
            a.total_features == r.total_features == f.total_features == 65_536
            and b.total_features == 4_096
            and a.candidates_per_token
            == r.candidates_per_token
            == b.candidates_per_token
            == 4_096
            and f.candidates_per_token == 65_536
            and all(row.selected_slots_per_token == 32 for row in self.layouts)
        ):
            raise ComponentClosureConfigError("closure geometry no longer closes")

    def phase(self, name: str) -> ClosurePhase:
        if name != PHASE_NAME:
            raise ComponentClosureConfigError(f"unknown closure phase {name!r}")
        return self.closure_phase

    def layouts_for_phase(self, name: str) -> tuple[ClosureLayout, ...]:
        self.phase(name)
        return self.layouts

    def layout_for_condition(self, condition: str) -> ClosureLayout:
        matches = tuple(row for row in self.layouts if row.condition == condition)
        if len(matches) != 1:
            raise ComponentClosureConfigError(
                f"condition {condition!r} does not name one layout"
            )
        return matches[0]

    def training_seeds_for_condition(self, condition: str) -> tuple[int, ...]:
        return self.layout_for_condition(condition).training_seeds

    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy(dict(self.raw))


def component_closure_config_from_mapping(
    value: Any,
    *,
    expected_panel: str | None = None,
) -> ComponentClosureConfig:
    top = _mapping(
        value,
        name="config",
        keys={
            "schema_version",
            "protocol_id",
            "panel",
            "provenance",
            "model",
            "artifacts",
            "training",
            "optimizer",
            "layouts",
            "existing_a_reference",
            "execution",
            "evaluation",
            "report_contract",
        },
    )
    _exact(top["schema_version"], SCHEMA_VERSION, name="schema_version")
    _exact(top["protocol_id"], PROTOCOL_ID, name="protocol_id")
    panel = top["panel"]
    if panel not in PANELS or (expected_panel is not None and panel != expected_panel):
        raise ComponentClosureConfigError("panel identity is invalid")

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
    for key, expected in {
        "target_layer_one_indexed": 15,
        "hidden_size": 2_816,
        "num_routed_experts": 128,
        "experts_per_token": 8,
        "shared_experts": 1,
    }.items():
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
    expected_artifacts = {
        "new_harvest_allowed": False,
        "target_schema_version": "gemma4-expert-contribution-v2",
        "source_overlay_schema_version": (
            None if panel == "sae" else "gemma4-routed-input-overlay-v1"
        ),
        "train_target_manifest_sha256": (
            "f29aaf45d9ff93be08cc3d175cdcea9fa4aecc37fc0e256d09720e88e1ae838c"
        ),
        "eval_target_manifest_sha256": (
            "31454e5ce8693521c41a2a95f2ef88d40c5d26c726a0efac5c26d882ed341f5e"
        ),
        "train_source_overlay_manifest_sha256": (
            None
            if panel == "sae"
            else "8f27dcf9c1dc0c7f3ffa2e2941daec13d8a72c7204077f09420e15aacb02ffe0"
        ),
        "eval_source_overlay_manifest_sha256": (
            None
            if panel == "sae"
            else "f68e2fe6e4cf568e694925697dd5d65fe10160a52d357c641892de85ee5c3da7"
        ),
        "source_overlay_capture_contract_sha256": (
            None
            if panel == "sae"
            else "fe0b004ca6e48c9eb8163237a5ea20f45b33acdbd9a3a267a9fb4e670ed2c954"
        ),
        "recomposition_sidecar_content_sha256": (
            "37c7b8c012f8fc0c1f63210986c0b3071a261261c88ee4bc1f502220ce3b8526"
        ),
        "recomposition_sidecar_file_sha256": (
            "6486288677e81665b21815879729c424c6acb8c7d0e45be264b961a9d652fe3d"
        ),
        "train_tokens": TRAIN_TOKENS,
        "eval_tokens": EVAL_TOKENS,
    }
    for key, expected in expected_artifacts.items():
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
            "encoder_activation",
            "selection_policy",
            "topk_requires_strictly_positive",
            "topk_tie_break",
            "capacity_margin_fraction",
            "capacity_overflow_policy",
            "token_drop_policy",
        },
    )
    expected_training = {
        "phase_name": PHASE_NAME,
        "layout_set": "a512_component_closure",
        "seeds": list(SEEDS),
        "initialization": "from_scratch_controls_only",
        "train_tokens": TRAIN_TOKENS,
        "eval_tokens": EVAL_TOKENS,
        "effective_batch_tokens": EFFECTIVE_BATCH_TOKENS,
        "microbatch_tokens": MICROBATCH_TOKENS,
        "planned_updates": PLANNED_UPDATES,
        "warmup_updates": WARMUP_UPDATES,
        "tail_mask_tokens": TAIL_MASK_TOKENS,
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
        "encoder_activation": "relu",
        "selection_policy": "inclusive_zero_topk",
        "topk_requires_strictly_positive": False,
        "topk_tie_break": "global_latent_id_ascending",
        "capacity_margin_fraction": 0.10,
        "capacity_overflow_policy": "hard_error",
        "token_drop_policy": "hard_error",
    }
    for key, expected in expected_training.items():
        _exact(training[key], expected, name=f"training.{key}")

    optimizer_raw = _mapping(
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
    for key, expected in expected_optimizer.items():
        _exact(optimizer_raw[key], expected, name=f"optimizer.{key}")

    raw_layouts = top["layouts"]
    if not isinstance(raw_layouts, list) or len(raw_layouts) != 4:
        raise ComponentClosureConfigError("layouts must contain exactly A/R/B/F")
    shapes = {
        "A": ("true_expert_id", 128, 512, ()),
        "R": ("per_sequence_randomized_expert_id", 128, 512, SEEDS),
        "B": ("router_rank", 8, 512, SEEDS),
        "F": ("router_rank", 8, 8_192, SEEDS),
    }
    layouts: list[ClosureLayout] = []
    for index, condition in enumerate(CONDITIONS):
        row = _mapping(
            raw_layouts[index],
            name=f"layouts[{index}]",
            keys={
                "condition",
                "target",
                "bank_identity",
                "bank_count",
                "bank_width",
                "components_per_token",
                "topk_per_component",
                "total_features",
                "candidates_per_token",
                "selected_slots_per_token",
                "encoder_activation",
                "selection_policy",
                "topk_requires_strictly_positive",
                "training_seeds",
            },
        )
        identity, banks, width, seeds = shapes[condition]
        expected = {
            "condition": condition,
            "target": "raw_selected_expert_output",
            "bank_identity": identity,
            "bank_count": banks,
            "bank_width": width,
            "components_per_token": 8,
            "topk_per_component": 4,
            "total_features": banks * width,
            "candidates_per_token": 8 * width,
            "selected_slots_per_token": 32,
            "encoder_activation": "relu",
            "selection_policy": "inclusive_zero_topk",
            "topk_requires_strictly_positive": False,
            "training_seeds": list(seeds),
        }
        for key, selected in expected.items():
            _exact(row[key], selected, name=f"layouts[{index}].{key}")
        layouts.append(
            ClosureLayout(
                condition=condition,
                target="raw_selected_expert_output",
                bank_identity=identity,
                bank_count=banks,
                bank_width=width,
                components_per_token=8,
                topk_per_component=4,
                encoder_activation="relu",
                selection_policy="inclusive_zero_topk",
                topk_requires_strictly_positive=False,
                training_seeds=seeds,
            )
        )

    a_raw = _mapping(
        top["existing_a_reference"],
        name="existing_a_reference",
        keys={
            "source_protocol_id",
            "source_schema_version",
            "source_config_sha256",
            "source_config_file_sha256",
            "source_phase_name",
            "source_condition",
            "import_policy",
            "seeds",
            "checkpoint_tokens",
        },
    )
    expected_a = {
        "source_protocol_id": SOURCE_PROTOCOL_ID,
        "source_schema_version": (
            "gemma4-experiment5-routed-activation-ladder-run-config-v3"
        ),
        "source_config_sha256": SOURCE_CONFIG_SHA256[panel],
        "source_config_file_sha256": SOURCE_CONFIG_FILE_SHA256[panel],
        "source_phase_name": SOURCE_PHASE_NAME,
        "source_condition": SOURCE_CONDITION,
        "import_policy": "immutable_read_only_no_copy_no_relabel",
        "seeds": list(SEEDS),
        "checkpoint_tokens": list(CHECKPOINT_TOKENS),
    }
    for key, expected in expected_a.items():
        _exact(a_raw[key], expected, name=f"existing_a_reference.{key}")

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
        "direct_runtime_max_seconds": 43_200,
        "restart_checkpoint_tokens": list(RESTART_CHECKPOINT_TOKENS),
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
            "conditions",
            "primary_endpoint",
            "objective",
            "common_target_sst",
            "bootstrap_unit",
            "bootstrap_replicates",
            "paired_confidence_level",
            "primary_contrasts",
            "generalized_resident_bundle",
        },
    )
    expected_evaluation = {
        "checkpoint_tokens": list(CHECKPOINT_TOKENS),
        "conditions": list(CONDITIONS),
        "primary_endpoint": "component",
        "objective": "normalized_router_weighted_contribution_sse",
        "common_target_sst": True,
        "bootstrap_unit": "packed_sequence",
        "bootstrap_replicates": 20_000,
        "paired_confidence_level": 0.95,
        "primary_contrasts": list(PRIMARY_CONTRASTS),
        "generalized_resident_bundle": "one_panel_seed_four_conditions_three_milestones",
    }
    for key, expected in expected_evaluation.items():
        _exact(evaluation[key], expected, name=f"evaluation.{key}")

    report = _mapping(
        top["report_contract"],
        name="report_contract",
        keys={
            "primary_estimand",
            "geometry_matches",
            "g_definition",
            "g_undefined_when",
            "g_unstable_when",
            "cost_comparability",
        },
    )
    expected_report = {
        "primary_estimand": "component_target_explained_variance",
        "geometry_matches": {
            "A_R_F": "total_features_and_trainable_parameters",
            "A_R_B": "candidates_per_token",
            "all": "selected_slots_per_token",
        },
        "g_definition": "100*(A-B)/(F-B)",
        "g_undefined_when": "F-B_nonpositive",
        "g_unstable_when": "paired_interval_for_F-B_includes_zero",
        "cost_comparability": (
            "same_accelerator_generation_and_identical_timing_scope_only"
        ),
    }
    for key, expected in expected_report.items():
        _exact(report[key], expected, name=f"report_contract.{key}")

    if (
        math.ceil(TRAIN_TOKENS / EFFECTIVE_BATCH_TOKENS) != PLANNED_UPDATES
        or PLANNED_UPDATES * EFFECTIVE_BATCH_TOKENS - TRAIN_TOKENS != TAIL_MASK_TOKENS
    ):
        raise AssertionError("closure token arithmetic drifted")
    canonical = json.loads(
        json.dumps(top, sort_keys=True, separators=(",", ":"), allow_nan=False)
    )
    result = ComponentClosureConfig(
        raw=canonical,
        config_hash=canonical_sha256(canonical),
        protocol_id=PROTOCOL_ID,
        panel=panel,
        scientific_model=panel,
        encoder_activation="relu",
        provenance=ClosureProvenance(
            maxtext_fork_commit=str(provenance["maxtext_fork_commit"])
        ),
        model=ClosureModel(
            hidden_size=int(model["hidden_size"]),
            num_routed_experts=int(model["num_routed_experts"]),
            experts_per_token=int(model["experts_per_token"]),
        ),
        runtime=ClosureRuntime(
            effective_batch_tokens=EFFECTIVE_BATCH_TOKENS,
            microbatch_tokens=MICROBATCH_TOKENS,
            capacity_margin_fraction=0.10,
            accelerator_min_useful_mfu=0.10,
            accelerator_qualification_max_seconds=600,
            direct_runtime_max_seconds=43_200,
            topk_requires_strictly_positive=False,
        ),
        optimizer=ClosureOptimizer(
            parameter_dtype="bfloat16",
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
            global_gradient_clip=1.0,
            peak_learning_rate=3e-4,
            minimum_learning_rate=3e-5,
        ),
        layouts=tuple(layouts),
        parents={
            "train_target_manifest_sha256": artifacts["train_target_manifest_sha256"],
            "eval_target_manifest_sha256": artifacts["eval_target_manifest_sha256"],
            "train_source_overlay_manifest_sha256": artifacts[
                "train_source_overlay_manifest_sha256"
            ],
            "eval_source_overlay_manifest_sha256": artifacts[
                "eval_source_overlay_manifest_sha256"
            ],
            "source_overlay_capture_contract_sha256": artifacts[
                "source_overlay_capture_contract_sha256"
            ],
        },
        closure_phase=ClosurePhase(
            name=PHASE_NAME,
            layout_set="a512_component_closure",
            seeds=SEEDS,
            train_tokens=TRAIN_TOKENS,
            eval_tokens=EVAL_TOKENS,
            planned_updates=PLANNED_UPDATES,
            warmup_updates=WARMUP_UPDATES,
            tail_mask_tokens=TAIL_MASK_TOKENS,
            restart_checkpoint_tokens=RESTART_CHECKPOINT_TOKENS,
            evaluation_checkpoint_tokens=CHECKPOINT_TOKENS,
            initialization="from_scratch_controls_only",
        ),
        existing_a_reference=ExistingAReference(
            source_protocol_id=SOURCE_PROTOCOL_ID,
            source_schema_version=expected_a["source_schema_version"],
            source_config_sha256=SOURCE_CONFIG_SHA256[panel],
            source_config_file_sha256=SOURCE_CONFIG_FILE_SHA256[panel],
            source_phase_name=SOURCE_PHASE_NAME,
            source_condition=SOURCE_CONDITION,
            import_policy="immutable_read_only_no_copy_no_relabel",
            seeds=SEEDS,
            checkpoint_tokens=CHECKPOINT_TOKENS,
        ),
        report_contract=copy.deepcopy(report),
        recomposition_sidecar_sha256=str(
            artifacts["recomposition_sidecar_content_sha256"]
        ),
    )
    result.validate()
    return result


def load_component_closure_config(
    path: str | Path,
    *,
    expected_panel: Literal["sae", "transcoder"] | None = None,
) -> ComponentClosureConfig:
    source = Path(path).expanduser().resolve()
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise ComponentClosureConfigError(
            f"cannot read component-closure config {source}"
        ) from error
    return component_closure_config_from_mapping(
        raw,
        expected_panel=expected_panel,
    )


__all__ = [
    "CHECKPOINT_TOKENS",
    "CONDITIONS",
    "ComponentClosureConfig",
    "ComponentClosureConfigError",
    "EVAL_TOKENS",
    "PANELS",
    "PHASE_NAME",
    "PRIMARY_CONTRASTS",
    "PROTOCOL_ID",
    "SCHEMA_VERSION",
    "SEEDS",
    "SOURCE_CONDITION",
    "SOURCE_CONFIG_FILE_SHA256",
    "SOURCE_CONFIG_SHA256",
    "SOURCE_PHASE_NAME",
    "SOURCE_PROTOCOL_ID",
    "TRAINED_CONDITIONS",
    "TRAIN_TOKENS",
    "component_closure_config_from_mapping",
    "load_component_closure_config",
]
