"""Strict, minimal Experiment-3 run configuration.

Experiment 3 intentionally does not pass its YAML through the larger
Experiment-2 phase machine.  This loader validates the complete checked-in
document, exposes only the small worker-facing compatibility surface, and
names the canonical YAML content in every workload identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from moe_sae.contribution_artifacts import canonical_sha256

SCHEMA_VERSION = "gemma4-contribution-transcoder-run-config-v2"
PROTOCOL_ID = "gemma4-contribution-transcoder-v2-20260727"
PHASE_NAME = "mechanism_pilot"
CONDITIONS = ("A", "R", "B", "F")


class ContributionTranscoderConfigError(ValueError):
    """The Experiment-3 document differs from its frozen protocol."""


def _mapping(value: Any, *, owner: str, keys: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContributionTranscoderConfigError(f"{owner} must be a mapping")
    if set(value) != keys:
        raise ContributionTranscoderConfigError(
            f"{owner} fields differ: expected {sorted(keys)}, got {sorted(value)}"
        )
    return value


def _exact(value: Any, expected: Any, *, name: str) -> None:
    if value != expected:
        raise ContributionTranscoderConfigError(
            f"{name} is frozen to {expected!r}, got {value!r}"
        )


@dataclass(frozen=True, slots=True)
class TranscoderProvenance:
    maxtext_fork_commit: str


@dataclass(frozen=True, slots=True)
class TranscoderModel:
    hidden_size: int
    num_routed_experts: int
    experts_per_token: int


@dataclass(frozen=True, slots=True)
class TranscoderRuntime:
    effective_batch_tokens: int
    microbatch_tokens: int
    capacity_margin_fraction: float
    accelerator_min_useful_mfu: float = 0.10
    accelerator_qualification_max_seconds: int = 600


@dataclass(frozen=True, slots=True)
class TranscoderOptimizer:
    parameter_dtype: str
    adam_beta1: float
    adam_beta2: float
    adam_epsilon: float
    global_gradient_clip: float
    peak_learning_rate: float
    minimum_learning_rate: float


@dataclass(frozen=True, slots=True)
class TranscoderLayout:
    condition: str
    bank_width: int
    topk_per_component: int


@dataclass(frozen=True, slots=True)
class TranscoderPhase:
    name: str = PHASE_NAME
    layout_set: str = "pilot"
    seeds: tuple[int, ...] = (0,)
    train_tokens: int = 2_000_000
    eval_tokens: int = 250_000
    planned_updates: int = 123
    warmup_updates: int = 12
    checkpoint_tokens: tuple[int, ...] = (2_000_000,)


@dataclass(frozen=True, slots=True)
class Experiment3Config:
    """Validated worker view plus the canonical source-document hash."""

    raw: Mapping[str, Any]
    config_hash: str
    protocol_id: str
    encoder_activation: str
    provenance: TranscoderProvenance
    model: TranscoderModel
    runtime: TranscoderRuntime
    optimizer: TranscoderOptimizer
    layouts: tuple[TranscoderLayout, ...]
    parents: Mapping[str, Any]

    def validate(self) -> None:
        # Construction is private and exhaustive; the hash check also detects
        # callers attempting to replace ``raw`` after loading.
        if canonical_sha256(self.raw) != self.config_hash:
            raise ContributionTranscoderConfigError(
                "transcoder config canonical identity changed"
            )

    def phase(self, name: str) -> TranscoderPhase:
        if name != PHASE_NAME:
            raise ContributionTranscoderConfigError(
                f"Experiment 3 has one phase named {PHASE_NAME!r}"
            )
        return TranscoderPhase()

    def layouts_for_phase(self, name: str) -> tuple[TranscoderLayout, ...]:
        self.phase(name)
        return self.layouts


def _validate_document(value: Any) -> Experiment3Config:
    top = _mapping(
        value,
        owner="config",
        keys={
            "schema_version",
            "protocol_id",
            "parent_protocol_id",
            "provenance",
            "model",
            "parents",
            "source_overlay",
            "training",
            "execution",
            "optimizer",
            "layouts",
            "evaluation",
        },
    )
    _exact(top["schema_version"], SCHEMA_VERSION, name="schema_version")
    _exact(top["protocol_id"], PROTOCOL_ID, name="protocol_id")
    _exact(
        top["parent_protocol_id"],
        "gemma4-contribution-v1-20260725",
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
    frozen_provenance = {
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
    for name, expected in frozen_provenance.items():
        _exact(provenance[name], expected, name=f"provenance.{name}")

    model = _mapping(
        top["model"],
        owner="model",
        keys={
            "target_layer_one_indexed",
            "hidden_size",
            "num_routed_experts",
            "experts_per_token",
            "sequence_length",
            "dtype",
            "use_multimodal",
            "scan_layers",
            "pure_nnx_decoder",
            "attention",
            "dropout_rate",
            "load_balance_loss_weight",
        },
    )
    frozen_model = {
        "target_layer_one_indexed": 15,
        "hidden_size": 2_816,
        "num_routed_experts": 128,
        "experts_per_token": 8,
        "sequence_length": 1_024,
        "dtype": "bfloat16",
        "use_multimodal": False,
        "scan_layers": False,
        "pure_nnx_decoder": True,
        "attention": "autoselected",
        "dropout_rate": 0.0,
        "load_balance_loss_weight": 0.0,
    }
    for name, expected in frozen_model.items():
        _exact(model[name], expected, name=f"model.{name}")

    parents = _mapping(
        top["parents"],
        owner="parents",
        keys={
            "target_schema_version",
            "train_target_manifest_sha256",
            "eval_target_manifest_sha256",
            "train_source_overlay_manifest_sha256",
            "eval_source_overlay_manifest_sha256",
            "train_tokens",
            "eval_tokens",
            "recomposition_sidecar_schema_version",
        },
    )
    frozen_parents = {
        "target_schema_version": "gemma4-expert-contribution-v2",
        "train_target_manifest_sha256": (
            "a2f23aef6b9121e81a5bfea473c64039ca5bf6b073e4f70f4703938a1d8a6903"
        ),
        "eval_target_manifest_sha256": (
            "31454e5ce8693521c41a2a95f2ef88d40c5d26c726a0efac5c26d882ed341f5e"
        ),
        "train_source_overlay_manifest_sha256": (
            "8d947629e57d64d054a3086e2729090eed1e9f6afc4568b76e5c90da5f378e3c"
        ),
        "eval_source_overlay_manifest_sha256": (
            "f68e2fe6e4cf568e694925697dd5d65fe10160a52d357c641892de85ee5c3da7"
        ),
        "train_tokens": 2_000_000,
        "eval_tokens": 250_000,
        "recomposition_sidecar_schema_version": (
            "gemma4-contribution-recomposition-v1"
        ),
    }
    for name, expected in frozen_parents.items():
        _exact(parents[name], expected, name=f"parents.{name}")

    overlay = _mapping(
        top["source_overlay"],
        owner="source_overlay",
        keys={
            "schema_version",
            "field",
            "capture_site",
            "dtype",
            "hidden_size",
            "max_shard_tokens",
            "statistics_dtype",
            "alignment_fields",
            "parent_binding",
            "mismatch_policy",
        },
    )
    frozen_overlay = {
        "schema_version": "gemma4-routed-input-overlay-v1",
        "field": "routed_input",
        "capture_site": (
            "target.mlp.pre_feedforward_layernorm_2(" "block15.post_attention_residual)"
        ),
        "dtype": "bfloat16",
        "hidden_size": 2_816,
        "max_shard_tokens": 16_384,
        "statistics_dtype": "float64",
        "alignment_fields": ["sequence_id", "position"],
        "parent_binding": "exact_target_manifest_and_logical_range",
        "mismatch_policy": "hard_error",
    }
    for name, expected in frozen_overlay.items():
        _exact(overlay[name], expected, name=f"source_overlay.{name}")

    training = _mapping(
        top["training"],
        owner="training",
        keys={
            "mode",
            "encoder_activation",
            "seed",
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
    frozen_training = {
        "mode": "contribution_transcoder",
        "encoder_activation": "softplus",
        "seed": 0,
        "train_tokens": 2_000_000,
        "eval_tokens": 250_000,
        "effective_batch_tokens": 16_384,
        "microbatch_tokens": 4_096,
        "planned_updates": 123,
        "warmup_updates": 12,
        "tail_mask_tokens": 15_232,
        "source_centering": "one_global_routed_input_mean",
        "target_centering": "one_pooled_raw_expert_output_mean",
        "output_bias": "none",
        "contribution_loss": "normalized_router_weighted_sse",
        "router_weight_math_dtype": "float32",
        "topk_requires_strictly_positive": True,
        "topk_tie_break": "global_latent_id_ascending",
        "capacity_margin_fraction": 0.10,
        "capacity_overflow_policy": "hard_error",
    }
    for name, expected in frozen_training.items():
        _exact(training[name], expected, name=f"training.{name}")

    execution = _mapping(
        top["execution"],
        owner="execution",
        keys={
            "process_count",
            "accelerator_min_useful_mfu",
            "source_harvest_ungated_max_seconds",
            "per_condition_max_seconds",
            "checkpoint_tokens",
        },
    )
    frozen_execution = {
        "process_count": 4,
        "accelerator_min_useful_mfu": 0.10,
        "source_harvest_ungated_max_seconds": 600,
        "per_condition_max_seconds": 3_600,
        "checkpoint_tokens": [2_000_000],
    }
    for name, expected in frozen_execution.items():
        _exact(execution[name], expected, name=f"execution.{name}")

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
    frozen_optimizer = {
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
    for name, expected in frozen_optimizer.items():
        _exact(optimizer[name], expected, name=f"optimizer.{name}")

    raw_layouts = top["layouts"]
    if not isinstance(raw_layouts, list) or len(raw_layouts) != 4:
        raise ContributionTranscoderConfigError("layouts must contain exactly A/R/B/F")
    expected_layouts = {
        "A": ("true_expert_id", 128, 256),
        "R": ("per_sequence_randomized_expert_id", 128, 256),
        "B": ("router_rank", 8, 256),
        "F": ("router_rank", 8, 4_096),
    }
    layouts: list[TranscoderLayout] = []
    for index, condition in enumerate(CONDITIONS):
        item = _mapping(
            raw_layouts[index],
            owner=f"layouts[{index}]",
            keys={
                "condition",
                "bank_identity",
                "bank_count",
                "bank_width",
                "components_per_token",
                "topk_per_component",
            },
        )
        bank_identity, bank_count, bank_width = expected_layouts[condition]
        for name, expected in {
            "condition": condition,
            "bank_identity": bank_identity,
            "bank_count": bank_count,
            "bank_width": bank_width,
            "components_per_token": 8,
            "topk_per_component": 2,
        }.items():
            _exact(item[name], expected, name=f"layouts[{index}].{name}")
        layouts.append(
            TranscoderLayout(
                condition=condition,
                bank_width=bank_width,
                topk_per_component=2,
            )
        )

    evaluation = _mapping(
        top["evaluation"],
        owner="evaluation",
        keys={
            "shared_branch",
            "common_target_sst",
            "bootstrap_unit",
            "bootstrap_replicates",
            "paired_confidence_level",
            "primary_contrasts",
            "minimum_g3_percent",
            "undefined_denominator_policy",
            "effect_retention_reference",
        },
    )
    frozen_evaluation = {
        "shared_branch": "sealed_actual_oracle",
        "common_target_sst": True,
        "bootstrap_unit": "packed_sequence",
        "bootstrap_replicates": 20_000,
        "paired_confidence_level": 0.95,
        "primary_contrasts": ["A-R", "A-B", "A-F", "F-B"],
        "minimum_g3_percent": 10.0,
        "undefined_denominator_policy": "report_undefined",
        "effect_retention_reference": {
            "experiment2_a_minus_r": 0.2194296741,
            "experiment2_a_minus_b": 0.0216359968,
        },
    }
    for name, expected in frozen_evaluation.items():
        _exact(evaluation[name], expected, name=f"evaluation.{name}")

    canonical = dict(top)
    result = Experiment3Config(
        raw=canonical,
        config_hash=canonical_sha256(canonical),
        protocol_id=PROTOCOL_ID,
        encoder_activation=str(training["encoder_activation"]),
        provenance=TranscoderProvenance(
            maxtext_fork_commit=str(provenance["maxtext_fork_commit"])
        ),
        model=TranscoderModel(
            hidden_size=int(model["hidden_size"]),
            num_routed_experts=int(model["num_routed_experts"]),
            experts_per_token=int(model["experts_per_token"]),
        ),
        runtime=TranscoderRuntime(
            effective_batch_tokens=int(training["effective_batch_tokens"]),
            microbatch_tokens=int(training["microbatch_tokens"]),
            capacity_margin_fraction=float(training["capacity_margin_fraction"]),
            accelerator_min_useful_mfu=float(execution["accelerator_min_useful_mfu"]),
            accelerator_qualification_max_seconds=int(
                execution["source_harvest_ungated_max_seconds"]
            ),
        ),
        optimizer=TranscoderOptimizer(
            parameter_dtype=str(optimizer["parameter_dtype"]),
            adam_beta1=float(optimizer["adam_beta1"]),
            adam_beta2=float(optimizer["adam_beta2"]),
            adam_epsilon=float(optimizer["adam_epsilon"]),
            global_gradient_clip=float(optimizer["global_gradient_clip"]),
            peak_learning_rate=float(optimizer["peak_learning_rate"]),
            minimum_learning_rate=float(optimizer["minimum_learning_rate"]),
        ),
        layouts=tuple(layouts),
        parents=dict(parents),
    )
    result.validate()
    return result


def load_contribution_transcoder_config(
    path: str | Path,
) -> Experiment3Config:
    selected = Path(path).expanduser().resolve()
    try:
        raw = yaml.safe_load(selected.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise ContributionTranscoderConfigError(
            f"cannot load transcoder config {selected}"
        ) from error
    return _validate_document(raw)


__all__ = [
    "CONDITIONS",
    "Experiment3Config",
    "PHASE_NAME",
    "PROTOCOL_ID",
    "SCHEMA_VERSION",
    "ContributionTranscoderConfigError",
    "load_contribution_transcoder_config",
]
