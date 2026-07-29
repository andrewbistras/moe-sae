"""Strict protocol-v2 configuration for the corrected routed ladder.

Protocol v1 accidentally omitted the historical Experiment-4 ``A`` geometry
from the three-condition activation/width ladder.  V2 makes the intended
scientific matrix explicit:

* ``A256ReLU``: 128 routed banks of width 256, TopK-2 per visible bank;
* ``A512ReLU``: 128 routed banks of width 512, TopK-4 per visible bank;
* ``A512Softplus``: the same 512/TopK-4 geometry with Softplus; and
* read-only ``H`` checkpoints from the completed Experiment-5 core.

Only the Experiment-4 SAE seed 0 is compatible with ``A256ReLU``.  It is an
immutable reference, not three-seed evidence, so the SAE panel trains seeds 1
and 2.  The historical transcoder seed 0 used Softplus and is deliberately
excluded; the transcoder panel trains all three A256ReLU seeds from scratch.
V2 requires all three exact cells in either panel before an across-seed A256
comparison can be reported.

The validator derives and checks the unchanged v1 A512/H/artifact contract
through :mod:`moe_sae.experiment5_routed_ladder_config`.  This keeps the v2
delta small while retaining exhaustive validation of the shared scientific
surface.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from moe_sae.contribution_artifacts import canonical_sha256
from moe_sae.experiment5_routed_ladder_config import (
    EXPERIMENT4_PROTOCOL_ID,
    EXPERIMENT5_PROTOCOL_ID,
    H_CONFIG_SHA256,
    RoutedLadderConfigError,
    RoutedLadderModel,
    RoutedLadderOptimizer,
    RoutedLadderPhase,
    RoutedLadderProvenance,
    RoutedLadderRuntime,
    ExistingHReference,
    _validate_document as _validate_v1_document,
)

SCHEMA_VERSION = "gemma4-experiment5-routed-activation-ladder-run-config-v2"
PROTOCOL_ID = "gemma4-experiment5-routed-activation-ladder-v2-20260729"
PHASE_NAME = "experiment5_routed_activation_ladder_v2_10m"
LAYOUT_SET = "true_id_routed_activation_ladder_v2"
CONDITIONS = ("A256ReLU", "A512ReLU", "A512Softplus")
ANALYZED_CONDITIONS = (*CONDITIONS, "H")
PANELS = ("sae", "transcoder")
SEEDS = (0, 1, 2)
CHECKPOINT_TOKENS = (2_000_000, 5_000_000, 10_000_000)

A256_SAE_CONFIG_SHA256 = (
    "035e1afc91f7351fda3128582ac52e2051fcb0741f6f71913f5141aaa1c7ab59"
)
A256_SAE_REUSED_SEEDS = (0,)
A256_SAE_NEW_TRAINING_SEEDS = (1, 2)
ALL_NEW_TRAINING_SEEDS = (0, 1, 2)

_V1_SCHEMA_VERSION = "gemma4-experiment5-routed-activation-ladder-run-config-v1"
_V1_PROTOCOL_ID = "gemma4-experiment5-routed-activation-ladder-v1-20260729"
_V1_PHASE_NAME = "experiment5_routed_activation_ladder_10m"
_V1_LAYOUT_SET = "true_id_routed_activation_ladder"


class RoutedLadderV2ConfigError(RoutedLadderConfigError):
    """A v2 routed-ladder document differs from its immutable protocol."""


def _mapping(
    value: Any,
    *,
    name: str,
    keys: set[str] | None = None,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RoutedLadderV2ConfigError(f"{name} must be a mapping")
    if keys is not None and set(value) != keys:
        raise RoutedLadderV2ConfigError(
            f"{name} fields differ: expected {sorted(keys)}, got {sorted(value)}"
        )
    return value


def _exact(value: Any, expected: Any, *, name: str) -> None:
    if value != expected:
        raise RoutedLadderV2ConfigError(
            f"{name} is frozen to {expected!r}, got {value!r}"
        )


@dataclass(frozen=True, slots=True)
class RoutedLadderV2Layout:
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
    training_seeds: tuple[int, ...]

    @property
    def worker_condition(self) -> str:
        return self.base_condition


@dataclass(frozen=True, slots=True)
class ExistingA256Reference:
    """The one historical seed plus the exact missing-seed training policy."""

    condition: str
    source_condition: str
    source_protocol_id: str
    source_config_sha256: str
    source_phase_name: str
    training_policy: str
    reused_seeds: tuple[int, ...]
    new_training_seeds: tuple[int, ...]
    encoder_activation: str
    objective: str
    bank_width: int
    topk_per_component: int
    checkpoint_tokens: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RoutedLadderV2Config:
    """Worker-compatible view of one immutable corrected-ladder panel."""

    raw: Mapping[str, Any]
    config_hash: str
    protocol_id: str
    panel: Literal["sae", "transcoder"]
    scientific_model: Literal["sae", "transcoder"]
    provenance: RoutedLadderProvenance
    model: RoutedLadderModel
    runtime: RoutedLadderRuntime
    optimizer: RoutedLadderOptimizer
    layouts: tuple[RoutedLadderV2Layout, ...]
    parents: Mapping[str, Any]
    parent_protocol_ids: Mapping[str, str]
    ladder_phase: RoutedLadderPhase
    existing_a256_reference: ExistingA256Reference | None
    existing_h_reference: ExistingHReference
    recomposition_sidecar_sha256: str

    def validate(self) -> None:
        if canonical_sha256(self.raw) != self.config_hash:
            raise RoutedLadderV2ConfigError("routed-ladder v2 config identity changed")
        if self.protocol_id != PROTOCOL_ID or self.panel not in PANELS:
            raise RoutedLadderV2ConfigError(
                "routed-ladder v2 protocol identity drifted"
            )
        if tuple(item.condition for item in self.layouts) != CONDITIONS:
            raise RoutedLadderV2ConfigError(
                "layouts must be exactly A256ReLU/A512ReLU/A512Softplus"
            )
        if any(item.worker_condition != "A" for item in self.layouts):
            raise RoutedLadderV2ConfigError(
                "routed-ladder layouts must reuse condition-A worker semantics"
            )
        reference = self.existing_a256_reference
        if self.panel == "sae":
            if (
                reference is None
                or reference.condition != "A256ReLU"
                or reference.reused_seeds != A256_SAE_REUSED_SEEDS
                or reference.new_training_seeds
                != A256_SAE_NEW_TRAINING_SEEDS
                or self.layouts[0].training_seeds
                != A256_SAE_NEW_TRAINING_SEEDS
            ):
                raise RoutedLadderV2ConfigError(
                    "SAE A256 seed 0 must be reused and only seeds 1/2 trained"
                )
        elif (
            reference is not None
            or self.layouts[0].training_seeds != ALL_NEW_TRAINING_SEEDS
        ):
            raise RoutedLadderV2ConfigError(
                "transcoder A256ReLU must train seeds 0/1/2 from scratch"
            )
        if any(
            layout.training_seeds != ALL_NEW_TRAINING_SEEDS
            for layout in self.layouts[1:]
        ):
            raise RoutedLadderV2ConfigError(
                "both A512 conditions must train seeds 0/1/2"
            )
        if (
            self.existing_h_reference.condition != "H"
            or self.existing_h_reference.training_policy
            != "reuse_existing_artifacts_only"
        ):
            raise RoutedLadderV2ConfigError("H must remain a read-only reference")

    def phase(self, name: str) -> RoutedLadderPhase:
        if name != PHASE_NAME:
            raise RoutedLadderV2ConfigError(
                f"routed ladder v2 has one phase named {PHASE_NAME!r}"
            )
        return self.ladder_phase

    def layouts_for_phase(self, name: str) -> tuple[RoutedLadderV2Layout, ...]:
        self.phase(name)
        return self.layouts

    def layout_for_condition(self, condition: str) -> RoutedLadderV2Layout:
        matches = tuple(item for item in self.layouts if item.condition == condition)
        if len(matches) != 1:
            raise RoutedLadderV2ConfigError(
                f"condition {condition!r} does not name one trainable layout"
            )
        return matches[0]

    def training_seeds_for_condition(self, condition: str) -> tuple[int, ...]:
        return self.layout_for_condition(condition).training_seeds

    def as_dict(self) -> dict[str, Any]:
        return json.loads(
            json.dumps(
                self.raw,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )


def _validate_layout(
    value: Any,
    *,
    index: int,
    panel: str,
) -> RoutedLadderV2Layout:
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
            "training_seeds",
        },
    )
    condition = CONDITIONS[index]
    width = (256, 512, 512)[index]
    topk = (2, 4, 4)[index]
    activation = ("relu", "relu", "softplus")[index]
    training_seeds = (
        A256_SAE_NEW_TRAINING_SEEDS
        if index == 0 and panel == "sae"
        else ALL_NEW_TRAINING_SEEDS
    )
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
        "training_seeds": list(training_seeds),
    }
    for key, expected_value in expected.items():
        _exact(raw[key], expected_value, name=f"layouts[{index}].{key}")
    return RoutedLadderV2Layout(
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
        training_seeds=training_seeds,
    )


def _legacy_projection(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Project the unchanged A512/H surface into the audited v1 validator."""

    projected = json.loads(
        json.dumps(raw, sort_keys=True, separators=(",", ":"), allow_nan=False)
    )
    projected.pop("existing_a256_reference")
    projected["schema_version"] = _V1_SCHEMA_VERSION
    projected["protocol_id"] = _V1_PROTOCOL_ID
    projected["training"]["phase_name"] = _V1_PHASE_NAME
    projected["training"]["layout_set"] = _V1_LAYOUT_SET
    projected["training"]["initialization"] = "from_scratch"
    projected["layouts"] = projected["layouts"][1:]
    for layout in projected["layouts"]:
        layout.pop("training_seeds")
    projected["evaluation"]["ladder_conditions"] = [
        "A512ReLU",
        "A512Softplus",
        "H",
    ]
    projected["evaluation"].pop("a256_reference_training_allowed")
    return projected


def _validate_document(
    value: Any,
    *,
    expected_panel: str | None,
) -> RoutedLadderV2Config:
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
            "existing_a256_reference",
            "existing_h_reference",
            "execution",
            "evaluation",
        },
    )
    _exact(top["schema_version"], SCHEMA_VERSION, name="schema_version")
    _exact(top["protocol_id"], PROTOCOL_ID, name="protocol_id")
    panel = top["panel"]
    if panel not in PANELS:
        raise RoutedLadderV2ConfigError(f"panel must be one of {PANELS}")
    if expected_panel is not None and panel != expected_panel:
        raise RoutedLadderV2ConfigError(
            f"expected a {expected_panel} routed-ladder v2 config, got {panel}"
        )

    training = _mapping(top["training"], name="training")
    _exact(training.get("phase_name"), PHASE_NAME, name="training.phase_name")
    _exact(training.get("layout_set"), LAYOUT_SET, name="training.layout_set")
    _exact(
        training.get("initialization"),
        "from_scratch_for_new_trajectories",
        name="training.initialization",
    )
    _exact(training.get("seeds"), list(SEEDS), name="training.seeds")

    raw_layouts = top["layouts"]
    if not isinstance(raw_layouts, list) or len(raw_layouts) != len(CONDITIONS):
        raise RoutedLadderV2ConfigError(
            "layouts must contain exactly A256ReLU/A512ReLU/A512Softplus"
        )
    layouts = tuple(
        _validate_layout(item, index=index, panel=panel)
        for index, item in enumerate(raw_layouts)
    )

    existing_a256_reference: ExistingA256Reference | None
    if panel == "sae":
        raw_reference = _mapping(
            top["existing_a256_reference"],
            name="existing_a256_reference",
            keys={
                "condition",
                "source_condition",
                "source_protocol_id",
                "source_config_sha256",
                "source_phase_name",
                "training_policy",
                "reused_seeds",
                "new_training_seeds",
                "encoder_activation",
                "objective",
                "bank_width",
                "topk_per_component",
                "checkpoint_tokens",
            },
        )
        expected_reference = {
            "condition": "A256ReLU",
            "source_condition": "A",
            "source_protocol_id": EXPERIMENT4_PROTOCOL_ID,
            "source_config_sha256": A256_SAE_CONFIG_SHA256,
            "source_phase_name": "learning_horizon_10m",
            "training_policy": "reuse_seed0_train_missing_seeds",
            "reused_seeds": list(A256_SAE_REUSED_SEEDS),
            "new_training_seeds": list(A256_SAE_NEW_TRAINING_SEEDS),
            "encoder_activation": "relu",
            "objective": "normalized_router_weighted_contribution_sse",
            "bank_width": 256,
            "topk_per_component": 2,
            "checkpoint_tokens": list(CHECKPOINT_TOKENS),
        }
        for key, expected in expected_reference.items():
            _exact(
                raw_reference[key],
                expected,
                name=f"existing_a256_reference.{key}",
            )
        existing_a256_reference = ExistingA256Reference(
            condition="A256ReLU",
            source_condition="A",
            source_protocol_id=EXPERIMENT4_PROTOCOL_ID,
            source_config_sha256=A256_SAE_CONFIG_SHA256,
            source_phase_name="learning_horizon_10m",
            training_policy="reuse_seed0_train_missing_seeds",
            reused_seeds=A256_SAE_REUSED_SEEDS,
            new_training_seeds=A256_SAE_NEW_TRAINING_SEEDS,
            encoder_activation="relu",
            objective="normalized_router_weighted_contribution_sse",
            bank_width=256,
            topk_per_component=2,
            checkpoint_tokens=CHECKPOINT_TOKENS,
        )
    else:
        _exact(
            top["existing_a256_reference"],
            None,
            name="existing_a256_reference",
        )
        existing_a256_reference = None

    evaluation = _mapping(top["evaluation"], name="evaluation")
    _exact(
        evaluation.get("ladder_conditions"),
        list(ANALYZED_CONDITIONS),
        name="evaluation.ladder_conditions",
    )
    _exact(
        evaluation.get("a256_reference_training_allowed"),
        False,
        name="evaluation.a256_reference_training_allowed",
    )

    # Validate every unchanged artifact, optimizer, A512, H, runtime, and
    # evaluation field with the already-audited v1 parser.
    try:
        legacy = _validate_v1_document(
            _legacy_projection(top),
            expected_panel=panel,
        )
    except RoutedLadderConfigError as error:
        raise RoutedLadderV2ConfigError(
            "routed-ladder v2 shared v1 contract is invalid"
        ) from error

    raw_copy = json.loads(
        json.dumps(top, sort_keys=True, separators=(",", ":"), allow_nan=False)
    )
    result = RoutedLadderV2Config(
        raw=raw_copy,
        config_hash=canonical_sha256(raw_copy),
        protocol_id=PROTOCOL_ID,
        panel=panel,
        scientific_model=panel,
        provenance=legacy.provenance,
        model=legacy.model,
        runtime=legacy.runtime,
        optimizer=legacy.optimizer,
        layouts=layouts,
        parents=legacy.parents,
        parent_protocol_ids=legacy.parent_protocol_ids,
        ladder_phase=RoutedLadderPhase(
            name=PHASE_NAME,
            layout_set=LAYOUT_SET,
            seeds=SEEDS,
            train_tokens=legacy.ladder_phase.train_tokens,
            eval_tokens=legacy.ladder_phase.eval_tokens,
            planned_updates=legacy.ladder_phase.planned_updates,
            warmup_updates=legacy.ladder_phase.warmup_updates,
            tail_mask_tokens=legacy.ladder_phase.tail_mask_tokens,
            restart_checkpoint_tokens=(
                legacy.ladder_phase.restart_checkpoint_tokens
            ),
            evaluation_checkpoint_tokens=(
                legacy.ladder_phase.evaluation_checkpoint_tokens
            ),
            initialization="from_scratch_for_new_trajectories",
        ),
        existing_a256_reference=existing_a256_reference,
        existing_h_reference=legacy.existing_h_reference,
        recomposition_sidecar_sha256=legacy.recomposition_sidecar_sha256,
    )
    result.validate()
    return result


def load_routed_ladder_v2_config(
    path: str | Path,
    *,
    expected_panel: Literal["sae", "transcoder"] | None = None,
) -> RoutedLadderV2Config:
    """Load and exhaustively validate one corrected-ladder YAML contract."""

    source = Path(path).expanduser().resolve()
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise RoutedLadderV2ConfigError(
            f"cannot read routed-ladder v2 config {source}"
        ) from error
    return _validate_document(raw, expected_panel=expected_panel)


__all__ = [
    "A256_SAE_CONFIG_SHA256",
    "A256_SAE_NEW_TRAINING_SEEDS",
    "A256_SAE_REUSED_SEEDS",
    "ANALYZED_CONDITIONS",
    "CHECKPOINT_TOKENS",
    "CONDITIONS",
    "ExistingA256Reference",
    "LAYOUT_SET",
    "PANELS",
    "PHASE_NAME",
    "PROTOCOL_ID",
    "RoutedLadderV2Config",
    "RoutedLadderV2ConfigError",
    "RoutedLadderV2Layout",
    "SCHEMA_VERSION",
    "SEEDS",
    "load_routed_ladder_v2_config",
]
