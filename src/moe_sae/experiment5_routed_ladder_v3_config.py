"""Immutable Experiment-5 v3 activation-policy configuration.

V2 tried to make ReLU satisfy an exact-positive TopK contract.  The
transcoder trajectories demonstrated that this is not a durable feasibility
condition: a visible bank can have fewer than ``k`` strictly positive ReLU
activations even while the model state and reconstruction loss remain finite.

V3 asks the scientifically useful question without relabelling zeros as active:

* ``A512ReLUInclusiveTopK`` selects exactly four ReLU slots per visible bank,
  allowing deterministic zero-valued selections when fewer than four
  activations are positive;
* ``A512Softplus`` retains the exact-positive four-slot policy; and
* ``A256Native`` is panel-native, read-only Experiment-4 context (ReLU for the
  SAE panel and Softplus for the transcoder panel).

The two A512 conditions match width, parameters, candidates, and maximum
selected slots.  They do *not* match exact positive L0.  Reports must
therefore expose selected-slot count, positive L0, and zero-selected-slot
fraction separately.  Completed v2 trajectories are imported only for
Softplus, where the activation, selection policy, and training graph are
identical.  No v2 ReLU trajectory crosses the v3 selection-policy boundary.

The unchanged model, artifacts, optimizer, runtime, H reference, and endpoint
contract are projected through the audited v1 validator.  V2 identities and
artifacts are never rewritten; only the six explicitly mapped, completed,
graph-identical Softplus trajectories enter the v3 quality grid.
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

SCHEMA_VERSION = "gemma4-experiment5-routed-activation-ladder-run-config-v3"
PROTOCOL_ID = "gemma4-experiment5-routed-activation-ladder-v3-20260729"
PHASE_NAME = "experiment5_routed_activation_ladder_v3_10m"
LAYOUT_SET = "true_id_routed_activation_policy_v3"
CONDITIONS = ("A512ReLUInclusiveTopK", "A512Softplus")
ANALYZED_CONDITIONS = ("A256Native", *CONDITIONS, "H")
PANELS = ("sae", "transcoder")
SEEDS = (0, 1, 2)
CHECKPOINT_TOKENS = (2_000_000, 5_000_000, 10_000_000)
REQUIRED_SPARSITY_METRICS = (
    "selected_slot_count",
    "positive_l0",
    "zero_selected_slot_fraction",
)
PRIMARY_CONTRAST_LABEL = "A512Softplus-A512ReLUInclusiveTopK"
PRIMARY_ESTIMAND = (
    "activation_policy_effect_at_matched_geometry_and_max_selected_slots"
)

A256_NATIVE_CONFIG_SHA256 = {
    "sae": "035e1afc91f7351fda3128582ac52e2051fcb0741f6f71913f5141aaa1c7ab59",
    "transcoder": (
        "1742ebc1b7993dffd0b35665695a074c3d10025f5d410021dd66ccb1bb221ec9"
    ),
}
A256_NATIVE_LABEL = {
    "sae": "A256ReLU",
    "transcoder": "A256Softplus",
}
A256_NATIVE_ACTIVATION = {
    "sae": "relu",
    "transcoder": "softplus",
}
A256_NATIVE_SELECTION_POLICY = {
    "sae": "strict_positive_topk",
    "transcoder": "exact_positive_topk",
}
A256_SCALE_INTERPRETATION = {
    "sae": "descriptive_pending_inclusive_policy_replay_equivalence",
    "transcoder": "activation_matched_historical_scale_context",
}

V2_SCHEMA_VERSION = "gemma4-experiment5-routed-activation-ladder-run-config-v2"
V2_PROTOCOL_ID = "gemma4-experiment5-routed-activation-ladder-v2-20260729"
V2_CONFIG_SHA256 = {
    "sae": "eec72b479ef470fa02917bef78a731503a4ad890e86a4571eae5ba64e9217f46",
    "transcoder": (
        "12062fbc4073bc93d72a55643ff3031d6166996e8454b35e557596e6361a9787"
    ),
}

_V1_SCHEMA_VERSION = "gemma4-experiment5-routed-activation-ladder-run-config-v1"
_V1_PROTOCOL_ID = "gemma4-experiment5-routed-activation-ladder-v1-20260729"
_V1_PHASE_NAME = "experiment5_routed_activation_ladder_10m"
_V1_LAYOUT_SET = "true_id_routed_activation_ladder"


class RoutedLadderV3ConfigError(RoutedLadderConfigError):
    """A v3 activation-policy document differs from its immutable protocol."""


def _mapping(
    value: Any,
    *,
    name: str,
    keys: set[str] | None = None,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RoutedLadderV3ConfigError(f"{name} must be a mapping")
    if keys is not None and set(value) != keys:
        raise RoutedLadderV3ConfigError(
            f"{name} fields differ: expected {sorted(keys)}, got {sorted(value)}"
        )
    return value


def _exact(value: Any, expected: Any, *, name: str) -> None:
    if value != expected:
        raise RoutedLadderV3ConfigError(
            f"{name} is frozen to {expected!r}, got {value!r}"
        )


@dataclass(frozen=True, slots=True)
class RoutedLadderV3Layout:
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
    selected_slots_per_token: int
    positive_l0_contract: str
    maximum_positive_l0: int
    trainable_parameters: int
    encoder_activation: str
    selection_policy: str
    topk_requires_strictly_positive: bool
    softplus_positive_floor: str | None
    training_seeds: tuple[int, ...]

    @property
    def worker_condition(self) -> str:
        return self.base_condition

    @property
    def total_l0(self) -> int:
        """Compatibility alias for the selected-slot budget, not positive L0."""

        return self.selected_slots_per_token


@dataclass(frozen=True, slots=True)
class ExistingA256NativeReference:
    condition: str
    source_label: str
    source_condition: str
    source_protocol_id: str
    source_config_sha256: str
    source_phase_name: str
    training_policy: str
    reused_seeds: tuple[int, ...]
    encoder_activation: str
    selection_policy: str
    objective: str
    bank_width: int
    topk_per_component: int
    checkpoint_tokens: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PriorV2Reference:
    schema_version: str
    protocol_id: str
    config_sha256: str
    artifact_policy: str
    unmapped_quality_grid_allowed: bool
    evidentiary_role: str
    terminal_reason: str


@dataclass(frozen=True, slots=True)
class V2ConditionImport:
    condition: str
    source_condition: str
    source_config_sha256: str
    import_policy: str
    reused_seeds: tuple[int, ...]
    new_training_seeds: tuple[int, ...]
    equivalence_basis: str
    checkpoint_tokens: tuple[int, ...]
    failed_or_partial_import_allowed: bool


@dataclass(frozen=True, slots=True)
class RoutedLadderV3ReportContract:
    primary_contrast_label: str
    estimand: str
    paired_seeds: tuple[int, ...]
    fixed_bank_width: bool
    fixed_parameter_count: bool
    fixed_candidate_count: bool
    fixed_max_selected_slots: bool
    fixed_exact_positive_l0: bool
    required_sparsity_metrics: tuple[str, ...]
    a256_condition: str
    a256_panel_native: bool
    a256_scale_interpretation: str
    h_role: str


@dataclass(frozen=True, slots=True)
class RoutedLadderV3Config:
    """Worker-compatible view of one immutable v3 panel."""

    raw: Mapping[str, Any]
    config_hash: str
    protocol_id: str
    panel: Literal["sae", "transcoder"]
    scientific_model: Literal["sae", "transcoder"]
    provenance: RoutedLadderProvenance
    model: RoutedLadderModel
    runtime: RoutedLadderRuntime
    optimizer: RoutedLadderOptimizer
    layouts: tuple[RoutedLadderV3Layout, ...]
    parents: Mapping[str, Any]
    parent_protocol_ids: Mapping[str, str]
    ladder_phase: RoutedLadderPhase
    existing_a256_native_reference: ExistingA256NativeReference
    existing_h_reference: ExistingHReference
    prior_v2_reference: PriorV2Reference
    v2_condition_imports: tuple[V2ConditionImport, ...]
    report_contract: RoutedLadderV3ReportContract
    recomposition_sidecar_sha256: str

    def validate(self) -> None:
        if canonical_sha256(self.raw) != self.config_hash:
            raise RoutedLadderV3ConfigError("routed-ladder v3 config identity changed")
        if self.protocol_id != PROTOCOL_ID or self.panel not in PANELS:
            raise RoutedLadderV3ConfigError(
                "routed-ladder v3 protocol identity drifted"
            )
        if tuple(item.condition for item in self.layouts) != CONDITIONS:
            raise RoutedLadderV3ConfigError(
                "layouts must be exactly "
                "A512ReLUInclusiveTopK/A512Softplus"
            )
        if any(item.worker_condition != "A" for item in self.layouts):
            raise RoutedLadderV3ConfigError(
                "v3 layouts must reuse condition-A grouped-kernel semantics"
            )
        expected_training_seeds = (SEEDS, ())
        if tuple(item.training_seeds for item in self.layouts) != (
            expected_training_seeds
        ):
            raise RoutedLadderV3ConfigError(
                "v3 training seeds differ from the exact import plan"
            )
        relu, softplus = self.layouts
        if (
            relu.selected_slots_per_token != softplus.selected_slots_per_token
            or relu.maximum_positive_l0 != softplus.maximum_positive_l0
            or relu.positive_l0_contract == softplus.positive_l0_contract
        ):
            raise RoutedLadderV3ConfigError(
                "v3 must match selected-slot budgets but distinguish positive L0"
            )
        if (
            self.existing_a256_native_reference.condition != "A256Native"
            or self.existing_a256_native_reference.training_policy
            != "reuse_existing_seed0_only"
        ):
            raise RoutedLadderV3ConfigError(
                "A256Native must remain panel-native read-only context"
            )
        if (
            self.prior_v2_reference.artifact_policy != "immutable_read_only"
            or self.prior_v2_reference.unmapped_quality_grid_allowed
        ):
            raise RoutedLadderV3ConfigError(
                "unmapped v2 evidence must remain outside the quality grid"
            )
        relu_import, softplus_import = self.v2_condition_imports
        if (
            relu_import.condition != "A512ReLUInclusiveTopK"
            or softplus_import.condition != "A512Softplus"
            or softplus_import.reused_seeds != SEEDS
            or softplus_import.new_training_seeds
            or softplus_import.failed_or_partial_import_allowed
        ):
            raise RoutedLadderV3ConfigError(
                "v3 Softplus must reuse only completed equivalent v2 trajectories"
            )
        if (
            relu_import.reused_seeds
            or relu_import.new_training_seeds != SEEDS
            or relu_import.import_policy != "train_v3_from_scratch"
        ):
            raise RoutedLadderV3ConfigError(
                "both panels must train all ReLU-inclusive v3 seeds from scratch"
            )
        if self.report_contract.fixed_exact_positive_l0:
            raise RoutedLadderV3ConfigError(
                "the v3 activation contrast does not fix exact positive L0"
            )

    def phase(self, name: str) -> RoutedLadderPhase:
        if name != PHASE_NAME:
            raise RoutedLadderV3ConfigError(
                f"routed ladder v3 has one phase named {PHASE_NAME!r}"
            )
        return self.ladder_phase

    def layouts_for_phase(self, name: str) -> tuple[RoutedLadderV3Layout, ...]:
        self.phase(name)
        return self.layouts

    def layout_for_condition(self, condition: str) -> RoutedLadderV3Layout:
        matches = tuple(item for item in self.layouts if item.condition == condition)
        if len(matches) != 1:
            raise RoutedLadderV3ConfigError(
                f"condition {condition!r} does not name one trainable v3 layout"
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
) -> RoutedLadderV3Layout:
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
            "selected_slots_per_token",
            "positive_l0_contract",
            "maximum_positive_l0",
            "trainable_parameters",
            "encoder_activation",
            "selection_policy",
            "topk_requires_strictly_positive",
            "softplus_positive_floor",
            "training_seeds",
        },
    )
    condition = CONDITIONS[index]
    activation = ("relu", "softplus")[index]
    selection_policy = ("inclusive_zero_topk", "exact_positive_topk")[index]
    positive_l0_contract = (
        "bounded_zero_to_selected_slots",
        "exact_selected_slots",
    )[index]
    strict = index == 1
    training_seeds = (
        ()
        if condition == "A512Softplus"
        else SEEDS
    )
    expected = {
        "condition": condition,
        "base_condition": "A",
        "target": "raw_selected_expert_output",
        "bank_identity": "true_expert_id",
        "bank_count": 128,
        "bank_width": 512,
        "components_per_token": 8,
        "topk_per_component": 4,
        "total_features": 65_536,
        "candidates_per_token": 4_096,
        "selected_slots_per_token": 32,
        "positive_l0_contract": positive_l0_contract,
        "maximum_positive_l0": 32,
        "trainable_parameters": 369_164_288,
        "encoder_activation": activation,
        "selection_policy": selection_policy,
        "topk_requires_strictly_positive": strict,
        "softplus_positive_floor": (
            "smallest_normal_float32" if strict else None
        ),
        "training_seeds": list(training_seeds),
    }
    for key, expected_value in expected.items():
        _exact(raw[key], expected_value, name=f"layouts[{index}].{key}")
    return RoutedLadderV3Layout(
        condition=condition,
        base_condition="A",
        target="raw_selected_expert_output",
        bank_identity="true_expert_id",
        bank_count=128,
        bank_width=512,
        components_per_token=8,
        topk_per_component=4,
        total_features=65_536,
        candidates_per_token=4_096,
        selected_slots_per_token=32,
        positive_l0_contract=positive_l0_contract,
        maximum_positive_l0=32,
        trainable_parameters=369_164_288,
        encoder_activation=activation,
        selection_policy=selection_policy,
        topk_requires_strictly_positive=strict,
        softplus_positive_floor=("smallest_normal_float32" if strict else None),
        training_seeds=training_seeds,
    )


def _validate_a256_reference(
    value: Any,
    *,
    panel: str,
) -> ExistingA256NativeReference:
    raw = _mapping(
        value,
        name="existing_a256_native_reference",
        keys={
            "condition",
            "source_label",
            "source_condition",
            "source_protocol_id",
            "source_config_sha256",
            "source_phase_name",
            "training_policy",
            "reused_seeds",
            "encoder_activation",
            "selection_policy",
            "objective",
            "bank_width",
            "topk_per_component",
            "checkpoint_tokens",
        },
    )
    expected = {
        "condition": "A256Native",
        "source_label": A256_NATIVE_LABEL[panel],
        "source_condition": "A",
        "source_protocol_id": EXPERIMENT4_PROTOCOL_ID,
        "source_config_sha256": A256_NATIVE_CONFIG_SHA256[panel],
        "source_phase_name": "learning_horizon_10m",
        "training_policy": "reuse_existing_seed0_only",
        "reused_seeds": [0],
        "encoder_activation": A256_NATIVE_ACTIVATION[panel],
        "selection_policy": A256_NATIVE_SELECTION_POLICY[panel],
        "objective": "normalized_router_weighted_contribution_sse",
        "bank_width": 256,
        "topk_per_component": 2,
        "checkpoint_tokens": list(CHECKPOINT_TOKENS),
    }
    for key, expected_value in expected.items():
        _exact(
            raw[key],
            expected_value,
            name=f"existing_a256_native_reference.{key}",
        )
    return ExistingA256NativeReference(
        condition="A256Native",
        source_label=A256_NATIVE_LABEL[panel],
        source_condition="A",
        source_protocol_id=EXPERIMENT4_PROTOCOL_ID,
        source_config_sha256=A256_NATIVE_CONFIG_SHA256[panel],
        source_phase_name="learning_horizon_10m",
        training_policy="reuse_existing_seed0_only",
        reused_seeds=(0,),
        encoder_activation=A256_NATIVE_ACTIVATION[panel],
        selection_policy=A256_NATIVE_SELECTION_POLICY[panel],
        objective="normalized_router_weighted_contribution_sse",
        bank_width=256,
        topk_per_component=2,
        checkpoint_tokens=CHECKPOINT_TOKENS,
    )


def _validate_prior_v2(value: Any, *, panel: str) -> PriorV2Reference:
    raw = _mapping(
        value,
        name="prior_v2_reference",
        keys={
            "schema_version",
            "protocol_id",
            "config_sha256",
            "artifact_policy",
            "unmapped_quality_grid_allowed",
            "evidentiary_role",
            "terminal_reason",
        },
    )
    expected = {
        "schema_version": V2_SCHEMA_VERSION,
        "protocol_id": V2_PROTOCOL_ID,
        "config_sha256": V2_CONFIG_SHA256[panel],
        "artifact_policy": "immutable_read_only",
        "unmapped_quality_grid_allowed": False,
        "evidentiary_role": "terminal_feasibility_evidence_only",
        "terminal_reason": "fewer_than_k_positive_candidates",
    }
    for key, expected_value in expected.items():
        _exact(raw[key], expected_value, name=f"prior_v2_reference.{key}")
    return PriorV2Reference(**expected)


def _validate_v2_condition_imports(
    value: Any,
    *,
    panel: str,
) -> tuple[V2ConditionImport, ...]:
    if not isinstance(value, list) or len(value) != len(CONDITIONS):
        raise RoutedLadderV3ConfigError(
            "v2_condition_imports must contain exactly the two A512 conditions"
        )
    results: list[V2ConditionImport] = []
    for index, item in enumerate(value):
        raw = _mapping(
            item,
            name=f"v2_condition_imports[{index}]",
            keys={
                "condition",
                "source_condition",
                "source_config_sha256",
                "import_policy",
                "reused_seeds",
                "new_training_seeds",
                "equivalence_basis",
                "checkpoint_tokens",
                "failed_or_partial_import_allowed",
            },
        )
        condition = CONDITIONS[index]
        if condition == "A512Softplus":
            source_condition = "A512Softplus"
            import_policy = "reuse_completed_v2_exact_graph"
            reused_seeds = SEEDS
            new_training_seeds: tuple[int, ...] = ()
            equivalence_basis = "identical_activation_selection_and_training_graph"
        else:
            source_condition = "A512ReLU"
            import_policy = "train_v3_from_scratch"
            reused_seeds = ()
            new_training_seeds = SEEDS
            equivalence_basis = "relu_selection_policy_boundary_not_importable"
        expected = {
            "condition": condition,
            "source_condition": source_condition,
            "source_config_sha256": V2_CONFIG_SHA256[panel],
            "import_policy": import_policy,
            "reused_seeds": list(reused_seeds),
            "new_training_seeds": list(new_training_seeds),
            "equivalence_basis": equivalence_basis,
            "checkpoint_tokens": list(CHECKPOINT_TOKENS),
            "failed_or_partial_import_allowed": False,
        }
        for key, expected_value in expected.items():
            _exact(
                raw[key],
                expected_value,
                name=f"v2_condition_imports[{index}].{key}",
            )
        results.append(
            V2ConditionImport(
                condition=condition,
                source_condition=source_condition,
                source_config_sha256=V2_CONFIG_SHA256[panel],
                import_policy=import_policy,
                reused_seeds=reused_seeds,
                new_training_seeds=new_training_seeds,
                equivalence_basis=equivalence_basis,
                checkpoint_tokens=CHECKPOINT_TOKENS,
                failed_or_partial_import_allowed=False,
            )
        )
    return tuple(results)


def _validate_report_contract(
    value: Any,
    *,
    panel: str,
) -> RoutedLadderV3ReportContract:
    raw = _mapping(
        value,
        name="report_contract",
        keys={
            "primary_contrast",
            "matched_quantities",
            "required_sparsity_metrics",
            "a256_scale_context",
            "h_context",
        },
    )
    primary = _mapping(
        raw["primary_contrast"],
        name="report_contract.primary_contrast",
        keys={"label", "estimand", "paired_seeds"},
    )
    matched = _mapping(
        raw["matched_quantities"],
        name="report_contract.matched_quantities",
        keys={
            "bank_width",
            "trainable_parameters",
            "candidates_per_token",
            "maximum_selected_slots",
            "exact_positive_l0",
        },
    )
    a256 = _mapping(
        raw["a256_scale_context"],
        name="report_contract.a256_scale_context",
        keys={"condition", "panel_native", "interpretation"},
    )
    h_context = _mapping(
        raw["h_context"],
        name="report_contract.h_context",
        keys={"condition", "role"},
    )
    expected_primary = {
        "label": PRIMARY_CONTRAST_LABEL,
        "estimand": PRIMARY_ESTIMAND,
        "paired_seeds": list(SEEDS),
    }
    expected_matched = {
        "bank_width": True,
        "trainable_parameters": True,
        "candidates_per_token": True,
        "maximum_selected_slots": True,
        "exact_positive_l0": False,
    }
    expected_a256 = {
        "condition": "A256Native",
        "panel_native": True,
        "interpretation": A256_SCALE_INTERPRETATION[panel],
    }
    expected_h = {"condition": "H", "role": "read_only_descriptive_context"}
    for name, selected, expected in (
        ("primary_contrast", primary, expected_primary),
        ("matched_quantities", matched, expected_matched),
        ("a256_scale_context", a256, expected_a256),
        ("h_context", h_context, expected_h),
    ):
        for key, expected_value in expected.items():
            _exact(
                selected[key],
                expected_value,
                name=f"report_contract.{name}.{key}",
            )
    _exact(
        raw["required_sparsity_metrics"],
        list(REQUIRED_SPARSITY_METRICS),
        name="report_contract.required_sparsity_metrics",
    )
    return RoutedLadderV3ReportContract(
        primary_contrast_label=PRIMARY_CONTRAST_LABEL,
        estimand=PRIMARY_ESTIMAND,
        paired_seeds=SEEDS,
        fixed_bank_width=True,
        fixed_parameter_count=True,
        fixed_candidate_count=True,
        fixed_max_selected_slots=True,
        fixed_exact_positive_l0=False,
        required_sparsity_metrics=REQUIRED_SPARSITY_METRICS,
        a256_condition="A256Native",
        a256_panel_native=True,
        a256_scale_interpretation=A256_SCALE_INTERPRETATION[panel],
        h_role="read_only_descriptive_context",
    )


def _legacy_projection(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Project the unchanged A512/H surface into the audited v1 validator."""

    projected = json.loads(
        json.dumps(raw, sort_keys=True, separators=(",", ":"), allow_nan=False)
    )
    projected.pop("existing_a256_native_reference")
    projected.pop("prior_v2_reference")
    projected.pop("v2_condition_imports")
    projected.pop("report_contract")
    projected["schema_version"] = _V1_SCHEMA_VERSION
    projected["protocol_id"] = _V1_PROTOCOL_ID
    projected["training"]["phase_name"] = _V1_PHASE_NAME
    projected["training"]["layout_set"] = _V1_LAYOUT_SET
    projected["training"]["initialization"] = "from_scratch"
    projected["training"]["topk_requires_strictly_positive"] = True
    projected["layouts"] = [
        {
            "condition": (
                "A512ReLU"
                if layout["condition"] == "A512ReLUInclusiveTopK"
                else "A512Softplus"
            ),
            "base_condition": layout["base_condition"],
            "target": layout["target"],
            "bank_identity": layout["bank_identity"],
            "bank_count": layout["bank_count"],
            "bank_width": layout["bank_width"],
            "components_per_token": layout["components_per_token"],
            "topk_per_component": layout["topk_per_component"],
            "total_features": layout["total_features"],
            "candidates_per_token": layout["candidates_per_token"],
            "exact_total_l0": layout["selected_slots_per_token"],
            "trainable_parameters": layout["trainable_parameters"],
            "encoder_activation": layout["encoder_activation"],
            "softplus_positive_floor": layout["softplus_positive_floor"],
        }
        for layout in projected["layouts"]
    ]
    projected["evaluation"]["ladder_conditions"] = [
        "A512ReLU",
        "A512Softplus",
        "H",
    ]
    for key in (
        "a256_native_training_allowed",
        "condition_specific_v2_import_policy",
        "required_sparsity_metrics",
        "primary_contrast_label",
        "fixed_exact_positive_l0",
    ):
        projected["evaluation"].pop(key)
    return projected


def _validate_document(
    value: Any,
    *,
    expected_panel: str | None,
) -> RoutedLadderV3Config:
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
            "existing_a256_native_reference",
            "existing_h_reference",
            "prior_v2_reference",
            "v2_condition_imports",
            "execution",
            "evaluation",
            "report_contract",
        },
    )
    _exact(top["schema_version"], SCHEMA_VERSION, name="schema_version")
    _exact(top["protocol_id"], PROTOCOL_ID, name="protocol_id")
    panel = top["panel"]
    if panel not in PANELS:
        raise RoutedLadderV3ConfigError(f"panel must be one of {PANELS}")
    if expected_panel is not None and panel != expected_panel:
        raise RoutedLadderV3ConfigError(
            f"expected a {expected_panel} routed-ladder v3 config, got {panel}"
        )

    training = _mapping(top["training"], name="training")
    _exact(training.get("phase_name"), PHASE_NAME, name="training.phase_name")
    _exact(training.get("layout_set"), LAYOUT_SET, name="training.layout_set")
    _exact(
        training.get("initialization"),
        "from_scratch",
        name="training.initialization",
    )
    _exact(training.get("seeds"), list(SEEDS), name="training.seeds")
    _exact(
        training.get("topk_requires_strictly_positive"),
        "condition_specific_by_layout",
        name="training.topk_requires_strictly_positive",
    )

    raw_layouts = top["layouts"]
    if not isinstance(raw_layouts, list) or len(raw_layouts) != len(CONDITIONS):
        raise RoutedLadderV3ConfigError(
            "layouts must contain exactly "
            "A512ReLUInclusiveTopK/A512Softplus"
        )
    layouts = tuple(
        _validate_layout(item, index=index, panel=panel)
        for index, item in enumerate(raw_layouts)
    )
    a256_reference = _validate_a256_reference(
        top["existing_a256_native_reference"],
        panel=panel,
    )
    prior_v2_reference = _validate_prior_v2(
        top["prior_v2_reference"],
        panel=panel,
    )
    v2_condition_imports = _validate_v2_condition_imports(
        top["v2_condition_imports"],
        panel=panel,
    )
    report_contract = _validate_report_contract(
        top["report_contract"],
        panel=panel,
    )

    evaluation = _mapping(top["evaluation"], name="evaluation")
    expected_evaluation_delta = {
        "ladder_conditions": list(ANALYZED_CONDITIONS),
        "a256_native_training_allowed": False,
        "condition_specific_v2_import_policy": (
            "completed_exact_graph_only"
        ),
        "required_sparsity_metrics": list(REQUIRED_SPARSITY_METRICS),
        "primary_contrast_label": PRIMARY_CONTRAST_LABEL,
        "fixed_exact_positive_l0": False,
    }
    for key, expected in expected_evaluation_delta.items():
        _exact(evaluation.get(key), expected, name=f"evaluation.{key}")

    try:
        legacy = _validate_v1_document(
            _legacy_projection(top),
            expected_panel=panel,
        )
    except RoutedLadderConfigError as error:
        raise RoutedLadderV3ConfigError(
            "routed-ladder v3 shared v1 contract is invalid"
        ) from error

    raw_copy = json.loads(
        json.dumps(top, sort_keys=True, separators=(",", ":"), allow_nan=False)
    )
    result = RoutedLadderV3Config(
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
            initialization="from_scratch",
        ),
        existing_a256_native_reference=a256_reference,
        existing_h_reference=legacy.existing_h_reference,
        prior_v2_reference=prior_v2_reference,
        v2_condition_imports=v2_condition_imports,
        report_contract=report_contract,
        recomposition_sidecar_sha256=legacy.recomposition_sidecar_sha256,
    )
    result.validate()
    return result


def load_routed_ladder_v3_config(
    path: str | Path,
    *,
    expected_panel: Literal["sae", "transcoder"] | None = None,
) -> RoutedLadderV3Config:
    """Load and exhaustively validate one v3 activation-policy contract."""

    source = Path(path).expanduser().resolve()
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise RoutedLadderV3ConfigError(
            f"cannot read routed-ladder v3 config {source}"
        ) from error
    return _validate_document(raw, expected_panel=expected_panel)


__all__ = [
    "A256_NATIVE_ACTIVATION",
    "A256_NATIVE_CONFIG_SHA256",
    "A256_NATIVE_LABEL",
    "A256_NATIVE_SELECTION_POLICY",
    "A256_SCALE_INTERPRETATION",
    "ANALYZED_CONDITIONS",
    "CHECKPOINT_TOKENS",
    "CONDITIONS",
    "ExistingA256NativeReference",
    "LAYOUT_SET",
    "PANELS",
    "PHASE_NAME",
    "PRIMARY_CONTRAST_LABEL",
    "PRIMARY_ESTIMAND",
    "PROTOCOL_ID",
    "PriorV2Reference",
    "REQUIRED_SPARSITY_METRICS",
    "RoutedLadderV3Config",
    "RoutedLadderV3ConfigError",
    "RoutedLadderV3Layout",
    "RoutedLadderV3ReportContract",
    "SCHEMA_VERSION",
    "SEEDS",
    "V2_CONFIG_SHA256",
    "V2ConditionImport",
    "V2_PROTOCOL_ID",
    "V2_SCHEMA_VERSION",
    "load_routed_ladder_v3_config",
]
