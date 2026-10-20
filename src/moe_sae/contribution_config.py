"""Strict run contract for the staged Gemma 4 contribution experiment.

This module is deliberately separate from :mod:`moe_sae.config`.  Experiment
1 aggregate captures and Experiment 2 expert-contribution captures are
different scientific objects, with different phase plans and shard limits.
Silently loading one experiment's configuration into the other is therefore
an error.

The checked-in configuration pins the immutable MaxText base, selected-output
patch content, and the exact published fork commit containing that patch.
Every authoritative TPU capture must observe that same 40-character revision.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "gemma4-contribution-run-config-v1"
PROTOCOL_ID = "gemma4-contribution-v1-20260725"

MODEL_ID = "google/gemma-4-26B-A4B"
MODEL_REVISION = "24548b62aa021d562695c04aaf7758a1ea47990b"
MAXTEXT_BASE_COMMIT = "48c7da3e28e8f57fe36ddf09d00eec53072a2ea0"
MAXTEXT_SELECTED_OUTPUTS_PATCH_SHA256 = (
    "9da1ec2a0792a569123093c8840c20d9ee09891854f2665a972d3d7340d8c42d"
)
MAXTEXT_ROUTED_MOE_PATCHED_SHA256 = (
    "e731a50cf30f6bd4295f7891e97c8fdcb104092abf63268bc39aa5e6c8bdc531"
)
MAXTEXT_FORK_COMMIT = "73bafe53569eb17dc18542f2568dd591a8db9e28"
C4_REPOSITORY = "allenai/c4"
C4_REVISION = "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"

CAPTURE_SCHEMA_VERSION = "gemma4-expert-contribution-v2"
CAPTURE_SITE = "routed_moe.selected_outputs.pre_router_weight"
SHARED_CAPTURE_SITE = "shared_expert.raw_output"

CONDITIONS = ("A", "R", "B", "F")
LAYOUT_SETS = ("none", "pilot", "full")
PHASE_NAMES = (
    "phase1_smoke",
    "phase2_mechanism_pilot",
    "phase3_full_width_pilot",
    "phase4_confirmatory",
)

PHASE1_GATES = (
    "all_capture_exactness_checks_pass",
    "bf16_roundtrip_bit_exact",
    "interrupted_resume_shard_identity",
    "multihost_ranges_no_gaps_or_duplicates",
    "no_routed_token_drops",
    "writer_queue_bounded",
    "persisted_goodput_ge_0.60_device_output",
    "no_recompilation_after_warmup",
    "auxiliary_tokens_per_second_ge_0.85_writer_disabled",
)
PHASE2_GATES = (
    "all_conditions_finite_exact_l0_no_overflow_falling_loss",
    "a_minus_r_positive",
    "a_minus_r_paired_sequence_ci_excludes_zero",
    "a_minus_b_positive",
    "a_minus_b_paired_sequence_ci_excludes_zero",
    "f_minus_b_positive",
    "f_minus_b_paired_sequence_ci_excludes_zero",
    "g2_ge_10_percent",
    "a_minus_b_sign_agrees_on_recomposed_routed_ev",
    "coverage_requirements_pass",
)
PHASE3_GATES = (
    "all_phase2_directional_gates_pass",
    "g2_ge_10_percent",
    "c2_lt_50_percent",
    "a_learning_curve_has_no_material_late_collapse",
)
PHASE4_GATES = (
    "all_conditions_pass_health_before_unblinding",
    "three_seed_report_complete",
    "unstable_denominator_labeled_when_required",
)

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class ContributionConfigError(ValueError):
    """An Experiment-2 configuration violates the frozen run contract."""


def _require_exact_keys(
    value: Any,
    expected: set[str],
    *,
    owner: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContributionConfigError(f"{owner} must be a mapping")
    keys = set(value)
    unknown = sorted(keys - expected)
    missing = sorted(expected - keys)
    if unknown:
        raise ContributionConfigError(
            f"{owner} has unknown keys: {', '.join(unknown)}"
        )
    if missing:
        raise ContributionConfigError(
            f"{owner} is missing keys: {', '.join(missing)}"
        )
    return value


def _strict_dataclass(cls: type[Any], value: Any, *, owner: str) -> Any:
    expected = {item.name for item in fields(cls)}
    checked = _require_exact_keys(value, expected, owner=owner)
    return cls(**dict(checked))


def _require_declared_defaults(
    value: Any,
    *,
    owner: str,
    exclude: tuple[str, ...] = (),
) -> None:
    """Validate a frozen dataclass against its literal field defaults."""

    for item in fields(value):
        if item.name in exclude:
            continue
        if item.default is MISSING:
            raise TypeError(
                f"{owner}.{item.name} has no frozen default to validate"
            )
        actual = getattr(value, item.name)
        if actual != item.default:
            raise ContributionConfigError(
                f"{owner}.{item.name} is pinned to {item.default!r}, "
                f"got {actual!r}"
            )


def _positive_int(name: str, value: Any, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
    ):
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ContributionConfigError(
            f"{name} must be a {qualifier} integer, got {value!r}"
        )
    return value


@dataclass(frozen=True, slots=True)
class ContributionProvenance:
    """Immutable source identities including the published fork revision."""

    model_id: str = MODEL_ID
    model_revision: str = MODEL_REVISION
    tokenizer_path: str = MODEL_ID
    maxtext_base_commit: str = MAXTEXT_BASE_COMMIT
    maxtext_selected_outputs_patch_sha256: str = (
        MAXTEXT_SELECTED_OUTPUTS_PATCH_SHA256
    )
    maxtext_fork_commit: str = MAXTEXT_FORK_COMMIT
    dataset_name: str = C4_REPOSITORY
    dataset_config: str = "en"
    dataset_revision: str = C4_REVISION
    train_split: str = "train"
    eval_split: str = "validation"

    def __post_init__(self) -> None:
        _require_declared_defaults(
            self,
            owner="provenance",
            exclude=("maxtext_fork_commit",),
        )
        fork = self.maxtext_fork_commit
        if not isinstance(fork, str) or _COMMIT_RE.fullmatch(fork) is None:
            raise ContributionConfigError(
                "maxtext_fork_commit must be a lowercase 40-character Git commit"
            )
        if fork == self.maxtext_base_commit:
            raise ContributionConfigError(
                "maxtext_fork_commit cannot equal the unpatched base commit"
            )
        if fork != MAXTEXT_FORK_COMMIT:
            raise ContributionConfigError(
                f"maxtext_fork_commit is pinned to {MAXTEXT_FORK_COMMIT!r}, "
                f"got {fork!r}"
            )

    @property
    def authoritative_capture_ready(self) -> bool:
        return self.maxtext_fork_commit == MAXTEXT_FORK_COMMIT


@dataclass(frozen=True, slots=True)
class ContributionModelContract:
    """Exact Gemma/MaxText path qualified for selected-output capture."""

    maxtext_model_name: str = "gemma4-26b"
    num_decoder_layers: int = 30
    hidden_size: int = 2_816
    num_query_heads: int = 16
    num_kv_heads: int = 8
    global_num_kv_heads: int = 2
    head_dim: int = 256
    global_head_dim: int = 512
    mlp_dim: int = 2_112
    moe_mlp_dim: int = 704
    num_routed_experts: int = 128
    experts_per_token: int = 8
    num_shared_experts: int = 1
    target_layer_one_indexed: int = 15
    sequence_length: int = 1_024
    sliding_window_size: int = 1_024
    dtype: str = "bfloat16"
    weight_dtype: str = "bfloat16"
    use_multimodal: bool = False
    scan_layers: bool = False
    pure_nnx_decoder: bool = True
    share_kv_projections: bool = True
    use_post_attn_norm: bool = True
    use_post_ffw_norm: bool = True
    dropout_rate: float = 0.0
    load_balance_loss_weight: float = 0.0
    norm_topk_prob: bool = True
    use_random_routing: bool = False
    float32_gate_logits: bool = False
    capacity_factor: float = -1.0
    attention: str = "autoselected"
    sparse_matmul: bool = True
    use_ring_of_experts: bool = False
    ragged_buffer_factor: float = -1.0
    tensor_parallelism: int = 1
    hash_routing: bool = False
    serve_quantization: bool = False
    float32_weight_sum: bool = True
    matmul_precision: str = "default"

    def __post_init__(self) -> None:
        _require_declared_defaults(self, owner="model")


@dataclass(frozen=True, slots=True)
class ContributionRuntime:
    """Shared capture/training/report runtime and numerical gates."""

    capture_schema_version: str = CAPTURE_SCHEMA_VERSION
    capture_site: str = CAPTURE_SITE
    shared_capture_site: str = SHARED_CAPTURE_SITE
    max_shard_tokens: int = 16_384
    effective_batch_tokens: int = 16_384
    microbatch_tokens: int = 4_096
    writer_backlog_max_batches: int = 2
    phase1_min_persisted_goodput_fraction: float = 0.60
    phase1_min_auxiliary_throughput_fraction: float = 0.85
    accelerator_mfu_policy: str = (
        "gemma_experiment2_direct_3600s_v2"
    )
    accelerator_min_useful_mfu: float = 0.03
    accelerator_ungated_max_seconds: int = 3_600
    accelerator_qualification_max_seconds: int = 600
    router_order: str = "weight_desc_expert_id_asc"
    pooled_centering: str = "one_raw_expert_output_mean"
    contribution_loss: str = "normalized_router_weighted_sse"
    router_weight_math_dtype: str = "float32"
    topk_requires_strictly_positive: bool = True
    topk_tie_break: str = "global_latent_id_ascending"
    capacity_margin_fraction: float = 0.10
    capacity_overflow_policy: str = "hard_error"
    phase2_min_g2_percent: float = 10.0
    phase3_min_g2_percent: float = 10.0
    phase3_max_c2_percent: float = 50.0
    bootstrap_replicates: int = 20_000
    paired_confidence_level: float = 0.95

    def __post_init__(self) -> None:
        _require_declared_defaults(self, owner="runtime")
        if self.effective_batch_tokens % self.microbatch_tokens:
            raise ContributionConfigError(
                "effective_batch_tokens must be divisible by microbatch_tokens"
            )


@dataclass(frozen=True, slots=True)
class ContributionOptimizer:
    """Optimizer and initialization shared by every routed condition and Q."""

    parameter_dtype: str = "bfloat16"
    matmul_dtype: str = "bfloat16"
    moment_dtype: str = "float32"
    reduction_dtype: str = "float32"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    weight_decay: float = 0.0
    global_gradient_clip: float = 1.0
    peak_learning_rate: float = 3e-4
    minimum_learning_rate: float = 3e-5
    schedule: str = "linear_warmup_cosine_decay"
    decoder_init: str = "unit_norm_columns"
    encoder_init: str = "decoder_transpose"
    bias_init: str = "zeros"
    tangent_decoder_gradient_projection: bool = True
    decoder_column_renormalization: bool = True

    def __post_init__(self) -> None:
        _require_declared_defaults(self, owner="optimizer")


@dataclass(frozen=True, slots=True)
class ContributionLayout:
    """One objective-matched dictionary layout."""

    condition: str
    target: str
    bank_identity: str
    bank_count: int
    bank_width: int
    components_per_token: int
    topk_per_component: int

    def __post_init__(self) -> None:
        if self.condition not in (*CONDITIONS, "Q"):
            raise ContributionConfigError(
                f"unsupported layout condition {self.condition!r}"
            )
        for name in (
            "bank_count",
            "bank_width",
            "components_per_token",
            "topk_per_component",
        ):
            _positive_int(f"layout.{self.condition}.{name}", getattr(self, name))
        if self.topk_per_component > self.bank_width:
            raise ContributionConfigError(
                f"layout {self.condition} TopK exceeds its bank width"
            )
        if self.condition == "Q":
            expected = (
                "normalized_shared_branch",
                "global_shared",
                1,
                1,
            )
        else:
            identities = {
                "A": "true_expert_id",
                "R": "per_sequence_randomized_expert_id",
                "B": "router_rank",
                "F": "router_rank",
            }
            expected = (
                "raw_selected_expert_output",
                identities[self.condition],
                8 if self.condition in ("B", "F") else 128,
                8,
            )
        actual = (
            self.target,
            self.bank_identity,
            self.bank_count,
            self.components_per_token,
        )
        if actual != expected:
            raise ContributionConfigError(
                f"layout {self.condition} identity/shape contract must be "
                f"{expected!r}, got {actual!r}"
            )

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
class ContributionPhase:
    """One monotonic capture/training stage and its escalation contract."""

    ordinal: int
    name: str
    extends_phase: str | None
    train_tokens: int
    eval_tokens: int
    seeds: tuple[int, ...]
    layout_set: str
    planned_updates: int | None
    warmup_updates: int | None
    tail_mask_tokens: int | None
    checkpoint_tokens: tuple[int, ...]
    escalation_gates: tuple[str, ...]
    blind_relative_quality_until_health: bool
    require_f_compile_smoke: bool
    shared_q_resume_phase: str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "seeds", tuple(self.seeds))
        object.__setattr__(
            self,
            "checkpoint_tokens",
            tuple(self.checkpoint_tokens),
        )
        object.__setattr__(
            self,
            "escalation_gates",
            tuple(self.escalation_gates),
        )
        _positive_int("phase.ordinal", self.ordinal)
        _positive_int("phase.train_tokens", self.train_tokens)
        _positive_int("phase.eval_tokens", self.eval_tokens)
        if not isinstance(self.name, str) or not self.name:
            raise ContributionConfigError("phase name must be nonempty")
        if self.extends_phase is not None and (
            not isinstance(self.extends_phase, str) or not self.extends_phase
        ):
            raise ContributionConfigError(
                "phase extends_phase must be null or a nonempty name"
            )
        if self.shared_q_resume_phase is not None and (
            not isinstance(self.shared_q_resume_phase, str)
            or not self.shared_q_resume_phase
        ):
            raise ContributionConfigError(
                "phase shared_q_resume_phase must be null or a nonempty name"
            )
        if self.layout_set not in LAYOUT_SETS:
            raise ContributionConfigError(
                f"phase layout_set must be one of {LAYOUT_SETS}"
            )
        if not isinstance(self.blind_relative_quality_until_health, bool):
            raise ContributionConfigError(
                "phase blind_relative_quality_until_health must be boolean"
            )
        if not isinstance(self.require_f_compile_smoke, bool):
            raise ContributionConfigError(
                "phase require_f_compile_smoke must be boolean"
            )
        if len(set(self.seeds)) != len(self.seeds) or any(
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or seed < 0
            for seed in self.seeds
        ):
            raise ContributionConfigError(
                "phase seeds must be unique nonnegative integers"
            )
        if len(set(self.checkpoint_tokens)) != len(self.checkpoint_tokens):
            raise ContributionConfigError("phase checkpoint tokens must be unique")
        if tuple(sorted(self.checkpoint_tokens)) != self.checkpoint_tokens:
            raise ContributionConfigError(
                "phase checkpoint tokens must be strictly increasing"
            )
        if any(
            isinstance(token, bool)
            or not isinstance(token, int)
            or token <= 0
            or token > self.train_tokens
            for token in self.checkpoint_tokens
        ):
            raise ContributionConfigError(
                "phase checkpoint tokens must lie in (0, train_tokens]"
            )
        if not self.escalation_gates or len(set(self.escalation_gates)) != len(
            self.escalation_gates
        ):
            raise ContributionConfigError(
                "phase escalation gates must be nonempty and unique"
            )
        if any(
            not isinstance(gate, str) or not gate
            for gate in self.escalation_gates
        ):
            raise ContributionConfigError(
                "phase escalation gate IDs must be nonempty strings"
            )


def _pilot_layouts() -> tuple[ContributionLayout, ...]:
    return (
        ContributionLayout(
            "A",
            "raw_selected_expert_output",
            "true_expert_id",
            128,
            256,
            8,
            2,
        ),
        ContributionLayout(
            "R",
            "raw_selected_expert_output",
            "per_sequence_randomized_expert_id",
            128,
            256,
            8,
            2,
        ),
        ContributionLayout(
            "B",
            "raw_selected_expert_output",
            "router_rank",
            8,
            256,
            8,
            2,
        ),
        ContributionLayout(
            "F",
            "raw_selected_expert_output",
            "router_rank",
            8,
            4_096,
            8,
            2,
        ),
    )


def _full_layouts() -> tuple[ContributionLayout, ...]:
    return (
        ContributionLayout(
            "A",
            "raw_selected_expert_output",
            "true_expert_id",
            128,
            1_024,
            8,
            8,
        ),
        ContributionLayout(
            "R",
            "raw_selected_expert_output",
            "per_sequence_randomized_expert_id",
            128,
            1_024,
            8,
            8,
        ),
        ContributionLayout(
            "B",
            "raw_selected_expert_output",
            "router_rank",
            8,
            1_024,
            8,
            8,
        ),
        ContributionLayout(
            "F",
            "raw_selected_expert_output",
            "router_rank",
            8,
            16_384,
            8,
            8,
        ),
    )


def _shared_q_layout() -> ContributionLayout:
    return ContributionLayout(
        "Q",
        "normalized_shared_branch",
        "global_shared",
        1,
        8_192,
        1,
        64,
    )


def _phase_plans() -> tuple[ContributionPhase, ...]:
    return (
        ContributionPhase(
            ordinal=1,
            name="phase1_smoke",
            extends_phase=None,
            train_tokens=100_000,
            eval_tokens=25_000,
            seeds=(),
            layout_set="none",
            planned_updates=None,
            warmup_updates=None,
            tail_mask_tokens=None,
            checkpoint_tokens=(),
            escalation_gates=PHASE1_GATES,
            blind_relative_quality_until_health=False,
            require_f_compile_smoke=False,
            shared_q_resume_phase=None,
        ),
        ContributionPhase(
            ordinal=2,
            name="phase2_mechanism_pilot",
            extends_phase="phase1_smoke",
            train_tokens=2_000_000,
            eval_tokens=250_000,
            seeds=(0,),
            layout_set="pilot",
            planned_updates=123,
            warmup_updates=12,
            tail_mask_tokens=15_232,
            checkpoint_tokens=(2_000_000,),
            escalation_gates=PHASE2_GATES,
            blind_relative_quality_until_health=True,
            require_f_compile_smoke=True,
            shared_q_resume_phase=None,
        ),
        ContributionPhase(
            ordinal=3,
            name="phase3_full_width_pilot",
            extends_phase="phase2_mechanism_pilot",
            train_tokens=10_000_000,
            eval_tokens=1_000_000,
            seeds=(0,),
            layout_set="full",
            planned_updates=611,
            warmup_updates=61,
            tail_mask_tokens=10_624,
            checkpoint_tokens=(2_000_000, 5_000_000, 10_000_000),
            escalation_gates=PHASE3_GATES,
            blind_relative_quality_until_health=True,
            require_f_compile_smoke=False,
            shared_q_resume_phase="phase2_mechanism_pilot",
        ),
        ContributionPhase(
            ordinal=4,
            name="phase4_confirmatory",
            extends_phase="phase3_full_width_pilot",
            train_tokens=50_000_000,
            eval_tokens=2_000_000,
            seeds=(1, 2, 3),
            layout_set="full",
            planned_updates=3_052,
            warmup_updates=250,
            tail_mask_tokens=3_968,
            checkpoint_tokens=(
                2_000_000,
                5_000_000,
                10_000_000,
                20_000_000,
                30_000_000,
                40_000_000,
                50_000_000,
            ),
            escalation_gates=PHASE4_GATES,
            blind_relative_quality_until_health=True,
            require_f_compile_smoke=False,
            shared_q_resume_phase=None,
        ),
    )


@dataclass(frozen=True, slots=True)
class Experiment2Config:
    """Complete immutable Experiment-2 plan.

    Every field is a protocol pin rather than a launch-time tuning surface.
    """

    schema_version: str = SCHEMA_VERSION
    protocol_id: str = PROTOCOL_ID
    provenance: ContributionProvenance = field(
        default_factory=ContributionProvenance
    )
    model: ContributionModelContract = field(
        default_factory=ContributionModelContract
    )
    runtime: ContributionRuntime = field(default_factory=ContributionRuntime)
    optimizer: ContributionOptimizer = field(
        default_factory=ContributionOptimizer
    )
    pilot_layouts: tuple[ContributionLayout, ...] = field(
        default_factory=_pilot_layouts
    )
    full_layouts: tuple[ContributionLayout, ...] = field(
        default_factory=_full_layouts
    )
    shared_q: ContributionLayout = field(default_factory=_shared_q_layout)
    phases: tuple[ContributionPhase, ...] = field(default_factory=_phase_plans)

    def __post_init__(self) -> None:
        object.__setattr__(self, "pilot_layouts", tuple(self.pilot_layouts))
        object.__setattr__(self, "full_layouts", tuple(self.full_layouts))
        object.__setattr__(self, "phases", tuple(self.phases))
        self.validate()

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ContributionConfigError(
                f"schema_version is pinned to {SCHEMA_VERSION!r}"
            )
        if self.protocol_id != PROTOCOL_ID:
            raise ContributionConfigError(
                f"protocol_id is pinned to {PROTOCOL_ID!r}"
            )
        if self.pilot_layouts != _pilot_layouts():
            raise ContributionConfigError(
                "pilot layouts must be the exact A0/R0/B0/F0 matrix"
            )
        if self.full_layouts != _full_layouts():
            raise ContributionConfigError(
                "full layouts must be the exact A/R/B/F matrix"
            )
        if self.shared_q != _shared_q_layout():
            raise ContributionConfigError(
                "shared Q is pinned to 8,192 global features and TopK-64"
            )
        self._validate_monotonic_phase_plan()
        if self.phases != _phase_plans():
            raise ContributionConfigError(
                "phase plan differs from the frozen Phase-1--4 contract"
            )

    def _validate_monotonic_phase_plan(self) -> None:
        previous: ContributionPhase | None = None
        for index, phase in enumerate(self.phases, start=1):
            if phase.ordinal != index or phase.name != PHASE_NAMES[index - 1]:
                raise ContributionConfigError(
                    "phases must have contiguous ordinals and canonical names"
                )
            expected_parent = None if previous is None else previous.name
            if phase.extends_phase != expected_parent:
                raise ContributionConfigError(
                    f"{phase.name} must extend {expected_parent!r}"
                )
            if phase.shared_q_resume_phase is not None:
                prior = {
                    candidate.name: candidate
                    for candidate in self.phases[: index - 1]
                }
                if phase.shared_q_resume_phase not in prior:
                    raise ContributionConfigError(
                        f"{phase.name} shared Q can resume only from a "
                        "predecessor phase"
                    )
                if prior[phase.shared_q_resume_phase].seeds != phase.seeds:
                    raise ContributionConfigError(
                        f"{phase.name} cannot resume shared Q across seed sets"
                    )
            if previous is not None and (
                phase.train_tokens <= previous.train_tokens
                or phase.eval_tokens <= previous.eval_tokens
            ):
                raise ContributionConfigError(
                    "phase token manifests must extend monotonically"
                )
            if phase.layout_set == "none":
                if any(
                    value is not None
                    for value in (
                        phase.planned_updates,
                        phase.warmup_updates,
                        phase.tail_mask_tokens,
                    )
                ):
                    raise ContributionConfigError(
                        "capture-only Phase 1 cannot define an SAE schedule"
                    )
            else:
                _positive_int(
                    f"{phase.name}.planned_updates",
                    phase.planned_updates,
                )
                _positive_int(
                    f"{phase.name}.warmup_updates",
                    phase.warmup_updates,
                )
                _positive_int(
                    f"{phase.name}.tail_mask_tokens",
                    phase.tail_mask_tokens,
                    allow_zero=True,
                )
                expected_updates = math.ceil(
                    phase.train_tokens / self.runtime.effective_batch_tokens
                )
                expected_tail = (
                    expected_updates * self.runtime.effective_batch_tokens
                    - phase.train_tokens
                )
                if phase.planned_updates != expected_updates:
                    raise ContributionConfigError(
                        f"{phase.name} planned_updates must be {expected_updates}"
                    )
                if phase.tail_mask_tokens != expected_tail:
                    raise ContributionConfigError(
                        f"{phase.name} tail_mask_tokens must be {expected_tail}"
                    )
                if not 0 < phase.warmup_updates < phase.planned_updates:
                    raise ContributionConfigError(
                        f"{phase.name} warmup must finish before its final update"
                    )
            previous = phase

    @property
    def config_hash(self) -> str:
        payload = json.dumps(
            self.as_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        return hashlib.sha256(payload).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def phase(self, phase: int | str) -> ContributionPhase:
        if isinstance(phase, bool):
            raise ContributionConfigError("phase selector cannot be boolean")
        if isinstance(phase, int):
            if 1 <= phase <= len(self.phases):
                return self.phases[phase - 1]
            raise ContributionConfigError(f"unknown phase ordinal {phase}")
        if isinstance(phase, str):
            for plan in self.phases:
                if plan.name == phase:
                    return plan
            raise ContributionConfigError(f"unknown phase name {phase!r}")
        raise ContributionConfigError("phase selector must be an ordinal or name")

    def layouts_for_phase(
        self,
        phase: int | str,
    ) -> tuple[ContributionLayout, ...]:
        plan = self.phase(phase)
        if plan.layout_set == "none":
            return ()
        routed = (
            self.pilot_layouts
            if plan.layout_set == "pilot"
            else self.full_layouts
        )
        return (*routed, self.shared_q)

    def with_maxtext_fork_commit(self, commit: str) -> Experiment2Config:
        """Accept only the already-pinned published fork identity."""

        if commit != self.provenance.maxtext_fork_commit:
            raise ContributionConfigError(
                "maxtext_fork_commit cannot drift from the published "
                "Experiment-2 fork"
            )
        return self

    def validate_authoritative_tpu_capture(
        self,
        phase: int | str,
        *,
        observed_maxtext_fork_commit: str | None = None,
    ) -> ContributionPhase:
        """Bind the configured fork to the checkout observed by the launcher."""

        plan = self.phase(phase)
        if (
            observed_maxtext_fork_commit
            != self.provenance.maxtext_fork_commit
        ):
            raise ContributionConfigError(
                "authoritative TPU capture is blocked: the launcher's observed "
                "MaxText fork commit must exactly match the pinned "
                "maxtext_fork_commit"
            )
        return plan

    def validate_escalation(
        self,
        target_phase: int | str,
        gate_results: Mapping[str, Mapping[str, bool]],
    ) -> ContributionPhase:
        """Require every preceding phase's exact gate set to be true."""

        target = self.phase(target_phase)
        if not isinstance(gate_results, Mapping):
            raise ContributionConfigError("gate_results must be a mapping")
        required_predecessors = self.phases[: target.ordinal - 1]
        allowed_names = {phase.name for phase in required_predecessors}
        unknown_phases = sorted(set(gate_results) - allowed_names)
        if unknown_phases:
            raise ContributionConfigError(
                "gate_results contains non-predecessor phases: "
                + ", ".join(unknown_phases)
            )
        for phase in required_predecessors:
            raw = gate_results.get(phase.name)
            if not isinstance(raw, Mapping):
                raise ContributionConfigError(
                    f"missing gate results for {phase.name}"
                )
            expected = set(phase.escalation_gates)
            actual = set(raw)
            unknown = sorted(actual - expected)
            missing = sorted(expected - actual)
            if unknown:
                raise ContributionConfigError(
                    f"{phase.name} has unknown gate results: {', '.join(unknown)}"
                )
            if missing:
                raise ContributionConfigError(
                    f"{phase.name} is missing gate results: {', '.join(missing)}"
                )
            failed = sorted(
                gate
                for gate in phase.escalation_gates
                if raw[gate] is not True
            )
            if failed:
                raise ContributionConfigError(
                    f"{phase.name} has failed gates: {', '.join(failed)}"
                )
        return target

    @classmethod
    def from_mapping(cls, value: Any) -> Experiment2Config:
        expected = {item.name for item in fields(cls)}
        raw = _require_exact_keys(value, expected, owner="experiment2")

        provenance = _strict_dataclass(
            ContributionProvenance,
            raw["provenance"],
            owner="experiment2.provenance",
        )
        model = _strict_dataclass(
            ContributionModelContract,
            raw["model"],
            owner="experiment2.model",
        )
        runtime = _strict_dataclass(
            ContributionRuntime,
            raw["runtime"],
            owner="experiment2.runtime",
        )
        optimizer = _strict_dataclass(
            ContributionOptimizer,
            raw["optimizer"],
            owner="experiment2.optimizer",
        )

        def layouts(name: str) -> tuple[ContributionLayout, ...]:
            entries = raw[name]
            if not isinstance(entries, Sequence) or isinstance(
                entries, (str, bytes)
            ):
                raise ContributionConfigError(
                    f"experiment2.{name} must be a sequence"
                )
            return tuple(
                _strict_dataclass(
                    ContributionLayout,
                    entry,
                    owner=f"experiment2.{name}[{index}]",
                )
                for index, entry in enumerate(entries)
            )

        shared_q = _strict_dataclass(
            ContributionLayout,
            raw["shared_q"],
            owner="experiment2.shared_q",
        )
        phase_entries = raw["phases"]
        if not isinstance(phase_entries, Sequence) or isinstance(
            phase_entries, (str, bytes)
        ):
            raise ContributionConfigError("experiment2.phases must be a sequence")
        phases = tuple(
            _strict_dataclass(
                ContributionPhase,
                entry,
                owner=f"experiment2.phases[{index}]",
            )
            for index, entry in enumerate(phase_entries)
        )
        return cls(
            schema_version=raw["schema_version"],
            protocol_id=raw["protocol_id"],
            provenance=provenance,
            model=model,
            runtime=runtime,
            optimizer=optimizer,
            pilot_layouts=layouts("pilot_layouts"),
            full_layouts=layouts("full_layouts"),
            shared_q=shared_q,
            phases=phases,
        )


def load_contribution_config(path: str | Path) -> Any:
    """Load a strict contribution-SAE run contract.

    The completed Experiment-2 schema remains frozen.  The only successor
    accepted here is the explicitly SAE-typed pilot learning-horizon schema;
    its checked-in template is rejected until sealed artifact hashes have been
    finalized.
    """

    source = Path(path)
    text = source.read_text(encoding="utf-8")
    if source.suffix.lower() == ".json":
        raw = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "PyYAML is required to load Experiment-2 YAML"
            ) from exc
        raw = yaml.safe_load(text)
    if (
        isinstance(raw, Mapping)
        and raw.get("schema_version")
        == "gemma4-pilot-learning-horizon-run-config-v1"
    ):
        from moe_sae.learning_horizon_config import (
            learning_horizon_config_from_mapping,
        )

        return learning_horizon_config_from_mapping(
            raw,
            expected_scientific_model="sae",
        )
    return Experiment2Config.from_mapping(raw)


def save_contribution_config(
    config: Experiment2Config,
    path: str | Path,
) -> None:
    """Atomically persist one validated, canonically hashable run contract."""

    config.validate()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    if destination.suffix.lower() == ".json":
        payload = json.dumps(config.as_dict(), indent=2, sort_keys=True) + "\n"
    else:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "PyYAML is required to save Experiment-2 YAML"
            ) from exc
        payload = yaml.safe_dump(config.as_dict(), sort_keys=False)
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(destination)


__all__ = [
    "C4_REPOSITORY",
    "C4_REVISION",
    "CAPTURE_SCHEMA_VERSION",
    "CONDITIONS",
    "MAXTEXT_BASE_COMMIT",
    "MAXTEXT_FORK_COMMIT",
    "MAXTEXT_SELECTED_OUTPUTS_PATCH_SHA256",
    "MODEL_ID",
    "MODEL_REVISION",
    "PHASE_NAMES",
    "PROTOCOL_ID",
    "SCHEMA_VERSION",
    "ContributionConfigError",
    "ContributionLayout",
    "ContributionModelContract",
    "ContributionOptimizer",
    "ContributionPhase",
    "ContributionProvenance",
    "ContributionRuntime",
    "Experiment2Config",
    "load_contribution_config",
    "save_contribution_config",
]
