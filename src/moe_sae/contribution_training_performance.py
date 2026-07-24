"""Useful-FLOP admission gate for Experiment-2 SAE training.

Harvester and SAE graphs have different arithmetic.  This module supplies an
honest SAE ledger, then adapts a deterministic training-step qualification to
the repository-wide exact-identity receipt format.  The shared receipt
validator still enforces the same hard floor: no graph below 10% useful MFU
may authorize accelerator work beyond 600 seconds.

Only algorithmic reconstruction FLOPs are counted.  Capacity padding,
rematerialization, TopK, collectives, normalization, Adam, and input stalls
remain in wall time but not the numerator.  The resulting MFU is conservative
and comparable across A/R/B/F/Q execution choices.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Literal

from moe_sae.contribution_artifacts import canonical_sha256
from moe_sae.contribution_performance import (
    MAX_QUALIFICATION_SECONDS,
    ExactPerformanceIdentity,
    QualificationEvidence,
    build_qualification_receipt,
)
from moe_sae.contribution_sae import ContributionSAEConfig


SAE_FLOPS_LEDGER_SCHEMA_VERSION = "gemma4-contribution-sae-useful-flops-v1"
SAE_QUALIFICATION_OBSERVATION_SCHEMA_VERSION = (
    "gemma4-contribution-sae-qualification-observation-v1"
)


class ContributionTrainingPerformanceError(ValueError):
    """SAE performance evidence is incomplete or internally inconsistent."""


@dataclass(frozen=True, slots=True)
class SAEArithmetic:
    """Static candidate geometry for routed A/R/B/F or shared Q."""

    workload: Literal["routed", "shared_q"]
    condition: str
    hidden_dim: int
    candidates_per_token: int
    selected_features_per_token: int

    def __post_init__(self) -> None:
        if self.workload not in {"routed", "shared_q"}:
            raise ContributionTrainingPerformanceError("unknown SAE workload")
        if not isinstance(self.condition, str) or not self.condition:
            raise ContributionTrainingPerformanceError("condition is empty")
        for name in (
            "hidden_dim",
            "candidates_per_token",
            "selected_features_per_token",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ContributionTrainingPerformanceError(
                    f"{name} must be positive"
                )
        if self.selected_features_per_token > self.candidates_per_token:
            raise ContributionTrainingPerformanceError(
                "selected features exceed candidates"
            )

    @classmethod
    def from_contribution_layout(
        cls,
        layout: ContributionSAEConfig,
    ) -> "SAEArithmetic":
        return cls(
            workload="routed",
            condition=layout.condition,
            hidden_dim=layout.hidden_dim,
            candidates_per_token=layout.candidates_per_token,
            selected_features_per_token=layout.total_l0,
        )

    @classmethod
    def shared_q(
        cls,
        *,
        hidden_dim: int = 2_816,
        total_features: int = 8_192,
        k: int = 64,
    ) -> "SAEArithmetic":
        return cls(
            workload="shared_q",
            condition="Q",
            hidden_dim=hidden_dim,
            candidates_per_token=total_features,
            selected_features_per_token=k,
        )


def sae_training_useful_flops(
    arithmetic: SAEArithmetic,
    *,
    scientific_tokens: int,
) -> int:
    """Forward+backward encoder and sparse-decoder matmul FLOPs.

    A trainable matrix used once in a forward pass costs six FLOPs per weight
    interaction over forward, input-gradient, and parameter-gradient matmuls.
    The dense encoder touches every visible candidate; the decoder touches only
    selected columns.
    """

    if (
        isinstance(scientific_tokens, bool)
        or not isinstance(scientific_tokens, int)
        or scientific_tokens <= 0
    ):
        raise ContributionTrainingPerformanceError(
            "scientific_tokens must be positive"
        )
    encoder = (
        6
        * scientific_tokens
        * arithmetic.hidden_dim
        * arithmetic.candidates_per_token
    )
    decoder = (
        6
        * scientific_tokens
        * arithmetic.hidden_dim
        * arithmetic.selected_features_per_token
    )
    return encoder + decoder


def sae_useful_flops_ledger(
    arithmetic: SAEArithmetic,
    *,
    scientific_tokens: int,
    measured_steps: int,
) -> dict[str, Any]:
    """Seal the conservative arithmetic used for the SAE MFU gate."""

    if (
        isinstance(measured_steps, bool)
        or not isinstance(measured_steps, int)
        or measured_steps <= 0
        or scientific_tokens % measured_steps
    ):
        raise ContributionTrainingPerformanceError(
            "measured tokens must divide evenly across positive measured steps"
        )
    per_step_tokens = scientific_tokens // measured_steps
    per_step = sae_training_useful_flops(
        arithmetic,
        scientific_tokens=per_step_tokens,
    )
    return {
        "schema_version": SAE_FLOPS_LEDGER_SCHEMA_VERSION,
        "arithmetic": asdict(arithmetic),
        "scientific_tokens": scientific_tokens,
        "measured_steps": measured_steps,
        "scientific_tokens_per_step": per_step_tokens,
        "encoder_flops_per_token": (
            6 * arithmetic.hidden_dim * arithmetic.candidates_per_token
        ),
        "sparse_decoder_flops_per_token": (
            6
            * arithmetic.hidden_dim
            * arithmetic.selected_features_per_token
        ),
        "useful_flops_per_step": per_step,
        "useful_flops": per_step * measured_steps,
        "excluded_from_useful_flops": [
            "capacity_padding",
            "rematerialized_forward_work",
            "topk",
            "collectives",
            "loss_and_normalization",
            "adam",
            "decoder_tangent_projection",
            "decoder_column_renormalization",
            "input_wait",
        ],
    }


@dataclass(frozen=True, slots=True)
class SAEQualificationObservation:
    """Synchronized, deterministic-replay evidence for one SAE graph."""

    identity: ExactPerformanceIdentity
    qualification_wall_seconds: float
    measured_device_seconds: float
    scientific_tokens: int
    warmup_steps: int
    measured_steps: int
    compile_count: int
    recompilations_after_warmup: int
    oom_events: int
    nonfinite_events: int
    capacity_overflow_events: int
    exact_l0: bool
    state_progress_exact: bool
    parameters_resident_before_timing: bool
    inputs_resident_before_timing: bool
    synchronized_device_timing: bool
    transition_output_sha256: str
    replay_output_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "qualification_wall_seconds",
            "measured_device_seconds",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ContributionTrainingPerformanceError(
                    f"{name} must be finite and positive"
                )
        if self.measured_device_seconds > self.qualification_wall_seconds:
            raise ContributionTrainingPerformanceError(
                "device time exceeds qualification wall time"
            )
        for name in (
            "scientific_tokens",
            "warmup_steps",
            "measured_steps",
            "compile_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ContributionTrainingPerformanceError(
                    f"{name} must be positive"
                )
        for name in (
            "recompilations_after_warmup",
            "oom_events",
            "nonfinite_events",
            "capacity_overflow_events",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ContributionTrainingPerformanceError(
                    f"{name} must be nonnegative"
                )
        for name in (
            "exact_l0",
            "state_progress_exact",
            "parameters_resident_before_timing",
            "inputs_resident_before_timing",
            "synchronized_device_timing",
        ):
            if type(getattr(self, name)) is not bool:
                raise ContributionTrainingPerformanceError(
                    f"{name} must be boolean"
                )
        for name in ("transition_output_sha256", "replay_output_sha256"):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ContributionTrainingPerformanceError(
                    f"{name} is not a SHA-256"
                )


def build_sae_qualification_receipt(
    observation: SAEQualificationObservation,
    arithmetic: SAEArithmetic,
    *,
    peak_flops_per_second: int,
    minimum_useful_mfu: float = 0.10,
) -> tuple[QualificationEvidence, dict[str, Any]]:
    """Validate a deterministic SAE probe and seal the shared gate receipt."""

    if type(observation) is not SAEQualificationObservation:
        raise ContributionTrainingPerformanceError(
            "SAE qualification observation has the wrong type"
        )
    if not observation.exact_l0:
        raise ContributionTrainingPerformanceError(
            "SAE qualifier did not preserve exact L0"
        )
    if not observation.state_progress_exact:
        raise ContributionTrainingPerformanceError(
            "SAE qualifier state did not advance by the exact token/step plan"
        )
    if observation.transition_output_sha256 != observation.replay_output_sha256:
        raise ContributionTrainingPerformanceError(
            "SAE transition is not deterministic under exact replay"
        )
    ledger = sae_useful_flops_ledger(
        arithmetic,
        scientific_tokens=observation.scientific_tokens,
        measured_steps=observation.measured_steps,
    )
    source = {
        "schema_version": SAE_QUALIFICATION_OBSERVATION_SCHEMA_VERSION,
        "observation": {
            **asdict(observation),
            "identity": observation.identity.to_dict(),
        },
        "arithmetic": asdict(arithmetic),
    }
    # Training has no output writer in the optimizer timing window.  The
    # updated, synchronized state is its durable logical output; map that
    # one-for-one onto the generic receipt's output/persistence fields.
    evidence = QualificationEvidence(
        identity=observation.identity,
        source_evidence_sha256=canonical_sha256(source),
        useful_flops_ledger_sha256=canonical_sha256(ledger),
        qualification_wall_seconds=observation.qualification_wall_seconds,
        measured_device_seconds=observation.measured_device_seconds,
        persisted_window_seconds=observation.measured_device_seconds,
        useful_flops=int(ledger["useful_flops"]),
        peak_flops_per_second=peak_flops_per_second,
        warmup_steps=observation.warmup_steps,
        synchronized_measurement_steps=observation.measured_steps,
        measured_graph_compile_count=observation.compile_count,
        auxiliary_validation_compile_count=0,
        total_compile_count=observation.compile_count,
        recompilations_after_warmup=(
            observation.recompilations_after_warmup
        ),
        oom_events=observation.oom_events,
        nonfinite_events=observation.nonfinite_events,
        capacity_overflow_events=observation.capacity_overflow_events,
        dropped_routed_tokens=0,
        output_tokens=observation.scientific_tokens,
        persisted_tokens=observation.scientific_tokens,
        max_writer_backlog_batches=0,
        writer_backlog_limit_batches=1,
        writer_failures=0,
        writer_queue_drained=True,
        static_shapes=True,
        parameters_resident_before_timing=(
            observation.parameters_resident_before_timing
        ),
        inputs_resident_before_timing=(
            observation.inputs_resident_before_timing
        ),
        synchronized_device_timing=(
            observation.synchronized_device_timing
        ),
        instrumentation_disabled_output_sha256=(
            observation.transition_output_sha256
        ),
        instrumentation_enabled_output_sha256=(
            observation.replay_output_sha256
        ),
        writer_disabled_output_tokens=observation.scientific_tokens,
        writer_disabled_device_seconds=observation.measured_device_seconds,
    )
    receipt = build_qualification_receipt(
        evidence,
        minimum_useful_mfu=minimum_useful_mfu,
        qualification_max_seconds=MAX_QUALIFICATION_SECONDS,
    )
    return evidence, {
        "source_evidence": source,
        "useful_flops_ledger": ledger,
        "qualification_receipt": receipt,
    }


__all__ = [
    "SAE_FLOPS_LEDGER_SCHEMA_VERSION",
    "SAE_QUALIFICATION_OBSERVATION_SCHEMA_VERSION",
    "ContributionTrainingPerformanceError",
    "SAEArithmetic",
    "SAEQualificationObservation",
    "build_sae_qualification_receipt",
    "sae_training_useful_flops",
    "sae_useful_flops_ledger",
]
