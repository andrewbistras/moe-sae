"""Fail-closed accelerator qualification for Experiment 2.

The benchmark runner supplies hashes of the exact executable graph, hardware
inventory, and complete runtime configuration it observed.  This module
derives the performance gates, seals a self-hashed receipt, and admits only:

* a qualification run bounded to at most ten minutes;
* any other deliberately bounded run of at most ten minutes; or
* a longer run whose exact graph/hardware/configuration has a passing receipt.

This is an admission seam, not a process supervisor.  The launcher must still
enforce the authorized wall-clock deadline.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from typing import Any, Literal

from moe_sae.contribution_artifacts import (
    ContributionSchemaError,
    canonical_sha256,
)

QUALIFICATION_RECEIPT_SCHEMA_VERSION = (
    "gemma4-contribution-performance-qualification-v2"
)
PERFORMANCE_IDENTITY_SCHEMA_VERSION = "gemma4-contribution-performance-identity-v1"

# Gemma Experiment 2 only.  These admission floors are intentionally lower
# than the separate DeepSeek performance target and are bound into every
# exact-identity receipt.  They were frozen after the real B128 qualification
# measured 0.0365598 useful MFU, 0.6388945 persisted/device goodput, and
# 0.9008767 auxiliary/baseline throughput.
MINIMUM_USEFUL_PREFIX_MFU = 0.03
MAX_UNGATED_ACCELERATOR_SECONDS = 600
MAX_QUALIFICATION_SECONDS = 600
MINIMUM_PERSISTED_GOODPUT_FRACTION = 0.60
MINIMUM_AUXILIARY_THROUGHPUT_FRACTION = 0.85

PERSISTED_GOODPUT_ADMISSION_MESSAGE = (
    "persisted goodput is below the Gemma Experiment-2 60% "
    "device-output admission floor"
)
AUXILIARY_THROUGHPUT_ADMISSION_MESSAGE = (
    "auxiliary capture throughput is below the Gemma Experiment-2 85% "
    "writer-disabled admission floor"
)
USEFUL_MFU_ADMISSION_MESSAGE = (
    "useful-prefix MFU is below the Gemma Experiment-2 3% admission floor"
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PURPOSES = ("qualification", "workload")


class ContributionPerformanceError(ValueError):
    """Performance evidence or accelerator admission failed closed."""


def _sha256(name: str, value: Any) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ContributionPerformanceError(f"{name} must be a lowercase SHA-256")
    return value


def _strict_bool(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise ContributionPerformanceError(f"{name} must be boolean")
    return value


def _nonnegative_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContributionPerformanceError(f"{name} must be a nonnegative integer")
    return value


def _positive_int(name: str, value: Any) -> int:
    selected = _nonnegative_int(name, value)
    if selected == 0:
        raise ContributionPerformanceError(f"{name} must be positive")
    return selected


def _finite_positive(name: str, value: Any) -> float:
    if isinstance(value, bool):
        raise ContributionPerformanceError(f"{name} must be numeric")
    try:
        selected = float(value)
    except (TypeError, ValueError) as error:
        raise ContributionPerformanceError(f"{name} must be numeric") from error
    if not math.isfinite(selected) or selected <= 0.0:
        raise ContributionPerformanceError(
            f"{name} must be finite and strictly positive"
        )
    return selected


def _exact_mapping(
    name: str,
    value: Any,
    expected: set[str],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContributionPerformanceError(f"{name} must be a mapping")
    if set(value) != expected:
        missing = sorted(expected.difference(value))
        extra = sorted(set(value).difference(expected))
        raise ContributionPerformanceError(
            f"{name} fields are not exact; missing={missing}, extra={extra}"
        )
    return value


def _same_float(left: float, right: float) -> bool:
    scale = max(1.0, abs(left), abs(right))
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12 * scale)


def _canonical_sha256(value: Any, *, name: str) -> str:
    try:
        return canonical_sha256(value)
    except ContributionSchemaError as error:
        raise ContributionPerformanceError(
            f"{name} is not finite canonical JSON"
        ) from error


def _validate_policy(
    *,
    minimum_useful_mfu: Any,
    ungated_max_seconds: Any,
    qualification_max_seconds: Any,
) -> tuple[float, int, int]:
    minimum = _finite_positive("minimum_useful_mfu", minimum_useful_mfu)
    ungated = _positive_int("ungated_max_seconds", ungated_max_seconds)
    qualification = _positive_int(
        "qualification_max_seconds",
        qualification_max_seconds,
    )
    if minimum < MINIMUM_USEFUL_PREFIX_MFU:
        raise ContributionPerformanceError(
            "minimum_useful_mfu cannot weaken the Gemma Experiment-2 "
            "3% admission floor"
        )
    if ungated > MAX_UNGATED_ACCELERATOR_SECONDS:
        raise ContributionPerformanceError(
            "ungated accelerator duration cannot exceed 600 seconds"
        )
    if qualification > MAX_QUALIFICATION_SECONDS:
        raise ContributionPerformanceError(
            "qualification duration cannot exceed 600 seconds"
        )
    return minimum, ungated, qualification


@dataclass(frozen=True, slots=True)
class ExactPerformanceIdentity:
    """Hashes of the three things for which performance was measured."""

    graph_sha256: str
    hardware_sha256: str
    config_sha256: str

    def __post_init__(self) -> None:
        _sha256("graph_sha256", self.graph_sha256)
        _sha256("hardware_sha256", self.hardware_sha256)
        _sha256("config_sha256", self.config_sha256)

    @classmethod
    def from_components(
        cls,
        *,
        graph: Mapping[str, Any],
        hardware: Mapping[str, Any],
        config: Mapping[str, Any],
    ) -> ExactPerformanceIdentity:
        """Hash canonical manifests rather than trusting ambient identity."""

        values = {"graph": graph, "hardware": hardware, "config": config}
        for name, value in values.items():
            if not isinstance(value, Mapping) or not value:
                raise ContributionPerformanceError(
                    f"{name} identity component must be a nonempty mapping"
                )
        return cls(
            graph_sha256=_canonical_sha256(dict(graph), name="graph identity"),
            hardware_sha256=_canonical_sha256(
                dict(hardware),
                name="hardware identity",
            ),
            config_sha256=_canonical_sha256(dict(config), name="config identity"),
        )

    @property
    def identity_sha256(self) -> str:
        return _canonical_sha256(self.to_dict(), name="performance identity")

    def to_dict(self) -> dict[str, str]:
        return {
            "schema_version": PERFORMANCE_IDENTITY_SCHEMA_VERSION,
            "graph_sha256": self.graph_sha256,
            "hardware_sha256": self.hardware_sha256,
            "config_sha256": self.config_sha256,
        }


@dataclass(frozen=True, slots=True)
class QualificationEvidence:
    """Raw bounded-benchmark evidence used to derive one passing receipt."""

    identity: ExactPerformanceIdentity
    source_evidence_sha256: str
    useful_flops_ledger_sha256: str
    qualification_wall_seconds: float
    measured_device_seconds: float
    persisted_window_seconds: float
    useful_flops: int
    peak_flops_per_second: int
    warmup_steps: int
    synchronized_measurement_steps: int
    measured_graph_compile_count: int
    auxiliary_validation_compile_count: int
    total_compile_count: int
    recompilations_after_warmup: int
    oom_events: int
    nonfinite_events: int
    capacity_overflow_events: int
    dropped_routed_tokens: int
    output_tokens: int
    persisted_tokens: int
    max_writer_backlog_batches: int
    writer_backlog_limit_batches: int
    writer_failures: int
    writer_queue_drained: bool
    static_shapes: bool
    parameters_resident_before_timing: bool
    inputs_resident_before_timing: bool
    synchronized_device_timing: bool
    instrumentation_disabled_output_sha256: str
    instrumentation_enabled_output_sha256: str
    writer_disabled_output_tokens: int
    writer_disabled_device_seconds: float

    @property
    def useful_prefix_mfu(self) -> float:
        return self.useful_flops / (
            self.peak_flops_per_second * self.measured_device_seconds
        )

    @property
    def device_output_tokens_per_second(self) -> float:
        return self.output_tokens / self.measured_device_seconds

    @property
    def persisted_tokens_per_second(self) -> float:
        return self.persisted_tokens / self.persisted_window_seconds

    @property
    def writer_disabled_tokens_per_second(self) -> float:
        return self.writer_disabled_output_tokens / self.writer_disabled_device_seconds

    @property
    def auxiliary_throughput_fraction(self) -> float:
        return (
            self.device_output_tokens_per_second
            / self.writer_disabled_tokens_per_second
        )

@dataclass(frozen=True, slots=True)
class AcceleratorRunAuthorization:
    """One in-memory decision a launcher can turn into a hard deadline."""

    purpose: Literal["qualification", "workload"]
    maximum_runtime_seconds: int
    identity_sha256: str
    basis: Literal[
        "bounded_qualification",
        "bounded_ungated_workload",
        "qualified_exact_identity",
    ]
    qualification_receipt_sha256: str | None


_RAW_EVIDENCE_FIELDS = tuple(
    item.name
    for item in fields(QualificationEvidence)
    if item.name
    not in {
        "identity",
        "source_evidence_sha256",
        "useful_flops_ledger_sha256",
    }
)
_PERFORMANCE_FIELDS = (
    "useful_prefix_mfu",
    "device_output_tokens_per_second",
    "persisted_tokens_per_second",
    "writer_disabled_tokens_per_second",
    "auxiliary_throughput_fraction",
)


def _validate_evidence_shape(
    evidence: QualificationEvidence,
) -> QualificationEvidence:
    if type(evidence) is not QualificationEvidence:
        raise ContributionPerformanceError("qualification evidence has the wrong type")
    if type(evidence.identity) is not ExactPerformanceIdentity:
        raise ContributionPerformanceError("qualification identity has the wrong type")
    for name in (
        "source_evidence_sha256",
        "useful_flops_ledger_sha256",
        "instrumentation_disabled_output_sha256",
        "instrumentation_enabled_output_sha256",
    ):
        _sha256(name, getattr(evidence, name))
    for name in (
        "qualification_wall_seconds",
        "measured_device_seconds",
        "persisted_window_seconds",
        "writer_disabled_device_seconds",
    ):
        _finite_positive(name, getattr(evidence, name))
    for name in (
        "useful_flops",
        "peak_flops_per_second",
        "warmup_steps",
        "synchronized_measurement_steps",
        "measured_graph_compile_count",
        "total_compile_count",
        "output_tokens",
        "persisted_tokens",
        "writer_backlog_limit_batches",
        "writer_disabled_output_tokens",
    ):
        _positive_int(name, getattr(evidence, name))
    for name in (
        "recompilations_after_warmup",
        "auxiliary_validation_compile_count",
        "oom_events",
        "nonfinite_events",
        "capacity_overflow_events",
        "dropped_routed_tokens",
        "max_writer_backlog_batches",
        "writer_failures",
    ):
        _nonnegative_int(name, getattr(evidence, name))
    for name in (
        "writer_queue_drained",
        "static_shapes",
        "parameters_resident_before_timing",
        "inputs_resident_before_timing",
        "synchronized_device_timing",
    ):
        _strict_bool(name, getattr(evidence, name))
    if evidence.measured_device_seconds > evidence.qualification_wall_seconds:
        raise ContributionPerformanceError(
            "measured device time exceeds total qualification time"
        )
    if evidence.persisted_window_seconds > evidence.qualification_wall_seconds:
        raise ContributionPerformanceError(
            "persisted window exceeds total qualification time"
        )
    if evidence.total_compile_count != (
        evidence.measured_graph_compile_count
        + evidence.auxiliary_validation_compile_count
    ):
        raise ContributionPerformanceError(
            "total compile count is not derived from measured plus auxiliary "
            "validation counts"
        )
    return evidence


def _require_passing_evidence(
    evidence: QualificationEvidence,
    *,
    minimum_useful_mfu: float,
    qualification_max_seconds: int,
) -> None:
    if evidence.qualification_wall_seconds > qualification_max_seconds:
        raise ContributionPerformanceError(
            "qualification exceeded its ten-minute wall-clock budget"
        )
    failures = (
        (
            evidence.measured_graph_compile_count != 1,
            "qualification must compile exactly one measured executable",
        ),
        (
            evidence.recompilations_after_warmup != 0,
            "qualification observed recompilation after warmup",
        ),
        (evidence.oom_events != 0, "qualification observed an OOM"),
        (
            evidence.nonfinite_events != 0,
            "qualification observed non-finite state",
        ),
        (
            evidence.capacity_overflow_events != 0,
            "qualification observed capacity overflow",
        ),
        (
            evidence.dropped_routed_tokens != 0,
            "qualification dropped routed tokens",
        ),
        (
            evidence.writer_failures != 0,
            "qualification observed a writer failure",
        ),
        (
            not evidence.writer_queue_drained,
            "writer queue was not drained before qualification completed",
        ),
        (
            evidence.max_writer_backlog_batches
            >= evidence.writer_backlog_limit_batches,
            "writer backlog reached or exceeded its hard bound",
        ),
        (
            not evidence.static_shapes,
            "qualification did not use one static-shape graph",
        ),
        (
            not evidence.parameters_resident_before_timing,
            "parameters were not device-resident before timing",
        ),
        (
            not evidence.inputs_resident_before_timing,
            "benchmark inputs were not device-resident before timing",
        ),
        (
            not evidence.synchronized_device_timing,
            "MFU timing was not synchronized through device completion",
        ),
        (
            evidence.instrumentation_disabled_output_sha256
            != evidence.instrumentation_enabled_output_sha256,
            "instrumentation changed the uninstrumented model output",
        ),
        (
            evidence.output_tokens != evidence.persisted_tokens,
            "persisted token count differs from device output",
        ),
    )
    for failed, message in failures:
        if failed:
            raise ContributionPerformanceError(message)
    if (
        evidence.persisted_tokens_per_second
        < MINIMUM_PERSISTED_GOODPUT_FRACTION * evidence.device_output_tokens_per_second
    ):
        raise ContributionPerformanceError(PERSISTED_GOODPUT_ADMISSION_MESSAGE)
    if evidence.auxiliary_throughput_fraction < MINIMUM_AUXILIARY_THROUGHPUT_FRACTION:
        raise ContributionPerformanceError(AUXILIARY_THROUGHPUT_ADMISSION_MESSAGE)
    mfu = evidence.useful_prefix_mfu
    if not math.isfinite(mfu) or not 0.0 < mfu <= 1.0:
        raise ContributionPerformanceError(
            "derived useful-prefix MFU is outside (0, 1]"
        )
    if mfu < minimum_useful_mfu:
        raise ContributionPerformanceError(USEFUL_MFU_ADMISSION_MESSAGE)


def _performance(evidence: QualificationEvidence) -> dict[str, float]:
    return {name: float(getattr(evidence, name)) for name in _PERFORMANCE_FIELDS}


def _raw_evidence(evidence: QualificationEvidence) -> dict[str, Any]:
    return {name: getattr(evidence, name) for name in _RAW_EVIDENCE_FIELDS}


def _parse_identity(value: Any) -> ExactPerformanceIdentity:
    selected = _exact_mapping(
        "qualification identity",
        value,
        {
            "schema_version",
            "graph_sha256",
            "hardware_sha256",
            "config_sha256",
        },
    )
    if selected["schema_version"] != PERFORMANCE_IDENTITY_SCHEMA_VERSION:
        raise ContributionPerformanceError(
            "performance identity schema is incompatible"
        )
    return ExactPerformanceIdentity(
        graph_sha256=_sha256("identity.graph_sha256", selected["graph_sha256"]),
        hardware_sha256=_sha256(
            "identity.hardware_sha256",
            selected["hardware_sha256"],
        ),
        config_sha256=_sha256(
            "identity.config_sha256",
            selected["config_sha256"],
        ),
    )


def build_qualification_receipt(
    evidence: QualificationEvidence,
    *,
    minimum_useful_mfu: float = MINIMUM_USEFUL_PREFIX_MFU,
    qualification_max_seconds: int = MAX_QUALIFICATION_SECONDS,
) -> dict[str, Any]:
    """Validate raw evidence and seal one exact-identity benchmark receipt."""

    minimum, _, maximum = _validate_policy(
        minimum_useful_mfu=minimum_useful_mfu,
        ungated_max_seconds=MAX_UNGATED_ACCELERATOR_SECONDS,
        qualification_max_seconds=qualification_max_seconds,
    )
    selected = _validate_evidence_shape(evidence)
    _require_passing_evidence(
        selected,
        minimum_useful_mfu=minimum,
        qualification_max_seconds=maximum,
    )
    unsigned = {
        "schema_version": QUALIFICATION_RECEIPT_SCHEMA_VERSION,
        "status": "passed",
        "identity": selected.identity.to_dict(),
        "identity_sha256": selected.identity.identity_sha256,
        "source_evidence_sha256": selected.source_evidence_sha256,
        "useful_flops_ledger_sha256": selected.useful_flops_ledger_sha256,
        "policy": {
            "minimum_useful_prefix_mfu": minimum,
            "qualification_max_seconds": maximum,
            "minimum_persisted_goodput_fraction": (MINIMUM_PERSISTED_GOODPUT_FRACTION),
            "minimum_auxiliary_throughput_fraction": (
                MINIMUM_AUXILIARY_THROUGHPUT_FRACTION
            ),
        },
        "evidence": _raw_evidence(selected),
        "performance": _performance(selected),
    }
    return {
        **unsigned,
        "receipt_sha256": _canonical_sha256(
            unsigned,
            name="qualification receipt",
        ),
    }


_RECEIPT_KEYS = {
    "schema_version",
    "status",
    "identity",
    "identity_sha256",
    "source_evidence_sha256",
    "useful_flops_ledger_sha256",
    "policy",
    "evidence",
    "performance",
    "receipt_sha256",
}


def validate_qualification_receipt(
    value: Mapping[str, Any],
    *,
    expected_identity: ExactPerformanceIdentity | None = None,
    minimum_useful_mfu: float = MINIMUM_USEFUL_PREFIX_MFU,
    qualification_max_seconds: int = MAX_QUALIFICATION_SECONDS,
) -> dict[str, Any]:
    """Re-derive every hard gate and return a validated receipt."""

    minimum, _, maximum = _validate_policy(
        minimum_useful_mfu=minimum_useful_mfu,
        ungated_max_seconds=MAX_UNGATED_ACCELERATOR_SECONDS,
        qualification_max_seconds=qualification_max_seconds,
    )
    selected = dict(_exact_mapping("qualification receipt", value, _RECEIPT_KEYS))
    claimed = _sha256("receipt_sha256", selected.pop("receipt_sha256"))
    if selected.get("schema_version") != QUALIFICATION_RECEIPT_SCHEMA_VERSION:
        raise ContributionPerformanceError(
            "qualification receipt schema is incompatible"
        )
    if selected.get("status") != "passed":
        raise ContributionPerformanceError("qualification receipt is not passing")
    if _canonical_sha256(selected, name="qualification receipt") != claimed:
        raise ContributionPerformanceError("qualification receipt self-hash is invalid")

    identity = _parse_identity(selected["identity"])
    if selected["identity_sha256"] != identity.identity_sha256:
        raise ContributionPerformanceError("qualification identity hash is invalid")
    if expected_identity is not None:
        if type(expected_identity) is not ExactPerformanceIdentity:
            raise ContributionPerformanceError("expected identity has the wrong type")
        if identity != expected_identity:
            raise ContributionPerformanceError(
                "qualification receipt does not match the exact "
                "graph/hardware/configuration"
            )
    source_evidence_sha256 = _sha256(
        "source_evidence_sha256",
        selected["source_evidence_sha256"],
    )
    useful_flops_ledger_sha256 = _sha256(
        "useful_flops_ledger_sha256",
        selected["useful_flops_ledger_sha256"],
    )

    policy = _exact_mapping(
        "qualification policy",
        selected["policy"],
        {
            "minimum_useful_prefix_mfu",
            "qualification_max_seconds",
            "minimum_persisted_goodput_fraction",
            "minimum_auxiliary_throughput_fraction",
        },
    )
    receipt_minimum = _finite_positive(
        "receipt minimum_useful_prefix_mfu",
        policy["minimum_useful_prefix_mfu"],
    )
    receipt_maximum = _positive_int(
        "receipt qualification_max_seconds",
        policy["qualification_max_seconds"],
    )
    if receipt_minimum < minimum:
        raise ContributionPerformanceError(
            "receipt MFU policy is weaker than launch policy"
        )
    if receipt_maximum > maximum:
        raise ContributionPerformanceError(
            "receipt qualification budget is weaker than launch policy"
        )
    throughput_policy = (
        (
            "minimum_persisted_goodput_fraction",
            MINIMUM_PERSISTED_GOODPUT_FRACTION,
        ),
        (
            "minimum_auxiliary_throughput_fraction",
            MINIMUM_AUXILIARY_THROUGHPUT_FRACTION,
        ),
    )
    for name, expected in throughput_policy:
        actual = _finite_positive(f"receipt {name}", policy[name])
        if not _same_float(actual, expected):
            raise ContributionPerformanceError(
                "qualification throughput policy is incompatible"
            )

    raw = _exact_mapping(
        "qualification evidence",
        selected["evidence"],
        set(_RAW_EVIDENCE_FIELDS),
    )
    evidence = QualificationEvidence(
        identity=identity,
        source_evidence_sha256=source_evidence_sha256,
        useful_flops_ledger_sha256=useful_flops_ledger_sha256,
        **dict(raw),
    )
    _validate_evidence_shape(evidence)
    _require_passing_evidence(
        evidence,
        minimum_useful_mfu=max(minimum, receipt_minimum),
        qualification_max_seconds=min(maximum, receipt_maximum),
    )
    performance = _exact_mapping(
        "qualification performance",
        selected["performance"],
        set(_PERFORMANCE_FIELDS),
    )
    for name, wanted in _performance(evidence).items():
        actual = _finite_positive(f"receipt performance.{name}", performance[name])
        if not _same_float(actual, wanted):
            raise ContributionPerformanceError(
                f"receipt performance.{name} is not derived from raw evidence"
            )

    selected["receipt_sha256"] = claimed
    return selected


def authorize_accelerator_run(
    *,
    identity: ExactPerformanceIdentity,
    maximum_runtime_seconds: int,
    purpose: Literal["qualification", "workload"],
    qualification_receipt: Mapping[str, Any] | None = None,
    minimum_useful_mfu: float = MINIMUM_USEFUL_PREFIX_MFU,
    ungated_max_seconds: int = MAX_UNGATED_ACCELERATOR_SECONDS,
    qualification_max_seconds: int = MAX_QUALIFICATION_SECONDS,
) -> AcceleratorRunAuthorization:
    """Authorize only a bounded probe or an exactly qualified longer run."""

    if type(identity) is not ExactPerformanceIdentity:
        raise ContributionPerformanceError("accelerator identity has the wrong type")
    runtime = _positive_int("maximum_runtime_seconds", maximum_runtime_seconds)
    if purpose not in _PURPOSES:
        raise ContributionPerformanceError(f"purpose must be one of {_PURPOSES!r}")
    minimum, ungated, qualification = _validate_policy(
        minimum_useful_mfu=minimum_useful_mfu,
        ungated_max_seconds=ungated_max_seconds,
        qualification_max_seconds=qualification_max_seconds,
    )

    if purpose == "qualification":
        if runtime > qualification:
            raise ContributionPerformanceError(
                "qualification launch exceeds its ten-minute hard limit"
            )
        return AcceleratorRunAuthorization(
            purpose=purpose,
            maximum_runtime_seconds=runtime,
            identity_sha256=identity.identity_sha256,
            basis="bounded_qualification",
            qualification_receipt_sha256=None,
        )
    if runtime <= ungated and qualification_receipt is None:
        return AcceleratorRunAuthorization(
            purpose=purpose,
            maximum_runtime_seconds=runtime,
            identity_sha256=identity.identity_sha256,
            basis="bounded_ungated_workload",
            qualification_receipt_sha256=None,
        )
    if qualification_receipt is None:
        raise ContributionPerformanceError(
            "accelerator work beyond ten minutes requires an exact "
            "qualification receipt"
        )
    validated = validate_qualification_receipt(
        qualification_receipt,
        expected_identity=identity,
        minimum_useful_mfu=minimum,
        qualification_max_seconds=qualification,
    )
    return AcceleratorRunAuthorization(
        purpose=purpose,
        maximum_runtime_seconds=runtime,
        identity_sha256=identity.identity_sha256,
        basis="qualified_exact_identity",
        qualification_receipt_sha256=validated["receipt_sha256"],
    )


def authorization_to_dict(
    value: AcceleratorRunAuthorization,
) -> dict[str, Any]:
    """Return a plain record suitable for an operator log."""

    if type(value) is not AcceleratorRunAuthorization:
        raise ContributionPerformanceError(
            "accelerator authorization has the wrong type"
        )
    return asdict(value)


__all__ = [
    "AUXILIARY_THROUGHPUT_ADMISSION_MESSAGE",
    "MAX_QUALIFICATION_SECONDS",
    "MAX_UNGATED_ACCELERATOR_SECONDS",
    "MINIMUM_AUXILIARY_THROUGHPUT_FRACTION",
    "MINIMUM_PERSISTED_GOODPUT_FRACTION",
    "MINIMUM_USEFUL_PREFIX_MFU",
    "PERFORMANCE_IDENTITY_SCHEMA_VERSION",
    "PERSISTED_GOODPUT_ADMISSION_MESSAGE",
    "QUALIFICATION_RECEIPT_SCHEMA_VERSION",
    "USEFUL_MFU_ADMISSION_MESSAGE",
    "AcceleratorRunAuthorization",
    "ContributionPerformanceError",
    "ExactPerformanceIdentity",
    "QualificationEvidence",
    "authorization_to_dict",
    "authorize_accelerator_run",
    "build_qualification_receipt",
    "validate_qualification_receipt",
]
