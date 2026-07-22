"""Auditable MFU records for the DeepSeek-R1 performance gates.

The module intentionally does not estimate DeepSeek FLOPs from a model name.
An adapter must produce a versioned operation inventory from the pinned tensor
inventory and compiled program.  This module only validates that inventory and
reduces synchronized measurement windows against the documented hardware peak.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Literal, Sequence

from .protocol import canonical_sha256


V6E_CHIP_BF16_PEAK_FLOPS = 918_000_000_000_000
V6E64_CHIPS = 64
V6E64_BF16_PEAK_FLOPS = V6E_CHIP_BF16_PEAK_FLOPS * V6E64_CHIPS
R1_MINIMUM_MFU = 0.51
R1_MINIMUM_WINDOWS = 3
R1_MINIMUM_STEPS_PER_WINDOW = 500
R1_MINIMUM_WALL_SECONDS_PER_WINDOW = 60.0
R1_MAXIMUM_INPUT_WAIT_FRACTION = 0.05
R1_MINIMUM_REAL_TO_SYNTHETIC_RATIO = 0.90
R1_MINIMUM_PERSISTED_TO_DEVICE_RATIO = 0.90
R1_MINIMUM_HBM_HEADROOM_FRACTION = 0.15
R1_MAXIMUM_INSTRUMENTATION_MFU_DELTA = 0.02
R1_MAXIMUM_P95_TO_P50_RATIO = 1.10
R1_MAXIMUM_SLOWEST_TO_MEDIAN_HOST_RATIO = 1.05
R1_MAXIMUM_ROUTING_CAPACITY_AMPLIFICATION = 1.10

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PHASES = frozenset({"harvest_prefix", "sae_dense_calibration", "sae_f"})


class MFUContractError(ValueError):
    """Raised when performance evidence does not satisfy the frozen contract."""


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise MFUContractError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MFUContractError(f"{name} must be a non-negative integer")
    return value


def _finite_float(name: str, value: float, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MFUContractError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise MFUContractError(f"{name} must be a finite number")
    if minimum is not None and result < minimum:
        raise MFUContractError(f"{name} must be at least {minimum}")
    return result


def _sha256(name: str, value: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise MFUContractError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _identifier(name: str, value: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 160:
        raise MFUContractError(
            f"{name} must be a non-empty string of at most 160 characters"
        )
    return value


@dataclass(frozen=True, slots=True)
class FlopTermV1:
    """One useful-matmul term derived from the accepted execution inventory."""

    name: str
    useful_flops: int
    derivation: str

    def __post_init__(self) -> None:
        _identifier("name", self.name)
        _positive_int("useful_flops", self.useful_flops)
        _identifier("derivation", self.derivation)

    def to_dict(self) -> dict[str, object]:
        return {
            "derivation": self.derivation,
            "name": self.name,
            "useful_flops": self.useful_flops,
        }


@dataclass(frozen=True, slots=True)
class UsefulFlopInventoryV1:
    """Content-addressed useful-FLOP definition for one static step shape."""

    phase: Literal["harvest_prefix", "sae_dense_calibration", "sae_f"]
    model_revision_sha256: str
    tensor_inventory_sha256: str
    hlo_sha256: str
    terms: tuple[FlopTermV1, ...]
    excluded_operations: tuple[str, ...]
    schema_version: str = "r1-useful-flops-v1"

    def __post_init__(self) -> None:
        if self.phase not in _PHASES:
            raise MFUContractError(f"unsupported MFU phase: {self.phase!r}")
        _sha256("model_revision_sha256", self.model_revision_sha256)
        _sha256("tensor_inventory_sha256", self.tensor_inventory_sha256)
        _sha256("hlo_sha256", self.hlo_sha256)
        if not self.terms:
            raise MFUContractError("terms must not be empty")
        names = [term.name for term in self.terms]
        if len(names) != len(set(names)):
            raise MFUContractError("FLOP term names must be unique")
        if tuple(sorted(names)) != tuple(names):
            raise MFUContractError("FLOP terms must be sorted by name")
        if not self.excluded_operations:
            raise MFUContractError("excluded_operations must be explicit")
        for operation in self.excluded_operations:
            _identifier("excluded operation", operation)
        if tuple(sorted(set(self.excluded_operations))) != self.excluded_operations:
            raise MFUContractError("excluded_operations must be unique and sorted")
        if self.schema_version != "r1-useful-flops-v1":
            raise MFUContractError("unsupported useful-FLOP schema")

    @property
    def useful_flops_per_step(self) -> int:
        return sum(term.useful_flops for term in self.terms)

    def to_dict(self) -> dict[str, object]:
        return {
            "excluded_operations": list(self.excluded_operations),
            "hlo_sha256": self.hlo_sha256,
            "model_revision_sha256": self.model_revision_sha256,
            "phase": self.phase,
            "schema_version": self.schema_version,
            "tensor_inventory_sha256": self.tensor_inventory_sha256,
            "terms": [term.to_dict() for term in self.terms],
            "useful_flops_per_step": self.useful_flops_per_step,
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class MFUWindowV1:
    """One synchronized, post-warmup real-data measurement window."""

    window_id: str
    phase: Literal["harvest_prefix", "sae_dense_calibration", "sae_f"]
    flop_inventory_sha256: str
    hlo_sha256: str
    profiler_sha256: str
    step_start: int
    step_end: int
    useful_flops: int
    scientific_tokens: int
    wall_seconds: float
    input_wait_seconds: float
    real_tokens_per_second: float
    synthetic_tokens_per_second: float
    device_output_tokens_per_second: float
    persisted_tokens_per_second: float | None
    instrumentation_mfu_delta: float
    p50_step_seconds: float
    p95_step_seconds: float
    median_host_step_seconds: float
    slowest_host_step_seconds: float
    routing_capacity_amplification: float
    peak_hbm_bytes_per_chip: int
    hbm_capacity_bytes_per_chip: int
    recompilations: int
    dropped_routed_tokens: int
    capacity_overflows: int
    finite_state: bool
    physical_chips: int = V6E64_CHIPS
    peak_flops_per_chip: int = V6E_CHIP_BF16_PEAK_FLOPS
    schema_version: str = "r1-mfu-window-v1"

    def __post_init__(self) -> None:
        _identifier("window_id", self.window_id)
        if self.phase not in _PHASES:
            raise MFUContractError(f"unsupported MFU phase: {self.phase!r}")
        _sha256("flop_inventory_sha256", self.flop_inventory_sha256)
        _sha256("hlo_sha256", self.hlo_sha256)
        _sha256("profiler_sha256", self.profiler_sha256)
        _nonnegative_int("step_start", self.step_start)
        _positive_int("step_end", self.step_end)
        if self.step_end <= self.step_start:
            raise MFUContractError("step_end must be greater than step_start")
        _positive_int("useful_flops", self.useful_flops)
        _positive_int("scientific_tokens", self.scientific_tokens)
        _finite_float("wall_seconds", self.wall_seconds, minimum=0.0)
        if self.wall_seconds == 0.0:
            raise MFUContractError("wall_seconds must be positive")
        _finite_float("input_wait_seconds", self.input_wait_seconds, minimum=0.0)
        if self.input_wait_seconds > self.wall_seconds:
            raise MFUContractError("input_wait_seconds cannot exceed wall_seconds")
        for name in (
            "real_tokens_per_second",
            "synthetic_tokens_per_second",
            "device_output_tokens_per_second",
        ):
            if _finite_float(name, getattr(self, name), minimum=0.0) == 0.0:
                raise MFUContractError(f"{name} must be positive")
        if self.persisted_tokens_per_second is not None:
            if (
                _finite_float(
                    "persisted_tokens_per_second",
                    self.persisted_tokens_per_second,
                    minimum=0.0,
                )
                == 0.0
            ):
                raise MFUContractError("persisted_tokens_per_second must be positive")
        _finite_float(
            "instrumentation_mfu_delta", self.instrumentation_mfu_delta, minimum=0.0
        )
        for name in (
            "p50_step_seconds",
            "p95_step_seconds",
            "median_host_step_seconds",
            "slowest_host_step_seconds",
            "routing_capacity_amplification",
        ):
            if _finite_float(name, getattr(self, name), minimum=0.0) == 0.0:
                raise MFUContractError(f"{name} must be positive")
        if self.p95_step_seconds < self.p50_step_seconds:
            raise MFUContractError(
                "p95_step_seconds must not be below p50_step_seconds"
            )
        if self.slowest_host_step_seconds < self.median_host_step_seconds:
            raise MFUContractError(
                "slowest_host_step_seconds must not be below median_host_step_seconds"
            )
        _positive_int("peak_hbm_bytes_per_chip", self.peak_hbm_bytes_per_chip)
        _positive_int("hbm_capacity_bytes_per_chip", self.hbm_capacity_bytes_per_chip)
        if self.peak_hbm_bytes_per_chip >= self.hbm_capacity_bytes_per_chip:
            raise MFUContractError("peak HBM must be below physical capacity")
        _nonnegative_int("recompilations", self.recompilations)
        _nonnegative_int("dropped_routed_tokens", self.dropped_routed_tokens)
        _nonnegative_int("capacity_overflows", self.capacity_overflows)
        if not isinstance(self.finite_state, bool):
            raise MFUContractError("finite_state must be boolean")
        _positive_int("physical_chips", self.physical_chips)
        _positive_int("peak_flops_per_chip", self.peak_flops_per_chip)
        if self.schema_version != "r1-mfu-window-v1":
            raise MFUContractError("unsupported MFU-window schema")

    @property
    def measured_steps(self) -> int:
        return self.step_end - self.step_start

    @property
    def mfu(self) -> float:
        return self.useful_flops / (
            self.wall_seconds * self.physical_chips * self.peak_flops_per_chip
        )

    @property
    def input_wait_fraction(self) -> float:
        return self.input_wait_seconds / self.wall_seconds

    @property
    def real_to_synthetic_ratio(self) -> float:
        return self.real_tokens_per_second / self.synthetic_tokens_per_second

    @property
    def persisted_to_device_ratio(self) -> float | None:
        if self.persisted_tokens_per_second is None:
            return None
        return self.persisted_tokens_per_second / self.device_output_tokens_per_second

    @property
    def hbm_headroom_fraction(self) -> float:
        return 1.0 - self.peak_hbm_bytes_per_chip / self.hbm_capacity_bytes_per_chip


@dataclass(frozen=True, slots=True)
class MFUQualificationV1:
    """Accepted reduction of independent windows for one R1 hard gate."""

    phase: Literal["harvest_prefix", "sae_dense_calibration", "sae_f"]
    window_ids: tuple[str, ...]
    flop_inventory_sha256: str
    hlo_sha256: str
    aggregate_mfu: float
    minimum_window_mfu: float
    total_steps: int
    total_scientific_tokens: int
    threshold: float
    schema_version: str = "r1-mfu-qualification-v1"


def qualify_mfu_windows(
    windows: Sequence[MFUWindowV1],
    *,
    phase: Literal["harvest_prefix", "sae_dense_calibration", "sae_f"],
    minimum_mfu: float = R1_MINIMUM_MFU,
    minimum_windows: int = R1_MINIMUM_WINDOWS,
    minimum_steps_per_window: int = R1_MINIMUM_STEPS_PER_WINDOW,
    minimum_wall_seconds_per_window: float = R1_MINIMUM_WALL_SECONDS_PER_WINDOW,
    maximum_input_wait_fraction: float = R1_MAXIMUM_INPUT_WAIT_FRACTION,
    minimum_real_to_synthetic_ratio: float = R1_MINIMUM_REAL_TO_SYNTHETIC_RATIO,
    minimum_persisted_to_device_ratio: float = R1_MINIMUM_PERSISTED_TO_DEVICE_RATIO,
    minimum_hbm_headroom_fraction: float = R1_MINIMUM_HBM_HEADROOM_FRACTION,
    maximum_instrumentation_mfu_delta: float = R1_MAXIMUM_INSTRUMENTATION_MFU_DELTA,
    maximum_p95_to_p50_ratio: float = R1_MAXIMUM_P95_TO_P50_RATIO,
    maximum_slowest_to_median_host_ratio: float = R1_MAXIMUM_SLOWEST_TO_MEDIAN_HOST_RATIO,
    maximum_routing_capacity_amplification: float = R1_MAXIMUM_ROUTING_CAPACITY_AMPLIFICATION,
) -> MFUQualificationV1:
    """Fail closed unless every independent window passes the R1 MFU gate."""

    if phase not in _PHASES:
        raise MFUContractError(f"unsupported MFU phase: {phase!r}")
    _finite_float("minimum_mfu", minimum_mfu, minimum=R1_MINIMUM_MFU)
    if _positive_int("minimum_windows", minimum_windows) < R1_MINIMUM_WINDOWS:
        raise MFUContractError(
            f"minimum_windows cannot be weakened below {R1_MINIMUM_WINDOWS}"
        )
    if (
        _positive_int("minimum_steps_per_window", minimum_steps_per_window)
        < R1_MINIMUM_STEPS_PER_WINDOW
    ):
        raise MFUContractError(
            "minimum_steps_per_window cannot be weakened below "
            f"{R1_MINIMUM_STEPS_PER_WINDOW}"
        )
    if (
        _finite_float(
            "minimum_wall_seconds_per_window",
            minimum_wall_seconds_per_window,
            minimum=0.0,
        )
        < R1_MINIMUM_WALL_SECONDS_PER_WINDOW
    ):
        raise MFUContractError(
            "minimum_wall_seconds_per_window cannot be weakened below "
            f"{R1_MINIMUM_WALL_SECONDS_PER_WINDOW:g}"
        )
    if (
        _finite_float(
            "maximum_input_wait_fraction",
            maximum_input_wait_fraction,
            minimum=0.0,
        )
        > R1_MAXIMUM_INPUT_WAIT_FRACTION
    ):
        raise MFUContractError(
            "maximum_input_wait_fraction cannot be weakened above "
            f"{R1_MAXIMUM_INPUT_WAIT_FRACTION}"
        )
    if (
        _finite_float(
            "minimum_real_to_synthetic_ratio",
            minimum_real_to_synthetic_ratio,
            minimum=0.0,
        )
        < R1_MINIMUM_REAL_TO_SYNTHETIC_RATIO
    ):
        raise MFUContractError(
            "minimum_real_to_synthetic_ratio cannot be weakened below "
            f"{R1_MINIMUM_REAL_TO_SYNTHETIC_RATIO}"
        )
    if (
        _finite_float(
            "minimum_persisted_to_device_ratio",
            minimum_persisted_to_device_ratio,
            minimum=0.0,
        )
        < R1_MINIMUM_PERSISTED_TO_DEVICE_RATIO
    ):
        raise MFUContractError(
            "minimum_persisted_to_device_ratio cannot be weakened below "
            f"{R1_MINIMUM_PERSISTED_TO_DEVICE_RATIO}"
        )
    if (
        _finite_float(
            "minimum_hbm_headroom_fraction",
            minimum_hbm_headroom_fraction,
            minimum=0.0,
        )
        < R1_MINIMUM_HBM_HEADROOM_FRACTION
    ):
        raise MFUContractError(
            "minimum_hbm_headroom_fraction cannot be weakened below "
            f"{R1_MINIMUM_HBM_HEADROOM_FRACTION}"
        )
    if (
        _finite_float(
            "maximum_instrumentation_mfu_delta",
            maximum_instrumentation_mfu_delta,
            minimum=0.0,
        )
        > R1_MAXIMUM_INSTRUMENTATION_MFU_DELTA
    ):
        raise MFUContractError(
            "maximum_instrumentation_mfu_delta cannot be weakened above "
            f"{R1_MAXIMUM_INSTRUMENTATION_MFU_DELTA}"
        )
    if (
        _finite_float(
            "maximum_p95_to_p50_ratio",
            maximum_p95_to_p50_ratio,
            minimum=1.0,
        )
        > R1_MAXIMUM_P95_TO_P50_RATIO
    ):
        raise MFUContractError(
            "maximum_p95_to_p50_ratio cannot be weakened above "
            f"{R1_MAXIMUM_P95_TO_P50_RATIO}"
        )
    if (
        _finite_float(
            "maximum_slowest_to_median_host_ratio",
            maximum_slowest_to_median_host_ratio,
            minimum=1.0,
        )
        > R1_MAXIMUM_SLOWEST_TO_MEDIAN_HOST_RATIO
    ):
        raise MFUContractError(
            "maximum_slowest_to_median_host_ratio cannot be weakened above "
            f"{R1_MAXIMUM_SLOWEST_TO_MEDIAN_HOST_RATIO}"
        )
    if (
        _finite_float(
            "maximum_routing_capacity_amplification",
            maximum_routing_capacity_amplification,
            minimum=1.0,
        )
        > R1_MAXIMUM_ROUTING_CAPACITY_AMPLIFICATION
    ):
        raise MFUContractError(
            "maximum_routing_capacity_amplification cannot be weakened above "
            f"{R1_MAXIMUM_ROUTING_CAPACITY_AMPLIFICATION}"
        )
    if len(windows) < minimum_windows:
        raise MFUContractError(
            f"need at least {minimum_windows} independent windows; got {len(windows)}"
        )

    ordered = tuple(
        sorted(
            windows, key=lambda item: (item.step_start, item.step_end, item.window_id)
        )
    )
    ids = tuple(window.window_id for window in ordered)
    if len(ids) != len(set(ids)):
        raise MFUContractError("measurement window IDs must be unique")
    for previous, current in zip(ordered, ordered[1:]):
        if current.step_start < previous.step_end:
            raise MFUContractError("measurement windows must not overlap")

    inventory_digests = {window.flop_inventory_sha256 for window in ordered}
    hlo_digests = {window.hlo_sha256 for window in ordered}
    chip_shapes = {
        (window.physical_chips, window.peak_flops_per_chip) for window in ordered
    }
    if len(inventory_digests) != 1:
        raise MFUContractError("all windows must use one useful-FLOP inventory")
    if len(hlo_digests) != 1:
        raise MFUContractError("all windows must use one static HLO")
    if len(chip_shapes) != 1 or chip_shapes != {
        (V6E64_CHIPS, V6E_CHIP_BF16_PEAK_FLOPS)
    }:
        raise MFUContractError(
            "R1 qualification requires a physical v6e-64 denominator"
        )

    for window in ordered:
        if window.phase != phase:
            raise MFUContractError("all windows must match the requested phase")
        if window.measured_steps < minimum_steps_per_window:
            raise MFUContractError(
                f"window {window.window_id} has only {window.measured_steps} measured steps"
            )
        if window.wall_seconds < minimum_wall_seconds_per_window:
            raise MFUContractError(
                f"window {window.window_id} is shorter than the 60-second minimum"
            )
        if window.mfu < minimum_mfu:
            raise MFUContractError(
                f"window {window.window_id} MFU {window.mfu:.6%} is below {minimum_mfu:.2%}"
            )
        if window.mfu > 1.0:
            raise MFUContractError(
                f"window {window.window_id} exceeds physical peak; audit the FLOP ledger"
            )
        if window.recompilations != 0:
            raise MFUContractError(f"window {window.window_id} recompiled")
        if window.dropped_routed_tokens != 0:
            raise MFUContractError(f"window {window.window_id} dropped routed tokens")
        if window.capacity_overflows != 0:
            raise MFUContractError(f"window {window.window_id} overflowed capacity")
        if not window.finite_state:
            raise MFUContractError(f"window {window.window_id} has non-finite state")
        if window.input_wait_fraction >= maximum_input_wait_fraction:
            raise MFUContractError(
                f"window {window.window_id} input wait is not below "
                f"{maximum_input_wait_fraction:.2%}"
            )
        if window.real_to_synthetic_ratio < minimum_real_to_synthetic_ratio:
            raise MFUContractError(
                f"window {window.window_id} real/synthetic goodput is below "
                f"{minimum_real_to_synthetic_ratio:.2%}"
            )
        if window.hbm_headroom_fraction < minimum_hbm_headroom_fraction:
            raise MFUContractError(
                f"window {window.window_id} has insufficient HBM headroom"
            )
        if window.instrumentation_mfu_delta > maximum_instrumentation_mfu_delta:
            raise MFUContractError(
                f"window {window.window_id} instrumentation MFU delta exceeds "
                f"{maximum_instrumentation_mfu_delta:.2%}"
            )
        if window.p95_step_seconds / window.p50_step_seconds > maximum_p95_to_p50_ratio:
            raise MFUContractError(
                f"window {window.window_id} has unstable step latency"
            )
        if (
            window.slowest_host_step_seconds / window.median_host_step_seconds
            > maximum_slowest_to_median_host_ratio
        ):
            raise MFUContractError(
                f"window {window.window_id} has a slow host straggler"
            )
        if (
            window.routing_capacity_amplification
            > maximum_routing_capacity_amplification
        ):
            raise MFUContractError(
                f"window {window.window_id} has excessive routing amplification"
            )
        if phase == "harvest_prefix":
            ratio = window.persisted_to_device_ratio
            if ratio is None or ratio < minimum_persisted_to_device_ratio:
                raise MFUContractError(
                    f"window {window.window_id} persistence goodput is too low"
                )

    total_flops = sum(window.useful_flops for window in ordered)
    total_wall = sum(window.wall_seconds for window in ordered)
    aggregate_mfu = total_flops / (total_wall * V6E64_BF16_PEAK_FLOPS)
    if aggregate_mfu < minimum_mfu:
        raise MFUContractError("aggregate MFU is below the non-waivable threshold")
    return MFUQualificationV1(
        phase=phase,
        window_ids=ids,
        flop_inventory_sha256=next(iter(inventory_digests)),
        hlo_sha256=next(iter(hlo_digests)),
        aggregate_mfu=aggregate_mfu,
        minimum_window_mfu=min(window.mfu for window in ordered),
        total_steps=sum(window.measured_steps for window in ordered),
        total_scientific_tokens=sum(window.scientific_tokens for window in ordered),
        threshold=minimum_mfu,
    )


__all__ = [
    "FlopTermV1",
    "MFUContractError",
    "MFUQualificationV1",
    "MFUWindowV1",
    "R1_MINIMUM_MFU",
    "R1_MAXIMUM_INPUT_WAIT_FRACTION",
    "R1_MAXIMUM_INSTRUMENTATION_MFU_DELTA",
    "R1_MAXIMUM_P95_TO_P50_RATIO",
    "R1_MAXIMUM_ROUTING_CAPACITY_AMPLIFICATION",
    "R1_MAXIMUM_SLOWEST_TO_MEDIAN_HOST_RATIO",
    "R1_MINIMUM_HBM_HEADROOM_FRACTION",
    "R1_MINIMUM_PERSISTED_TO_DEVICE_RATIO",
    "R1_MINIMUM_REAL_TO_SYNTHETIC_RATIO",
    "R1_MINIMUM_STEPS_PER_WINDOW",
    "R1_MINIMUM_WALL_SECONDS_PER_WINDOW",
    "R1_MINIMUM_WINDOWS",
    "UsefulFlopInventoryV1",
    "V6E64_BF16_PEAK_FLOPS",
    "V6E64_CHIPS",
    "V6E_CHIP_BF16_PEAK_FLOPS",
    "qualify_mfu_windows",
]
