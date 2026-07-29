"""Standalone bounded or qualified training worker for Experiment-2 SAEs.

This module is intentionally independent of :mod:`moe_sae.cli`.  It owns the
small operational boundary which the scientific trainers do not:

* run historical fixed resident executables directly under a <=3600-second
  deadline and the strict 10M horizon executable under a <=12-hour deadline;
* optionally qualify resident and rematerialized executables for longer runs;
* bind the faster passing executable to an exact graph/hardware/config receipt;
* refuse other accelerator work beyond 3600 seconds without that >=3% receipt;
* advance a sealed v2 input stream and immutable checkpoint pointers in exact
  update order.

The orchestration core is dependency injected.  Unit tests can therefore prove
admission and resume semantics on CPU without pretending to emulate a TPU.
Production dependencies are imported lazily so the outer supervisor process
does not initialize JAX before it starts the hard-deadlined child.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import yaml

from moe_sae.contribution_artifacts import canonical_json_bytes, canonical_sha256
from moe_sae.contribution_config import (
    Experiment2Config,
    load_contribution_config,
)
from moe_sae.contribution_performance import (
    MAX_LEARNING_HORIZON_ACCELERATOR_SECONDS,
    MAX_QUALIFICATION_SECONDS,
    MAX_UNGATED_ACCELERATOR_SECONDS,
    MINIMUM_USEFUL_PREFIX_MFU,
    ExactPerformanceIdentity,
    authorize_accelerator_run,
    validate_qualification_receipt,
)
from moe_sae.contribution_training_performance import (
    SAEArithmetic,
    SAEQualificationObservation,
    build_sae_qualification_receipt,
)

TRAINING_WORKER_SCHEMA_VERSION = "gemma4-contribution-training-worker-v1"
TRAINING_MODE_SELECTION_SCHEMA_VERSION = "gemma4-contribution-sae-mode-selection-v1"
TRAINING_RUN_IDENTITY_SCHEMA_VERSION = "gemma4-contribution-sae-run-identity-v2"
TRAINING_RESULT_SCHEMA_VERSION = "gemma4-contribution-sae-training-result-v1"
TRAINING_EXECUTION_CONFIG_SCHEMA_VERSION = "gemma4-contribution-sae-execution-config-v1"
TRAINING_GRAPH_SCHEMA_VERSION = "gemma4-contribution-sae-stablehlo-v1"
TRAINING_HARDWARE_SCHEMA_VERSION = "gemma4-contribution-sae-hardware-v1"
TRAINING_SUPERVISION_SCHEMA_VERSION = (
    "gemma4-contribution-sae-supervised-qualification-v1"
)
PREFETCH_RECEIPT_SCHEMA_VERSION = "gemma4-contribution-sae-process-lane-prefetch-v1"
TRANSCODER_PREFETCH_RECEIPT_SCHEMA_VERSION = (
    "gemma4-contribution-transcoder-paired-process-lane-prefetch-v1"
)
HISTORICAL_PREFETCH_MAX_SECONDS = 600
HORIZON_PREFETCH_MAX_SECONDS = 1_800
HORIZON_DIRECT_ACCELERATOR_TYPES = frozenset(
    {"v4-32", "v6e-16", "v5litepod-16"}
)
CHIP_SECONDS_SCOPE = (
    "fixed_capacity_scan_in_training_invocation_plus_synchronized_"
    "optimizer_loop_including_input_wait"
)
CHIP_SECONDS_EXCLUDES = (
    "prefetch",
    "qualification",
    "lowering",
    "compilation",
    "checkpoint_serialization",
    "evaluation",
)


def _finalized_horizon_config(config_path: str | Path) -> Any | None:
    """Return a strict long-horizon config, or ``None`` for another schema."""

    selected = Path(config_path).expanduser().resolve()
    try:
        raw = yaml.safe_load(selected.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError, yaml.YAMLError) as error:
        raise ContributionTrainingWorkerError(
            "cannot inspect the selected config for its execution policy"
        ) from error
    if not isinstance(raw, Mapping):
        return None

    schema = raw.get("schema_version")
    if schema == "gemma4-routed-shared-hybrid-run-config-v1":
        from moe_sae.experiment5_config import (
            Experiment5ConfigError,
            load_experiment5_config,
        )

        try:
            return load_experiment5_config(selected)
        except Experiment5ConfigError as error:
            raise ContributionTrainingWorkerError(
                "Experiment-5 execution requires its strict finalized config"
            ) from error
    if schema != "gemma4-pilot-learning-horizon-run-config-v1":
        return None

    # A filename is not an execution identity. Re-load the whole document
    # through the strict finalized boundary before granting either extension.
    from moe_sae.learning_horizon_config import (
        LearningHorizonConfigError,
        load_learning_horizon_config,
    )

    try:
        return load_learning_horizon_config(selected, require_finalized=True)
    except LearningHorizonConfigError as error:
        raise ContributionTrainingWorkerError(
            "learning-horizon execution requires a finalized strict config"
        ) from error


def prefetch_runtime_limit_seconds(
    *,
    config_path: str | Path,
    accelerator_type: str,
    phase: str | None = None,
) -> int:
    """Return the exact host-prefetch cap for one immutable run shape."""

    if (
        accelerator_type in HORIZON_DIRECT_ACCELERATOR_TYPES
        and phase in {"learning_horizon_10m", "routed_shared_hybrid_10m"}
        and _finalized_horizon_config(config_path) is not None
    ):
        return HORIZON_PREFETCH_MAX_SECONDS
    return HISTORICAL_PREFETCH_MAX_SECONDS


def direct_resident_runtime_limit_seconds(
    *,
    config_path: str | Path,
    phase: str,
    accelerator_type: str,
) -> int:
    """Return the direct resident cap without weakening historical protocols.

    Only the exact finalized ``learning_horizon_10m`` contract on an audited
    ``v4-32``, ``v6e-16``, or ``v5litepod-16`` runtime receives the twelve-hour
    run-to-completion window. Historical phases and unsupported generations
    retain the one-hour direct bound.
    """

    if (
        accelerator_type not in HORIZON_DIRECT_ACCELERATOR_TYPES
        or phase not in {"learning_horizon_10m", "routed_shared_hybrid_10m"}
    ):
        return MAX_UNGATED_ACCELERATOR_SECONDS
    config = _finalized_horizon_config(config_path)
    if config is None:
        raise ContributionTrainingWorkerError(
            "learning_horizon_10m requires its strict finalized config"
        )
    maximum = _positive_int(
        "execution.direct_runtime_max_seconds",
        config.runtime.direct_runtime_max_seconds,
    )
    if maximum != MAX_LEARNING_HORIZON_ACCELERATOR_SECONDS:
        raise ContributionTrainingWorkerError(
            "learning-horizon direct runtime policy drifted"
        )
    return maximum

InputMode = Literal["resident", "rematerialized"]
INPUT_MODES: tuple[InputMode, ...] = ("resident", "rematerialized")
ScientificModel = Literal["sae", "transcoder"]
SCIENTIFIC_MODELS: tuple[ScientificModel, ...] = ("sae", "transcoder")
CONDITIONS = ("A", "H", "R", "B", "F", "Q")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ContributionTrainingWorkerError(RuntimeError):
    """The qualified training boundary failed closed."""


def _sha256(name: str, value: Any) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ContributionTrainingWorkerError(f"{name} must be a SHA-256")
    return value


def _positive_int(name: str, value: Any, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ContributionTrainingWorkerError(f"{name} must be a {qualifier} integer")
    return value


def _finite_nonnegative(name: str, value: Any) -> float:
    selected = float(value)
    if not math.isfinite(selected) or selected < 0.0:
        raise ContributionTrainingWorkerError(f"{name} must be finite and nonnegative")
    return selected


def _read_json(path: str | Path, *, name: str) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file() or source.is_symlink():
        raise ContributionTrainingWorkerError(f"{name} is absent or not regular")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContributionTrainingWorkerError(f"cannot read {name}") from error
    if not isinstance(value, dict):
        raise ContributionTrainingWorkerError(f"{name} must be a JSON object")
    return value


def _write_create_only_json(path: str | Path, value: Mapping[str, Any]) -> Path:
    """Publish one immutable, fsynced JSON record without replacement."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(dict(value)) + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(destination, flags, 0o640)
    except FileExistsError:
        try:
            existing = json.loads(destination.read_text(encoding="ascii"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ContributionTrainingWorkerError(
                f"immutable record already exists and cannot be read: {destination}"
            ) from error
        if existing != dict(value):
            raise ContributionTrainingWorkerError(
                f"immutable record differs from replay: {destination}"
            )
        return destination
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return destination


@dataclass(frozen=True, slots=True)
class TrainingWorkerPlan:
    """One exact phase/condition/seed token and schedule plan."""

    phase_name: str
    condition: str
    seed: int
    total_tokens: int
    planned_updates: int
    warmup_updates: int
    global_microbatch_tokens: int
    microbatches_per_update: int
    checkpoint_tokens: tuple[int, ...]
    randomization_key: int

    def __post_init__(self) -> None:
        if not isinstance(self.phase_name, str) or not self.phase_name:
            raise ContributionTrainingWorkerError("phase_name is empty")
        if self.condition not in CONDITIONS:
            raise ContributionTrainingWorkerError("unsupported SAE condition")
        for name in (
            "seed",
            "total_tokens",
            "planned_updates",
            "warmup_updates",
            "global_microbatch_tokens",
            "microbatches_per_update",
            "randomization_key",
        ):
            _positive_int(
                name,
                getattr(self, name),
                allow_zero=name in {"seed", "warmup_updates", "randomization_key"},
            )
        if self.warmup_updates >= self.planned_updates:
            raise ContributionTrainingWorkerError(
                "warmup must finish before the final update"
            )
        expected_updates = math.ceil(self.total_tokens / self.effective_batch_tokens)
        if self.planned_updates != expected_updates:
            raise ContributionTrainingWorkerError(
                "planned updates differ from the exact token plan"
            )
        if tuple(sorted(set(self.checkpoint_tokens))) != self.checkpoint_tokens or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 < value <= self.total_tokens
            for value in self.checkpoint_tokens
        ):
            raise ContributionTrainingWorkerError(
                "checkpoint tokens must be unique, ordered, and in range"
            )

    @property
    def effective_batch_tokens(self) -> int:
        return self.global_microbatch_tokens * self.microbatches_per_update

    def expected_tokens_after(self, updates: int) -> int:
        _positive_int("updates", updates, allow_zero=True)
        if updates > self.planned_updates:
            raise ContributionTrainingWorkerError("update exceeds the token plan")
        return min(updates * self.effective_batch_tokens, self.total_tokens)

    @classmethod
    def from_config(
        cls,
        config: Experiment2Config,
        *,
        phase: str,
        condition: str,
        seed: int,
    ) -> TrainingWorkerPlan:
        config.validate()
        selected = config.phase(phase)
        if selected.layout_set == "none":
            raise ContributionTrainingWorkerError(
                "selected phase has no SAE training layouts"
            )
        if condition not in CONDITIONS:
            raise ContributionTrainingWorkerError("unsupported SAE condition")
        if seed not in selected.seeds:
            raise ContributionTrainingWorkerError(
                "seed is not authorized by the selected phase"
            )
        if condition not in {
            layout.condition for layout in config.layouts_for_phase(phase)
        }:
            raise ContributionTrainingWorkerError(
                "condition is absent from the selected phase"
            )
        assert selected.planned_updates is not None
        assert selected.warmup_updates is not None
        randomization_key = seed
        from moe_sae.experiment5_config import (
            Experiment5Config,
            experiment5_randomization_key,
        )

        if isinstance(config, Experiment5Config):
            randomization_key = experiment5_randomization_key(seed)
        return cls(
            phase_name=selected.name,
            condition=condition,
            seed=seed,
            total_tokens=selected.train_tokens,
            planned_updates=selected.planned_updates,
            warmup_updates=selected.warmup_updates,
            global_microbatch_tokens=config.runtime.microbatch_tokens,
            microbatches_per_update=(
                config.runtime.effective_batch_tokens
                // config.runtime.microbatch_tokens
            ),
            checkpoint_tokens=selected.checkpoint_tokens,
            randomization_key=randomization_key,
        )


@dataclass(frozen=True, slots=True)
class PreparedTrainingExecutable:
    """Opaque lowered executable plus its observed exact identity."""

    handle: Any
    identity: ExactPerformanceIdentity
    graph_manifest: Mapping[str, Any]
    hardware_manifest: Mapping[str, Any]
    execution_config: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class RestoredTrainingState:
    state: Any
    metadata: Mapping[str, Any]


class TrainingSource(Protocol):
    total_tokens: int
    planned_updates: int

    def iter_updates(self) -> Any: ...


@dataclass(frozen=True, slots=True)
class TrainingLoopDependencies:
    """Injected state/executable/checkpoint operations used by the core loop."""

    initialize_state: Callable[[], Any]
    lower: Callable[[InputMode, Any, Any, Any], PreparedTrainingExecutable]
    compile: Callable[[PreparedTrainingExecutable], float]
    step: Callable[
        [PreparedTrainingExecutable, Any, Any, Any],
        tuple[Any, Mapping[str, Any]],
    ]
    synchronize: Callable[[Any], None]
    progress: Callable[[Any], tuple[int, int]]
    restore: Callable[
        [Any, Mapping[str, Any]],
        RestoredTrainingState | None,
    ]
    save: Callable[
        [Any, Mapping[str, Any], Mapping[str, Any], bool],
        Mapping[str, Any],
    ]
    canonicalize_seconds: Callable[[float], float] = float
    clock: Callable[[], float] = time.perf_counter


def _mode_candidate(
    mode: InputMode,
    observation: SAEQualificationObservation,
    arithmetic: SAEArithmetic,
    *,
    total_wall_seconds: float,
    peak_flops_per_second: int,
    minimum_useful_mfu: float,
) -> dict[str, Any]:
    adjusted = replace(
        observation,
        qualification_wall_seconds=max(
            float(observation.measured_device_seconds),
            total_wall_seconds,
        ),
    )
    evidence, bundle = build_sae_qualification_receipt(
        adjusted,
        arithmetic,
        peak_flops_per_second=peak_flops_per_second,
        minimum_useful_mfu=minimum_useful_mfu,
    )
    unsigned = {
        "mode": mode,
        "status": "passed",
        "identity": adjusted.identity.to_dict(),
        "identity_sha256": adjusted.identity.identity_sha256,
        "scientific_tokens": adjusted.scientific_tokens,
        "measured_device_seconds": adjusted.measured_device_seconds,
        "tokens_per_second": (
            adjusted.scientific_tokens / adjusted.measured_device_seconds
        ),
        "useful_mfu": evidence.useful_prefix_mfu,
        "source_evidence": bundle["source_evidence"],
        "useful_flops_ledger": bundle["useful_flops_ledger"],
        "qualification_receipt": bundle["qualification_receipt"],
    }
    return {
        **unsigned,
        "candidate_sha256": canonical_sha256(unsigned),
    }


def qualify_training_modes(
    probe: Callable[[InputMode], SAEQualificationObservation],
    arithmetic: SAEArithmetic,
    *,
    peak_flops_per_second: int,
    minimum_useful_mfu: float = MINIMUM_USEFUL_PREFIX_MFU,
    maximum_seconds: int = MAX_QUALIFICATION_SECONDS,
    workload_context: Mapping[str, Any] | None = None,
    qualification_started: float | None = None,
    elapsed_transform: Callable[[float], float] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Benchmark both graph modes and seal the fastest passing exact receipt.

    The total wall time—not two independent per-mode budgets—is written into
    every passing receipt.  A caller must still place this function inside the
    process-group supervisor; the local check is evidence, not a kill switch.
    """

    if maximum_seconds > MAX_QUALIFICATION_SECONDS:
        raise ContributionTrainingWorkerError(
            "SAE qualification cannot exceed 600 seconds"
        )
    _positive_int("maximum_seconds", maximum_seconds)
    _positive_int("peak_flops_per_second", peak_flops_per_second)
    started = clock() if qualification_started is None else float(qualification_started)
    if not math.isfinite(started):
        raise ContributionTrainingWorkerError("qualification start time is invalid")
    raw: list[tuple[InputMode, SAEQualificationObservation | BaseException]] = []
    for mode in INPUT_MODES:
        try:
            observation = probe(mode)
            if type(observation) is not SAEQualificationObservation:
                raise ContributionTrainingWorkerError(
                    "mode probe returned the wrong observation type"
                )
            raw.append((mode, observation))
        # Retain terminal evidence for both predeclared modes; the outer
        # process-group supervisor still bounds the combined attempt.
        except Exception as error:  # noqa: BLE001
            raw.append((mode, error))
    elapsed = float(clock() - started)
    if elapsed_transform is not None:
        elapsed = float(elapsed_transform(elapsed))
    if not math.isfinite(elapsed) or elapsed < 0:
        raise ContributionTrainingWorkerError(
            "qualification clock returned an invalid duration"
        )
    if elapsed > maximum_seconds:
        raise ContributionTrainingWorkerError(
            "SAE mode qualification exceeded its 600-second total deadline"
        )

    candidates: list[dict[str, Any]] = []
    for mode, result in raw:
        if isinstance(result, BaseException):
            unsigned = {
                "mode": mode,
                "status": "failed",
                "error_type": type(result).__name__,
                "error_message": str(result),
            }
            candidates.append(
                {
                    **unsigned,
                    "candidate_sha256": canonical_sha256(unsigned),
                }
            )
            continue
        try:
            candidates.append(
                _mode_candidate(
                    mode,
                    result,
                    arithmetic,
                    total_wall_seconds=elapsed,
                    peak_flops_per_second=peak_flops_per_second,
                    minimum_useful_mfu=minimum_useful_mfu,
                )
            )
        # A validation failure rejects this mode without hiding the other
        # predeclared candidate's independently measured evidence.
        except Exception as error:  # noqa: BLE001
            unsigned = {
                "mode": mode,
                "status": "failed",
                "identity": result.identity.to_dict(),
                "identity_sha256": result.identity.identity_sha256,
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
            candidates.append(
                {
                    **unsigned,
                    "candidate_sha256": canonical_sha256(unsigned),
                }
            )
    passing = [item for item in candidates if item["status"] == "passed"]
    if not passing:
        failures = "; ".join(
            f"{item['mode']}: {item['error_type']}: {item['error_message']}"
            for item in candidates
        )
        raise ContributionTrainingWorkerError(
            f"neither SAE input mode qualified: {failures}"
        )
    # Deterministic tie-break favors resident-forward: it performs no
    # deliberately duplicated rematerialized arithmetic.
    chosen = max(
        passing,
        key=lambda item: (
            float(item["tokens_per_second"]),
            item["mode"] == "resident",
        ),
    )
    unsigned = {
        "schema_version": TRAINING_MODE_SELECTION_SCHEMA_VERSION,
        "qualification_total_wall_seconds": elapsed,
        "qualification_max_seconds": maximum_seconds,
        "minimum_useful_mfu": float(minimum_useful_mfu),
        "arithmetic": asdict(arithmetic),
        "workload_context": dict(workload_context or {}),
        "candidate_order": list(INPUT_MODES),
        "candidates": candidates,
        "chosen_mode": chosen["mode"],
        "chosen_identity_sha256": chosen["identity_sha256"],
        "chosen_receipt_sha256": chosen["qualification_receipt"]["receipt_sha256"],
    }
    return {
        **unsigned,
        "selection_sha256": canonical_sha256(unsigned),
    }


def validate_training_mode_selection(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Revalidate hashes, both attempts, chosen mode, and hard-gate receipt."""

    expected = {
        "schema_version",
        "qualification_total_wall_seconds",
        "qualification_max_seconds",
        "minimum_useful_mfu",
        "arithmetic",
        "workload_context",
        "candidate_order",
        "candidates",
        "chosen_mode",
        "chosen_identity_sha256",
        "chosen_receipt_sha256",
        "selection_sha256",
    }
    if set(value) != expected:
        raise ContributionTrainingWorkerError(
            "mode-selection fields do not match the schema"
        )
    selected = dict(value)
    claimed = _sha256("selection_sha256", selected.pop("selection_sha256"))
    if selected["schema_version"] != TRAINING_MODE_SELECTION_SCHEMA_VERSION:
        raise ContributionTrainingWorkerError("mode-selection schema changed")
    if canonical_sha256(selected) != claimed:
        raise ContributionTrainingWorkerError("mode-selection self-hash is invalid")
    elapsed = _finite_nonnegative(
        "qualification_total_wall_seconds",
        selected["qualification_total_wall_seconds"],
    )
    maximum = _positive_int(
        "qualification_max_seconds",
        selected["qualification_max_seconds"],
    )
    if maximum > MAX_QUALIFICATION_SECONDS or elapsed > maximum:
        raise ContributionTrainingWorkerError(
            "mode selection exceeded the hard qualification deadline"
        )
    if selected["candidate_order"] != list(INPUT_MODES):
        raise ContributionTrainingWorkerError(
            "mode selection did not benchmark both frozen modes"
        )
    if not isinstance(selected["workload_context"], Mapping):
        raise ContributionTrainingWorkerError(
            "mode-selection workload context is malformed"
        )
    candidates = selected["candidates"]
    if not isinstance(candidates, list) or [
        item.get("mode") for item in candidates
    ] != list(INPUT_MODES):
        raise ContributionTrainingWorkerError(
            "mode-selection candidate order is invalid"
        )
    chosen_mode = selected["chosen_mode"]
    if chosen_mode not in INPUT_MODES:
        raise ContributionTrainingWorkerError("chosen input mode is invalid")
    chosen: Mapping[str, Any] | None = None
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise ContributionTrainingWorkerError(
                "mode-selection candidate is malformed"
            )
        candidate_copy = dict(candidate)
        candidate_hash = _sha256(
            "candidate_sha256",
            candidate_copy.pop("candidate_sha256", None),
        )
        if canonical_sha256(candidate_copy) != candidate_hash:
            raise ContributionTrainingWorkerError(
                "mode-selection candidate self-hash is invalid"
            )
        if candidate.get("status") == "passed":
            raw_identity = candidate.get("identity")
            if not isinstance(raw_identity, Mapping):
                raise ContributionTrainingWorkerError(
                    "passing candidate identity is malformed"
                )
            identity = ExactPerformanceIdentity(
                graph_sha256=_sha256(
                    "candidate graph SHA-256",
                    raw_identity.get("graph_sha256"),
                ),
                hardware_sha256=_sha256(
                    "candidate hardware SHA-256",
                    raw_identity.get("hardware_sha256"),
                ),
                config_sha256=_sha256(
                    "candidate config SHA-256",
                    raw_identity.get("config_sha256"),
                ),
            )
            receipt = candidate.get("qualification_receipt")
            if not isinstance(receipt, Mapping):
                raise ContributionTrainingWorkerError(
                    "passing candidate lacks a receipt"
                )
            validate_qualification_receipt(
                receipt,
                expected_identity=identity,
                minimum_useful_mfu=float(selected["minimum_useful_mfu"]),
                qualification_max_seconds=maximum,
            )
        if candidate.get("mode") == chosen_mode:
            chosen = candidate
    if chosen is None or chosen.get("status") != "passed":
        raise ContributionTrainingWorkerError("chosen mode is not passing")
    if (
        chosen.get("identity_sha256") != selected["chosen_identity_sha256"]
        or chosen["qualification_receipt"].get("receipt_sha256")
        != selected["chosen_receipt_sha256"]
    ):
        raise ContributionTrainingWorkerError(
            "chosen mode does not match its identity/receipt"
        )
    selected["selection_sha256"] = claimed
    return selected


def chosen_mode_evidence(
    selection: Mapping[str, Any],
) -> tuple[InputMode, ExactPerformanceIdentity, dict[str, Any]]:
    checked = validate_training_mode_selection(selection)
    mode: InputMode = checked["chosen_mode"]
    candidate = next(item for item in checked["candidates"] if item["mode"] == mode)
    raw = candidate["identity"]
    identity = ExactPerformanceIdentity(
        graph_sha256=raw["graph_sha256"],
        hardware_sha256=raw["hardware_sha256"],
        config_sha256=raw["config_sha256"],
    )
    return mode, identity, dict(candidate["qualification_receipt"])


def _metric_record(value: Mapping[str, Any], *, expected_l0: int) -> dict[str, Any]:
    required = {
        "loss",
        "sse",
        "sst",
        "l0",
        "valid_tokens",
        "topk_valid",
        "grad_norm",
        "learning_rate",
    }
    if set(value) != required:
        raise ContributionTrainingWorkerError(
            "training metric fields do not match the worker schema"
        )
    result = dict(value)
    for name in ("loss", "sse", "sst", "l0", "grad_norm", "learning_rate"):
        selected = float(result[name])
        if not math.isfinite(selected):
            raise ContributionTrainingWorkerError(
                f"training metric {name} is non-finite"
            )
        result[name] = selected
    tokens = _positive_int("metric valid_tokens", result["valid_tokens"])
    result["valid_tokens"] = tokens
    if type(result["topk_valid"]) is not bool or not result["topk_valid"]:
        raise ContributionTrainingWorkerError(
            "training update violated exact positive TopK"
        )
    if not math.isclose(float(result["l0"]), expected_l0, abs_tol=1e-5):
        raise ContributionTrainingWorkerError(
            "training update L0 differs from the frozen layout"
        )
    if result["sst"] <= 0.0:
        raise ContributionTrainingWorkerError("training update SST is not positive")
    return result


def _run_identity(
    plan: TrainingWorkerPlan,
    *,
    identity: ExactPerformanceIdentity,
    mode: InputMode,
    admission_basis: str,
    admission_evidence_sha256: str | None,
) -> dict[str, Any]:
    if admission_basis not in {
        "qualified_exact_identity",
        "bounded_ungated_resident",
    }:
        raise ContributionTrainingWorkerError("training run admission basis is invalid")
    if admission_evidence_sha256 is not None:
        _sha256(
            "admission evidence SHA-256",
            admission_evidence_sha256,
        )
    if (admission_basis == "qualified_exact_identity") != (
        admission_evidence_sha256 is not None
    ):
        raise ContributionTrainingWorkerError(
            "training run admission evidence is inconsistent"
        )
    unsigned = {
        "schema_version": TRAINING_RUN_IDENTITY_SCHEMA_VERSION,
        "plan": {
            **asdict(plan),
            "checkpoint_tokens": list(plan.checkpoint_tokens),
        },
        "input_mode": mode,
        "performance_identity": identity.to_dict(),
        "performance_identity_sha256": identity.identity_sha256,
        "admission": {
            "basis": admission_basis,
            "evidence_sha256": admission_evidence_sha256,
        },
    }
    return {
        **unsigned,
        "run_identity_sha256": canonical_sha256(unsigned),
    }


def _canonical_elapsed_seconds(
    dependencies: TrainingLoopDependencies,
    *,
    name: str,
    local_seconds: float,
) -> float:
    """Return one timing value shared by every process in the training job."""

    local = _finite_nonnegative(name, local_seconds)
    try:
        canonical = dependencies.canonicalize_seconds(local)
    except Exception as error:
        raise ContributionTrainingWorkerError(
            f"could not canonicalize {name} across training processes"
        ) from error
    return _finite_nonnegative(f"canonical {name}", canonical)


def _restored_timing(
    restored: RestoredTrainingState | None,
) -> dict[str, float | int]:
    """Validate cumulative cost evidence embedded in a restored checkpoint."""

    if restored is None:
        return {}
    raw = dict(restored.metadata).get("timing")
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ContributionTrainingWorkerError(
            "restored checkpoint timing must be a mapping"
        )
    result: dict[str, float | int] = {}
    for name in (
        "compile_seconds",
        "lowering_seconds",
        "preprocessing_wall_seconds",
        "optimizer_wall_seconds",
        "checkpoint_seconds_before_save",
    ):
        if name in raw:
            result[name] = _finite_nonnegative(
                f"restored {name}",
                raw[name],
            )
    if "global_chip_count" in raw:
        result["global_chip_count"] = _positive_int(
            "restored global_chip_count",
            raw["global_chip_count"],
        )
    return result


def _training_chip_count(
    prepared: PreparedTrainingExecutable | None,
    *,
    supplied: int | None,
    restored_timing: Mapping[str, float | int],
) -> int:
    """Resolve and cross-check the one global chip count used by the ledger."""

    candidates: list[int] = []
    if supplied is not None:
        candidates.append(_positive_int("global_chip_count", supplied))
    if prepared is not None:
        candidates.append(
            _positive_int(
                "prepared global_chip_count",
                prepared.hardware_manifest.get("global_chip_count"),
            )
        )
    restored = restored_timing.get("global_chip_count")
    if restored is not None:
        candidates.append(_positive_int("restored global_chip_count", restored))
    if not candidates:
        raise ContributionTrainingWorkerError(
            "training cost ledger has no global chip count"
        )
    if len(set(candidates)) != 1:
        raise ContributionTrainingWorkerError(
            "training cost ledger global chip counts disagree"
        )
    return candidates[0]


def run_qualified_training(
    plan: TrainingWorkerPlan,
    source: TrainingSource,
    mean: Any,
    selection: Mapping[str, Any] | None,
    dependencies: TrainingLoopDependencies,
    *,
    expected_l0: int,
    maximum_runtime_seconds: int,
    metric_sink: Callable[[Mapping[str, Any]], None] | None = None,
    direct_bounded_resident: bool = False,
    direct_runtime_max_seconds: int = MAX_UNGATED_ACCELERATOR_SECONDS,
    preprocessing_wall_seconds: float = 0.0,
    global_chip_count: int | None = None,
) -> dict[str, Any]:
    """Run or resume one exact fixed-order condition.

    Checkpoint callbacks receive the complete run identity and cumulative
    timing.  They must publish immutable payloads before changing a mutable
    ``latest`` pointer.  The core never infers progress from payload listings.

    ``direct_bounded_resident`` deliberately bypasses mode benchmarking and
    receipt admission only while the complete child is hard-limited by its
    explicit direct policy. Historical plans retain 3600 seconds; only the
    strict 10M horizon may pass 43200. The run still derives its checkpoint
    identity from the actually lowered StableHLO/hardware/config tuple.
    """

    if source.total_tokens != plan.total_tokens:
        raise ContributionTrainingWorkerError(
            "training source token count differs from the phase plan"
        )
    if source.planned_updates != plan.planned_updates:
        raise ContributionTrainingWorkerError(
            "training source update count differs from the phase plan"
        )
    _positive_int("expected_l0", expected_l0)
    maximum = _positive_int("maximum_runtime_seconds", maximum_runtime_seconds)
    direct_maximum = _positive_int(
        "direct_runtime_max_seconds",
        direct_runtime_max_seconds,
    )
    if direct_maximum not in {
        MAX_UNGATED_ACCELERATOR_SECONDS,
        MAX_LEARNING_HORIZON_ACCELERATOR_SECONDS,
    }:
        raise ContributionTrainingWorkerError(
            "direct resident runtime policy is unsupported"
        )
    if (
        direct_maximum == MAX_LEARNING_HORIZON_ACCELERATOR_SECONDS
        and plan.phase_name
        not in {"learning_horizon_10m", "routed_shared_hybrid_10m"}
    ):
        raise ContributionTrainingWorkerError(
            "the twelve-hour direct policy is restricted to finalized "
            "10M learning-horizon protocols"
        )
    if direct_bounded_resident:
        if maximum > direct_maximum:
            raise ContributionTrainingWorkerError(
                "direct resident training exceeds its explicit runtime policy"
            )
        if selection is not None:
            raise ContributionTrainingWorkerError(
                "direct resident training must not receive a mode selection"
            )
        mode: InputMode = "resident"
        receipt_identity: ExactPerformanceIdentity | None = None
        receipt: dict[str, Any] | None = None
        admission_basis = "bounded_ungated_resident"
        admission_evidence_sha256 = None
    else:
        if selection is None:
            raise ContributionTrainingWorkerError(
                "qualified training requires a mode selection"
            )
        mode, receipt_identity, receipt = chosen_mode_evidence(selection)
        checked_selection = validate_training_mode_selection(selection)
        admission_basis = "qualified_exact_identity"
        admission_evidence_sha256 = checked_selection["selection_sha256"]

    current_preprocessing_seconds = _canonical_elapsed_seconds(
        dependencies,
        name="preprocessing_wall_seconds",
        local_seconds=preprocessing_wall_seconds,
    )
    initial = dependencies.initialize_state()
    prepared: PreparedTrainingExecutable | None = None
    current_lowering_seconds = 0.0
    if direct_bounded_resident:
        # Lower once from the deterministic first fixed-order batch so the
        # direct run obtains an exact identity before looking up a resumable
        # checkpoint namespace.  Values do not affect StableHLO identity.
        try:
            identity_batch = next(iter(source.iter_updates()))
        except StopIteration as error:
            raise ContributionTrainingWorkerError(
                "training source contains no updates"
            ) from error
        lowered_started = dependencies.clock()
        prepared = dependencies.lower(mode, initial, identity_batch, mean)
        current_lowering_seconds = _canonical_elapsed_seconds(
            dependencies,
            name="lowering_seconds",
            local_seconds=dependencies.clock() - lowered_started,
        )
        provisional_identity = _run_identity(
            plan,
            identity=prepared.identity,
            mode=mode,
            admission_basis=admission_basis,
            admission_evidence_sha256=admission_evidence_sha256,
        )
    else:
        assert receipt_identity is not None
        # A qualified run can identify its checkpoint namespace before
        # lowering because the chosen receipt already binds that identity.
        provisional_identity = _run_identity(
            plan,
            identity=receipt_identity,
            mode=mode,
            admission_basis=admission_basis,
            admission_evidence_sha256=admission_evidence_sha256,
        )
    restored = dependencies.restore(initial, provisional_identity)
    prior_timing = _restored_timing(restored)
    state = initial if restored is None else restored.state
    if (
        restored is not None
        and dict(restored.metadata).get("run_identity_sha256")
        != provisional_identity["run_identity_sha256"]
    ):
        raise ContributionTrainingWorkerError(
            "restored checkpoint metadata belongs to another run"
        )
    start_step, start_tokens = dependencies.progress(state)
    _positive_int("restored step", start_step, allow_zero=True)
    _positive_int("restored tokens", start_tokens, allow_zero=True)
    expected_start = plan.expected_tokens_after(start_step)
    if start_tokens != expected_start:
        raise ContributionTrainingWorkerError(
            "restored state is not on the deterministic fixed-order boundary"
        )
    if start_step > plan.planned_updates:
        raise ContributionTrainingWorkerError(
            "restored state exceeds the phase update plan"
        )

    preprocessing_seconds = (
        float(prior_timing.get("preprocessing_wall_seconds", 0.0))
        + current_preprocessing_seconds
    )
    lowering_seconds = (
        float(prior_timing.get("lowering_seconds", 0.0)) + current_lowering_seconds
    )
    compile_seconds = float(prior_timing.get("compile_seconds", 0.0))
    optimizer_seconds = float(prior_timing.get("optimizer_wall_seconds", 0.0))
    checkpoint_seconds = float(prior_timing.get("checkpoint_seconds_before_save", 0.0))

    direct_seek = getattr(source, "iter_updates_from", None)
    iterator = (
        iter(direct_seek(start_step))
        if callable(direct_seek)
        else itertools.islice(source.iter_updates(), start_step, None)
    )
    if start_step == plan.planned_updates:
        if optimizer_seconds <= 0.0:
            raise ContributionTrainingWorkerError(
                "completed checkpoint lacks cumulative optimizer timing"
            )
        chip_count = _training_chip_count(
            prepared,
            supplied=global_chip_count,
            restored_timing=prior_timing,
        )
        marginal_seconds = preprocessing_seconds + optimizer_seconds
        unsigned = {
            "schema_version": TRAINING_RESULT_SCHEMA_VERSION,
            "status": "already_complete",
            "run_identity": provisional_identity,
            "input_mode": mode,
            "steps": start_step,
            "tokens": start_tokens,
            "compile_seconds": compile_seconds,
            "lowering_seconds": lowering_seconds,
            "preprocessing_wall_seconds": preprocessing_seconds,
            "optimizer_wall_seconds": optimizer_seconds,
            "marginal_wall_seconds": marginal_seconds,
            "checkpoint_seconds": checkpoint_seconds,
            "chip_seconds": marginal_seconds * chip_count,
            "chip_seconds_scope": CHIP_SECONDS_SCOPE,
            "chip_seconds_excludes": list(CHIP_SECONDS_EXCLUDES),
            "tokens_per_second": start_tokens / marginal_seconds,
            "metric_count": 0,
            "last_metric": None,
            "authorization_basis": "existing_complete_checkpoint",
            "qualification_receipt_sha256": None,
            "checkpoint_receipts": [],
        }
        return {
            **unsigned,
            "result_sha256": canonical_sha256(unsigned),
        }

    input_started = dependencies.clock()
    try:
        first = next(iterator)
    except StopIteration as error:
        raise ContributionTrainingWorkerError(
            "training source ended before the restored phase boundary"
        ) from error
    first_input_seconds = dependencies.clock() - input_started
    if first_input_seconds < 0:
        raise ContributionTrainingWorkerError("training clock moved backwards")

    if prepared is None:
        lowered_started = dependencies.clock()
        prepared = dependencies.lower(mode, state, first, mean)
        lowering_seconds += _canonical_elapsed_seconds(
            dependencies,
            name="lowering_seconds",
            local_seconds=dependencies.clock() - lowered_started,
        )
        if prepared.identity != receipt_identity:
            raise ContributionTrainingWorkerError(
                "lowered graph/hardware/config differs from the chosen " "qualification"
            )
    authorization = authorize_accelerator_run(
        identity=prepared.identity,
        maximum_runtime_seconds=maximum,
        purpose="workload",
        qualification_receipt=(
            receipt
            if not direct_bounded_resident and maximum > MAX_QUALIFICATION_SECONDS
            else None
        ),
        workload_profile=(
            "learning_horizon_10m"
            if direct_bounded_resident
            and direct_maximum == MAX_LEARNING_HORIZON_ACCELERATOR_SECONDS
            else "historical"
        ),
    )
    compile_seconds += _canonical_elapsed_seconds(
        dependencies,
        name="compile_seconds",
        local_seconds=dependencies.compile(prepared),
    )

    metrics: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    next_checkpoints = iter(plan.checkpoint_tokens)
    checkpoint_target = next(next_checkpoints, None)
    while checkpoint_target is not None and checkpoint_target <= start_tokens:
        checkpoint_target = next(next_checkpoints, None)

    pending: Any | None = first
    previous_step = start_step
    previous_tokens = start_tokens
    while previous_step < plan.planned_updates:
        step_started = dependencies.clock()
        if pending is None:
            try:
                group = next(iterator)
            except StopIteration as error:
                raise ContributionTrainingWorkerError(
                    "training source ended before the exact phase plan"
                ) from error
            input_seconds = 0.0
        else:
            group = pending
            pending = None
            input_seconds = first_input_seconds
        state, raw_metric = dependencies.step(prepared, state, group, mean)
        dependencies.synchronize(state)
        elapsed = _canonical_elapsed_seconds(
            dependencies,
            name="optimizer update seconds",
            local_seconds=input_seconds + (dependencies.clock() - step_started),
        )
        optimizer_seconds += elapsed
        metric = _metric_record(raw_metric, expected_l0=expected_l0)
        step, tokens = dependencies.progress(state)
        if step != previous_step + 1:
            raise ContributionTrainingWorkerError(
                "optimizer state did not advance exactly one update"
            )
        if tokens != previous_tokens + metric["valid_tokens"]:
            raise ContributionTrainingWorkerError(
                "optimizer token state differs from the scientific batch count"
            )
        if tokens != plan.expected_tokens_after(step):
            raise ContributionTrainingWorkerError(
                "optimizer state left the fixed phase token plan"
            )
        metric_row = {
            **metric,
            "step": step,
            "tokens_seen": tokens,
            "optimizer_wall_seconds": optimizer_seconds,
        }
        metrics.append(metric_row)
        if metric_sink is not None:
            metric_sink(metric_row)
        previous_step, previous_tokens = step, tokens

        should_checkpoint = (
            checkpoint_target is not None and tokens >= checkpoint_target
        )
        is_final = step == plan.planned_updates
        if should_checkpoint or is_final:
            checkpoint_started = dependencies.clock()
            timing = {
                "compile_seconds": compile_seconds,
                "lowering_seconds": lowering_seconds,
                "preprocessing_wall_seconds": preprocessing_seconds,
                "optimizer_wall_seconds": optimizer_seconds,
                "checkpoint_seconds_before_save": checkpoint_seconds,
                "global_chip_count": _training_chip_count(
                    prepared,
                    supplied=global_chip_count,
                    restored_timing=prior_timing,
                ),
            }
            record = dict(
                dependencies.save(
                    state,
                    provisional_identity,
                    timing,
                    is_final,
                )
            )
            checkpoints.append(record)
            checkpoint_seconds += _canonical_elapsed_seconds(
                dependencies,
                name="checkpoint serialization seconds",
                local_seconds=dependencies.clock() - checkpoint_started,
            )
            while checkpoint_target is not None and checkpoint_target <= tokens:
                checkpoint_target = next(next_checkpoints, None)

    try:
        next(iterator)
    except StopIteration:
        pass
    else:
        raise ContributionTrainingWorkerError(
            "training source yielded rows beyond the exact phase plan"
        )
    if previous_tokens != plan.total_tokens:
        raise ContributionTrainingWorkerError(
            "training finished without consuming the exact token count"
        )

    # MFU remains diagnostic during production training.  The primary measured
    # ledger includes the fixed A/R capacity scan performed by this training
    # invocation plus synchronized optimizer work and its input waits.  The
    # explicitly named exclusions remain separately visible.
    chip_count = _training_chip_count(
        prepared,
        supplied=global_chip_count,
        restored_timing=prior_timing,
    )
    marginal_seconds = preprocessing_seconds + optimizer_seconds
    unsigned = {
        "schema_version": TRAINING_RESULT_SCHEMA_VERSION,
        "status": "completed",
        "run_identity": provisional_identity,
        "input_mode": mode,
        "steps": previous_step,
        "tokens": previous_tokens,
        "compile_seconds": compile_seconds,
        "lowering_seconds": lowering_seconds,
        "preprocessing_wall_seconds": preprocessing_seconds,
        "optimizer_wall_seconds": optimizer_seconds,
        "marginal_wall_seconds": marginal_seconds,
        "checkpoint_seconds": checkpoint_seconds,
        "chip_seconds": marginal_seconds * chip_count,
        "chip_seconds_scope": CHIP_SECONDS_SCOPE,
        "chip_seconds_excludes": list(CHIP_SECONDS_EXCLUDES),
        "tokens_per_second": previous_tokens / marginal_seconds,
        "metric_count": len(metrics),
        "last_metric": metrics[-1],
        "authorization_basis": authorization.basis,
        "qualification_receipt_sha256": (authorization.qualification_receipt_sha256),
        "checkpoint_receipts": checkpoints,
    }
    return {
        **unsigned,
        "result_sha256": canonical_sha256(unsigned),
    }


# ---------------------------------------------------------------------------
# Production TPU adapter


class _ResidentOnlyObjectStore:
    """Sentinel proving that the training process cannot refill from GCS."""

    def _blocked(self, *_args: Any, **_kwargs: Any) -> Any:
        raise ContributionTrainingWorkerError(
            "qualified resident v2 cache attempted an object-store operation"
        )

    put_if_absent = _blocked
    download = _blocked
    verify = _blocked
    delete = _blocked


def _prefetch_object_summary(
    dataset: Any,
    *,
    range_start: int,
    range_end: int,
) -> dict[str, Any]:
    identities = tuple(dataset.token_range_object_identities(range_start, range_end))
    if not identities:
        raise ContributionTrainingWorkerError(
            "process lane intersects no sealed contribution objects"
        )
    encoded = [identity.as_dict() for identity in identities]
    if len({(item["uri"], item["generation"]) for item in encoded}) != len(encoded):
        raise ContributionTrainingWorkerError(
            "process lane contains duplicate immutable object identities"
        )
    return {
        "object_count": len(encoded),
        "object_bytes": sum(int(item["size_bytes"]) for item in encoded),
        "object_set_sha256": canonical_sha256(encoded),
    }


def validate_prefetch_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one non-JAX, process-local resident-cache proof."""

    fields = {
        "schema_version",
        "config_sha256",
        "phase",
        "source_revision",
        "capture_manifest_sha256",
        "object_prefix",
        "cache_root",
        "cache_max_bytes",
        "process_partition",
        "object_count",
        "object_bytes",
        "object_set_sha256",
        "elapsed_seconds",
        "resident_only_complete",
        "receipt_sha256",
    }
    selected = dict(value)
    if set(selected) != fields:
        raise ContributionTrainingWorkerError("prefetch receipt fields are not exact")
    claimed = _sha256("prefetch receipt_sha256", selected.pop("receipt_sha256"))
    if selected["schema_version"] != PREFETCH_RECEIPT_SCHEMA_VERSION:
        raise ContributionTrainingWorkerError("prefetch receipt schema is incompatible")
    for name in (
        "config_sha256",
        "capture_manifest_sha256",
        "object_set_sha256",
    ):
        _sha256(f"prefetch {name}", selected[name])
    if (
        not isinstance(selected["source_revision"], str)
        or re.fullmatch(r"[0-9a-f]{40}", selected["source_revision"]) is None
    ):
        raise ContributionTrainingWorkerError(
            "prefetch source_revision must be a 40-character Git commit"
        )
    if not isinstance(selected["phase"], str) or not selected["phase"]:
        raise ContributionTrainingWorkerError("prefetch phase is empty")
    if not isinstance(selected["object_prefix"], str) or not selected[
        "object_prefix"
    ].startswith("gs://"):
        raise ContributionTrainingWorkerError("prefetch object_prefix is invalid")
    cache_root = Path(str(selected["cache_root"]))
    if not cache_root.is_absolute():
        raise ContributionTrainingWorkerError("prefetch cache_root is not absolute")
    _positive_int("prefetch cache_max_bytes", selected["cache_max_bytes"])
    _positive_int("prefetch object_count", selected["object_count"])
    _positive_int("prefetch object_bytes", selected["object_bytes"])
    _finite_nonnegative("prefetch elapsed_seconds", selected["elapsed_seconds"])
    if selected["resident_only_complete"] is not True:
        raise ContributionTrainingWorkerError(
            "prefetch receipt lacks a resident-only completeness proof"
        )
    partition = selected["process_partition"]
    if not isinstance(partition, Mapping) or set(partition) != {
        "strategy",
        "process_count",
        "process_index",
        "range_start",
        "range_end",
        "tokens",
    }:
        raise ContributionTrainingWorkerError(
            "prefetch process partition fields are incompatible"
        )
    from moe_sae.contribution_input import PROCESS_PARTITION_STRATEGY

    if partition["strategy"] != PROCESS_PARTITION_STRATEGY:
        raise ContributionTrainingWorkerError(
            "prefetch process partition strategy is incompatible"
        )
    # Uneven lanes cannot recover total_tokens from one lane alone.  Exact
    # topology/range validation is therefore repeated against the phase plan
    # by _prove_prefetch_for_plan; this check still rejects malformed bounds.
    for name in ("process_count", "tokens"):
        _positive_int(f"prefetch process_partition.{name}", partition[name])
    _positive_int(
        "prefetch process_partition.process_index",
        partition["process_index"],
        allow_zero=True,
    )
    _positive_int(
        "prefetch process_partition.range_start",
        partition["range_start"],
        allow_zero=True,
    )
    _positive_int(
        "prefetch process_partition.range_end",
        partition["range_end"],
    )
    if (
        int(partition["process_index"]) >= int(partition["process_count"])
        or int(partition["range_end"]) <= int(partition["range_start"])
        or int(partition["tokens"])
        != int(partition["range_end"]) - int(partition["range_start"])
    ):
        raise ContributionTrainingWorkerError(
            "prefetch process partition bounds are invalid"
        )
    if canonical_sha256(selected) != claimed:
        raise ContributionTrainingWorkerError("prefetch receipt self-hash is invalid")
    return {**selected, "receipt_sha256": claimed}


def prefetch_process_lane(
    *,
    config: Experiment2Config,
    phase: str,
    source_revision: str,
    capture_control_root: str | Path,
    object_prefix: str,
    cache_root: str | Path,
    cache_max_bytes: int,
    process_count: int,
    process_index: int,
    recomposition_sidecar: Mapping[str, Any] | None,
    object_store: Any | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Hash-verify and materialize exactly one process-contiguous token lane."""

    from moe_sae.contribution_input import process_partition_identity
    from moe_sae.contribution_storage import ContributionDataset
    from moe_sae.object_store import GCSObjectStore, ShardCache

    config.validate()
    selected_phase = config.phase(phase)
    if selected_phase.layout_set == "none":
        raise ContributionTrainingWorkerError(
            "prefetch requires a phase with SAE training layouts"
        )
    cache_max_bytes = _positive_int("cache_max_bytes", cache_max_bytes)
    partition = process_partition_identity(
        selected_phase.train_tokens,
        process_count=process_count,
        process_index=process_index,
    )
    selected_cache_root = Path(cache_root).expanduser().resolve()
    store = GCSObjectStore() if object_store is None else object_store
    cache = ShardCache(
        selected_cache_root,
        store,
        max_bytes=cache_max_bytes,
        resident_only=False,
    )
    dataset = ContributionDataset(
        capture_control_root,
        object_prefix=object_prefix,
        cache=cache,
        recomposition_sidecar=recomposition_sidecar,
    )
    if int(dataset.total_tokens) != int(selected_phase.train_tokens):
        raise ContributionTrainingWorkerError(
            "prefetch requires the exact phase-specific sealed train capture"
        )
    parents = getattr(config, "parents", None)
    expected_manifest = (
        parents.get("train_target_manifest_sha256")
        if isinstance(parents, Mapping)
        else None
    )
    if (
        expected_manifest is not None
        and dataset.manifest_sha256 != expected_manifest
    ):
        raise ContributionTrainingWorkerError(
            "prefetch capture differs from the config-pinned train manifest"
        )
    summary = _prefetch_object_summary(
        dataset,
        range_start=int(partition["range_start"]),
        range_end=int(partition["range_end"]),
    )
    if int(summary["object_bytes"]) > cache_max_bytes:
        raise ContributionTrainingWorkerError(
            "process lane objects exceed resident-cache max bytes; provision "
            "larger local storage before prefetch"
        )
    started = clock()
    try:
        materialized = dataset.prefetch_token_range(
            int(partition["range_start"]),
            int(partition["range_end"]),
        )
        resident_cache = ShardCache(
            selected_cache_root,
            _ResidentOnlyObjectStore(),
            max_bytes=cache_max_bytes,
            resident_only=True,
        )
        resident_dataset = ContributionDataset(
            capture_control_root,
            object_prefix=object_prefix,
            cache=resident_cache,
            recomposition_sidecar=recomposition_sidecar,
        )
        resident = resident_dataset.require_resident_token_range(
            int(partition["range_start"]),
            int(partition["range_end"]),
        )
    except Exception as error:
        raise ContributionTrainingWorkerError(
            "process-lane prefetch or resident-only proof failed"
        ) from error
    elapsed = max(0.0, float(clock() - started))
    expected_counts = {
        "objects": int(summary["object_count"]),
        "bytes": int(summary["object_bytes"]),
    }
    if materialized != expected_counts or resident != expected_counts:
        raise ContributionTrainingWorkerError(
            "prefetch/materialized/resident object counts differ"
        )
    unsigned = {
        "schema_version": PREFETCH_RECEIPT_SCHEMA_VERSION,
        "config_sha256": config.config_hash,
        "phase": selected_phase.name,
        "source_revision": source_revision,
        "capture_manifest_sha256": dataset.manifest_sha256,
        "object_prefix": object_prefix.rstrip("/"),
        "cache_root": str(selected_cache_root),
        "cache_max_bytes": cache_max_bytes,
        "process_partition": partition,
        **summary,
        "elapsed_seconds": elapsed,
        "resident_only_complete": True,
    }
    return validate_prefetch_receipt(
        {**unsigned, "receipt_sha256": canonical_sha256(unsigned)}
    )


def _prove_prefetch_for_plan(
    receipt: Mapping[str, Any],
    *,
    config: Experiment2Config,
    plan: TrainingWorkerPlan,
    source_revision: str,
    capture_control_root: str | Path,
    object_prefix: str,
    cache_root: str | Path,
    cache_max_bytes: int,
    recomposition_sidecar: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Re-derive one receipt and prove it remains complete without GCS."""

    from moe_sae.contribution_input import process_partition_identity
    from moe_sae.contribution_storage import ContributionDataset
    from moe_sae.object_store import ShardCache

    selected = validate_prefetch_receipt(receipt)
    partition = selected["process_partition"]
    expected_partition = process_partition_identity(
        plan.total_tokens,
        process_count=int(partition["process_count"]),
        process_index=int(partition["process_index"]),
    )
    fixed = {
        "config_sha256": config.config_hash,
        "phase": plan.phase_name,
        "source_revision": source_revision,
        "object_prefix": object_prefix.rstrip("/"),
        "cache_root": str(Path(cache_root).expanduser().resolve()),
        "cache_max_bytes": int(cache_max_bytes),
        "process_partition": expected_partition,
    }
    if any(selected[name] != value for name, value in fixed.items()):
        raise ContributionTrainingWorkerError(
            "prefetch receipt differs from the requested training workload"
        )
    cache = ShardCache(
        cache_root,
        _ResidentOnlyObjectStore(),
        max_bytes=cache_max_bytes,
        resident_only=True,
    )
    dataset = ContributionDataset(
        capture_control_root,
        object_prefix=object_prefix,
        cache=cache,
        recomposition_sidecar=recomposition_sidecar,
    )
    if (
        dataset.manifest_sha256 != selected["capture_manifest_sha256"]
        or int(dataset.total_tokens) != plan.total_tokens
    ):
        raise ContributionTrainingWorkerError(
            "prefetch receipt and sealed contribution capture differ"
        )
    parents = getattr(config, "parents", None)
    expected_manifest = (
        parents.get("train_target_manifest_sha256")
        if isinstance(parents, Mapping)
        else None
    )
    if (
        expected_manifest is not None
        and dataset.manifest_sha256 != expected_manifest
    ):
        raise ContributionTrainingWorkerError(
            "resident capture differs from the config-pinned train manifest"
        )
    summary = _prefetch_object_summary(
        dataset,
        range_start=int(expected_partition["range_start"]),
        range_end=int(expected_partition["range_end"]),
    )
    for name in ("object_count", "object_bytes", "object_set_sha256"):
        if selected[name] != summary[name]:
            raise ContributionTrainingWorkerError(
                "prefetch receipt object set differs from the sealed process lane"
            )
    try:
        resident = dataset.require_resident_token_range(
            int(expected_partition["range_start"]),
            int(expected_partition["range_end"]),
        )
    except Exception as error:
        raise ContributionTrainingWorkerError(
            "prefetched process lane is no longer resident and hash-valid"
        ) from error
    if resident != {
        "objects": int(summary["object_count"]),
        "bytes": int(summary["object_bytes"]),
    }:
        raise ContributionTrainingWorkerError(
            "resident process-lane proof has inconsistent counts"
        )
    return selected


def validate_transcoder_prefetch_receipt(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one paired target+source resident-cache proof."""

    fields = {
        "schema_version",
        "config_sha256",
        "phase",
        "source_revision",
        "target_manifest_sha256",
        "source_manifest_sha256",
        "target_object_prefix",
        "source_object_prefix",
        "cache_root",
        "cache_max_bytes",
        "process_partition",
        "target_objects",
        "source_objects",
        "combined_object_count",
        "combined_object_bytes",
        "combined_object_set_sha256",
        "elapsed_seconds",
        "resident_only_complete",
        "receipt_sha256",
    }
    selected = dict(value)
    if set(selected) != fields:
        raise ContributionTrainingWorkerError(
            "transcoder prefetch receipt fields are not exact"
        )
    claimed = _sha256(
        "transcoder prefetch receipt_sha256",
        selected.pop("receipt_sha256"),
    )
    if selected["schema_version"] != TRANSCODER_PREFETCH_RECEIPT_SCHEMA_VERSION:
        raise ContributionTrainingWorkerError(
            "transcoder prefetch receipt schema is incompatible"
        )
    for name in (
        "config_sha256",
        "target_manifest_sha256",
        "source_manifest_sha256",
        "combined_object_set_sha256",
    ):
        _sha256(f"transcoder prefetch {name}", selected[name])
    if (
        not isinstance(selected["source_revision"], str)
        or re.fullmatch(r"[0-9a-f]{40}", selected["source_revision"]) is None
    ):
        raise ContributionTrainingWorkerError(
            "transcoder prefetch source revision is invalid"
        )
    if not isinstance(selected["phase"], str) or not selected["phase"]:
        raise ContributionTrainingWorkerError("transcoder prefetch phase is empty")
    for name in ("target_object_prefix", "source_object_prefix"):
        if not isinstance(selected[name], str) or not selected[name].startswith(
            "gs://"
        ):
            raise ContributionTrainingWorkerError(
                f"transcoder prefetch {name} is invalid"
            )
    if not Path(str(selected["cache_root"])).is_absolute():
        raise ContributionTrainingWorkerError(
            "transcoder prefetch cache_root is not absolute"
        )
    _positive_int(
        "transcoder prefetch cache_max_bytes",
        selected["cache_max_bytes"],
    )
    partition = selected["process_partition"]
    if not isinstance(partition, Mapping):
        raise ContributionTrainingWorkerError(
            "transcoder prefetch process partition is malformed"
        )
    from moe_sae.contribution_input import PROCESS_PARTITION_STRATEGY

    if (
        set(partition)
        != {
            "strategy",
            "process_count",
            "process_index",
            "range_start",
            "range_end",
            "tokens",
        }
        or partition.get("strategy") != PROCESS_PARTITION_STRATEGY
    ):
        raise ContributionTrainingWorkerError(
            "transcoder prefetch process partition is incompatible"
        )
    for name in ("process_count", "range_end", "tokens"):
        _positive_int(f"transcoder prefetch partition {name}", partition[name])
    for name in ("process_index", "range_start"):
        _positive_int(
            f"transcoder prefetch partition {name}",
            partition[name],
            allow_zero=True,
        )
    if (
        int(partition["process_index"]) >= int(partition["process_count"])
        or int(partition["range_end"]) <= int(partition["range_start"])
        or int(partition["tokens"])
        != int(partition["range_end"]) - int(partition["range_start"])
    ):
        raise ContributionTrainingWorkerError(
            "transcoder prefetch process partition bounds are invalid"
        )
    for role in ("target", "source"):
        summary = selected[f"{role}_objects"]
        if not isinstance(summary, Mapping) or set(summary) != {
            "object_count",
            "object_bytes",
            "object_set_sha256",
        }:
            raise ContributionTrainingWorkerError(
                f"transcoder prefetch {role} object summary is malformed"
            )
        _positive_int(f"{role} object_count", summary["object_count"])
        _positive_int(f"{role} object_bytes", summary["object_bytes"])
        _sha256(f"{role} object_set_sha256", summary["object_set_sha256"])
    expected_count = sum(
        int(selected[f"{role}_objects"]["object_count"])
        for role in ("target", "source")
    )
    expected_bytes = sum(
        int(selected[f"{role}_objects"]["object_bytes"])
        for role in ("target", "source")
    )
    expected_set_sha = canonical_sha256(
        {
            "target": selected["target_objects"]["object_set_sha256"],
            "source": selected["source_objects"]["object_set_sha256"],
        }
    )
    if (
        selected["combined_object_count"] != expected_count
        or selected["combined_object_bytes"] != expected_bytes
        or selected["combined_object_set_sha256"] != expected_set_sha
    ):
        raise ContributionTrainingWorkerError(
            "transcoder prefetch combined object totals are inconsistent"
        )
    if expected_bytes > int(selected["cache_max_bytes"]):
        raise ContributionTrainingWorkerError(
            "transcoder prefetch object bytes exceed cache capacity"
        )
    if selected["resident_only_complete"] is not True:
        raise ContributionTrainingWorkerError(
            "transcoder prefetch lacks a resident-only completeness proof"
        )
    _finite_nonnegative(
        "transcoder prefetch elapsed_seconds",
        selected["elapsed_seconds"],
    )
    if canonical_sha256(selected) != claimed:
        raise ContributionTrainingWorkerError(
            "transcoder prefetch receipt self-hash is invalid"
        )
    return {**selected, "receipt_sha256": claimed}


def _require_transcoder_overlay_capture_contract(
    config: Any,
    source_dataset: Any,
) -> None:
    parents = getattr(config, "parents", None)
    expected = (
        parents.get("source_overlay_capture_contract_sha256")
        if isinstance(parents, Mapping)
        else None
    )
    try:
        from moe_sae.contribution_transcoder_input import (
            ContributionTranscoderInputError,
            require_overlay_capture_contract,
        )

        require_overlay_capture_contract(source_dataset, expected)
    except ContributionTranscoderInputError as error:
        raise ContributionTrainingWorkerError(str(error)) from error


def prefetch_transcoder_process_lane(
    *,
    config: Any,
    phase: str,
    source_revision: str,
    capture_control_root: str | Path,
    object_prefix: str,
    source_overlay_control_root: str | Path,
    source_overlay_object_prefix: str,
    cache_root: str | Path,
    cache_max_bytes: int,
    process_count: int,
    process_index: int,
    recomposition_sidecar: Mapping[str, Any] | None,
    object_store: Any | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Materialize and re-prove one exact paired process lane."""

    from moe_sae.contribution_input import process_partition_identity
    from moe_sae.contribution_storage import ContributionDataset
    from moe_sae.contribution_transcoder_storage import RoutedInputDataset
    from moe_sae.object_store import GCSObjectStore, ShardCache

    config.validate()
    selected_phase = config.phase(phase)
    cache_max_bytes = _positive_int("cache_max_bytes", cache_max_bytes)
    partition = process_partition_identity(
        selected_phase.train_tokens,
        process_count=process_count,
        process_index=process_index,
    )
    selected_cache_root = Path(cache_root).expanduser().resolve()
    store = GCSObjectStore() if object_store is None else object_store
    cache = ShardCache(
        selected_cache_root,
        store,
        max_bytes=cache_max_bytes,
        resident_only=False,
    )
    target = ContributionDataset(
        capture_control_root,
        object_prefix=object_prefix,
        cache=cache,
        recomposition_sidecar=recomposition_sidecar,
    )
    if (
        int(target.total_tokens) != selected_phase.train_tokens
        or target.manifest_sha256 != config.parents["train_target_manifest_sha256"]
    ):
        raise ContributionTrainingWorkerError(
            "transcoder prefetch target differs from its config-pinned parent"
        )
    source = RoutedInputDataset(
        source_overlay_control_root,
        object_prefix=source_overlay_object_prefix,
        cache=cache,
        target_manifest=target.manifest,
    )
    _require_transcoder_overlay_capture_contract(config, source)
    if (
        int(source.total_tokens) != selected_phase.train_tokens
        or source.manifest_sha256
        != config.parents["train_source_overlay_manifest_sha256"]
    ):
        raise ContributionTrainingWorkerError(
            "transcoder prefetch source differs from its config-pinned parent"
        )
    start = int(partition["range_start"])
    end = int(partition["range_end"])
    target_summary = _prefetch_object_summary(
        target,
        range_start=start,
        range_end=end,
    )
    source_summary = _prefetch_object_summary(
        source,
        range_start=start,
        range_end=end,
    )
    combined_bytes = int(target_summary["object_bytes"]) + int(
        source_summary["object_bytes"]
    )
    if combined_bytes > cache_max_bytes:
        raise ContributionTrainingWorkerError(
            "paired process-lane objects exceed resident cache capacity"
        )
    started = clock()
    try:
        target_materialized = target.prefetch_token_range(start, end)
        source_materialized = source.prefetch_token_range(start, end)
        resident_cache = ShardCache(
            selected_cache_root,
            _ResidentOnlyObjectStore(),
            max_bytes=cache_max_bytes,
            resident_only=True,
        )
        resident_target = ContributionDataset(
            capture_control_root,
            object_prefix=object_prefix,
            cache=resident_cache,
            recomposition_sidecar=recomposition_sidecar,
        )
        resident_source = RoutedInputDataset(
            source_overlay_control_root,
            object_prefix=source_overlay_object_prefix,
            cache=resident_cache,
            target_manifest=resident_target.manifest,
        )
        target_resident = resident_target.require_resident_token_range(start, end)
        source_resident = resident_source.require_resident_token_range(start, end)
    except Exception as error:
        raise ContributionTrainingWorkerError(
            "paired process-lane prefetch or resident proof failed"
        ) from error
    for observed, summary in (
        (target_materialized, target_summary),
        (target_resident, target_summary),
        (source_materialized, source_summary),
        (source_resident, source_summary),
    ):
        if observed != {
            "objects": int(summary["object_count"]),
            "bytes": int(summary["object_bytes"]),
        }:
            raise ContributionTrainingWorkerError(
                "paired prefetch object counts differ from sealed manifests"
            )
    combined_set = canonical_sha256(
        {
            "target": target_summary["object_set_sha256"],
            "source": source_summary["object_set_sha256"],
        }
    )
    unsigned = {
        "schema_version": TRANSCODER_PREFETCH_RECEIPT_SCHEMA_VERSION,
        "config_sha256": config.config_hash,
        "phase": selected_phase.name,
        "source_revision": source_revision,
        "target_manifest_sha256": target.manifest_sha256,
        "source_manifest_sha256": source.manifest_sha256,
        "target_object_prefix": object_prefix.rstrip("/"),
        "source_object_prefix": source_overlay_object_prefix.rstrip("/"),
        "cache_root": str(selected_cache_root),
        "cache_max_bytes": cache_max_bytes,
        "process_partition": partition,
        "target_objects": target_summary,
        "source_objects": source_summary,
        "combined_object_count": int(target_summary["object_count"])
        + int(source_summary["object_count"]),
        "combined_object_bytes": combined_bytes,
        "combined_object_set_sha256": combined_set,
        "elapsed_seconds": max(0.0, float(clock() - started)),
        "resident_only_complete": True,
    }
    return validate_transcoder_prefetch_receipt(
        {**unsigned, "receipt_sha256": canonical_sha256(unsigned)}
    )


def _prove_transcoder_prefetch_for_plan(
    receipt: Mapping[str, Any],
    *,
    config: Any,
    plan: TrainingWorkerPlan,
    source_revision: str,
    capture_control_root: str | Path,
    object_prefix: str,
    source_overlay_control_root: str | Path,
    source_overlay_object_prefix: str,
    cache_root: str | Path,
    cache_max_bytes: int,
    recomposition_sidecar: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Re-derive both object sets and prove the pair remains resident."""

    from moe_sae.contribution_input import process_partition_identity
    from moe_sae.contribution_storage import ContributionDataset
    from moe_sae.contribution_transcoder_storage import RoutedInputDataset
    from moe_sae.object_store import ShardCache

    selected = validate_transcoder_prefetch_receipt(receipt)
    partition = selected["process_partition"]
    expected_partition = process_partition_identity(
        plan.total_tokens,
        process_count=int(partition["process_count"]),
        process_index=int(partition["process_index"]),
    )
    expected = {
        "config_sha256": config.config_hash,
        "phase": plan.phase_name,
        "source_revision": source_revision,
        "target_object_prefix": object_prefix.rstrip("/"),
        "source_object_prefix": source_overlay_object_prefix.rstrip("/"),
        "cache_root": str(Path(cache_root).expanduser().resolve()),
        "cache_max_bytes": int(cache_max_bytes),
        "process_partition": expected_partition,
    }
    if any(selected[name] != value for name, value in expected.items()):
        raise ContributionTrainingWorkerError(
            "paired prefetch receipt differs from the requested workload"
        )
    cache = ShardCache(
        cache_root,
        _ResidentOnlyObjectStore(),
        max_bytes=cache_max_bytes,
        resident_only=True,
    )
    target = ContributionDataset(
        capture_control_root,
        object_prefix=object_prefix,
        cache=cache,
        recomposition_sidecar=recomposition_sidecar,
    )
    source = RoutedInputDataset(
        source_overlay_control_root,
        object_prefix=source_overlay_object_prefix,
        cache=cache,
        target_manifest=target.manifest,
    )
    _require_transcoder_overlay_capture_contract(config, source)
    if (
        target.manifest_sha256 != selected["target_manifest_sha256"]
        or source.manifest_sha256 != selected["source_manifest_sha256"]
        or target.manifest_sha256 != config.parents["train_target_manifest_sha256"]
        or source.manifest_sha256
        != config.parents["train_source_overlay_manifest_sha256"]
        or int(target.total_tokens) != plan.total_tokens
        or int(source.total_tokens) != plan.total_tokens
    ):
        raise ContributionTrainingWorkerError(
            "paired prefetch receipt and sealed datasets differ"
        )
    start = int(expected_partition["range_start"])
    end = int(expected_partition["range_end"])
    for role, dataset in (("target", target), ("source", source)):
        summary = _prefetch_object_summary(
            dataset,
            range_start=start,
            range_end=end,
        )
        if summary != selected[f"{role}_objects"]:
            raise ContributionTrainingWorkerError(
                f"paired prefetch {role} object set changed"
            )
        try:
            resident = dataset.require_resident_token_range(start, end)
        except Exception as error:
            raise ContributionTrainingWorkerError(
                f"paired prefetch {role} lane is no longer resident"
            ) from error
        if resident != {
            "objects": int(summary["object_count"]),
            "bytes": int(summary["object_bytes"]),
        }:
            raise ContributionTrainingWorkerError(
                f"paired prefetch {role} resident proof is inconsistent"
            )
    return selected


@dataclass(slots=True)
class _PreparedHandle:
    trainer: Any
    lowered: Any
    device_batch: Any
    mean: Any
    process_local: bool
    experiment5_recomposition: Any | None = None


@dataclass(frozen=True, slots=True)
class _ProductionWorkload:
    scientific_model: ScientificModel
    source: Any
    mean: Any
    routed_layout: Any | None
    q_config: Any | None
    optimizer: Any | None
    distributed: Any
    mesh: Any
    bank_capacity: int | None
    arithmetic: SAEArithmetic
    workload_context: Mapping[str, Any]
    checkpoint_training_config: Any
    preprocessing_wall_seconds: float
    experiment5_recomposition: Any | None

    @property
    def expected_l0(self) -> int:
        if self.routed_layout is not None:
            return int(self.routed_layout.total_l0)
        assert self.q_config is not None
        return int(self.q_config.k)


def _collective_records(value: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """All-gather bounded canonical JSON records in process-index order."""

    import jax

    payload = canonical_json_bytes(dict(value))
    if len(payload) > 1024 * 1024:
        raise ContributionTrainingWorkerError(
            "collective training identity record exceeds one MiB"
        )
    if jax.process_count() == 1:
        return (dict(value),)
    from jax.experimental import multihost_utils

    lengths = np.asarray(
        multihost_utils.process_allgather(
            np.asarray([len(payload)], dtype=np.int32),
            tiled=False,
        ),
        dtype=np.int32,
    ).reshape(-1)
    if lengths.size != jax.process_count() or np.any(lengths <= 0):
        raise ContributionTrainingWorkerError(
            "collective training identity lengths are invalid"
        )
    width = int(np.max(lengths))
    local = np.zeros((width,), dtype=np.uint8)
    local[: len(payload)] = np.frombuffer(payload, dtype=np.uint8)
    gathered = np.asarray(
        multihost_utils.process_allgather(local, tiled=False),
        dtype=np.uint8,
    ).reshape(jax.process_count(), width)
    records: list[dict[str, Any]] = []
    for index, length in enumerate(lengths):
        try:
            decoded = json.loads(gathered[index, : int(length)].tobytes())
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ContributionTrainingWorkerError(
                "collective training identity record cannot be decoded"
            ) from error
        if not isinstance(decoded, dict):
            raise ContributionTrainingWorkerError(
                "collective training identity record is not an object"
            )
        records.append(decoded)
    return tuple(records)


def _collective_max_seconds(value: float) -> float:
    import jax

    if jax.process_count() == 1:
        return float(value)
    from jax.experimental import multihost_utils

    gathered = multihost_utils.process_allgather(
        np.asarray([value], dtype=np.float64),
        tiled=False,
    )
    return float(np.max(np.asarray(gathered, dtype=np.float64)))


def _collective_barrier(label: str) -> None:
    import jax

    if jax.process_count() > 1:
        from jax.experimental import multihost_utils

        multihost_utils.sync_global_devices(label)


def _state_sha256(state: Any) -> str:
    """Hash scalable on-device sufficient checksums of every state leaf.

    The qualifier separately performs an exact device-side leaf comparison
    between transition and replay.  This digest is the compact durable name of
    that equality result; it deliberately avoids copying multi-gigabyte F
    parameters and Adam moments to host merely to hash their bytes.
    """

    import jax
    import jax.numpy as jnp

    records: list[dict[str, Any]] = []
    for leaf_index, leaf in enumerate(jax.tree_util.tree_leaves(state)):
        value = jnp.asarray(leaf)
        as_float = value.astype(jnp.float32)
        finite = jnp.all(jnp.isfinite(as_float))
        summary = jax.device_get(
            (
                jnp.sum(as_float, dtype=jnp.float32),
                jnp.sum(as_float * as_float, dtype=jnp.float32),
                jnp.min(as_float),
                jnp.max(as_float),
                finite,
            )
        )
        if not bool(summary[4]):
            raise ContributionTrainingWorkerError(
                "qualification state checksum observed non-finite state"
            )
        records.append(
            {
                "leaf": leaf_index,
                "shape": list(leaf.shape),
                "dtype": str(leaf.dtype),
                "sum": float(summary[0]),
                "sum_squares": float(summary[1]),
                "minimum": float(summary[2]),
                "maximum": float(summary[3]),
            }
        )
    return canonical_sha256(
        {
            "schema_version": "gemma4-contribution-sae-state-summary-v1",
            "leaves": records,
        }
    )


def _states_exact_equal(left: Any, right: Any) -> bool:
    import jax
    import jax.numpy as jnp

    left_leaves = jax.tree_util.tree_leaves(left)
    right_leaves = jax.tree_util.tree_leaves(right)
    if len(left_leaves) != len(right_leaves):
        return False
    checks = [
        jnp.array_equal(left_leaf, right_leaf)
        for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True)
    ]
    return all(bool(value) for value in jax.device_get(checks))


def _stablehlo_manifest(
    lowered: Any,
    *,
    condition: str,
    mode: InputMode,
    scientific_model: ScientificModel,
) -> dict[str, Any]:
    compiler_ir = getattr(lowered, "compiler_ir", None)
    if not callable(compiler_ir):
        raise ContributionTrainingWorkerError(
            "lowered SAE graph does not expose StableHLO"
        )
    try:
        stablehlo = str(compiler_ir(dialect="stablehlo"))
    except TypeError:
        stablehlo = str(compiler_ir())
    if not stablehlo or "stablehlo" not in stablehlo.lower():
        raise ContributionTrainingWorkerError(
            "lowered SAE graph produced no StableHLO text"
        )
    unsigned = {
        "schema_version": TRAINING_GRAPH_SCHEMA_VERSION,
        "condition": condition,
        "input_mode": mode,
        "scientific_model": scientific_model,
        "stablehlo_sha256": hashlib.sha256(stablehlo.encode("utf-8")).hexdigest(),
        "stablehlo_bytes": len(stablehlo.encode("utf-8")),
    }
    return {
        **unsigned,
        "graph_manifest_sha256": canonical_sha256(unsigned),
    }


def _hardware_manifest(
    mesh: Any,
    *,
    peak_flops_per_second: int,
    accelerator_type: str = "v4-32",
) -> dict[str, Any]:
    import jax

    devices = sorted(
        jax.devices(),
        key=lambda item: (int(item.process_index), int(item.id)),
    )
    records = []
    for device in devices:
        records.append(
            {
                "id": int(device.id),
                "process_index": int(device.process_index),
                "platform": str(device.platform),
                "device_kind": str(device.device_kind),
                "coords": (
                    [int(value) for value in device.coords]
                    if getattr(device, "coords", None) is not None
                    else None
                ),
                "core_on_chip": (
                    int(device.core_on_chip)
                    if getattr(device, "core_on_chip", None) is not None
                    else None
                ),
            }
        )
    unsigned = {
        "schema_version": TRAINING_HARDWARE_SCHEMA_VERSION,
        "accelerator_type": accelerator_type,
        "global_chip_count": len(devices),
        "process_count": int(jax.process_count()),
        "local_device_count": int(jax.local_device_count()),
        "peak_flops_per_second": _positive_int(
            "peak_flops_per_second",
            peak_flops_per_second,
        ),
        "mesh_shape": {name: int(size) for name, size in mesh.shape.items()},
        "mesh_axis_names": list(mesh.axis_names),
        "devices": records,
    }
    return {
        **unsigned,
        "hardware_manifest_sha256": canonical_sha256(unsigned),
    }


def _execution_config(
    config: Experiment2Config,
    plan: TrainingWorkerPlan,
    workload: _ProductionWorkload,
    *,
    mode: InputMode,
    source_revision: str,
) -> dict[str, Any]:
    import jax

    try:
        import jaxlib

        jaxlib_version = str(jaxlib.__version__)
    except (ImportError, AttributeError):
        jaxlib_version = "unknown"
    unsigned = {
        "schema_version": TRAINING_EXECUTION_CONFIG_SCHEMA_VERSION,
        "experiment_config_sha256": config.config_hash,
        "maxtext_fork_commit": config.provenance.maxtext_fork_commit,
        "source_revision": source_revision,
        "plan": {
            **asdict(plan),
            "checkpoint_tokens": list(plan.checkpoint_tokens),
        },
        "input_mode": mode,
        "rematerialize_microbatch": mode == "rematerialized",
        "routed_layout": (
            asdict(workload.routed_layout)
            if workload.routed_layout is not None
            else None
        ),
        "q_config": (
            asdict(workload.q_config) if workload.q_config is not None else None
        ),
        "optimizer": (
            asdict(workload.optimizer) if workload.optimizer is not None else None
        ),
        "distributed": asdict(workload.distributed),
        "workload_context": dict(workload.workload_context),
        "jax_version": str(jax.__version__),
        "jaxlib_version": jaxlib_version,
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "parameter_residency": "feature_sharded_device",
        "input_residency": (
            "paired_overlay_and_sealed_v2_resident_only_cache"
            if workload.scientific_model == "transcoder"
            else "sealed_v2_resident_only_cache"
        ),
    }
    return {
        **unsigned,
        "execution_config_sha256": canonical_sha256(unsigned),
    }


class _ProductionEngine:
    def __init__(
        self,
        config: Experiment2Config,
        plan: TrainingWorkerPlan,
        workload: _ProductionWorkload,
        *,
        source_revision: str,
        peak_flops_per_second: int,
        accelerator_type: str = "v4-32",
    ) -> None:
        self.config = config
        self.plan = plan
        self.workload = workload
        self.source_revision = source_revision
        self.peak_flops_per_second = peak_flops_per_second
        self.accelerator_type = accelerator_type

    def initialize_state(self) -> Any:
        import jax

        if self.workload.experiment5_recomposition is not None:
            from moe_sae.experiment5_distributed import (
                init_experiment5_distributed_state,
            )

            assert self.workload.routed_layout is not None
            return init_experiment5_distributed_state(
                jax.random.key(self.plan.seed),
                self.workload.mesh,
                self.workload.routed_layout,
                self.workload.optimizer,
                self.workload.distributed,
            )
        if self.workload.scientific_model == "transcoder":
            from moe_sae.contribution_transcoder_distributed import (
                init_contribution_transcoder_distributed_state,
            )

            assert self.workload.routed_layout is not None
            return init_contribution_transcoder_distributed_state(
                jax.random.key(self.plan.seed),
                self.workload.mesh,
                self.workload.routed_layout,
                self.workload.optimizer,
                self.workload.distributed,
            )
        if self.workload.routed_layout is not None:
            from moe_sae.contribution_distributed import (
                init_contribution_distributed_state,
            )

            return init_contribution_distributed_state(
                jax.random.key(self.plan.seed),
                self.workload.mesh,
                self.workload.routed_layout,
                self.workload.optimizer,
                self.workload.distributed,
            )
        from moe_sae.contribution_shared import init_shared_q_state

        assert self.workload.q_config is not None
        return init_shared_q_state(
            jax.random.key(self.plan.seed),
            self.workload.mesh,
            self.workload.q_config,
            self.workload.distributed,
        )

    def lower(
        self,
        mode: InputMode,
        state: Any,
        batch: Any,
        mean: Any,
    ) -> PreparedTrainingExecutable:
        rematerialize = mode == "rematerialized"
        process_local = int(self.workload.source.process_count) > 1
        if self.workload.experiment5_recomposition is not None:
            from moe_sae.experiment5_distributed import (
                Experiment5DistributedTrainer,
            )

            assert self.workload.routed_layout is not None
            trainer = Experiment5DistributedTrainer(
                self.workload.mesh,
                self.workload.routed_layout,
                self.workload.optimizer,
                self.workload.distributed,
                encoder_kind=self.workload.scientific_model,
                bank_capacity=self.workload.bank_capacity,
                rmsnorm_epsilon=float(
                    self.workload.experiment5_recomposition.epsilon
                ),
                rematerialize_microbatch=rematerialize,
                expected_physical_layout_sha256=self.workload.workload_context[
                    "physical_feature_layout_identity"
                ]["identity_sha256"],
            )
        elif self.workload.scientific_model == "transcoder":
            from moe_sae.contribution_transcoder_distributed import (
                ContributionTranscoderDistributedTrainer,
            )

            assert self.workload.routed_layout is not None
            trainer = ContributionTranscoderDistributedTrainer(
                self.workload.mesh,
                self.workload.routed_layout,
                self.workload.optimizer,
                self.workload.distributed,
                bank_capacity=self.workload.bank_capacity,
                rematerialize_microbatch=rematerialize,
            )
        elif self.workload.routed_layout is not None:
            from moe_sae.contribution_distributed import (
                ContributionDistributedTrainer,
            )

            trainer = ContributionDistributedTrainer(
                self.workload.mesh,
                self.workload.routed_layout,
                self.workload.optimizer,
                self.workload.distributed,
                bank_capacity=self.workload.bank_capacity,
                rematerialize_microbatch=rematerialize,
            )
        else:
            from moe_sae.contribution_shared import SharedQDistributedTrainer

            assert self.workload.q_config is not None
            trainer = SharedQDistributedTrainer(
                self.workload.mesh,
                self.workload.q_config,
                self.workload.distributed,
                rematerialize_microbatch=rematerialize,
            )
        if self.workload.experiment5_recomposition is not None:
            device_batch, device_constants = trainer._prepare(
                batch,
                mean,
                self.workload.experiment5_recomposition,
                process_local=process_local,
            )
            lowered = trainer._compiled.lower(
                state,
                device_batch,
                *device_constants,
            )
        else:
            device_batch, device_mean = trainer._prepare(
                batch,
                mean,
                process_local=process_local,
            )
        if self.workload.experiment5_recomposition is None and (
            self.workload.scientific_model == "transcoder"
        ):
            lowered = trainer._compiled.lower(
                state,
                device_batch,
                device_mean.source_mean,
                device_mean.target_mean,
            )
        elif self.workload.experiment5_recomposition is None:
            lowered = trainer._compiled.lower(state, device_batch, device_mean)
        graph = _stablehlo_manifest(
            lowered,
            condition=self.plan.condition,
            mode=mode,
            scientific_model=self.workload.scientific_model,
        )
        hardware = _hardware_manifest(
            self.workload.mesh,
            peak_flops_per_second=self.peak_flops_per_second,
            accelerator_type=self.accelerator_type,
        )
        execution = _execution_config(
            self.config,
            self.plan,
            self.workload,
            mode=mode,
            source_revision=self.source_revision,
        )
        identity = ExactPerformanceIdentity.from_components(
            graph=graph,
            hardware=hardware,
            config=execution,
        )
        return PreparedTrainingExecutable(
            handle=_PreparedHandle(
                trainer=trainer,
                lowered=lowered,
                device_batch=device_batch,
                mean=(
                    mean
                    if self.workload.experiment5_recomposition is not None
                    else device_mean
                ),
                process_local=process_local,
                experiment5_recomposition=self.workload.experiment5_recomposition,
            ),
            identity=identity,
            graph_manifest=graph,
            hardware_manifest=hardware,
            execution_config=execution,
        )

    @staticmethod
    def compile(prepared: PreparedTrainingExecutable) -> float:
        handle: _PreparedHandle = prepared.handle
        started = time.perf_counter()
        handle.trainer._executable = handle.lowered.compile()
        return time.perf_counter() - started

    @staticmethod
    def step(
        prepared: PreparedTrainingExecutable,
        state: Any,
        batch: Any,
        mean: Any,
    ) -> tuple[Any, Mapping[str, Any]]:
        handle: _PreparedHandle = prepared.handle
        if handle.experiment5_recomposition is not None:
            updated, metrics = handle.trainer.step(
                state,
                batch,
                mean,
                handle.experiment5_recomposition,
                process_local=handle.process_local,
            )
        else:
            updated, metrics = handle.trainer.step(
                state,
                batch,
                mean,
                process_local=handle.process_local,
            )
        import jax

        host = jax.device_get(metrics)
        return updated, {
            "loss": float(host.loss),
            "sse": float(host.sse),
            "sst": float(host.sst),
            "l0": float(host.l0),
            "valid_tokens": int(host.valid_tokens),
            "topk_valid": bool(host.topk_valid),
            "grad_norm": float(host.grad_norm),
            "learning_rate": float(host.learning_rate),
        }

    @staticmethod
    def resident_step(
        prepared: PreparedTrainingExecutable,
        state: Any,
    ) -> tuple[Any, Mapping[str, Any]]:
        """Qualifier-only step over the already device-resident static batch."""

        handle: _PreparedHandle = prepared.handle
        if handle.experiment5_recomposition is not None:
            updated, metrics = handle.trainer.step(
                state,
                handle.device_batch,
                handle.mean,
                handle.experiment5_recomposition,
                process_local=False,
            )
        else:
            updated, metrics = handle.trainer.step(
                state,
                handle.device_batch,
                handle.mean,
                process_local=False,
            )
        import jax

        host = jax.device_get(metrics)
        return updated, {
            "loss": float(host.loss),
            "sse": float(host.sse),
            "sst": float(host.sst),
            "l0": float(host.l0),
            "valid_tokens": int(host.valid_tokens),
            "topk_valid": bool(host.topk_valid),
            "grad_norm": float(host.grad_norm),
            "learning_rate": float(host.learning_rate),
        }

    @staticmethod
    def synchronize(state: Any) -> None:
        import jax

        jax.block_until_ready(state)

    @staticmethod
    def progress(state: Any) -> tuple[int, int]:
        import jax

        step, tokens = jax.device_get((state.step, state.tokens_seen))
        return int(step), int(tokens)


class _OrbaxCheckpointBackend:
    """Immutable Orbax payloads plus small atomic POSIX control pointers."""

    def __init__(
        self,
        *,
        control_root: str | Path,
        payload_root: str,
        mesh: Any,
        training_config: Any,
        retained_checkpoint_progress: Mapping[int, int] | None = None,
    ) -> None:
        from moe_sae.checkpointing import normalize_payload_root

        self.control_root = Path(control_root).expanduser().resolve()
        self.payload_root = normalize_payload_root(payload_root)
        self.mesh = mesh
        self.training_config = training_config
        progress = dict(retained_checkpoint_progress or {})
        retained = tuple(progress.values())
        if (
            tuple(sorted(set(retained))) != retained
            or any(
                isinstance(tokens, bool)
                or not isinstance(tokens, int)
                or tokens <= 0
                for tokens in retained
            )
        ):
            raise ContributionTrainingWorkerError(
                "retained checkpoint tokens must be unique, ordered, and positive"
            )
        if any(
            isinstance(tokens_seen, bool)
            or not isinstance(tokens_seen, int)
            or tokens_seen < target
            for tokens_seen, target in progress.items()
        ):
            raise ContributionTrainingWorkerError(
                "retained checkpoint progress is invalid"
            )
        self.retained_checkpoint_progress = progress

    def restore(
        self,
        template: Any,
        run_identity: Mapping[str, Any],
    ) -> RestoredTrainingState | None:
        import jax

        from moe_sae.checkpointing import pointer_path
        from moe_sae.distributed import restore_distributed_checkpoint

        pointer = pointer_path(self.control_root, "latest")
        local_exists = pointer.is_file()
        statuses = _collective_records(
            {
                "process_index": int(jax.process_index()),
                "latest_exists": local_exists,
            }
        )
        if len({bool(item["latest_exists"]) for item in statuses}) != 1:
            raise ContributionTrainingWorkerError(
                "checkpoint latest-pointer visibility differs across processes"
            )
        if not local_exists:
            return None
        restored, metadata = restore_distributed_checkpoint(
            pointer,
            template,
            self.mesh,
            self.training_config,
            checkpoint_payload_root=self.payload_root,
            checkpoint_identity=run_identity,
            checkpoint_metadata_identity={
                "run_identity_sha256": run_identity["run_identity_sha256"]
            },
        )
        return RestoredTrainingState(restored, metadata)

    def save(
        self,
        state: Any,
        run_identity: Mapping[str, Any],
        timing: Mapping[str, Any],
        final: bool,
    ) -> Mapping[str, Any]:
        import jax
        from jax.experimental import multihost_utils

        from moe_sae.checkpointing import (
            checkpoint_payload_uri,
            final_pointer_from_latest,
            make_pointer,
            milestone_pointer_path,
            new_attempt_id,
            pointer_path,
            publish_milestone_pointer,
            publish_pointer,
        )
        from moe_sae.distributed import save_distributed_checkpoint

        local_attempt = (
            bytes.fromhex(new_attempt_id()) if jax.process_index() == 0 else bytes(16)
        )
        attempt_bytes = np.asarray(
            multihost_utils.broadcast_one_to_all(
                np.frombuffer(local_attempt, dtype=np.uint8).copy(),
                is_source=jax.process_index() == 0,
            ),
            dtype=np.uint8,
        ).tobytes()
        attempt_id = attempt_bytes.hex()
        if attempt_id == "0" * 32:
            raise ContributionTrainingWorkerError(
                "checkpoint coordinator did not publish an attempt ID"
            )
        step, tokens = _ProductionEngine.progress(state)
        payload_uri = checkpoint_payload_uri(
            self.payload_root,
            run_identity,
            step=step,
            tokens_seen=tokens,
            attempt_id=attempt_id,
        )
        metadata = {
            "run_identity_sha256": run_identity["run_identity_sha256"],
            "training_worker_schema_version": TRAINING_WORKER_SCHEMA_VERSION,
            "timing": dict(timing),
        }
        save_distributed_checkpoint(
            payload_uri,
            state,
            self.mesh,
            self.training_config,
            metadata=metadata,
        )
        _collective_barrier(f"contribution-sae-payload-{step}-{tokens}")
        latest = make_pointer(
            kind="latest",
            payload_uri=payload_uri,
            step=step,
            tokens_seen=tokens,
            attempt_id=attempt_id,
            run_identity=run_identity,
        )
        milestone_target = self.retained_checkpoint_progress.get(tokens)
        if jax.process_index() == 0:
            publish_pointer(self.control_root, latest)
            if milestone_target is not None:
                publish_milestone_pointer(
                    self.control_root,
                    latest,
                    checkpoint_target_tokens=milestone_target,
                )
            if final:
                publish_pointer(
                    self.control_root,
                    final_pointer_from_latest(latest),
                )
        _collective_barrier(f"contribution-sae-pointer-{step}-{tokens}")
        # Every process derives the same canonical control paths only after
        # process zero has published them and the global visibility barrier
        # has completed.  Returning process-local ``None`` values here used to
        # make otherwise successful multi-host results hash differently.
        latest_path = str(pointer_path(self.control_root, "latest"))
        final_path = str(pointer_path(self.control_root, "final")) if final else None
        milestone_path = (
            str(milestone_pointer_path(self.control_root, milestone_target))
            if milestone_target is not None
            else None
        )
        return {
            "payload_uri": payload_uri,
            "step": step,
            "tokens": tokens,
            "attempt_id": attempt_id,
            "latest_pointer": latest_path,
            "final_pointer": final_path,
            "milestone_pointer": milestone_path,
            "milestone_target_tokens": milestone_target,
            "final": bool(final),
            "latest_pointer_sha256": latest["pointer_sha256"],
        }


def _pooled_routed_mean(dataset: Any) -> tuple[np.ndarray, str]:
    from moe_sae.contribution_artifacts import validate_contribution_statistics

    manifest = getattr(dataset, "manifest", None)
    statistics = manifest.get("statistics") if isinstance(manifest, Mapping) else None
    if not isinstance(statistics, Mapping):
        raise ContributionTrainingWorkerError(
            "sealed v2 training manifest lacks pooled statistics"
        )
    checked = validate_contribution_statistics(
        statistics,
        expected_tokens=int(dataset.total_tokens),
    )
    pooled = checked["pooled_expert_output"]
    count = int(pooled["count"])
    if count != int(dataset.total_tokens) * 8:
        raise ContributionTrainingWorkerError(
            "pooled expert statistic count differs from tokens × routes"
        )
    mean = (np.asarray(pooled["sum"], dtype=np.float64) / count).astype(np.float32)
    mean = np.ascontiguousarray(mean)
    digest = hashlib.sha256()
    digest.update(b"gemma4-contribution-centering-mean-v1\0")
    digest.update(mean.astype("<f4", copy=False).tobytes(order="C"))
    return mean, digest.hexdigest()


def _phase_layout(config: Experiment2Config, plan: TrainingWorkerPlan) -> Any:
    matches = [
        item
        for item in config.layouts_for_phase(plan.phase_name)
        if item.condition == plan.condition
    ]
    if len(matches) != 1:
        raise ContributionTrainingWorkerError(
            "selected phase/condition does not have one exact layout"
        )
    return matches[0]


def _build_experiment5_workload(
    config: Any,
    plan: TrainingWorkerPlan,
    *,
    scientific_model: ScientificModel,
    capture_control_root: str | Path,
    object_prefix: str,
    source_overlay_control_root: str | Path | None,
    source_overlay_object_prefix: str | None,
    resident_cache_root: str | Path,
    resident_cache_max_bytes: int,
    recomposition_sidecar: Mapping[str, Any] | None,
    prefetch_receipt: Mapping[str, Any],
    expected_context: Mapping[str, Any] | None,
) -> _ProductionWorkload:
    """Build the Experiment-5 full-FFN workload on the sealed parent data."""

    import jax

    from moe_sae.contribution_shared import build_shared_q_centering
    from moe_sae.contribution_storage import ContributionDataset
    from moe_sae.contribution_training import ContributionOptimizerConfig
    from moe_sae.distributed import DistributedConfig, create_mesh
    from moe_sae.experiment5_config import Experiment5Config
    from moe_sae.experiment5_hybrid import (
        Experiment5Centering,
        Experiment5Recomposition,
        experiment5_optimizer_sae_config,
        production_experiment5_layout,
    )
    from moe_sae.experiment5_distributed import experiment5_physical_layout
    from moe_sae.experiment5_input import (
        Experiment5PreprocessingReceipt,
        Experiment5ProcessInput,
        merge_experiment5_preprocessing,
    )
    from moe_sae.object_store import ShardCache
    from moe_sae.pipeline import TrainingConfig

    if not isinstance(config, Experiment5Config):
        raise ContributionTrainingWorkerError(
            "internal Experiment-5 workload dispatch received another config"
        )
    if recomposition_sidecar is None:
        raise ContributionTrainingWorkerError(
            "Experiment 5 requires the exact recomposition sidecar"
        )
    cache = ShardCache(
        resident_cache_root,
        _ResidentOnlyObjectStore(),
        max_bytes=_positive_int(
            "resident_cache_max_bytes",
            resident_cache_max_bytes,
        ),
        resident_only=True,
    )
    target = ContributionDataset(
        capture_control_root,
        object_prefix=object_prefix,
        cache=cache,
        recomposition_sidecar=recomposition_sidecar,
    )
    if (
        int(target.total_tokens) != plan.total_tokens
        or target.manifest_sha256
        != config.parents["train_target_manifest_sha256"]
    ):
        raise ContributionTrainingWorkerError(
            "Experiment-5 target differs from its config-pinned 10M capture"
        )
    source_dataset: Any | None = None
    if scientific_model == "transcoder":
        if source_overlay_control_root is None or source_overlay_object_prefix is None:
            raise ContributionTrainingWorkerError(
                "Experiment-5 transcoder requires its routed-input overlay"
            )
        from moe_sae.contribution_transcoder_input import (
            require_overlay_capture_contract,
            routed_input_mean,
        )
        from moe_sae.contribution_transcoder_storage import RoutedInputDataset

        source_dataset = RoutedInputDataset(
            source_overlay_control_root,
            object_prefix=source_overlay_object_prefix,
            cache=cache,
            target_manifest=target.manifest,
        )
        require_overlay_capture_contract(
            source_dataset,
            config.parents["source_overlay_capture_contract_sha256"],
        )
        if (
            int(source_dataset.total_tokens) != plan.total_tokens
            or source_dataset.manifest_sha256
            != config.parents["train_source_overlay_manifest_sha256"]
        ):
            raise ContributionTrainingWorkerError(
                "Experiment-5 source differs from its config-pinned overlay"
            )

    distributed = DistributedConfig(
        data_axis_size=int(jax.process_count()),
        feature_axis_size=int(jax.local_device_count()),
        microbatches_per_update=plan.microbatches_per_update,
    )
    mesh = create_mesh(distributed)
    layout = production_experiment5_layout(
        plan.condition,
        randomization_key=plan.randomization_key,
    )
    config_layout = _phase_layout(config, plan)
    if (
        layout.total_features != config_layout.total_features
        or layout.candidates_per_token != config_layout.candidates_per_token
        or layout.total_l0 != config_layout.total_l0
    ):
        raise ContributionTrainingWorkerError(
            "Experiment-5 executable layout differs from the frozen config"
        )
    input_source = Experiment5ProcessInput(
        target,
        recomposition_sidecar,
        layout,
        panel=scientific_model,
        total_tokens=plan.total_tokens,
        source_dataset=source_dataset,
        global_microbatch_tokens=plan.global_microbatch_tokens,
        microbatches_per_update=plan.microbatches_per_update,
        process_count=int(jax.process_count()),
        process_index=int(jax.process_index()),
    )
    checked_prefetch = (
        validate_transcoder_prefetch_receipt(prefetch_receipt)
        if scientific_model == "transcoder"
        else validate_prefetch_receipt(prefetch_receipt)
    )
    expected_partition = {
        "strategy": input_source.process_partition_strategy,
        "process_count": input_source.process_count,
        "process_index": input_source.process_index,
        "range_start": input_source.process_range_start,
        "range_end": input_source.process_range_end,
        "tokens": (
            input_source.process_range_end - input_source.process_range_start
        ),
    }
    if checked_prefetch["process_partition"] != expected_partition:
        raise ContributionTrainingWorkerError(
            "Experiment-5 input and resident prefetch partitions differ"
        )
    input_records = _collective_records(asdict(input_source.receipt))
    prefetch_records = _collective_records(checked_prefetch)
    for name, records in (
        ("input", input_records),
        ("prefetch", prefetch_records),
    ):
        observed = [
            int(
                item["process_index"]
                if name == "input"
                else item["process_partition"]["process_index"]
            )
            for item in records
        ]
        if observed != list(range(jax.process_count())):
            raise ContributionTrainingWorkerError(
                f"Experiment-5 {name} receipts are not process ordered"
            )

    preprocessing_started = time.perf_counter()
    local_preprocessing = input_source.scan_local_preprocessing()
    preprocessing_records = _collective_records(asdict(local_preprocessing))
    merged = merge_experiment5_preprocessing(
        tuple(
            Experiment5PreprocessingReceipt(**item)
            for item in preprocessing_records
        ),
        margin_fraction=config.runtime.capacity_margin_fraction,
    )
    preprocessing_wall_seconds = time.perf_counter() - preprocessing_started

    routed_mean, routed_mean_sha = _pooled_routed_mean(target)
    shared_centering = build_shared_q_centering(
        target,
        recomposition_sidecar,
    )
    shared_mean = shared_centering.mean
    if scientific_model == "transcoder":
        assert source_dataset is not None
        source_mean, source_mean_sha = routed_input_mean(source_dataset)
    else:
        source_mean = merged["full_target_mean"]
        source_mean_sha = merged["full_target_mean_sha256"]
    centering = Experiment5Centering(
        source_mean=np.ascontiguousarray(source_mean, dtype=np.float32),
        routed_target_mean=np.ascontiguousarray(routed_mean, dtype=np.float32),
        shared_target_mean=np.ascontiguousarray(shared_mean, dtype=np.float32),
        full_target_mean=np.ascontiguousarray(
            merged["full_target_mean"],
            dtype=np.float32,
        ),
    )
    from moe_sae.contribution_artifacts import bf16_bits_to_float32
    from moe_sae.contribution_recomposition import decode_recomposition_sidecar

    decoded_sidecar = decode_recomposition_sidecar(recomposition_sidecar)
    if decoded_sidecar.sidecar_sha256 != config.recomposition_sidecar_sha256:
        raise ContributionTrainingWorkerError(
            "Experiment-5 recomposition sidecar identity differs from config"
        )
    recomposition = Experiment5Recomposition(
        routed_scale=bf16_bits_to_float32(
            decoded_sidecar.routed_rmsnorm_scale_bf16_bits
        ),
        outer_scale=bf16_bits_to_float32(
            decoded_sidecar.outer_ffn_rmsnorm_scale_bf16_bits
        ),
        epsilon=float(decoded_sidecar.epsilon),
    )
    optimizer = ContributionOptimizerConfig(
        parameter_dtype=config.optimizer.parameter_dtype,
        adam_beta1=config.optimizer.adam_beta1,
        adam_beta2=config.optimizer.adam_beta2,
        adam_eps=config.optimizer.adam_epsilon,
        grad_clip=config.optimizer.global_gradient_clip,
        peak_lr=config.optimizer.peak_learning_rate,
        min_lr=config.optimizer.minimum_learning_rate,
        warmup_steps=plan.warmup_updates,
        total_steps=plan.planned_updates,
    )
    checkpoint_sae = experiment5_optimizer_sae_config(layout, optimizer)
    bank_capacity = (
        int(merged["bank_capacity"]) if layout.is_hybrid else None
    )
    centering_identity = {
        "source_mean_sha256": source_mean_sha,
        "routed_target_mean_sha256": routed_mean_sha,
        "shared_target_mean_sha256": shared_centering.mean_sha256,
        "full_target_mean_sha256": merged["full_target_mean_sha256"],
        "full_target_sst": merged["full_target_sst"],
        "recomposition_sidecar_sha256": decoded_sidecar.sidecar_sha256,
    }
    centering_values = {
        "source_mean": [
            float(item) for item in np.asarray(centering.source_mean, np.float32)
        ],
        "routed_target_mean": [
            float(item)
            for item in np.asarray(centering.routed_target_mean, np.float32)
        ],
        "shared_target_mean": [
            float(item)
            for item in np.asarray(centering.shared_target_mean, np.float32)
        ],
        "full_target_mean": [
            float(item)
            for item in np.asarray(centering.full_target_mean, np.float32)
        ],
    }
    feature_axis_size = int(mesh.shape[distributed.feature_axis])
    physical_layout = experiment5_physical_layout(layout, feature_axis_size)
    workload_context = {
        "experiment": 5,
        "panel": scientific_model,
        "target_manifest_sha256": target.manifest_sha256,
        "source_manifest_sha256": (
            source_dataset.manifest_sha256 if source_dataset is not None else None
        ),
        "input_receipts": list(input_records),
        "input_receipt_set_sha256": canonical_sha256(input_records),
        "prefetch_receipts": list(prefetch_records),
        "prefetch_receipt_set_sha256": canonical_sha256(prefetch_records),
        "preprocessing_receipts": list(preprocessing_records),
        "preprocessing_receipt_set_sha256": canonical_sha256(
            preprocessing_records
        ),
        "centering": centering_identity,
        "centering_values": centering_values,
        "bank_capacity": bank_capacity,
        "routed_bank_component_counts": (
            list(merged["routed_bank_component_counts"])
            if merged["routed_bank_component_counts"] is not None
            else None
        ),
        "routed_bank_component_counts_by_training_tokens": (
            [
                [tokens, list(counts)]
                for tokens, counts in merged[
                    "routed_bank_component_counts_by_training_tokens"
                ]
            ]
            if merged["routed_bank_component_counts_by_training_tokens"]
            is not None
            else None
        ),
        "physical_feature_layout": (
            "balanced_routed_banks_then_shared_slice_per_feature_shard_v1"
            if layout.is_hybrid
            else "contiguous_global_features_v1"
        ),
        "physical_feature_axis_size": feature_axis_size,
        "physical_feature_layout_identity": physical_layout.to_dict(),
        "resident_cache_policy": "resident_only_no_refill",
    }
    if expected_context is not None and workload_context != dict(expected_context):
        raise ContributionTrainingWorkerError(
            "Experiment-5 resident input or preprocessing context drifted"
        )
    arithmetic = SAEArithmetic(
        workload="routed" if layout.is_hybrid else "shared_q",
        condition=layout.condition,
        hidden_dim=layout.hidden_dim,
        candidates_per_token=layout.candidates_per_token,
        selected_features_per_token=layout.total_l0,
    )
    training_config = TrainingConfig(
        sae=checkpoint_sae,
        seed=plan.seed,
        total_tokens=plan.total_tokens,
        effective_batch_tokens=plan.effective_batch_tokens,
        microbatch_tokens=plan.global_microbatch_tokens,
        checkpoint_every_tokens=plan.total_tokens,
        health_gate_tokens=plan.total_tokens,
        a_capacity=bank_capacity,
    )
    return _ProductionWorkload(
        scientific_model=scientific_model,
        source=input_source,
        mean=centering,
        routed_layout=layout,
        q_config=None,
        optimizer=optimizer,
        distributed=distributed,
        mesh=mesh,
        bank_capacity=bank_capacity,
        arithmetic=arithmetic,
        workload_context=workload_context,
        checkpoint_training_config=training_config,
        preprocessing_wall_seconds=preprocessing_wall_seconds,
        experiment5_recomposition=recomposition,
    )


def _build_production_workload(
    config: Any,
    plan: TrainingWorkerPlan,
    *,
    scientific_model: ScientificModel = "sae",
    capture_control_root: str | Path,
    object_prefix: str,
    source_overlay_control_root: str | Path | None = None,
    source_overlay_object_prefix: str | None = None,
    resident_cache_root: str | Path,
    resident_cache_max_bytes: int,
    recomposition_sidecar: Mapping[str, Any] | None,
    prefetch_receipt: Mapping[str, Any],
    expected_context: Mapping[str, Any] | None = None,
) -> _ProductionWorkload:
    """Build a no-refill process-local source and its collective identity."""

    from moe_sae.experiment5_config import Experiment5Config

    if isinstance(config, Experiment5Config):
        return _build_experiment5_workload(
            config,
            plan,
            scientific_model=scientific_model,
            capture_control_root=capture_control_root,
            object_prefix=object_prefix,
            source_overlay_control_root=source_overlay_control_root,
            source_overlay_object_prefix=source_overlay_object_prefix,
            resident_cache_root=resident_cache_root,
            resident_cache_max_bytes=resident_cache_max_bytes,
            recomposition_sidecar=recomposition_sidecar,
            prefetch_receipt=prefetch_receipt,
            expected_context=expected_context,
        )

    import jax

    from moe_sae.contribution_input import (
        CapacityReceipt,
        ContributionProcessInput,
        merge_capacity_receipts,
    )
    from moe_sae.contribution_sae import ContributionSAEConfig
    from moe_sae.contribution_shared import (
        SharedQLayout,
        SharedQProcessInput,
        build_shared_q_centering,
        shared_q_sae_config,
    )
    from moe_sae.contribution_storage import ContributionDataset
    from moe_sae.contribution_training import ContributionOptimizerConfig
    from moe_sae.distributed import DistributedConfig, create_mesh
    from moe_sae.object_store import ShardCache
    from moe_sae.pipeline import TrainingConfig

    cache = ShardCache(
        resident_cache_root,
        _ResidentOnlyObjectStore(),
        max_bytes=_positive_int(
            "resident_cache_max_bytes",
            resident_cache_max_bytes,
        ),
        resident_only=True,
    )
    dataset = ContributionDataset(
        capture_control_root,
        object_prefix=object_prefix,
        cache=cache,
        recomposition_sidecar=recomposition_sidecar,
    )
    if int(dataset.total_tokens) != plan.total_tokens:
        raise ContributionTrainingWorkerError(
            "training requires the exact phase-specific sealed v2 manifest"
        )
    parents = getattr(config, "parents", None)
    expected_target_manifest = (
        parents.get("train_target_manifest_sha256")
        if isinstance(parents, Mapping)
        else None
    )
    if (
        expected_target_manifest is not None
        and dataset.manifest_sha256 != expected_target_manifest
    ):
        raise ContributionTrainingWorkerError(
            "training capture differs from the config-pinned train manifest"
        )
    source_dataset: Any | None = None
    if scientific_model == "transcoder":
        if plan.condition == "Q":
            raise ContributionTrainingWorkerError(
                "Experiment-3 routed transcoder mode excludes shared Q"
            )
        if source_overlay_control_root is None or source_overlay_object_prefix is None:
            raise ContributionTrainingWorkerError(
                "transcoder training requires an explicit routed-input overlay"
            )
        expected_target = config.parents.get("train_target_manifest_sha256")
        if dataset.manifest_sha256 != expected_target:
            raise ContributionTrainingWorkerError(
                "transcoder target differs from the config-pinned parent manifest"
            )
        from moe_sae.contribution_transcoder_storage import RoutedInputDataset

        source_dataset = RoutedInputDataset(
            source_overlay_control_root,
            object_prefix=source_overlay_object_prefix,
            cache=cache,
            target_manifest=dataset.manifest,
        )
        _require_transcoder_overlay_capture_contract(config, source_dataset)
        if (
            int(source_dataset.total_tokens) != plan.total_tokens
            or source_dataset.manifest_sha256
            != config.parents["train_source_overlay_manifest_sha256"]
        ):
            raise ContributionTrainingWorkerError(
                "routed-input overlay differs from the config-pinned parent"
            )
    distributed = DistributedConfig(
        data_axis_size=int(jax.process_count()),
        feature_axis_size=int(jax.local_device_count()),
        microbatches_per_update=plan.microbatches_per_update,
    )
    mesh = create_mesh(distributed)
    selected_layout = _phase_layout(config, plan)
    routed_layout: Any | None = None
    q_config: Any | None = None
    optimizer: Any | None = None
    bank_capacity: int | None = None
    capacity_records: tuple[dict[str, Any], ...] = ()

    if plan.condition == "Q":
        if recomposition_sidecar is None:
            raise ContributionTrainingWorkerError(
                "shared Q requires the recomposition sidecar"
            )
        centering = build_shared_q_centering(dataset, recomposition_sidecar)
        source = SharedQProcessInput(
            dataset,
            recomposition_sidecar,
            centering,
            role="train",
            total_tokens=plan.total_tokens,
            global_microbatch_tokens=plan.global_microbatch_tokens,
            microbatches_per_update=plan.microbatches_per_update,
            process_count=int(jax.process_count()),
            process_index=int(jax.process_index()),
        )
        q_layout = SharedQLayout(
            hidden_dim=config.model.hidden_size,
            total_features=selected_layout.bank_width,
            k=selected_layout.topk_per_component,
        )
        q_config = shared_q_sae_config(
            q_layout,
            parameter_dtype=config.optimizer.parameter_dtype,
            adam_beta1=config.optimizer.adam_beta1,
            adam_beta2=config.optimizer.adam_beta2,
            adam_eps=config.optimizer.adam_epsilon,
            grad_clip=config.optimizer.global_gradient_clip,
            peak_lr=config.optimizer.peak_learning_rate,
            min_lr=config.optimizer.minimum_learning_rate,
            warmup_steps=plan.warmup_updates,
            total_steps=plan.planned_updates,
        )
        mean = centering.mean
        mean_sha256 = centering.mean_sha256
        centering_identity = {
            "kind": "shared_q",
            "contract_sha256": centering.contract_sha256,
            "mean_sha256": centering.mean_sha256,
            "recomposition_sidecar_sha256": (centering.recomposition_sidecar_sha256),
        }
        arithmetic = SAEArithmetic.shared_q(
            hidden_dim=q_layout.hidden_dim,
            total_features=q_layout.total_features,
            k=q_layout.k,
        )
        checkpoint_sae = q_config
    else:
        routed_layout = ContributionSAEConfig(
            condition=plan.condition,
            hidden_dim=config.model.hidden_size,
            num_experts=config.model.num_routed_experts,
            top_routed=config.model.experts_per_token,
            bank_width=selected_layout.bank_width,
            k_per_contribution=selected_layout.topk_per_component,
            randomization_key=plan.randomization_key,
        )
        if scientific_model == "transcoder":
            from moe_sae.contribution_transcoder_input import (
                ContributionTranscoderProcessInput,
                routed_input_mean,
            )

            assert source_dataset is not None
            source = ContributionTranscoderProcessInput(
                source_dataset,
                dataset,
                routed_layout,
                total_tokens=plan.total_tokens,
                global_microbatch_tokens=plan.global_microbatch_tokens,
                microbatches_per_update=plan.microbatches_per_update,
                process_count=int(jax.process_count()),
                process_index=int(jax.process_index()),
            )
        else:
            source = ContributionProcessInput(
                dataset,
                routed_layout,
                total_tokens=plan.total_tokens,
                global_microbatch_tokens=plan.global_microbatch_tokens,
                microbatches_per_update=plan.microbatches_per_update,
                process_count=int(jax.process_count()),
                process_index=int(jax.process_index()),
            )
        optimizer = ContributionOptimizerConfig(
            parameter_dtype=config.optimizer.parameter_dtype,
            adam_beta1=config.optimizer.adam_beta1,
            adam_beta2=config.optimizer.adam_beta2,
            adam_eps=config.optimizer.adam_epsilon,
            grad_clip=config.optimizer.global_gradient_clip,
            peak_lr=config.optimizer.peak_learning_rate,
            min_lr=config.optimizer.minimum_learning_rate,
            warmup_steps=plan.warmup_updates,
            total_steps=plan.planned_updates,
        )
        target_mean, target_mean_sha256 = _pooled_routed_mean(dataset)
        if scientific_model == "transcoder":
            from moe_sae.contribution_transcoder_distributed import (
                ContributionTranscoderCentering,
            )

            assert source_dataset is not None
            source_mean, source_mean_sha256 = routed_input_mean(source_dataset)
            mean = ContributionTranscoderCentering(
                source_mean=source_mean,
                target_mean=target_mean,
            )
            mean_sha256 = canonical_sha256(
                {
                    "schema_version": ("gemma4-contribution-transcoder-centering-v1"),
                    "source_mean_sha256": source_mean_sha256,
                    "target_mean_sha256": target_mean_sha256,
                }
            )
            centering_identity = {
                "kind": "routed_input_to_pooled_raw_expert",
                "source_mean_sha256": source_mean_sha256,
                "target_mean_sha256": target_mean_sha256,
                "contract_sha256": mean_sha256,
            }
        else:
            mean = target_mean
            mean_sha256 = target_mean_sha256
            centering_identity = {
                "kind": "pooled_raw_expert",
                "mean_sha256": mean_sha256,
            }
        arithmetic = SAEArithmetic.from_contribution_layout(routed_layout)
        from moe_sae.contribution_training import optimizer_sae_config

        checkpoint_sae = optimizer_sae_config(routed_layout, optimizer)

    input_records = _collective_records(asdict(source.receipt))
    if [int(item["process_index"]) for item in input_records] != list(
        range(jax.process_count())
    ):
        raise ContributionTrainingWorkerError(
            "collective process input receipts are not process ordered"
        )
    if len({item["receipt_sha256"] for item in input_records}) != len(input_records):
        raise ContributionTrainingWorkerError(
            "collective process input receipts contain duplicates"
        )
    checked_prefetch = (
        validate_transcoder_prefetch_receipt(prefetch_receipt)
        if scientific_model == "transcoder"
        else validate_prefetch_receipt(prefetch_receipt)
    )
    expected_partition = {
        "strategy": source.process_partition_strategy,
        "process_count": source.process_count,
        "process_index": source.process_index,
        "range_start": source.process_range_start,
        "range_end": source.process_range_end,
        "tokens": source.process_range_end - source.process_range_start,
    }
    if checked_prefetch["process_partition"] != expected_partition:
        raise ContributionTrainingWorkerError(
            "prefetch receipt does not match the process-local training lane"
        )
    prefetch_records = _collective_records(checked_prefetch)
    if [
        int(item["process_partition"]["process_index"]) for item in prefetch_records
    ] != list(range(jax.process_count())):
        raise ContributionTrainingWorkerError(
            "collective prefetch receipts are not process ordered"
        )
    if len({item["receipt_sha256"] for item in prefetch_records}) != len(
        prefetch_records
    ):
        raise ContributionTrainingWorkerError(
            "collective prefetch receipts contain duplicates"
        )

    preprocessing_wall_seconds = 0.0
    if plan.condition in {"A", "R"}:
        if expected_context is None:
            preprocessing_started = time.perf_counter()
            local_capacity = source.scan_local_capacity(
                margin_fraction=config.runtime.capacity_margin_fraction,
            )
            capacity_records = _collective_records(asdict(local_capacity))
            capacity_receipts = tuple(
                CapacityReceipt(**item) for item in capacity_records
            )
            merged = merge_capacity_receipts(capacity_receipts)
            bank_capacity = int(merged.capacity)
            merged_capacity = merged.to_dict()
            preprocessing_wall_seconds = time.perf_counter() - preprocessing_started
        else:
            raw_capacity = expected_context.get("capacity_receipts")
            if not isinstance(raw_capacity, list):
                raise ContributionTrainingWorkerError(
                    "qualified A/R context lacks capacity receipts"
                )
            capacity_records = tuple(dict(item) for item in raw_capacity)
            capacity_receipts = tuple(
                CapacityReceipt(**item) for item in capacity_records
            )
            merged = merge_capacity_receipts(capacity_receipts)
            bank_capacity = int(merged.capacity)
            merged_capacity = merged.to_dict()
    else:
        merged_capacity = None

    workload_context = {
        "scientific_model": scientific_model,
        "capture_manifest_sha256": dataset.manifest_sha256,
        "target_manifest_sha256": dataset.manifest_sha256,
        "source_manifest_sha256": (
            source_dataset.manifest_sha256 if source_dataset is not None else None
        ),
        "input_receipts": list(input_records),
        "input_receipt_set_sha256": canonical_sha256(input_records),
        "prefetch_receipts": list(prefetch_records),
        "prefetch_receipt_set_sha256": canonical_sha256(prefetch_records),
        "centering": centering_identity,
        "mean_sha256": mean_sha256,
        "capacity_receipts": list(capacity_records),
        "merged_capacity_receipt": merged_capacity,
        "bank_capacity": bank_capacity,
        "resident_cache_policy": "resident_only_no_refill",
    }
    if expected_context is not None and workload_context != dict(expected_context):
        raise ContributionTrainingWorkerError(
            "resident v2 input/capacity/centering context drifted after qualification"
        )
    training_config = TrainingConfig(
        sae=checkpoint_sae,
        seed=plan.seed,
        total_tokens=plan.total_tokens,
        effective_batch_tokens=plan.effective_batch_tokens,
        microbatch_tokens=plan.global_microbatch_tokens,
        checkpoint_every_tokens=plan.total_tokens,
        health_gate_tokens=plan.total_tokens,
        a_capacity=bank_capacity,
    )
    return _ProductionWorkload(
        scientific_model=scientific_model,
        source=source,
        mean=mean,
        routed_layout=routed_layout,
        q_config=q_config,
        optimizer=optimizer,
        distributed=distributed,
        mesh=mesh,
        bank_capacity=bank_capacity,
        arithmetic=arithmetic,
        workload_context=workload_context,
        checkpoint_training_config=training_config,
        preprocessing_wall_seconds=preprocessing_wall_seconds,
        experiment5_recomposition=None,
    )


def _run_production_qualification(
    config: Experiment2Config,
    plan: TrainingWorkerPlan,
    workload: _ProductionWorkload,
    *,
    source_revision: str,
    peak_flops_per_second: int,
    accelerator_type: str = "v4-32",
    warmup_steps: int,
    measured_steps: int,
    qualification_started: float | None = None,
) -> dict[str, Any]:
    """Compile/replay both exact graphs over one device-resident v2 batch."""

    _positive_int("qualification warmup_steps", warmup_steps)
    _positive_int("qualification measured_steps", measured_steps)
    engine = _ProductionEngine(
        config,
        plan,
        workload,
        source_revision=source_revision,
        peak_flops_per_second=peak_flops_per_second,
        accelerator_type=accelerator_type,
    )
    try:
        static_batch = next(iter(workload.source.iter_updates()))
    except StopIteration as error:
        raise ContributionTrainingWorkerError(
            "qualified training source contains no updates"
        ) from error

    def probe(mode: InputMode) -> SAEQualificationObservation:
        mode_started = time.perf_counter()
        initial = engine.initialize_state()
        prepared = engine.lower(mode, initial, static_batch, workload.mean)
        compile_count = 0
        try:
            engine.compile(prepared)
            compile_count += 1
            warm = initial
            warm_metric: Mapping[str, Any] | None = None
            for _ in range(warmup_steps):
                warm, warm_metric = engine.resident_step(prepared, warm)
            engine.synchronize(warm)
            warm_step, warm_tokens = engine.progress(warm)

            transition = warm
            scientific_tokens = 0
            exact_l0 = True
            measured_started = time.perf_counter()
            for _ in range(measured_steps):
                transition, raw_metric = engine.resident_step(
                    prepared,
                    transition,
                )
                metric = _metric_record(
                    raw_metric,
                    expected_l0=workload.expected_l0,
                )
                scientific_tokens += int(metric["valid_tokens"])
                exact_l0 = exact_l0 and math.isclose(
                    float(metric["l0"]),
                    workload.expected_l0,
                    abs_tol=1e-5,
                )
            engine.synchronize(transition)
            measured_seconds = _collective_max_seconds(
                time.perf_counter() - measured_started
            )
            transition_step, transition_tokens = engine.progress(transition)

            replay = warm
            for _ in range(measured_steps):
                replay, raw_metric = engine.resident_step(prepared, replay)
                _metric_record(
                    raw_metric,
                    expected_l0=workload.expected_l0,
                )
            engine.synchronize(replay)
            replay_step, replay_tokens = engine.progress(replay)
            deterministic = _states_exact_equal(transition, replay)
            transition_sha = _state_sha256(transition)
            replay_sha = _state_sha256(replay)
            if not deterministic or transition_sha != replay_sha:
                raise ContributionTrainingWorkerError(
                    "qualified SAE transition is not exactly replayable"
                )
            state_progress_exact = (
                transition_step == warm_step + measured_steps
                and replay_step == transition_step
                and transition_tokens == warm_tokens + scientific_tokens
                and replay_tokens == transition_tokens
            )
            del warm_metric
            return SAEQualificationObservation(
                identity=prepared.identity,
                qualification_wall_seconds=_collective_max_seconds(
                    time.perf_counter() - mode_started
                ),
                measured_device_seconds=measured_seconds,
                scientific_tokens=scientific_tokens,
                warmup_steps=warmup_steps,
                measured_steps=measured_steps,
                compile_count=compile_count,
                recompilations_after_warmup=0,
                oom_events=0,
                nonfinite_events=0,
                capacity_overflow_events=0,
                exact_l0=exact_l0,
                state_progress_exact=state_progress_exact,
                parameters_resident_before_timing=True,
                inputs_resident_before_timing=True,
                synchronized_device_timing=True,
                transition_output_sha256=transition_sha,
                replay_output_sha256=replay_sha,
            )
        except Exception:  # noqa: TRY203
            raise

    return qualify_training_modes(
        probe,
        workload.arithmetic,
        peak_flops_per_second=peak_flops_per_second,
        minimum_useful_mfu=config.runtime.accelerator_min_useful_mfu,
        maximum_seconds=config.runtime.accelerator_qualification_max_seconds,
        workload_context=workload.workload_context,
        qualification_started=qualification_started,
        elapsed_transform=_collective_max_seconds,
    )


def _run_production_training(
    config: Any,
    plan: TrainingWorkerPlan,
    workload: _ProductionWorkload,
    selection: Mapping[str, Any] | None,
    *,
    source_revision: str,
    peak_flops_per_second: int,
    accelerator_type: str = "v4-32",
    checkpoint_control_root: str | Path,
    checkpoint_payload_root: str,
    maximum_runtime_seconds: int,
    metric_path: str | Path | None,
    direct_bounded_resident: bool = False,
    direct_runtime_max_seconds: int = MAX_UNGATED_ACCELERATOR_SECONDS,
) -> dict[str, Any]:
    engine = _ProductionEngine(
        config,
        plan,
        workload,
        source_revision=source_revision,
        peak_flops_per_second=peak_flops_per_second,
        accelerator_type=accelerator_type,
    )
    checkpoint = _OrbaxCheckpointBackend(
        control_root=checkpoint_control_root,
        payload_root=checkpoint_payload_root,
        mesh=workload.mesh,
        training_config=workload.checkpoint_training_config,
        retained_checkpoint_progress={
            plan.expected_tokens_after(
                math.ceil(target / plan.effective_batch_tokens)
            ): target
            for target in getattr(
                config.phase(plan.phase_name),
                "evaluation_checkpoint_tokens",
                (),
            )
        },
    )
    metric_destination = (
        None if metric_path is None else Path(metric_path).expanduser().resolve()
    )

    def metric_sink(value: Mapping[str, Any]) -> None:
        if metric_destination is None:
            return
        # Process zero is the sole control-plane writer; every process has
        # already synchronized and observed the same globally reduced metric.
        import jax

        if jax.process_index() != 0:
            return
        metric_destination.parent.mkdir(parents=True, exist_ok=True)
        with metric_destination.open("ab", buffering=0) as handle:
            handle.write(canonical_json_bytes(dict(value)) + b"\n")
            os.fsync(handle.fileno())

    dependencies = TrainingLoopDependencies(
        initialize_state=engine.initialize_state,
        lower=engine.lower,
        compile=engine.compile,
        step=engine.step,
        synchronize=engine.synchronize,
        progress=engine.progress,
        restore=checkpoint.restore,
        save=checkpoint.save,
        canonicalize_seconds=_collective_max_seconds,
    )
    result = run_qualified_training(
        plan,
        workload.source,
        workload.mean,
        selection,
        dependencies,
        expected_l0=workload.expected_l0,
        maximum_runtime_seconds=maximum_runtime_seconds,
        metric_sink=metric_sink,
        direct_bounded_resident=direct_bounded_resident,
        direct_runtime_max_seconds=direct_runtime_max_seconds,
        preprocessing_wall_seconds=workload.preprocessing_wall_seconds,
        global_chip_count=int(np.asarray(workload.mesh.devices, dtype=object).size),
    )
    if workload.experiment5_recomposition is None:
        return result
    unsigned = dict(result)
    unsigned.pop("result_sha256")
    unsigned["source_revision"] = source_revision
    unsigned["experiment5_context"] = dict(workload.workload_context)
    unsigned["experiment5_execution"] = {
        "accelerator_type": accelerator_type,
        "peak_flops_per_second": int(peak_flops_per_second),
        "global_chip_count": int(
            np.asarray(workload.mesh.devices, dtype=object).size
        ),
        "physical_feature_layout_identity": dict(
            workload.workload_context["physical_feature_layout_identity"]
        ),
    }
    return {
        **unsigned,
        "result_sha256": canonical_sha256(unsigned),
    }


# ---------------------------------------------------------------------------
# Standalone CLI and hard supervisor


def _source_revision(value: str) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise argparse.ArgumentTypeError(
            "source revision must be a lowercase 40-character Git commit"
        )
    return value


def _absolute_path(value: str) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("path must be absolute")
    return str(path.resolve(strict=False))


def _load_sidecar(path: str | None) -> dict[str, Any] | None:
    return None if path is None else _read_json(path, name="recomposition sidecar")


def _load_worker_config(
    path: str | Path,
    scientific_model: ScientificModel,
) -> Any:
    selected = Path(path).expanduser().resolve()
    try:
        raw = yaml.safe_load(selected.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise ContributionTrainingWorkerError(
            "cannot inspect the selected worker config"
        ) from error
    if (
        isinstance(raw, Mapping)
        and raw.get("schema_version")
        == "gemma4-routed-shared-hybrid-run-config-v1"
    ):
        from moe_sae.experiment5_config import load_experiment5_config

        return load_experiment5_config(
            selected,
            expected_panel=scientific_model,
        )
    if scientific_model == "transcoder":
        from moe_sae.contribution_transcoder_config import (
            load_contribution_transcoder_config,
        )

        return load_contribution_transcoder_config(selected)
    return load_contribution_config(selected)


def _require_transcoder_source_arguments(args: argparse.Namespace) -> None:
    if args.scientific_model != "transcoder":
        return
    if (
        args.source_overlay_control_root is None
        or args.source_overlay_object_prefix is None
    ):
        raise ContributionTrainingWorkerError(
            "transcoder mode requires --source-overlay-control-root and "
            "--source-overlay-object-prefix"
        )


def _verify_repository_revision(expected: str) -> None:
    """Require the child to execute the exact committed, clean source tree."""

    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError) as error:
        raise ContributionTrainingWorkerError(
            "cannot verify the training worker repository revision"
        ) from error
    if head != expected:
        raise ContributionTrainingWorkerError(
            f"source revision drifted: expected {expected}, observed {head}"
        )
    if dirty.strip():
        raise ContributionTrainingWorkerError(
            "authoritative SAE qualification/training requires a clean source tree"
        )


def _add_workload_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--scientific-model",
        "--mode",
        choices=SCIENTIFIC_MODELS,
        default="sae",
    )
    parser.add_argument(
        "--accelerator-type",
        choices=("v4-32", "v6e-16", "v5litepod-16"),
        default="v4-32",
    )
    parser.add_argument(
        "--config",
        default="configs/gemma4_contribution_v1.yaml",
    )
    parser.add_argument(
        "--phase",
        default="phase2_mechanism_pilot",
    )
    parser.add_argument("--condition", required=True, choices=CONDITIONS)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument(
        "--capture-control-root",
        required=True,
        type=_absolute_path,
    )
    parser.add_argument("--object-prefix", required=True)
    parser.add_argument(
        "--source-overlay-control-root",
        type=_absolute_path,
    )
    parser.add_argument("--source-overlay-object-prefix")
    parser.add_argument(
        "--resident-cache-root",
        required=True,
        type=_absolute_path,
    )
    parser.add_argument(
        "--resident-cache-max-bytes",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--prefetch-receipt-path",
        required=True,
        type=_absolute_path,
    )
    parser.add_argument(
        "--recomposition-sidecar",
        type=_absolute_path,
    )
    parser.add_argument(
        "--source-revision",
        required=True,
        type=_source_revision,
    )
    parser.add_argument(
        "--peak-flops-per-second",
        type=int,
        default=4_400_000_000_000_000,
    )


def _add_prefetch_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--scientific-model",
        "--mode",
        choices=SCIENTIFIC_MODELS,
        default="sae",
    )
    parser.add_argument(
        "--config",
        default="configs/gemma4_contribution_v1.yaml",
    )
    parser.add_argument(
        "--accelerator-type",
        choices=("v4-32", "v6e-16", "v5litepod-16"),
        default="v4-32",
    )
    parser.add_argument(
        "--phase",
        default="phase2_mechanism_pilot",
    )
    parser.add_argument(
        "--capture-control-root",
        required=True,
        type=_absolute_path,
    )
    parser.add_argument("--object-prefix", required=True)
    parser.add_argument(
        "--source-overlay-control-root",
        type=_absolute_path,
    )
    parser.add_argument("--source-overlay-object-prefix")
    parser.add_argument(
        "--resident-cache-root",
        required=True,
        type=_absolute_path,
    )
    parser.add_argument(
        "--resident-cache-max-bytes",
        required=True,
        type=int,
    )
    parser.add_argument("--process-count", required=True, type=int)
    parser.add_argument("--process-index", required=True, type=int)
    parser.add_argument(
        "--prefetch-receipt-path",
        required=True,
        type=_absolute_path,
    )
    parser.add_argument(
        "--recomposition-sidecar",
        type=_absolute_path,
    )
    parser.add_argument(
        "--source-revision",
        required=True,
        type=_source_revision,
    )
    parser.add_argument(
        "--regional-mirror-seal",
        type=_absolute_path,
        help=(
            "optional exact regional mirror seal; objects retain their "
            "original scientific identities in the resident cache"
        ),
    )


def _add_supervisor_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--supervision-dir",
        required=True,
        type=_absolute_path,
    )
    parser.add_argument("--host-id", required=True)
    parser.add_argument(
        "--working-directory",
        required=True,
        type=_absolute_path,
    )
    parser.add_argument(
        "--termination-grace-seconds",
        type=float,
        default=15.0,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="moe-sae-contribution-training",
        description="Qualified Experiment-2 SAE training worker",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    prefetch = commands.add_parser(
        "prefetch",
        help="hard-supervise one process-lane prefetch without JAX",
    )
    _add_prefetch_arguments(prefetch)
    _add_supervisor_arguments(prefetch)
    prefetch.add_argument(
        "--maximum-runtime-seconds",
        type=int,
        default=MAX_QUALIFICATION_SECONDS,
    )

    prefetch_child = commands.add_parser(
        "prefetch-child",
        help="internal child; invoke through prefetch",
    )
    _add_prefetch_arguments(prefetch_child)

    qualify = commands.add_parser(
        "qualify",
        help="hard-supervise a two-mode <=600s qualification",
    )
    _add_workload_arguments(qualify)
    _add_supervisor_arguments(qualify)
    qualify.add_argument("--selection-path", required=True, type=_absolute_path)
    qualify.add_argument("--warmup-steps", type=int, default=1)
    qualify.add_argument("--measured-steps", type=int, default=3)

    qualify_child = commands.add_parser(
        "qualify-child",
        help="internal child; invoke through qualify",
    )
    _add_workload_arguments(qualify_child)
    qualify_child.add_argument(
        "--selection-path",
        required=True,
        type=_absolute_path,
    )
    qualify_child.add_argument("--warmup-steps", type=int, default=1)
    qualify_child.add_argument("--measured-steps", type=int, default=3)

    train = commands.add_parser(
        "train",
        help="supervise an exactly qualified fixed-order training run",
    )
    _add_workload_arguments(train)
    _add_supervisor_arguments(train)
    train.add_argument("--selection-path", required=True, type=_absolute_path)
    train.add_argument(
        "--checkpoint-control-root",
        required=True,
        type=_absolute_path,
    )
    train.add_argument("--checkpoint-payload-root", required=True)
    train.add_argument("--maximum-runtime-seconds", required=True, type=int)
    train.add_argument("--result-path", required=True, type=_absolute_path)
    train.add_argument("--metric-path", type=_absolute_path)
    train.add_argument(
        "--direct-bounded-resident",
        action="store_true",
        help=(
            "run the resident graph directly without a mode-selection "
            "receipt; historical configs allow <=3600 seconds and the strict "
            "learning_horizon_10m v6e config allows <=43200 seconds"
        ),
    )

    train_child = commands.add_parser(
        "train-child",
        help="internal child; invoke through train",
    )
    _add_workload_arguments(train_child)
    train_child.add_argument(
        "--selection-path",
        required=True,
        type=_absolute_path,
    )
    train_child.add_argument(
        "--checkpoint-control-root",
        required=True,
        type=_absolute_path,
    )
    train_child.add_argument("--checkpoint-payload-root", required=True)
    train_child.add_argument("--maximum-runtime-seconds", required=True, type=int)
    train_child.add_argument("--result-path", required=True, type=_absolute_path)
    train_child.add_argument("--metric-path", type=_absolute_path)
    train_child.add_argument(
        "--direct-bounded-resident",
        action="store_true",
    )
    return parser


def _workload_argv(args: argparse.Namespace) -> list[str]:
    result = [
        "--scientific-model",
        args.scientific_model,
        "--accelerator-type",
        args.accelerator_type,
        "--config",
        str(Path(args.config).expanduser().resolve()),
        "--phase",
        args.phase,
        "--condition",
        args.condition,
        "--seed",
        str(args.seed),
        "--capture-control-root",
        args.capture_control_root,
        "--object-prefix",
        args.object_prefix,
        "--resident-cache-root",
        args.resident_cache_root,
        "--resident-cache-max-bytes",
        str(args.resident_cache_max_bytes),
        "--prefetch-receipt-path",
        args.prefetch_receipt_path,
        "--source-revision",
        args.source_revision,
        "--peak-flops-per-second",
        str(args.peak_flops_per_second),
    ]
    if args.recomposition_sidecar is not None:
        result.extend(["--recomposition-sidecar", args.recomposition_sidecar])
    if args.scientific_model == "transcoder":
        _require_transcoder_source_arguments(args)
        result.extend(
            [
                "--source-overlay-control-root",
                args.source_overlay_control_root,
                "--source-overlay-object-prefix",
                args.source_overlay_object_prefix,
            ]
        )
    return result


def _supervision_paths(
    args: argparse.Namespace,
    *,
    operation: str,
) -> dict[str, Path]:
    root = Path(args.supervision_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", args.host_id) is None:
        raise ContributionTrainingWorkerError("host-id is invalid")
    if operation == "prefetch":
        prefix = f"prefetch-p{int(args.process_index):05d}-{args.host_id}"
    else:
        prefix = f"{operation}-{args.condition}-seed{args.seed}-{args.host_id}"
    return {
        "log_path": root / f"{prefix}.log",
        "execution_record_path": root / f"{prefix}.execution.json",
        "result_path": root / f"{prefix}.supervision.json",
    }


def _prelaunch_identity(
    args: argparse.Namespace,
    *,
    child_command: Sequence[str],
    uses_accelerator: bool = True,
) -> ExactPerformanceIdentity:
    module_path = Path(__file__).resolve()
    module_sha = hashlib.sha256(module_path.read_bytes()).hexdigest()
    return ExactPerformanceIdentity.from_components(
        graph={
            "schema_version": "gemma4-contribution-sae-prelaunch-source-v1",
            "module_sha256": module_sha,
            "source_revision": args.source_revision,
        },
        hardware=(
            {
                "schema_version": ("gemma4-contribution-sae-requested-hardware-v1"),
                "platform": "tpu",
                "accelerator": args.accelerator_type,
                "global_chip_count": 16,
                "process_count": 4,
                "local_device_count": 4,
                "peak_flops_per_second": args.peak_flops_per_second,
            }
            if uses_accelerator
            else {
                "schema_version": ("gemma4-contribution-prefetch-host-runtime-v1"),
                "platform": "host",
                "accelerator": None,
                "jax_initialized": False,
            }
        ),
        config={
            "schema_version": "gemma4-contribution-sae-prelaunch-command-v1",
            "argv": list(child_command),
        },
    )


def _supervise(
    args: argparse.Namespace,
    *,
    child_command: Sequence[str],
    identity: ExactPerformanceIdentity,
    maximum_seconds: int,
    qualification_receipt: Mapping[str, Any] | None,
    operation: str,
    workload_profile: Literal[
        "historical",
        "learning_horizon_10m",
    ] = "historical",
) -> dict[str, Any]:
    from moe_sae.contribution_supervisor import supervise_authorized_process

    purpose = "qualification" if operation == "qualify" else "workload"
    authorization = authorize_accelerator_run(
        identity=identity,
        maximum_runtime_seconds=maximum_seconds,
        purpose=purpose,
        qualification_receipt=qualification_receipt,
        workload_profile=workload_profile,
    )
    paths = _supervision_paths(args, operation=operation)
    result = supervise_authorized_process(
        authorization=authorization,
        qualification_receipt=qualification_receipt,
        command=child_command,
        environment=dict(os.environ),
        working_directory=args.working_directory,
        termination_grace_seconds=args.termination_grace_seconds,
        **paths,
    )
    if result["status"] != "completed" or result["returncode"] != 0:
        raise ContributionTrainingWorkerError(
            f"{operation} child failed under supervisor: "
            f"status={result['status']}, returncode={result['returncode']}; "
            f"log={paths['log_path']}"
        )
    return result


def _prefetch_argv(args: argparse.Namespace) -> list[str]:
    result = [
        "--scientific-model",
        args.scientific_model,
        "--accelerator-type",
        args.accelerator_type,
        "--config",
        str(Path(args.config).expanduser().resolve()),
        "--phase",
        args.phase,
        "--capture-control-root",
        args.capture_control_root,
        "--object-prefix",
        args.object_prefix,
        "--resident-cache-root",
        args.resident_cache_root,
        "--resident-cache-max-bytes",
        str(args.resident_cache_max_bytes),
        "--process-count",
        str(args.process_count),
        "--process-index",
        str(args.process_index),
        "--prefetch-receipt-path",
        args.prefetch_receipt_path,
        "--source-revision",
        args.source_revision,
    ]
    if args.recomposition_sidecar is not None:
        result.extend(["--recomposition-sidecar", args.recomposition_sidecar])
    if args.regional_mirror_seal is not None:
        result.extend(
            ["--regional-mirror-seal", args.regional_mirror_seal]
        )
    if args.scientific_model == "transcoder":
        _require_transcoder_source_arguments(args)
        result.extend(
            [
                "--source-overlay-control-root",
                args.source_overlay_control_root,
                "--source-overlay-object-prefix",
                args.source_overlay_object_prefix,
            ]
        )
    return result


def _prefetch_command(args: argparse.Namespace) -> int:
    """Supervise the JAX-free transfer under one hard process-group deadline."""

    maximum = _positive_int(
        "maximum_runtime_seconds",
        args.maximum_runtime_seconds,
    )
    maximum_allowed = prefetch_runtime_limit_seconds(
        config_path=args.config,
        accelerator_type=args.accelerator_type,
        phase=args.phase,
    )
    if maximum > maximum_allowed:
        raise ContributionTrainingWorkerError(
            "process-lane prefetch cannot exceed "
            f"{maximum_allowed} seconds for this config/accelerator"
        )
    child = [
        sys.executable,
        "-m",
        "moe_sae.contribution_training_worker",
        "prefetch-child",
        *_prefetch_argv(args),
    ]
    identity = _prelaunch_identity(
        args,
        child_command=child,
        uses_accelerator=False,
    )
    result = _supervise(
        args,
        child_command=child,
        identity=identity,
        maximum_seconds=maximum,
        qualification_receipt=None,
        operation="prefetch",
    )
    raw_receipt = _read_json(
        args.prefetch_receipt_path,
        name="prefetch receipt",
    )
    receipt = (
        validate_transcoder_prefetch_receipt(raw_receipt)
        if args.scientific_model == "transcoder"
        else validate_prefetch_receipt(raw_receipt)
    )
    print(
        json.dumps(
            {
                "supervision_result_sha256": result["result_sha256"],
                "receipt_sha256": receipt["receipt_sha256"],
                "process_partition": receipt["process_partition"],
                "objects": receipt.get(
                    "object_count",
                    receipt.get("combined_object_count"),
                ),
                "bytes": receipt.get(
                    "object_bytes",
                    receipt.get("combined_object_bytes"),
                ),
                "elapsed_seconds": receipt["elapsed_seconds"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def _prefetch_child_command(args: argparse.Namespace) -> int:
    """Run the bounded-cache transfer without importing JAX."""

    _verify_repository_revision(args.source_revision)
    _require_transcoder_source_arguments(args)
    config = _load_worker_config(args.config, args.scientific_model)
    sidecar = _load_sidecar(args.recomposition_sidecar)
    object_store = None
    if args.regional_mirror_seal is not None:
        from moe_sae.object_store import GCSObjectStore
        from moe_sae.regional_execution import RegionalMirrorStore

        sealed_mirror = _read_json(
            args.regional_mirror_seal,
            name="regional mirror seal",
        )
        object_store = RegionalMirrorStore(
            sealed_mirror,
            GCSObjectStore(),
        )
    if args.scientific_model == "transcoder":
        receipt = prefetch_transcoder_process_lane(
            config=config,
            phase=args.phase,
            source_revision=args.source_revision,
            capture_control_root=args.capture_control_root,
            object_prefix=args.object_prefix,
            source_overlay_control_root=args.source_overlay_control_root,
            source_overlay_object_prefix=args.source_overlay_object_prefix,
            cache_root=args.resident_cache_root,
            cache_max_bytes=args.resident_cache_max_bytes,
            process_count=args.process_count,
            process_index=args.process_index,
            recomposition_sidecar=sidecar,
            object_store=object_store,
        )
    else:
        receipt = prefetch_process_lane(
            config=config,
            phase=args.phase,
            source_revision=args.source_revision,
            capture_control_root=args.capture_control_root,
            object_prefix=args.object_prefix,
            cache_root=args.resident_cache_root,
            cache_max_bytes=args.resident_cache_max_bytes,
            process_count=args.process_count,
            process_index=args.process_index,
            recomposition_sidecar=sidecar,
            object_store=object_store,
        )
    _write_create_only_json(args.prefetch_receipt_path, receipt)
    print(
        json.dumps(
            {
                "receipt_sha256": receipt["receipt_sha256"],
                "process_partition": receipt["process_partition"],
                "objects": receipt.get(
                    "object_count",
                    receipt.get("combined_object_count"),
                ),
                "bytes": receipt.get(
                    "object_bytes",
                    receipt.get("combined_object_bytes"),
                ),
                "elapsed_seconds": receipt["elapsed_seconds"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def _qualify_command(args: argparse.Namespace) -> int:
    child = [
        sys.executable,
        "-m",
        "moe_sae.contribution_training_worker",
        "qualify-child",
        *_workload_argv(args),
        "--selection-path",
        args.selection_path,
        "--warmup-steps",
        str(args.warmup_steps),
        "--measured-steps",
        str(args.measured_steps),
    ]
    identity = _prelaunch_identity(args, child_command=child)
    result = _supervise(
        args,
        child_command=child,
        identity=identity,
        maximum_seconds=MAX_QUALIFICATION_SECONDS,
        qualification_receipt=None,
        operation="qualify",
    )
    selection = validate_training_mode_selection(
        _read_json(args.selection_path, name="mode selection")
    )
    print(
        json.dumps(
            {
                "supervision_result_sha256": result["result_sha256"],
                "selection_sha256": selection["selection_sha256"],
                "chosen_mode": selection["chosen_mode"],
                "chosen_identity_sha256": selection["chosen_identity_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


def _qualify_child_command(args: argparse.Namespace) -> int:
    qualification_started = time.perf_counter()
    from moe_sae.runtime import (
        initialize_jax_distributed,
        require_accelerator_runtime,
    )

    _verify_repository_revision(args.source_revision)
    _require_transcoder_source_arguments(args)
    config = _load_worker_config(args.config, args.scientific_model)
    plan = TrainingWorkerPlan.from_config(
        config,
        phase=args.phase,
        condition=args.condition,
        seed=args.seed,
    )
    sidecar = _load_sidecar(args.recomposition_sidecar)
    raw_prefetch = _read_json(
        args.prefetch_receipt_path,
        name="prefetch receipt",
    )
    if args.scientific_model == "transcoder":
        prefetch_receipt = _prove_transcoder_prefetch_for_plan(
            raw_prefetch,
            config=config,
            plan=plan,
            source_revision=args.source_revision,
            capture_control_root=args.capture_control_root,
            object_prefix=args.object_prefix,
            source_overlay_control_root=args.source_overlay_control_root,
            source_overlay_object_prefix=args.source_overlay_object_prefix,
            cache_root=args.resident_cache_root,
            cache_max_bytes=args.resident_cache_max_bytes,
            recomposition_sidecar=sidecar,
        )
    else:
        prefetch_receipt = _prove_prefetch_for_plan(
            raw_prefetch,
            config=config,
            plan=plan,
            source_revision=args.source_revision,
            capture_control_root=args.capture_control_root,
            object_prefix=args.object_prefix,
            cache_root=args.resident_cache_root,
            cache_max_bytes=args.resident_cache_max_bytes,
            recomposition_sidecar=sidecar,
        )
    initialize_jax_distributed()
    require_accelerator_runtime(
        args.accelerator_type,
        peak_flops_per_second=args.peak_flops_per_second,
    )
    workload = _build_production_workload(
        config,
        plan,
        scientific_model=args.scientific_model,
        capture_control_root=args.capture_control_root,
        object_prefix=args.object_prefix,
        source_overlay_control_root=args.source_overlay_control_root,
        source_overlay_object_prefix=args.source_overlay_object_prefix,
        resident_cache_root=args.resident_cache_root,
        resident_cache_max_bytes=args.resident_cache_max_bytes,
        recomposition_sidecar=sidecar,
        prefetch_receipt=prefetch_receipt,
    )
    selection = _run_production_qualification(
        config,
        plan,
        workload,
        source_revision=args.source_revision,
        peak_flops_per_second=args.peak_flops_per_second,
        accelerator_type=args.accelerator_type,
        warmup_steps=args.warmup_steps,
        measured_steps=args.measured_steps,
        qualification_started=qualification_started,
    )
    import jax

    if jax.process_index() == 0:
        _write_create_only_json(args.selection_path, selection)
    _collective_barrier(f"contribution-sae-selection-{plan.condition}-{plan.seed}")
    observed = validate_training_mode_selection(
        _read_json(args.selection_path, name="mode selection")
    )
    if observed != selection:
        raise ContributionTrainingWorkerError(
            "published mode selection differs from this process's evidence"
        )
    print(
        json.dumps(
            {
                "process_index": int(jax.process_index()),
                "selection_sha256": selection["selection_sha256"],
                "chosen_mode": selection["chosen_mode"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def _train_command(args: argparse.Namespace) -> int:
    direct_maximum = direct_resident_runtime_limit_seconds(
        config_path=args.config,
        phase=args.phase,
        accelerator_type=args.accelerator_type,
    )
    if args.direct_bounded_resident and args.maximum_runtime_seconds > direct_maximum:
        raise ContributionTrainingWorkerError(
            "direct resident training exceeds its explicit runtime policy"
        )
    if args.direct_bounded_resident:
        selection = None
        receipt = None
    else:
        selection = validate_training_mode_selection(
            _read_json(args.selection_path, name="mode selection")
        )
        _mode, _selected_identity, receipt = chosen_mode_evidence(selection)
    child = [
        sys.executable,
        "-m",
        "moe_sae.contribution_training_worker",
        "train-child",
        *_workload_argv(args),
        "--selection-path",
        args.selection_path,
        "--checkpoint-control-root",
        args.checkpoint_control_root,
        "--checkpoint-payload-root",
        args.checkpoint_payload_root,
        "--maximum-runtime-seconds",
        str(args.maximum_runtime_seconds),
        "--result-path",
        args.result_path,
    ]
    if args.direct_bounded_resident:
        child.append("--direct-bounded-resident")
    if args.metric_path is not None:
        child.extend(["--metric-path", args.metric_path])
    identity = (
        _prelaunch_identity(args, child_command=child)
        if args.direct_bounded_resident
        else _selected_identity
    )
    result = _supervise(
        args,
        child_command=child,
        identity=identity,
        maximum_seconds=args.maximum_runtime_seconds,
        qualification_receipt=(
            receipt
            if not args.direct_bounded_resident
            and args.maximum_runtime_seconds > MAX_QUALIFICATION_SECONDS
            else None
        ),
        operation="train",
        workload_profile=(
            "learning_horizon_10m"
            if args.direct_bounded_resident
            and direct_maximum == MAX_LEARNING_HORIZON_ACCELERATOR_SECONDS
            else "historical"
        ),
    )
    training = _read_json(args.result_path, name="training result")
    print(
        json.dumps(
            {
                "supervision_result_sha256": result["result_sha256"],
                "training_result_sha256": training["result_sha256"],
                "status": training["status"],
                "tokens": training["tokens"],
            },
            sort_keys=True,
        )
    )
    return 0


def _train_child_command(args: argparse.Namespace) -> int:
    from moe_sae.runtime import (
        initialize_jax_distributed,
        require_accelerator_runtime,
    )

    _verify_repository_revision(args.source_revision)
    direct_maximum = direct_resident_runtime_limit_seconds(
        config_path=args.config,
        phase=args.phase,
        accelerator_type=args.accelerator_type,
    )
    if args.direct_bounded_resident and args.maximum_runtime_seconds > direct_maximum:
        raise ContributionTrainingWorkerError(
            "direct resident training exceeds its explicit runtime policy"
        )
    if args.direct_bounded_resident:
        selection = None
        context = None
    else:
        selection = validate_training_mode_selection(
            _read_json(args.selection_path, name="mode selection")
        )
        context = selection["workload_context"]
    _require_transcoder_source_arguments(args)
    config = _load_worker_config(args.config, args.scientific_model)
    plan = TrainingWorkerPlan.from_config(
        config,
        phase=args.phase,
        condition=args.condition,
        seed=args.seed,
    )
    sidecar = _load_sidecar(args.recomposition_sidecar)
    raw_prefetch = _read_json(
        args.prefetch_receipt_path,
        name="prefetch receipt",
    )
    if args.scientific_model == "transcoder":
        prefetch_receipt = _prove_transcoder_prefetch_for_plan(
            raw_prefetch,
            config=config,
            plan=plan,
            source_revision=args.source_revision,
            capture_control_root=args.capture_control_root,
            object_prefix=args.object_prefix,
            source_overlay_control_root=args.source_overlay_control_root,
            source_overlay_object_prefix=args.source_overlay_object_prefix,
            cache_root=args.resident_cache_root,
            cache_max_bytes=args.resident_cache_max_bytes,
            recomposition_sidecar=sidecar,
        )
    else:
        prefetch_receipt = _prove_prefetch_for_plan(
            raw_prefetch,
            config=config,
            plan=plan,
            source_revision=args.source_revision,
            capture_control_root=args.capture_control_root,
            object_prefix=args.object_prefix,
            cache_root=args.resident_cache_root,
            cache_max_bytes=args.resident_cache_max_bytes,
            recomposition_sidecar=sidecar,
        )
    initialize_jax_distributed()
    require_accelerator_runtime(
        args.accelerator_type,
        peak_flops_per_second=args.peak_flops_per_second,
    )
    workload = _build_production_workload(
        config,
        plan,
        scientific_model=args.scientific_model,
        capture_control_root=args.capture_control_root,
        object_prefix=args.object_prefix,
        source_overlay_control_root=args.source_overlay_control_root,
        source_overlay_object_prefix=args.source_overlay_object_prefix,
        resident_cache_root=args.resident_cache_root,
        resident_cache_max_bytes=args.resident_cache_max_bytes,
        recomposition_sidecar=sidecar,
        prefetch_receipt=prefetch_receipt,
        expected_context=context,
    )
    result = _run_production_training(
        config,
        plan,
        workload,
        selection,
        source_revision=args.source_revision,
        peak_flops_per_second=args.peak_flops_per_second,
        accelerator_type=args.accelerator_type,
        checkpoint_control_root=args.checkpoint_control_root,
        checkpoint_payload_root=args.checkpoint_payload_root,
        maximum_runtime_seconds=args.maximum_runtime_seconds,
        metric_path=args.metric_path,
        direct_bounded_resident=args.direct_bounded_resident,
        direct_runtime_max_seconds=direct_maximum,
    )
    import jax

    if jax.process_index() == 0:
        _write_create_only_json(args.result_path, result)
    _collective_barrier(f"contribution-sae-result-{plan.condition}-{plan.seed}")
    observed = _read_json(args.result_path, name="training result")
    if observed != result:
        raise ContributionTrainingWorkerError(
            "published training result differs across processes"
        )
    print(
        json.dumps(
            {
                "process_index": int(jax.process_index()),
                "result_sha256": result["result_sha256"],
                "status": result["status"],
                "tokens": result["tokens"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "prefetch":
            return _prefetch_command(args)
        if args.command == "prefetch-child":
            return _prefetch_child_command(args)
        if args.command == "qualify":
            return _qualify_command(args)
        if args.command == "qualify-child":
            return _qualify_child_command(args)
        if args.command == "train":
            return _train_command(args)
        if args.command == "train-child":
            return _train_child_command(args)
    except ContributionTrainingWorkerError as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        return 2
    raise AssertionError("unreachable contribution training command")


if __name__ == "__main__":  # pragma: no cover - exercised via module CLI
    raise SystemExit(main())


__all__ = [
    "CONDITIONS",
    "INPUT_MODES",
    "PREFETCH_RECEIPT_SCHEMA_VERSION",
    "TRAINING_MODE_SELECTION_SCHEMA_VERSION",
    "TRAINING_RESULT_SCHEMA_VERSION",
    "TRAINING_RUN_IDENTITY_SCHEMA_VERSION",
    "TRAINING_WORKER_SCHEMA_VERSION",
    "ContributionTrainingWorkerError",
    "PreparedTrainingExecutable",
    "RestoredTrainingState",
    "TrainingLoopDependencies",
    "TrainingWorkerPlan",
    "chosen_mode_evidence",
    "main",
    "prefetch_process_lane",
    "qualify_training_modes",
    "run_qualified_training",
    "validate_prefetch_receipt",
    "validate_training_mode_selection",
]
