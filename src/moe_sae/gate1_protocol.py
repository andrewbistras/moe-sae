"""Fail-closed protocol loader for the Gemma 4 Gate-1 audit.

Gate 1 deliberately reads historical immutable payloads directly instead of
following the production ``final`` checkpoint pointers.  That exception is
safe only when every input and diagnostic invocation is bound to one strict,
canonically hashed protocol.  This module is the JAX- and cloud-free boundary
which provides that binding.

The accepted schema is intentionally exact.  Unknown or missing keys,
duplicate YAML/JSON keys, non-canonical checkpoint payload names, overlapping
token ranges, dangling matrix references, and incomplete condition/seed
coverage all fail before any checkpoint or activation shard is opened.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from types import MappingProxyType
from typing import Any

from moe_sae.audit_checkpoint import AuditCheckpointExpectation
from moe_sae.config import CONDITIONS, Gemma4Config

GATE1_PROTOCOL_SCHEMA = "gemma4-gate1-audit-v1"
ROUTE_PERMUTATION_MODE = "keyed_sequence_permutation"
ROUTE_PERMUTATION_DOMAIN = "gemma4-gate1-router-permutation-v1"
GATE1_ROUTE_MODES = ("true", "shuffled", "all_banks")
GATE1_SEEDS = (1, 2, 3)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_PROTOCOL_ID_RE = re.compile(r"[a-z0-9](?:[a-z0-9-]{1,126}[a-z0-9])?")
_CHECKPOINT_LABEL_RE = re.compile(
    r"(?P<condition>[ABF])_seed(?P<seed>[1-3])_(?P<millions>[1-9][0-9]*)m"
)
_PAYLOAD_NAME_RE = re.compile(
    r"step-(?P<steps>[0-9]{8})-tokens-(?P<tokens>[0-9]{12})-"
    r"attempt-(?P<attempt>[0-9a-f]{32})"
)

_TOP_LEVEL_KEYS = {
    "schema_version",
    "protocol_id",
    "capture",
    "route_permutation",
    "statistics",
    "splits",
    "checkpoints",
    "matrix",
}
_CAPTURE_KEYS = {
    "config_sha256",
    "checkpoint_sha256",
    "train_manifest_sha256",
    "eval_manifest_sha256",
    "train_tokens",
    "eval_tokens",
}
_ROUTE_PERMUTATION_KEYS = {"mode", "domain", "key"}
_STATISTICS_KEYS = {
    "cluster",
    "bootstrap_replicates",
    "bootstrap_key",
    "old_f_minus_b_ev",
    "material_effect_ev",
}
_SPLIT_KEYS = {"source", "ranges"}
_CHECKPOINT_IDENTITY_KEYS = {
    "operation_sha256",
    "config_sha256",
    "capture_manifest_sha256",
    "capture_checkpoint_sha256",
    "condition",
    "seed",
    "run_instance_id",
}
_MATRIX_KEYS = {"learning_curve", "final_baselines", "router_ablation"}
_LEARNING_CURVE_KEYS = {"split", "route_mode", "checkpoints"}
_FINAL_BASELINES_KEYS = {"splits", "route_mode", "checkpoints"}
_ROUTER_ABLATION_KEYS = {"split", "route_modes", "checkpoints"}

_REQUIRED_SPLITS = ("heldout", "train_stratified")
_REQUIRED_CHECKPOINTS = (
    "A_seed1_5m",
    "B_seed1_5m",
    "F_seed1_5m",
    "A_seed1_40m",
    "A_seed1_50m",
    "A_seed2_50m",
    "A_seed3_50m",
    "B_seed1_50m",
    "B_seed2_50m",
    "B_seed3_50m",
)
_LEARNING_CURVE_CHECKPOINTS = (
    "A_seed1_5m",
    "B_seed1_5m",
    "F_seed1_5m",
    "A_seed1_40m",
)
_FINAL_BASELINE_CHECKPOINTS = (
    "A_seed1_50m",
    "A_seed2_50m",
    "A_seed3_50m",
    "B_seed1_50m",
    "B_seed2_50m",
    "B_seed3_50m",
)
_ROUTER_ABLATION_CHECKPOINTS = (
    "A_seed1_50m",
    "A_seed2_50m",
    "A_seed3_50m",
)

_UINT64_MAX = 2**64 - 1


class Gate1ProtocolError(ValueError):
    """The Gate-1 protocol is malformed or internally inconsistent."""


@dataclass(frozen=True, slots=True)
class CaptureSpec:
    config_sha256: str
    checkpoint_sha256: str
    train_manifest_sha256: str
    eval_manifest_sha256: str
    train_tokens: int
    eval_tokens: int

    def to_mapping(self) -> dict[str, Any]:
        return {
            "config_sha256": self.config_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "train_manifest_sha256": self.train_manifest_sha256,
            "eval_manifest_sha256": self.eval_manifest_sha256,
            "train_tokens": self.train_tokens,
            "eval_tokens": self.eval_tokens,
        }


@dataclass(frozen=True, slots=True)
class RoutePermutationSpec:
    mode: str
    domain: str
    key: int

    def to_mapping(self) -> dict[str, Any]:
        return {"mode": self.mode, "domain": self.domain, "key": self.key}


@dataclass(frozen=True, slots=True)
class StatisticsSpec:
    cluster: str
    bootstrap_replicates: int
    bootstrap_key: int
    old_f_minus_b_ev: float
    material_effect_ev: float

    def to_mapping(self) -> dict[str, Any]:
        return {
            "cluster": self.cluster,
            "bootstrap_replicates": self.bootstrap_replicates,
            "bootstrap_key": self.bootstrap_key,
            "old_f_minus_b_ev": self.old_f_minus_b_ev,
            "material_effect_ev": self.material_effect_ev,
        }


@dataclass(frozen=True, slots=True, order=True)
class TokenRange:
    """One half-open capture row range."""

    start: int
    stop: int

    @property
    def tokens(self) -> int:
        return self.stop - self.start

    def to_list(self) -> list[int]:
        return [self.start, self.stop]


@dataclass(frozen=True, slots=True)
class SplitSpec:
    name: str
    source: str
    ranges: tuple[TokenRange, ...]

    @property
    def tokens(self) -> int:
        return sum(item.tokens for item in self.ranges)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "ranges": [item.to_list() for item in self.ranges],
        }


@dataclass(frozen=True, slots=True)
class CheckpointSpec:
    label: str
    milestone_millions: int
    expectation: AuditCheckpointExpectation
    attempt_id: str
    run_identity_sha256: str

    @property
    def checkpoint_uri(self) -> str:
        return self.expectation.checkpoint_uri

    @property
    def condition(self) -> str:
        return self.expectation.condition

    @property
    def seed(self) -> int:
        return self.expectation.seed

    @property
    def tokens_seen(self) -> int:
        return self.expectation.tokens_seen

    @property
    def steps(self) -> int:
        return self.expectation.steps

    def to_mapping(self) -> dict[str, Any]:
        return self.expectation.to_mapping()


@dataclass(frozen=True, slots=True)
class LearningCurveSpec:
    split: str
    route_mode: str
    checkpoints: tuple[str, ...]

    def to_mapping(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "route_mode": self.route_mode,
            "checkpoints": list(self.checkpoints),
        }


@dataclass(frozen=True, slots=True)
class FinalBaselinesSpec:
    splits: tuple[str, ...]
    route_mode: str
    checkpoints: tuple[str, ...]

    def to_mapping(self) -> dict[str, Any]:
        return {
            "splits": list(self.splits),
            "route_mode": self.route_mode,
            "checkpoints": list(self.checkpoints),
        }


@dataclass(frozen=True, slots=True)
class RouterAblationSpec:
    split: str
    route_modes: tuple[str, ...]
    checkpoints: tuple[str, ...]

    def to_mapping(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "route_modes": list(self.route_modes),
            "checkpoints": list(self.checkpoints),
        }


@dataclass(frozen=True, slots=True)
class MatrixSpec:
    learning_curve: LearningCurveSpec
    final_baselines: FinalBaselinesSpec
    router_ablation: RouterAblationSpec

    def to_mapping(self) -> dict[str, Any]:
        return {
            "learning_curve": self.learning_curve.to_mapping(),
            "final_baselines": self.final_baselines.to_mapping(),
            "router_ablation": self.router_ablation.to_mapping(),
        }


@dataclass(frozen=True, slots=True)
class Gate1Protocol:
    """Validated immutable representation of the complete Gate-1 contract."""

    schema_version: str
    protocol_id: str
    capture: CaptureSpec
    route_permutation: RoutePermutationSpec
    statistics: StatisticsSpec
    splits: Mapping[str, SplitSpec]
    checkpoints: Mapping[str, CheckpointSpec]
    matrix: MatrixSpec

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "protocol_id": self.protocol_id,
            "capture": self.capture.to_mapping(),
            "route_permutation": self.route_permutation.to_mapping(),
            "statistics": self.statistics.to_mapping(),
            "splits": {
                name: self.splits[name].to_mapping() for name in _REQUIRED_SPLITS
            },
            "checkpoints": {
                label: self.checkpoints[label].to_mapping()
                for label in _REQUIRED_CHECKPOINTS
            },
            "matrix": self.matrix.to_mapping(),
        }

    @property
    def sha256(self) -> str:
        """Canonical digest used to bind every Gate-1 result."""

        return _canonical_sha256(self.to_mapping())


def _exact_mapping(value: Any, expected: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Gate1ProtocolError(f"{label} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise Gate1ProtocolError(f"{label} keys must all be strings")
    actual = set(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append(f"missing keys: {', '.join(missing)}")
        if unknown:
            details.append(f"unknown keys: {', '.join(unknown)}")
        raise Gate1ProtocolError(f"{label} has {'; '.join(details)}")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise Gate1ProtocolError(f"{label} must be a nonempty string")
    return value


def _integer(
    value: Any,
    label: str,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise Gate1ProtocolError(f"{label} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        suffix = f" and at most {maximum}" if maximum is not None else ""
        raise Gate1ProtocolError(f"{label} must be at least {minimum}{suffix}")
    return value


def _finite_number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Gate1ProtocolError(f"{label} must be numeric")
    selected = float(value)
    if not math.isfinite(selected):
        raise Gate1ProtocolError(f"{label} must be finite")
    if positive and selected <= 0.0:
        raise Gate1ProtocolError(f"{label} must be positive")
    return selected


def _sha256(value: Any, label: str) -> str:
    selected = _string(value, label)
    if _SHA256_RE.fullmatch(selected) is None:
        raise Gate1ProtocolError(f"{label} must be 64 lowercase hexadecimal characters")
    return selected


def _string_sequence(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise Gate1ProtocolError(f"{label} must be a sequence")
    selected = tuple(
        _string(item, f"{label}[{index}]") for index, item in enumerate(value)
    )
    if not selected:
        raise Gate1ProtocolError(f"{label} must not be empty")
    if len(set(selected)) != len(selected):
        raise Gate1ProtocolError(f"{label} must not contain duplicates")
    return selected


def _parse_capture(value: Any) -> CaptureSpec:
    raw = _exact_mapping(value, _CAPTURE_KEYS, "capture")
    capture = CaptureSpec(
        config_sha256=_sha256(raw["config_sha256"], "capture.config_sha256"),
        checkpoint_sha256=_sha256(
            raw["checkpoint_sha256"], "capture.checkpoint_sha256"
        ),
        train_manifest_sha256=_sha256(
            raw["train_manifest_sha256"], "capture.train_manifest_sha256"
        ),
        eval_manifest_sha256=_sha256(
            raw["eval_manifest_sha256"], "capture.eval_manifest_sha256"
        ),
        train_tokens=_integer(raw["train_tokens"], "capture.train_tokens", minimum=1),
        eval_tokens=_integer(raw["eval_tokens"], "capture.eval_tokens", minimum=1),
    )
    production = Gemma4Config()
    if capture.config_sha256 != production.config_hash:
        raise Gate1ProtocolError(
            "capture.config_sha256 does not identify the production Gemma config"
        )
    if capture.train_tokens != production.train_tokens:
        raise Gate1ProtocolError(
            f"capture.train_tokens must be {production.train_tokens}"
        )
    if capture.eval_tokens != production.eval_tokens:
        raise Gate1ProtocolError(
            f"capture.eval_tokens must be {production.eval_tokens}"
        )
    if capture.train_manifest_sha256 == capture.eval_manifest_sha256:
        raise Gate1ProtocolError("train and eval capture manifests must be distinct")
    return capture


def _parse_route_permutation(value: Any) -> RoutePermutationSpec:
    raw = _exact_mapping(value, _ROUTE_PERMUTATION_KEYS, "route_permutation")
    mode = _string(raw["mode"], "route_permutation.mode")
    if mode != ROUTE_PERMUTATION_MODE:
        raise Gate1ProtocolError(
            f"route_permutation.mode must be {ROUTE_PERMUTATION_MODE!r}"
        )
    domain = _string(raw["domain"], "route_permutation.domain")
    if domain != ROUTE_PERMUTATION_DOMAIN:
        raise Gate1ProtocolError(
            f"route_permutation.domain must be {ROUTE_PERMUTATION_DOMAIN!r}"
        )
    return RoutePermutationSpec(
        mode=mode,
        domain=domain,
        key=_integer(
            raw["key"],
            "route_permutation.key",
            minimum=0,
            maximum=_UINT64_MAX,
        ),
    )


def _parse_statistics(value: Any) -> StatisticsSpec:
    raw = _exact_mapping(value, _STATISTICS_KEYS, "statistics")
    cluster = _string(raw["cluster"], "statistics.cluster")
    if cluster != "sequence_id":
        raise Gate1ProtocolError("statistics.cluster must be 'sequence_id'")
    old_gain = _finite_number(
        raw["old_f_minus_b_ev"], "statistics.old_f_minus_b_ev", positive=True
    )
    material = _finite_number(
        raw["material_effect_ev"], "statistics.material_effect_ev", positive=True
    )
    expected_material = old_gain * 0.10
    if not math.isclose(material, expected_material, rel_tol=0.0, abs_tol=1e-15):
        raise Gate1ProtocolError(
            "statistics.material_effect_ev must equal 10% of old_f_minus_b_ev"
        )
    return StatisticsSpec(
        cluster=cluster,
        bootstrap_replicates=_integer(
            raw["bootstrap_replicates"],
            "statistics.bootstrap_replicates",
            minimum=1_000,
        ),
        bootstrap_key=_integer(
            raw["bootstrap_key"],
            "statistics.bootstrap_key",
            minimum=0,
            maximum=_UINT64_MAX,
        ),
        old_f_minus_b_ev=old_gain,
        material_effect_ev=material,
    )


def _parse_ranges(
    value: Any,
    *,
    label: str,
    limit: int,
) -> tuple[TokenRange, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or not value
    ):
        raise Gate1ProtocolError(f"{label} must be a nonempty sequence")
    ranges: list[TokenRange] = []
    for index, raw_range in enumerate(value):
        item_label = f"{label}[{index}]"
        if (
            not isinstance(raw_range, Sequence)
            or isinstance(raw_range, (str, bytes, bytearray))
            or len(raw_range) != 2
        ):
            raise Gate1ProtocolError(f"{item_label} must be a two-item half-open range")
        start = _integer(raw_range[0], f"{item_label}[0]", minimum=0)
        stop = _integer(raw_range[1], f"{item_label}[1]", minimum=1)
        if start >= stop:
            raise Gate1ProtocolError(f"{item_label} must satisfy start < stop")
        if stop > limit:
            raise Gate1ProtocolError(
                f"{item_label} stop exceeds its declared capture token count"
            )
        if (stop - start) % 4:
            raise Gate1ProtocolError(
                f"{item_label} length must be divisible by four host partitions"
            )
        ranges.append(TokenRange(start, stop))
    if ranges != sorted(ranges):
        raise Gate1ProtocolError(f"{label} must be ordered by start and stop")
    for previous, current in pairwise(ranges):
        if current.start < previous.stop:
            raise Gate1ProtocolError(f"{label} ranges must be disjoint")
    return tuple(ranges)


def _parse_splits(value: Any, capture: CaptureSpec) -> Mapping[str, SplitSpec]:
    raw = _exact_mapping(value, set(_REQUIRED_SPLITS), "splits")
    splits: dict[str, SplitSpec] = {}
    expected_sources = {"heldout": "eval", "train_stratified": "train"}
    limits = {"train": capture.train_tokens, "eval": capture.eval_tokens}
    occupied: dict[str, list[tuple[TokenRange, str]]] = {"train": [], "eval": []}
    for name in _REQUIRED_SPLITS:
        split_raw = _exact_mapping(raw[name], _SPLIT_KEYS, f"splits.{name}")
        source = _string(split_raw["source"], f"splits.{name}.source")
        if source != expected_sources[name]:
            raise Gate1ProtocolError(
                f"splits.{name}.source must be {expected_sources[name]!r}"
            )
        ranges = _parse_ranges(
            split_raw["ranges"],
            label=f"splits.{name}.ranges",
            limit=limits[source],
        )
        split = SplitSpec(name=name, source=source, ranges=ranges)
        splits[name] = split
        occupied[source].extend((item, name) for item in ranges)

    if splits["heldout"].ranges != (TokenRange(0, capture.eval_tokens),):
        raise Gate1ProtocolError(
            "splits.heldout must cover the complete declared eval capture"
        )
    for source, entries in occupied.items():
        ordered = sorted(entries, key=lambda item: item[0])
        for (previous, previous_name), (current, current_name) in pairwise(ordered):
            if current.start < previous.stop:
                raise Gate1ProtocolError(
                    f"splits {previous_name!r} and {current_name!r} overlap "
                    f"within capture source {source!r}"
                )
    return MappingProxyType(splits)


def _checkpoint_uri_identity(
    uri: Any,
    *,
    label: str,
    expected_steps: int,
    expected_tokens: int,
) -> tuple[str, str, str]:
    selected = _string(uri, f"checkpoints.{label}.uri")
    if (
        not selected.startswith("gs://")
        or selected.endswith("/")
        or "\\" in selected
        or "?" in selected
        or "#" in selected
        or any(character.isspace() for character in selected)
    ):
        raise Gate1ProtocolError(
            f"checkpoints.{label}.uri must be a canonical immutable gs:// URI"
        )
    remainder = selected[5:]
    bucket, separator, object_name = remainder.partition("/")
    if not bucket or not separator or not object_name:
        raise Gate1ProtocolError(
            f"checkpoints.{label}.uri must include a bucket and payload object"
        )
    components = object_name.split("/")
    if any(not component or component in {".", ".."} for component in components):
        raise Gate1ProtocolError(
            f"checkpoints.{label}.uri contains a non-canonical path component"
        )
    if len(components) < 3 or components[-3] != "runs":
        raise Gate1ProtocolError(
            f"checkpoints.{label}.uri must end in runs/<identity>/<payload>"
        )
    run_identity = components[-2]
    if _SHA256_RE.fullmatch(run_identity) is None:
        raise Gate1ProtocolError(
            f"checkpoints.{label}.uri run identity must be a lowercase SHA-256"
        )
    payload = components[-1]
    match = _PAYLOAD_NAME_RE.fullmatch(payload)
    if match is None:
        raise Gate1ProtocolError(
            f"checkpoints.{label}.uri must name an immutable checkpoint payload"
        )
    uri_steps = int(match.group("steps"))
    uri_tokens = int(match.group("tokens"))
    if uri_steps != expected_steps or uri_tokens != expected_tokens:
        raise Gate1ProtocolError(
            f"checkpoints.{label}.uri payload name disagrees with steps/tokens_seen"
        )
    return selected, match.group("attempt"), run_identity


def _parse_checkpoint(
    label: str,
    value: Any,
    *,
    capture: CaptureSpec,
) -> CheckpointSpec:
    label_match = _CHECKPOINT_LABEL_RE.fullmatch(label)
    if label_match is None:
        raise Gate1ProtocolError(
            f"checkpoint label {label!r} must encode condition, seed, and milestone"
        )
    if not isinstance(value, Mapping):
        raise Gate1ProtocolError(f"checkpoints.{label} must be a mapping")
    try:
        expectation = AuditCheckpointExpectation.from_mapping(value)
    except (TypeError, ValueError) as error:
        raise Gate1ProtocolError(
            f"checkpoints.{label} is not a valid audit checkpoint expectation: {error}"
        ) from error
    identity = expectation.checkpoint_identity
    if set(identity) != _CHECKPOINT_IDENTITY_KEYS:
        raise Gate1ProtocolError(
            f"checkpoints.{label}.checkpoint_identity fields do not match "
            "the frozen Gate-1 identity schema"
        )
    expected_capture_identity = {
        "config_sha256": capture.config_sha256,
        "capture_manifest_sha256": capture.train_manifest_sha256,
        "capture_checkpoint_sha256": capture.checkpoint_sha256,
    }
    for name, expected in expected_capture_identity.items():
        if identity[name] != expected:
            raise Gate1ProtocolError(
                f"checkpoints.{label}.checkpoint_identity.{name} "
                "does not match the capture contract"
            )
    condition = expectation.condition
    seed = expectation.seed
    label_condition = label_match.group("condition")
    label_seed = int(label_match.group("seed"))
    milestone_millions = int(label_match.group("millions"))
    if condition != label_condition or seed != label_seed:
        raise Gate1ProtocolError(
            f"checkpoint label {label!r} disagrees with condition/seed metadata"
        )
    tokens_seen = expectation.tokens_seen
    steps = expectation.steps
    if seed not in GATE1_SEEDS:
        raise Gate1ProtocolError(
            f"checkpoints.{label}.seed must be one of {GATE1_SEEDS}"
        )
    if tokens_seen > capture.train_tokens:
        raise Gate1ProtocolError(
            f"checkpoints.{label}.tokens_seen exceeds the train capture"
        )
    config = Gemma4Config()
    milestone_tokens = milestone_millions * 1_000_000
    expected_steps = math.ceil(milestone_tokens / config.effective_batch_tokens)
    expected_tokens = min(
        expected_steps * config.effective_batch_tokens,
        capture.train_tokens,
    )
    if steps != expected_steps or tokens_seen != expected_tokens:
        raise Gate1ProtocolError(
            f"checkpoint {label!r} steps/tokens_seen disagree with its milestone"
        )
    selected_uri, attempt_id, run_identity = _checkpoint_uri_identity(
        expectation.checkpoint_uri,
        label=label,
        expected_steps=steps,
        expected_tokens=tokens_seen,
    )
    if selected_uri != expectation.checkpoint_uri:  # pragma: no cover - defensive
        raise Gate1ProtocolError(
            f"checkpoints.{label}.checkpoint_uri was not preserved canonically"
        )
    return CheckpointSpec(
        label=label,
        milestone_millions=milestone_millions,
        expectation=expectation,
        attempt_id=attempt_id,
        run_identity_sha256=run_identity,
    )


def _parse_checkpoints(
    value: Any, capture: CaptureSpec
) -> Mapping[str, CheckpointSpec]:
    raw = _exact_mapping(value, set(_REQUIRED_CHECKPOINTS), "checkpoints")
    checkpoints = {
        label: _parse_checkpoint(label, raw[label], capture=capture)
        for label in _REQUIRED_CHECKPOINTS
    }
    uris = [checkpoint.checkpoint_uri for checkpoint in checkpoints.values()]
    if len(set(uris)) != len(uris):
        raise Gate1ProtocolError("checkpoint URIs must be unique")
    identities = [checkpoint.run_identity_sha256 for checkpoint in checkpoints.values()]
    if len(set(identities)) != len(identities):
        raise Gate1ProtocolError("checkpoint run identities must be unique")
    observed_conditions = {checkpoint.condition for checkpoint in checkpoints.values()}
    if observed_conditions != set(CONDITIONS):
        raise Gate1ProtocolError(
            f"checkpoint conditions must cover exactly {CONDITIONS}"
        )
    observed_final = {
        (checkpoint.condition, checkpoint.seed)
        for checkpoint in checkpoints.values()
        if checkpoint.milestone_millions == 50
    }
    required_final = {
        (condition, seed) for condition in ("A", "B") for seed in GATE1_SEEDS
    }
    if observed_final != required_final:
        raise Gate1ProtocolError(
            "50M checkpoints must cover A/B seeds 1, 2, and 3 exactly"
        )
    return MappingProxyType(checkpoints)


def _require_references(
    references: tuple[str, ...],
    available: Mapping[str, Any],
    *,
    label: str,
) -> None:
    missing = [reference for reference in references if reference not in available]
    if missing:
        raise Gate1ProtocolError(
            f"{label} contains unknown references: {', '.join(missing)}"
        )


def _parse_matrix(
    value: Any,
    *,
    splits: Mapping[str, SplitSpec],
    checkpoints: Mapping[str, CheckpointSpec],
) -> MatrixSpec:
    raw = _exact_mapping(value, _MATRIX_KEYS, "matrix")

    learning_raw = _exact_mapping(
        raw["learning_curve"], _LEARNING_CURVE_KEYS, "matrix.learning_curve"
    )
    learning_split = _string(learning_raw["split"], "matrix.learning_curve.split")
    learning_mode = _string(
        learning_raw["route_mode"], "matrix.learning_curve.route_mode"
    )
    learning_checkpoints = _string_sequence(
        learning_raw["checkpoints"], "matrix.learning_curve.checkpoints"
    )
    _require_references((learning_split,), splits, label="matrix.learning_curve.split")
    _require_references(
        learning_checkpoints,
        checkpoints,
        label="matrix.learning_curve.checkpoints",
    )
    if learning_split != "heldout":
        raise Gate1ProtocolError("matrix.learning_curve.split must be 'heldout'")
    if learning_mode != "true":
        raise Gate1ProtocolError("matrix.learning_curve.route_mode must be 'true'")
    if learning_checkpoints != _LEARNING_CURVE_CHECKPOINTS:
        raise Gate1ProtocolError(
            "matrix.learning_curve.checkpoints do not match the frozen audit"
        )

    baselines_raw = _exact_mapping(
        raw["final_baselines"], _FINAL_BASELINES_KEYS, "matrix.final_baselines"
    )
    baseline_splits = _string_sequence(
        baselines_raw["splits"], "matrix.final_baselines.splits"
    )
    baseline_mode = _string(
        baselines_raw["route_mode"], "matrix.final_baselines.route_mode"
    )
    baseline_checkpoints = _string_sequence(
        baselines_raw["checkpoints"], "matrix.final_baselines.checkpoints"
    )
    _require_references(baseline_splits, splits, label="matrix.final_baselines.splits")
    _require_references(
        baseline_checkpoints,
        checkpoints,
        label="matrix.final_baselines.checkpoints",
    )
    if baseline_splits != _REQUIRED_SPLITS:
        raise Gate1ProtocolError(
            "matrix.final_baselines.splits must be heldout and train_stratified"
        )
    if baseline_mode != "true":
        raise Gate1ProtocolError("matrix.final_baselines.route_mode must be 'true'")
    if baseline_checkpoints != _FINAL_BASELINE_CHECKPOINTS:
        raise Gate1ProtocolError(
            "matrix.final_baselines.checkpoints do not match the frozen audit"
        )

    ablation_raw = _exact_mapping(
        raw["router_ablation"], _ROUTER_ABLATION_KEYS, "matrix.router_ablation"
    )
    ablation_split = _string(ablation_raw["split"], "matrix.router_ablation.split")
    ablation_modes = _string_sequence(
        ablation_raw["route_modes"], "matrix.router_ablation.route_modes"
    )
    ablation_checkpoints = _string_sequence(
        ablation_raw["checkpoints"], "matrix.router_ablation.checkpoints"
    )
    _require_references((ablation_split,), splits, label="matrix.router_ablation.split")
    _require_references(
        ablation_checkpoints,
        checkpoints,
        label="matrix.router_ablation.checkpoints",
    )
    if ablation_split != "heldout":
        raise Gate1ProtocolError("matrix.router_ablation.split must be 'heldout'")
    if ablation_modes != GATE1_ROUTE_MODES:
        raise Gate1ProtocolError(
            f"matrix.router_ablation.route_modes must be exactly {GATE1_ROUTE_MODES}"
        )
    if ablation_checkpoints != _ROUTER_ABLATION_CHECKPOINTS:
        raise Gate1ProtocolError(
            "matrix.router_ablation.checkpoints do not match the frozen audit"
        )
    if any(checkpoints[label].condition != "A" for label in ablation_checkpoints):
        raise Gate1ProtocolError(
            "all_banks router ablation may reference only condition A"
        )

    referenced = (
        set(learning_checkpoints)
        | set(baseline_checkpoints)
        | set(ablation_checkpoints)
    )
    if referenced != set(checkpoints):
        raise Gate1ProtocolError(
            "every declared checkpoint must be referenced by the audit matrix"
        )
    return MatrixSpec(
        learning_curve=LearningCurveSpec(
            split=learning_split,
            route_mode=learning_mode,
            checkpoints=learning_checkpoints,
        ),
        final_baselines=FinalBaselinesSpec(
            splits=baseline_splits,
            route_mode=baseline_mode,
            checkpoints=baseline_checkpoints,
        ),
        router_ablation=RouterAblationSpec(
            split=ablation_split,
            route_modes=ablation_modes,
            checkpoints=ablation_checkpoints,
        ),
    )


def validate_gate1_protocol(value: Any) -> Gate1Protocol:
    """Validate one parsed protocol mapping and return its immutable form."""

    raw = _exact_mapping(value, _TOP_LEVEL_KEYS, "Gate-1 protocol")
    schema_version = _string(raw["schema_version"], "schema_version")
    if schema_version != GATE1_PROTOCOL_SCHEMA:
        raise Gate1ProtocolError(f"schema_version must be {GATE1_PROTOCOL_SCHEMA!r}")
    protocol_id = _string(raw["protocol_id"], "protocol_id")
    if _PROTOCOL_ID_RE.fullmatch(protocol_id) is None:
        raise Gate1ProtocolError(
            "protocol_id must be a canonical lowercase hyphenated identifier"
        )
    capture = _parse_capture(raw["capture"])
    route_permutation = _parse_route_permutation(raw["route_permutation"])
    statistics = _parse_statistics(raw["statistics"])
    splits = _parse_splits(raw["splits"], capture)
    checkpoints = _parse_checkpoints(raw["checkpoints"], capture)
    matrix = _parse_matrix(raw["matrix"], splits=splits, checkpoints=checkpoints)
    return Gate1Protocol(
        schema_version=schema_version,
        protocol_id=protocol_id,
        capture=capture,
        route_permutation=route_permutation,
        statistics=statistics,
        splits=splits,
        checkpoints=checkpoints,
        matrix=matrix,
    )


def _canonical_sha256(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise Gate1ProtocolError("Gate-1 protocol must be canonical JSON") from error
    return hashlib.sha256(encoded).hexdigest()


def canonical_gate1_protocol_sha256(
    value: Gate1Protocol | Mapping[str, Any],
) -> str:
    """Validate and hash a protocol using the repository canonical encoding."""

    protocol = (
        value if isinstance(value, Gate1Protocol) else validate_gate1_protocol(value)
    )
    return _canonical_sha256(protocol.to_mapping())


def _load_json(text: str) -> Any:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise Gate1ProtocolError(f"duplicate JSON key: {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(text, object_pairs_hook=unique_object)
    except json.JSONDecodeError as error:
        raise Gate1ProtocolError("cannot parse Gate-1 JSON protocol") from error


def _load_yaml(text: str) -> Any:
    try:
        import yaml
    except ImportError as error:  # pragma: no cover - a declared dependency
        raise RuntimeError("PyYAML is required to load a Gate-1 protocol") from error

    class UniqueKeySafeLoader(yaml.SafeLoader):
        pass

    def construct_mapping(
        loader: UniqueKeySafeLoader, node: Any, deep: bool = False
    ) -> dict[Any, Any]:
        if not isinstance(node, yaml.MappingNode):
            raise Gate1ProtocolError("YAML mapping node is malformed")
        result: dict[Any, Any] = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            try:
                duplicate = key in result
            except TypeError as error:
                raise Gate1ProtocolError("YAML mapping keys must be scalar") from error
            if duplicate:
                raise Gate1ProtocolError(f"duplicate YAML key: {key!r}")
            result[key] = loader.construct_object(value_node, deep=deep)
        return result

    UniqueKeySafeLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping
    )
    try:
        return yaml.load(text, Loader=UniqueKeySafeLoader)
    except Gate1ProtocolError:
        raise
    except yaml.YAMLError as error:
        raise Gate1ProtocolError("cannot parse Gate-1 YAML protocol") from error


def load_gate1_protocol(path: str | Path) -> Gate1Protocol:
    """Load, strictly validate, and canonically bind a YAML or JSON protocol."""

    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as error:
        raise Gate1ProtocolError(f"cannot read Gate-1 protocol: {source}") from error
    raw = _load_json(text) if source.suffix.lower() == ".json" else _load_yaml(text)
    return validate_gate1_protocol(raw)


__all__ = [
    "GATE1_PROTOCOL_SCHEMA",
    "GATE1_ROUTE_MODES",
    "GATE1_SEEDS",
    "ROUTE_PERMUTATION_DOMAIN",
    "ROUTE_PERMUTATION_MODE",
    "CaptureSpec",
    "CheckpointSpec",
    "FinalBaselinesSpec",
    "Gate1Protocol",
    "Gate1ProtocolError",
    "LearningCurveSpec",
    "MatrixSpec",
    "RoutePermutationSpec",
    "RouterAblationSpec",
    "SplitSpec",
    "StatisticsSpec",
    "TokenRange",
    "canonical_gate1_protocol_sha256",
    "load_gate1_protocol",
    "validate_gate1_protocol",
]
