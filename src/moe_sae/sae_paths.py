"""Content-addressed filesystem policy for one SAE operation.

The durable SAE controller already stores its state beneath::

    <artifact-root>/operator/sae/operations/<operation-sha256>/

This module keeps every numerical artifact under that same immutable operation
identity while separating it from the controller's own state files.  It is a
pure path policy: it neither creates directories nor imports JAX.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any


SAE_CONDITIONS = ("A", "B", "F")
SAE_SEEDS = (1, 2, 3)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class SAEPathError(ValueError):
    """An SAE artifact path request is invalid or escapes its operation root."""


def _condition(value: Any) -> str:
    if not isinstance(value, str) or value not in SAE_CONDITIONS:
        raise SAEPathError(
            f"condition must be one of {SAE_CONDITIONS}, got {value!r}"
        )
    return value


def _seed(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in SAE_SEEDS:
        raise SAEPathError(f"seed must be one of {SAE_SEEDS}, got {value!r}")
    return value


@dataclass(frozen=True, slots=True)
class SAEOperationPaths:
    """Validated paths for numerical artifacts belonging to one operation."""

    artifact_root: Path
    operation_sha256: str

    def __post_init__(self) -> None:
        root = Path(self.artifact_root).expanduser().resolve()
        digest = self.operation_sha256
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            raise SAEPathError(
                "operation_sha256 must be 64 lowercase hexadecimal characters"
            )
        object.__setattr__(self, "artifact_root", root)

    @property
    def operation_root(self) -> Path:
        """Root shared with the JAX-free SAE control plane."""

        return (
            self.artifact_root
            / "operator"
            / "sae"
            / "operations"
            / self.operation_sha256
        )

    @property
    def numerical_root(self) -> Path:
        """Root reserved for numerical children and their scientific outputs."""

        return self._within_operation(self.operation_root / "artifacts")

    def run_root(self, condition: str, seed: int) -> Path:
        """Authoritative checkpoint/control root for one condition and seed."""

        selected_condition = _condition(condition)
        selected_seed = _seed(seed)
        return self._within_operation(
            self.numerical_root
            / "training"
            / selected_condition
            / f"seed-{selected_seed}"
        )

    def evaluation_path(self, condition: str, seed: int) -> Path:
        """Final held-out evaluation record for one condition and seed."""

        selected_condition = _condition(condition)
        selected_seed = _seed(seed)
        return self._within_operation(
            self.numerical_root
            / "evaluations"
            / f"{selected_condition}-seed-{selected_seed}.json"
        )

    def metrics_path(self, condition: str, seed: int) -> Path:
        """Crash-safe metrics journal path for one authoritative run."""

        return self._within_operation(
            self.run_root(condition, seed) / "metrics.jsonl"
        )

    def smoke_root(self, condition: str) -> Path:
        """Non-authoritative full-shape smoke root, isolated from training."""

        selected_condition = _condition(condition)
        return self._within_operation(
            self.numerical_root / "smoke" / selected_condition
        )

    @property
    def report_path(self) -> Path:
        """Final nine-run efficiency report for this exact operation."""

        return self._within_operation(self.numerical_root / "report.json")

    def _within_operation(self, candidate: Path) -> Path:
        """Reject lexical or symlink-assisted escape from the operation root."""

        operation = self.operation_root.resolve()
        resolved = candidate.resolve()
        try:
            resolved.relative_to(operation)
        except ValueError as error:
            raise SAEPathError(
                f"SAE artifact path escapes its operation root: {candidate}"
            ) from error
        return resolved


__all__ = [
    "SAE_CONDITIONS",
    "SAE_SEEDS",
    "SAEOperationPaths",
    "SAEPathError",
]
