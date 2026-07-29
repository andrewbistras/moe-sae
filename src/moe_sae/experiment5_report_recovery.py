"""Append-only report recovery for a sealed Experiment-5 evaluation matrix.

This protocol deliberately does no accelerator work.  It parents an immutable
``EvaluationRecoveryPlan`` and its exact eighteen-cell evaluation barrier,
re-proves every result fingerprint, audits the held-out sequence/SST basis,
and then assembles the ordinary Experiment-5 report from the already-written
scientific artifacts and budget ledger.

The reporting checkout is a separate piece of provenance from the evaluator
checkout.  Every output is deterministic and create-only: an interrupted seal
may be rerun, but a pre-existing byte mismatch is terminal.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from moe_sae.contribution_artifacts import canonical_json_bytes, canonical_sha256
from moe_sae.experiment5_campaign import (
    EVALUATION_TASK_COUNT,
    MILESTONES,
    PANELS,
    PROTOCOL_ID,
    SEEDS,
    CompletionFingerprint,
    _read_record,
    inspect_evaluation_completion,
)
from moe_sae.experiment5_evaluation_recovery import (
    EvaluationRecoveryPlan,
    load_evaluation_recovery_plan,
)
from moe_sae.experiment5_report import (
    CROSS_HARDWARE_SST_RTOL,
    CROSS_HARDWARE_SST_SUM_RTOL,
    HARD_BUDGET_USD,
    Experiment5ReportError,
    ValidatedPairedScores,
    ValidatedResult,
    build_experiment5_report,
    render_experiment5_report,
    validate_experiment5_budget_ledger,
    validate_experiment5_paired_scores,
    validate_experiment5_report,
    validate_experiment5_result,
)

PLAN_SCHEMA_VERSION = "gemma4-experiment5-report-recovery-plan-v1"
PARENT_SCHEMA_VERSION = "gemma4-experiment5-report-recovery-parent-v1"
BASIS_AUDIT_SCHEMA_VERSION = "gemma4-experiment5-heldout-basis-audit-v1"
ENVELOPE_SCHEMA_VERSION = "gemma4-experiment5-report-recovery-envelope-v1"
FINAL_SCHEMA_VERSION = "gemma4-experiment5-report-recovery-final-v1"
HANDOFF_SCHEMA_VERSION = "gemma4-experiment5-report-recovery-handoff-v1"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class Experiment5ReportRecoveryError(RuntimeError):
    """The report-recovery plan, parent evidence, or output is invalid."""


@dataclass(frozen=True, slots=True)
class ArtifactBinding:
    path: Path
    file_sha256: str
    scientific_sha256: str

    def to_dict(self) -> dict[str, str]:
        return {
            "path": str(self.path),
            "file_sha256": self.file_sha256,
            "scientific_sha256": self.scientific_sha256,
        }


@dataclass(frozen=True, slots=True)
class ReportRecoveryParent:
    evaluation_plan: ArtifactBinding
    evaluation_source_revision: str
    evaluation_barrier: ArtifactBinding
    budget_ledger: ArtifactBinding


@dataclass(frozen=True, slots=True)
class ReportRecoveryPlan:
    recovery_id: str
    report_root: Path
    working_directory: Path
    reporting_source_revision: str
    parent: ReportRecoveryParent
    plan_sha256: str
    source: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class _VerifiedCell:
    task_id: str
    panel: str
    seed: int
    milestone: int
    result_path: Path
    paired_path: Path
    completion: CompletionFingerprint
    result: ValidatedResult
    paired: ValidatedPairedScores


@dataclass(frozen=True, slots=True)
class _VerifiedParent:
    evaluation_plan: EvaluationRecoveryPlan
    barrier: Mapping[str, Any]
    ledger: Mapping[str, Any]
    cells: tuple[_VerifiedCell, ...]


def _exact_mapping(value: Any, fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        observed = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise Experiment5ReportRecoveryError(
            f"{name} fields differ from the strict schema: {observed}"
        )
    if not all(isinstance(key, str) for key in value):
        raise Experiment5ReportRecoveryError(f"{name} keys must be strings")
    return dict(value)


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise Experiment5ReportRecoveryError(f"{name} must be a lowercase SHA-256")
    return value


def _git_sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or _GIT_SHA_RE.fullmatch(value) is None:
        raise Experiment5ReportRecoveryError(f"{name} must be a lowercase Git SHA")
    return value


def _safe_id(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SAFE_ID_RE.fullmatch(value) is None:
        raise Experiment5ReportRecoveryError(f"{name} is not a safe identifier")
    return value


def _absolute_path(value: Any, name: str) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise Experiment5ReportRecoveryError(f"{name} must be a path")
    selected = Path(value).expanduser()
    if not selected.is_absolute():
        raise Experiment5ReportRecoveryError(f"{name} must be absolute")
    return selected.resolve(strict=False)


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def _regular_file_bytes(path: Path, name: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise Experiment5ReportRecoveryError(
            f"{name} is not a regular immutable file: {path}"
        )
    try:
        return path.read_bytes()
    except OSError as error:
        raise Experiment5ReportRecoveryError(f"cannot read {name}: {path}") from error


def _read_json(path: Path, name: str) -> dict[str, Any]:
    payload = _regular_file_bytes(path, name)
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Experiment5ReportRecoveryError(
            f"{name} is not valid JSON: {path}"
        ) from error
    if not isinstance(value, Mapping):
        raise Experiment5ReportRecoveryError(f"{name} must be a JSON object")
    return dict(value)


def _file_sha256(path: Path, name: str) -> str:
    return hashlib.sha256(_regular_file_bytes(path, name)).hexdigest()


def _binding(
    value: Any,
    *,
    name: str,
    scientific_field_name: str,
) -> ArtifactBinding:
    selected = _exact_mapping(
        value,
        {"path", "file_sha256", scientific_field_name},
        name,
    )
    return ArtifactBinding(
        path=_absolute_path(selected["path"], f"{name}.path"),
        file_sha256=_sha256(selected["file_sha256"], f"{name}.file_sha256"),
        scientific_sha256=_sha256(
            selected[scientific_field_name],
            f"{name}.{scientific_field_name}",
        ),
    )


def _binding_dict(
    path: Path,
    *,
    scientific_field_name: str,
    scientific_sha256: str,
) -> dict[str, str]:
    return {
        "path": str(path),
        "file_sha256": _file_sha256(path, path.name),
        scientific_field_name: scientific_sha256,
    }


def _canonical_payload(value: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes(dict(value)) + b"\n"


def _seal_exact_bytes(path: Path, payload: bytes, *, name: str) -> str:
    """Create ``path`` once, or prove an existing file is byte-identical."""

    digest = hashlib.sha256(payload).hexdigest()
    if path.exists() or path.is_symlink():
        observed = _regular_file_bytes(path, name)
        if observed != payload:
            raise Experiment5ReportRecoveryError(f"existing {name} changed: {path}")
        return digest
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            # Hard-link publication is atomic and cannot replace a final file.
            os.link(temporary, path)
        except FileExistsError:
            return _seal_exact_bytes(path, payload, name=name)
    except BaseException:
        raise
    finally:
        try:
            temporary.unlink()
        except OSError:
            pass
    return digest


def _self_hashed(
    body: Mapping[str, Any],
    *,
    field: str,
) -> dict[str, Any]:
    selected = dict(body)
    return {**selected, field: canonical_sha256(selected)}


def _validate_self_hash(
    value: Any,
    *,
    field: str,
    name: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Experiment5ReportRecoveryError(f"{name} must be an object")
    body = dict(value)
    claimed = body.pop(field, None)
    if (
        not isinstance(claimed, str)
        or _SHA256_RE.fullmatch(claimed) is None
        or canonical_sha256(body) != claimed
    ):
        raise Experiment5ReportRecoveryError(f"{name} self-hash is invalid")
    return {**body, field: claimed}


def _verify_clean_checkout(
    working_directory: Path,
    reporting_source_revision: str,
) -> None:
    if working_directory.is_symlink() or not working_directory.is_dir():
        raise Experiment5ReportRecoveryError(
            "working_directory is not a real directory"
        )
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=working_directory,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=working_directory,
            check=True,
            text=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise Experiment5ReportRecoveryError(
            "cannot verify reporting checkout"
        ) from error
    if head != reporting_source_revision:
        raise Experiment5ReportRecoveryError(
            "reporting checkout revision differs from the plan"
        )
    if status:
        raise Experiment5ReportRecoveryError(
            "reporting checkout must be clean at reporting_source_revision"
        )


def parse_report_recovery_plan(value: Any) -> ReportRecoveryPlan:
    """Parse and structurally validate one immutable report-recovery plan."""

    selected = _exact_mapping(
        value,
        {
            "schema_version",
            "protocol_id",
            "recovery_id",
            "report_root",
            "working_directory",
            "reporting_source_revision",
            "parent",
            "plan_sha256",
        },
        "report_recovery_plan",
    )
    claimed = _sha256(selected.pop("plan_sha256"), "plan_sha256")
    if canonical_sha256(selected) != claimed:
        raise Experiment5ReportRecoveryError(
            "report-recovery plan self-hash is invalid"
        )
    if (
        selected["schema_version"] != PLAN_SCHEMA_VERSION
        or selected["protocol_id"] != PROTOCOL_ID
    ):
        raise Experiment5ReportRecoveryError("report-recovery plan identity changed")
    parent_raw = _exact_mapping(
        selected["parent"],
        {
            "schema_version",
            "evaluation_plan",
            "evaluation_source_revision",
            "evaluation_barrier",
            "budget_ledger",
        },
        "parent",
    )
    if parent_raw["schema_version"] != PARENT_SCHEMA_VERSION:
        raise Experiment5ReportRecoveryError("report-recovery parent identity changed")
    parent = ReportRecoveryParent(
        evaluation_plan=_binding(
            parent_raw["evaluation_plan"],
            name="parent.evaluation_plan",
            scientific_field_name="plan_sha256",
        ),
        evaluation_source_revision=_git_sha(
            parent_raw["evaluation_source_revision"],
            "parent.evaluation_source_revision",
        ),
        evaluation_barrier=_binding(
            parent_raw["evaluation_barrier"],
            name="parent.evaluation_barrier",
            scientific_field_name="record_sha256",
        ),
        budget_ledger=_binding(
            parent_raw["budget_ledger"],
            name="parent.budget_ledger",
            scientific_field_name="ledger_sha256",
        ),
    )
    reporting_revision = _git_sha(
        selected["reporting_source_revision"],
        "reporting_source_revision",
    )
    if reporting_revision == parent.evaluation_source_revision:
        raise Experiment5ReportRecoveryError(
            "reporting and evaluation revisions must be recorded separately"
        )
    report_root = _absolute_path(selected["report_root"], "report_root")
    working_directory = _absolute_path(
        selected["working_directory"], "working_directory"
    )
    if _paths_overlap(report_root, parent.evaluation_plan.path):
        raise Experiment5ReportRecoveryError("report root overlaps the evaluation plan")
    return ReportRecoveryPlan(
        recovery_id=_safe_id(selected["recovery_id"], "recovery_id"),
        report_root=report_root,
        working_directory=working_directory,
        reporting_source_revision=reporting_revision,
        parent=parent,
        plan_sha256=claimed,
        source={**selected, "plan_sha256": claimed},
    )


def load_report_recovery_plan(path: str | Path) -> ReportRecoveryPlan:
    selected = _absolute_path(path, "report-recovery plan path")
    payload = _regular_file_bytes(selected, "report-recovery plan")
    try:
        value = yaml.safe_load(payload)
    except yaml.YAMLError as error:
        raise Experiment5ReportRecoveryError(
            f"cannot parse report-recovery plan: {selected}"
        ) from error
    return parse_report_recovery_plan(value)


def _validate_barrier(
    plan: EvaluationRecoveryPlan,
    barrier: Mapping[str, Any],
) -> Mapping[str, Any]:
    rows = barrier.get("evaluations")
    expected_task_ids = {task.task_id for task in plan.evaluations}
    if (
        barrier.get("campaign_id") != plan.campaign_id
        or barrier.get("plan_sha256") != plan.plan_sha256
        or barrier.get("status") != "evaluation_completed"
        or not isinstance(rows, Mapping)
        or set(rows) != expected_task_ids
        or len(expected_task_ids) != EVALUATION_TASK_COUNT
    ):
        raise Experiment5ReportRecoveryError(
            "parent evaluation barrier is not the exact sealed 18-cell matrix"
        )
    return rows


def _verify_parent(plan: ReportRecoveryPlan) -> _VerifiedParent:
    """Re-prove all plan, barrier, result, paired-score, and ledger parents."""

    parent = plan.parent
    if (
        _file_sha256(parent.evaluation_plan.path, "evaluation recovery plan")
        != parent.evaluation_plan.file_sha256
    ):
        raise Experiment5ReportRecoveryError("parent evaluation plan file changed")
    try:
        evaluation_plan = load_evaluation_recovery_plan(parent.evaluation_plan.path)
    except Exception as error:
        raise Experiment5ReportRecoveryError(
            "parent evaluation plan failed validation"
        ) from error
    if (
        evaluation_plan.plan_sha256 != parent.evaluation_plan.scientific_sha256
        or evaluation_plan.project_commit != parent.evaluation_source_revision
    ):
        raise Experiment5ReportRecoveryError(
            "parent evaluator revision or plan hash changed"
        )
    expected_barrier_path = evaluation_plan.state_root / "evaluation-complete.json"
    if parent.evaluation_barrier.path != expected_barrier_path:
        raise Experiment5ReportRecoveryError("parent evaluation barrier path changed")
    if (
        _file_sha256(parent.evaluation_barrier.path, "evaluation barrier")
        != parent.evaluation_barrier.file_sha256
    ):
        raise Experiment5ReportRecoveryError("parent evaluation barrier file changed")
    try:
        barrier = _read_record(
            parent.evaluation_barrier.path,
            record_type="evaluation_phase_completion",
        )
    except Exception as error:
        raise Experiment5ReportRecoveryError(
            "parent evaluation barrier failed validation"
        ) from error
    if barrier.get("record_sha256") != parent.evaluation_barrier.scientific_sha256:
        raise Experiment5ReportRecoveryError(
            "parent evaluation barrier record hash changed"
        )
    barrier_rows = _validate_barrier(evaluation_plan, barrier)

    cells: list[_VerifiedCell] = []
    for task in sorted(
        evaluation_plan.evaluations,
        key=lambda row: (row.panel, row.seed, row.milestone_tokens),
    ):
        try:
            completion = inspect_evaluation_completion(evaluation_plan, task)
        except Exception as error:
            raise Experiment5ReportRecoveryError(
                f"parent completion failed validation: {task.task_id}"
            ) from error
        if completion is None or completion.to_dict() != barrier_rows.get(task.task_id):
            raise Experiment5ReportRecoveryError(
                f"parent completion fingerprint changed: {task.task_id}"
            )
        result_raw = _read_json(
            task.completion_path_controller,
            f"{task.task_id} scientific result",
        )
        paired_raw = _read_json(
            task.paired_scores_path_controller,
            f"{task.task_id} paired scores",
        )
        try:
            result = validate_experiment5_result(result_raw)
            paired = validate_experiment5_paired_scores(
                paired_raw,
                result=result,
            )
        except Experiment5ReportError as error:
            raise Experiment5ReportRecoveryError(
                f"parent scientific artifact is invalid: {task.task_id}: {error}"
            ) from error
        if result.body.get("source_revision") != parent.evaluation_source_revision:
            raise Experiment5ReportRecoveryError(
                f"parent evaluator revision changed: {task.task_id}"
            )
        cells.append(
            _VerifiedCell(
                task_id=task.task_id,
                panel=task.panel,
                seed=task.seed,
                milestone=task.milestone_tokens,
                result_path=task.completion_path_controller,
                paired_path=task.paired_scores_path_controller,
                completion=completion,
                result=result,
                paired=paired,
            )
        )
    expected_identities = {
        (panel, seed, milestone)
        for panel in PANELS
        for seed in SEEDS
        for milestone in MILESTONES
    }
    if (
        len(cells) != EVALUATION_TASK_COUNT
        or {(row.panel, row.seed, row.milestone) for row in cells}
        != expected_identities
    ):
        raise Experiment5ReportRecoveryError("verified evaluation matrix is incomplete")

    expected_ledger_path = (
        evaluation_plan.state_root / "report" / ("budget-ledger.json")
    )
    if parent.budget_ledger.path != expected_ledger_path:
        raise Experiment5ReportRecoveryError(
            "budget ledger is not the preexisting evaluation ledger"
        )
    if (
        _file_sha256(parent.budget_ledger.path, "budget ledger")
        != parent.budget_ledger.file_sha256
    ):
        raise Experiment5ReportRecoveryError("parent budget ledger file changed")
    ledger_raw = _read_json(parent.budget_ledger.path, "budget ledger")
    try:
        ledger = validate_experiment5_budget_ledger(ledger_raw)
    except Experiment5ReportError as error:
        raise Experiment5ReportRecoveryError(
            "parent budget ledger failed validation"
        ) from error
    if ledger.get("ledger_sha256") != parent.budget_ledger.scientific_sha256:
        raise Experiment5ReportRecoveryError(
            "parent budget ledger scientific hash changed"
        )
    spent = float(ledger["incremental_cash_spent"])
    if not math.isfinite(spent) or spent > HARD_BUDGET_USD:
        raise Experiment5ReportRecoveryError(
            "parent budget ledger exceeds the $50 ceiling"
        )

    old_paths = [
        evaluation_plan.state_root,
        *(path for _region, path in evaluation_plan.result_roots_controller),
    ]
    if any(_paths_overlap(plan.report_root, path) for path in old_paths):
        raise Experiment5ReportRecoveryError(
            "report-recovery output overlaps immutable evaluation state"
        )
    return _VerifiedParent(
        evaluation_plan=evaluation_plan,
        barrier=barrier,
        ledger=ledger,
        cells=tuple(cells),
    )


def build_report_recovery_plan(
    *,
    evaluation_plan_path: str | Path,
    budget_ledger_path: str | Path,
    report_root: str | Path,
    working_directory: str | Path,
    reporting_source_revision: str,
    recovery_id: str | None = None,
) -> dict[str, Any]:
    """Build a plan which pins all preexisting report inputs by hash."""

    evaluation_path = _absolute_path(evaluation_plan_path, "evaluation_plan_path")
    try:
        evaluation_plan = load_evaluation_recovery_plan(evaluation_path)
    except Exception as error:
        raise Experiment5ReportRecoveryError(
            "evaluation recovery plan failed validation"
        ) from error
    barrier_path = evaluation_plan.state_root / "evaluation-complete.json"
    try:
        barrier = _read_record(
            barrier_path,
            record_type="evaluation_phase_completion",
        )
    except Exception as error:
        raise Experiment5ReportRecoveryError(
            "evaluation barrier failed validation"
        ) from error
    _validate_barrier(evaluation_plan, barrier)
    for task in evaluation_plan.evaluations:
        try:
            completion = inspect_evaluation_completion(evaluation_plan, task)
        except Exception as error:
            raise Experiment5ReportRecoveryError(
                f"evaluation completion failed validation: {task.task_id}"
            ) from error
        if completion is None or completion.to_dict() != barrier["evaluations"].get(
            task.task_id
        ):
            raise Experiment5ReportRecoveryError(
                f"evaluation is not sealed: {task.task_id}"
            )
    ledger_path = _absolute_path(budget_ledger_path, "budget_ledger_path")
    expected_ledger_path = (
        evaluation_plan.state_root / "report" / ("budget-ledger.json")
    )
    if ledger_path != expected_ledger_path:
        raise Experiment5ReportRecoveryError(
            "budget ledger path differs from the sealed evaluation state"
        )
    try:
        ledger = validate_experiment5_budget_ledger(
            _read_json(ledger_path, "budget ledger")
        )
    except Experiment5ReportError as error:
        raise Experiment5ReportRecoveryError(
            "budget ledger failed validation"
        ) from error
    if float(ledger["incremental_cash_spent"]) > HARD_BUDGET_USD:
        raise Experiment5ReportRecoveryError("budget ledger exceeds the $50 ceiling")

    reporting_revision = _git_sha(
        reporting_source_revision, "reporting_source_revision"
    )
    body = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "recovery_id": (
            _safe_id(recovery_id, "recovery_id")
            if recovery_id is not None
            else f"experiment5-report-recovery-{reporting_revision[:12]}"
        ),
        "report_root": str(_absolute_path(report_root, "report_root")),
        "working_directory": str(
            _absolute_path(working_directory, "working_directory")
        ),
        "reporting_source_revision": reporting_revision,
        "parent": {
            "schema_version": PARENT_SCHEMA_VERSION,
            "evaluation_plan": _binding_dict(
                evaluation_path,
                scientific_field_name="plan_sha256",
                scientific_sha256=evaluation_plan.plan_sha256,
            ),
            "evaluation_source_revision": evaluation_plan.project_commit,
            "evaluation_barrier": _binding_dict(
                barrier_path,
                scientific_field_name="record_sha256",
                scientific_sha256=str(barrier["record_sha256"]),
            ),
            "budget_ledger": _binding_dict(
                ledger_path,
                scientific_field_name="ledger_sha256",
                scientific_sha256=str(ledger["ledger_sha256"]),
            ),
        },
    }
    raw = _self_hashed(body, field="plan_sha256")
    parse_report_recovery_plan(raw)
    return raw


def write_report_recovery_plan(
    path: str | Path,
    value: Mapping[str, Any],
) -> ReportRecoveryPlan:
    parsed = parse_report_recovery_plan(value)
    selected = _absolute_path(path, "report-recovery plan output")
    payload = yaml.safe_dump(
        dict(value),
        sort_keys=False,
        default_flow_style=False,
    ).encode("utf-8")
    _seal_exact_bytes(
        selected,
        payload,
        name="Experiment-5 report-recovery plan",
    )
    return parsed


def _float64_bytes_sha256(value: np.ndarray) -> str:
    selected = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    return hashlib.sha256(selected.tobytes(order="C")).hexdigest()


def _basis_audit(
    plan: ReportRecoveryPlan,
    parent: _VerifiedParent,
) -> dict[str, Any]:
    """Audit exact sample identity and bounded cross-hardware SST drift."""

    by_identity = {(row.panel, row.seed, row.milestone): row for row in parent.cells}
    globally_ordered = sorted(
        parent.cells,
        key=lambda row: (row.panel, row.seed, row.milestone),
    )
    global_ids = globally_ordered[0].paired.sequence_ids
    global_valid = globally_ordered[0].paired.valid_tokens
    for row in globally_ordered[1:]:
        if not np.array_equal(
            row.paired.sequence_ids, global_ids
        ) or not np.array_equal(row.paired.valid_tokens, global_valid):
            raise Experiment5ReportRecoveryError(
                f"held-out sequence basis changed: {row.task_id}"
            )
    basis_payload = {
        "packed_sequence_ids": global_ids.tolist(),
        "valid_tokens": global_valid.tolist(),
    }
    basis_sha256 = canonical_sha256(basis_payload)
    global_reference = globally_ordered[0]
    global_reference_sst = np.asarray(
        global_reference.paired.sst,
        dtype=np.float64,
    )
    global_reference_sum = float(
        global_reference_sst.sum(dtype=np.float64)
    )

    maximum_absolute = 0.0
    maximum_relative = 0.0
    maximum_aggregate_relative = 0.0
    cells: list[dict[str, Any]] = []
    for panel in PANELS:
        for milestone in MILESTONES:
            for seed in SEEDS:
                row = by_identity[(panel, seed, milestone)]
                current = np.asarray(row.paired.sst, dtype=np.float64)
                if not np.array_equal(
                    row.paired.sequence_ids,
                    global_reference.paired.sequence_ids,
                ) or not np.array_equal(
                    row.paired.valid_tokens,
                    global_reference.paired.valid_tokens,
                ):
                    raise Experiment5ReportRecoveryError(
                        f"held-out sequence basis differs from global reference: "
                        f"{row.task_id}"
                    )
                delta = current - global_reference_sst
                absolute = np.abs(delta)
                relative = absolute / np.abs(global_reference_sst)
                maximum_row_absolute = float(np.max(absolute))
                maximum_row_relative = float(np.max(relative))
                current_sum = float(current.sum(dtype=np.float64))
                aggregate_absolute = abs(current_sum - global_reference_sum)
                aggregate_relative = aggregate_absolute / abs(
                    global_reference_sum
                )
                if not np.allclose(
                    current,
                    global_reference_sst,
                    rtol=CROSS_HARDWARE_SST_RTOL,
                    atol=0.0,
                ):
                    raise Experiment5ReportRecoveryError(
                        f"per-sequence SST tolerance breached: {row.task_id}; "
                        f"max_relative={maximum_row_relative:.17g}"
                    )
                if not np.isclose(
                    current_sum,
                    global_reference_sum,
                    rtol=CROSS_HARDWARE_SST_SUM_RTOL,
                    atol=0.0,
                ):
                    raise Experiment5ReportRecoveryError(
                        f"aggregate SST tolerance breached: {row.task_id}; "
                        f"relative={aggregate_relative:.17g}"
                    )
                maximum_absolute = max(maximum_absolute, maximum_row_absolute)
                maximum_relative = max(maximum_relative, maximum_row_relative)
                maximum_aggregate_relative = max(
                    maximum_aggregate_relative, aggregate_relative
                )
                cells.append(
                    {
                        "task_id": row.task_id,
                        "panel": panel,
                        "seed": seed,
                        "milestone_tokens": milestone,
                        "reference_task_id": global_reference.task_id,
                        "packed_sequences": int(len(row.paired.sequence_ids)),
                        "valid_tokens": int(row.paired.valid_tokens.sum()),
                        "basis_sha256": basis_sha256,
                        "result": {
                            "path": str(row.result_path),
                            "file_sha256": row.completion.sha256,
                            "result_sha256": (row.completion.scientific_sha256),
                        },
                        "paired_scores": {
                            "path": str(row.paired_path),
                            "file_sha256": (row.completion.companion_sha256),
                            "artifact_sha256": (
                                row.completion.companion_scientific_sha256
                            ),
                        },
                        "sst": {
                            "vector_sha256": _float64_bytes_sha256(current),
                            "reference_vector_sha256": (
                                _float64_bytes_sha256(global_reference_sst)
                            ),
                            "delta_vector_sha256": (_float64_bytes_sha256(delta)),
                            "maximum_per_sequence_absolute_drift": (
                                maximum_row_absolute
                            ),
                            "maximum_per_sequence_relative_drift": (
                                maximum_row_relative
                            ),
                            "aggregate_absolute_drift": aggregate_absolute,
                            "aggregate_relative_drift": aggregate_relative,
                        },
                    }
                )
    body = {
        "schema_version": BASIS_AUDIT_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": "exact_basis_bounded_sst_drift",
        "recovery_id": plan.recovery_id,
        "report_recovery_plan_sha256": plan.plan_sha256,
        "evaluation_plan_sha256": (plan.parent.evaluation_plan.scientific_sha256),
        "evaluation_barrier_record_sha256": (
            plan.parent.evaluation_barrier.scientific_sha256
        ),
        "evaluation_source_revision": (plan.parent.evaluation_source_revision),
        "reporting_source_revision": plan.reporting_source_revision,
        "basis": {
            "basis_sha256": basis_sha256,
            "packed_sequences": int(len(global_ids)),
            "valid_tokens": int(global_valid.sum()),
            "packed_sequence_ids_sha256": canonical_sha256(global_ids.tolist()),
            "valid_tokens_sha256": canonical_sha256(global_valid.tolist()),
        },
        "tolerances": {
            "per_sequence_relative": CROSS_HARDWARE_SST_RTOL,
            "per_sequence_absolute": 0.0,
            "aggregate_relative": CROSS_HARDWARE_SST_SUM_RTOL,
            "aggregate_absolute": 0.0,
        },
        "observed_maxima": {
            "maximum_per_sequence_absolute_drift": maximum_absolute,
            "maximum_per_sequence_relative_drift": maximum_relative,
            "maximum_aggregate_relative_drift": (maximum_aggregate_relative),
        },
        "cells": cells,
    }
    return _self_hashed(body, field="audit_sha256")


def _output_paths(plan: ReportRecoveryPlan) -> dict[str, Path]:
    return {
        "basis_audit": plan.report_root / "heldout-basis-audit.json",
        "report": plan.report_root / "experiment5-result.json",
        "markdown": plan.report_root / "EXPERIMENT_5.md",
        "envelope": plan.report_root / "recovery-envelope.json",
        "final": plan.report_root / "final.json",
        "handoff": plan.report_root / "recovery-handoff.json",
    }


def _artifact_identity(
    path: Path,
    payload: bytes,
    *,
    scientific_field: str,
    scientific_sha256: str,
) -> dict[str, str]:
    return {
        "path": str(path),
        "file_sha256": hashlib.sha256(payload).hexdigest(),
        scientific_field: scientific_sha256,
    }


def _build_report_recovery_payloads(
    plan: ReportRecoveryPlan,
) -> tuple[dict[str, Any], tuple[tuple[str, bytes, str], ...]]:
    """Re-prove all parents and derive every deterministic sealed output."""

    parent = _verify_parent(plan)
    audit = _basis_audit(plan, parent)
    ordered = sorted(
        parent.cells,
        key=lambda row: (row.panel, row.seed, row.milestone),
    )
    try:
        report = build_experiment5_report(
            results=[row.result_path for row in ordered],
            paired_scores=[row.paired_path for row in ordered],
            budget_ledger=plan.parent.budget_ledger.path,
        )
        validate_experiment5_report(report)
        markdown = render_experiment5_report(report)
    except Experiment5ReportError as error:
        raise Experiment5ReportRecoveryError(
            f"Experiment-5 report assembly failed: {error}"
        ) from error

    paths = _output_paths(plan)
    audit_payload = _canonical_payload(audit)
    report_payload = _canonical_payload(report)
    markdown_payload = markdown.encode("utf-8")
    spent = float(parent.ledger["incremental_cash_spent"])

    envelope_body = {
        "schema_version": ENVELOPE_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": "report_recovery_validated",
        "recovery_id": plan.recovery_id,
        "report_recovery_plan_sha256": plan.plan_sha256,
        "reporting_source_revision": plan.reporting_source_revision,
        "evaluation_source_revision": (plan.parent.evaluation_source_revision),
        "parent": {
            "evaluation_plan": plan.parent.evaluation_plan.to_dict(),
            "evaluation_barrier": plan.parent.evaluation_barrier.to_dict(),
            "budget_ledger": plan.parent.budget_ledger.to_dict(),
        },
        "basis_audit": _artifact_identity(
            paths["basis_audit"],
            audit_payload,
            scientific_field="audit_sha256",
            scientific_sha256=str(audit["audit_sha256"]),
        ),
        "scientific_report": _artifact_identity(
            paths["report"],
            report_payload,
            scientific_field="report_sha256",
            scientific_sha256=str(report["report_sha256"]),
        ),
        "markdown": {
            "path": str(paths["markdown"]),
            "file_sha256": hashlib.sha256(markdown_payload).hexdigest(),
        },
        "spend": {
            "currency": "USD",
            "incremental_cash_spent": spent,
            "hard_limit": HARD_BUDGET_USD,
            "within_limit": spent <= HARD_BUDGET_USD,
        },
    }
    envelope = _self_hashed(envelope_body, field="envelope_sha256")
    envelope_payload = _canonical_payload(envelope)

    final_body = {
        "schema_version": FINAL_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": "report_recovery_sealed",
        "recovery_id": plan.recovery_id,
        "report_recovery_plan_sha256": plan.plan_sha256,
        "reporting_source_revision": plan.reporting_source_revision,
        "evaluation_source_revision": (plan.parent.evaluation_source_revision),
        "evaluation_plan_sha256": (plan.parent.evaluation_plan.scientific_sha256),
        "evaluation_barrier_record_sha256": (
            plan.parent.evaluation_barrier.scientific_sha256
        ),
        "budget_ledger_sha256": (plan.parent.budget_ledger.scientific_sha256),
        "basis_audit_sha256": audit["audit_sha256"],
        "report_sha256": report["report_sha256"],
        "envelope": _artifact_identity(
            paths["envelope"],
            envelope_payload,
            scientific_field="envelope_sha256",
            scientific_sha256=str(envelope["envelope_sha256"]),
        ),
        "incremental_cash_spent_usd": spent,
        "within_50_usd_limit": spent <= HARD_BUDGET_USD,
    }
    final = _self_hashed(final_body, field="final_sha256")
    final_payload = _canonical_payload(final)

    maxima = audit["observed_maxima"]
    handoff_body = {
        "schema_version": HANDOFF_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": "recovery_report_sealed",
        "recovery_id": plan.recovery_id,
        "report_recovery_plan_sha256": plan.plan_sha256,
        "reporting_source_revision": plan.reporting_source_revision,
        "evaluation_source_revision": (plan.parent.evaluation_source_revision),
        "evaluation_plan_file_sha256": (plan.parent.evaluation_plan.file_sha256),
        "evaluation_plan_sha256": (plan.parent.evaluation_plan.scientific_sha256),
        "evaluation_barrier_file_sha256": (plan.parent.evaluation_barrier.file_sha256),
        "evaluation_barrier_record_sha256": (
            plan.parent.evaluation_barrier.scientific_sha256
        ),
        "budget_ledger_file_sha256": (plan.parent.budget_ledger.file_sha256),
        "budget_ledger_sha256": (plan.parent.budget_ledger.scientific_sha256),
        "basis_audit_file_sha256": hashlib.sha256(audit_payload).hexdigest(),
        "basis_audit_sha256": audit["audit_sha256"],
        "report_file_sha256": hashlib.sha256(report_payload).hexdigest(),
        "report_sha256": report["report_sha256"],
        "final": _artifact_identity(
            paths["final"],
            final_payload,
            scientific_field="final_sha256",
            scientific_sha256=str(final["final_sha256"]),
        ),
        "sst_drift": {
            "per_sequence_relative_tolerance": (CROSS_HARDWARE_SST_RTOL),
            "aggregate_relative_tolerance": (CROSS_HARDWARE_SST_SUM_RTOL),
            **dict(maxima),
        },
        "incremental_cash_spent_usd": spent,
        "hard_limit_usd": HARD_BUDGET_USD,
        "within_limit": spent <= HARD_BUDGET_USD,
    }
    handoff = _self_hashed(handoff_body, field="handoff_sha256")
    validate_report_recovery_handoff(handoff, plan=plan)

    outputs = (
        ("basis_audit", audit_payload, "held-out basis audit"),
        ("report", report_payload, "Experiment-5 JSON report"),
        ("markdown", markdown_payload, "Experiment-5 Markdown report"),
        ("envelope", envelope_payload, "report-recovery envelope"),
        ("final", final_payload, "report-recovery final"),
        (
            "handoff",
            _canonical_payload(handoff),
            "report-recovery handoff",
        ),
    )
    return handoff, outputs


def seal_report_recovery(
    plan: ReportRecoveryPlan,
    *,
    verify_checkout: bool = True,
) -> dict[str, Any]:
    """Seal the audit, report, envelope, final, and handoff create-only."""

    if verify_checkout:
        _verify_clean_checkout(
            plan.working_directory,
            plan.reporting_source_revision,
        )
    handoff, outputs = _build_report_recovery_payloads(plan)
    paths = _output_paths(plan)
    for key, payload, name in outputs:
        _seal_exact_bytes(paths[key], payload, name=name)
    return handoff


def verify_report_recovery(
    plan: ReportRecoveryPlan,
    *,
    verify_checkout: bool = True,
) -> dict[str, Any]:
    """Reopen and re-prove every parent and output in one sealed recovery."""

    if verify_checkout:
        _verify_clean_checkout(
            plan.working_directory,
            plan.reporting_source_revision,
        )
    expected_handoff, outputs = _build_report_recovery_payloads(plan)
    paths = _output_paths(plan)
    for key, payload, name in outputs:
        observed = _regular_file_bytes(paths[key], name)
        if observed != payload:
            raise Experiment5ReportRecoveryError(
                f"sealed {name} changed: {paths[key]}"
            )

    # Return the reopened artifact rather than the in-memory candidate so that
    # callers receive evidence from the bytes which were actually verified.
    observed_handoff = validate_report_recovery_handoff(
        _read_json(paths["handoff"], "report-recovery handoff"),
        plan=plan,
    )
    if observed_handoff != expected_handoff:
        raise Experiment5ReportRecoveryError(
            "sealed report-recovery handoff changed"
        )
    return observed_handoff


def validate_report_recovery_handoff(
    value: Any,
    *,
    plan: ReportRecoveryPlan | None = None,
) -> dict[str, Any]:
    selected = _validate_self_hash(
        value,
        field="handoff_sha256",
        name="report-recovery handoff",
    )
    body = dict(selected)
    body.pop("handoff_sha256")
    required = {
        "schema_version",
        "protocol_id",
        "status",
        "recovery_id",
        "report_recovery_plan_sha256",
        "reporting_source_revision",
        "evaluation_source_revision",
        "evaluation_plan_file_sha256",
        "evaluation_plan_sha256",
        "evaluation_barrier_file_sha256",
        "evaluation_barrier_record_sha256",
        "budget_ledger_file_sha256",
        "budget_ledger_sha256",
        "basis_audit_file_sha256",
        "basis_audit_sha256",
        "report_file_sha256",
        "report_sha256",
        "final",
        "sst_drift",
        "incremental_cash_spent_usd",
        "hard_limit_usd",
        "within_limit",
    }
    if set(body) != required:
        raise Experiment5ReportRecoveryError("report-recovery handoff fields changed")
    if (
        body["schema_version"] != HANDOFF_SCHEMA_VERSION
        or body["protocol_id"] != PROTOCOL_ID
        or body["status"] != "recovery_report_sealed"
    ):
        raise Experiment5ReportRecoveryError("report-recovery handoff identity changed")
    _safe_id(body["recovery_id"], "handoff.recovery_id")
    for field in (
        "report_recovery_plan_sha256",
        "evaluation_plan_file_sha256",
        "evaluation_plan_sha256",
        "evaluation_barrier_file_sha256",
        "evaluation_barrier_record_sha256",
        "budget_ledger_file_sha256",
        "budget_ledger_sha256",
        "basis_audit_file_sha256",
        "basis_audit_sha256",
        "report_file_sha256",
        "report_sha256",
    ):
        _sha256(body[field], f"handoff.{field}")
    _git_sha(
        body["reporting_source_revision"],
        "handoff.reporting_source_revision",
    )
    _git_sha(
        body["evaluation_source_revision"],
        "handoff.evaluation_source_revision",
    )
    if (
        isinstance(body["incremental_cash_spent_usd"], bool)
        or not isinstance(body["incremental_cash_spent_usd"], (int, float))
        or not math.isfinite(float(body["incremental_cash_spent_usd"]))
        or not 0 <= float(body["incremental_cash_spent_usd"]) <= HARD_BUDGET_USD
        or body["hard_limit_usd"] != HARD_BUDGET_USD
        or body["within_limit"] is not True
    ):
        raise Experiment5ReportRecoveryError("report-recovery handoff spend is invalid")
    drift = body["sst_drift"]
    drift_fields = {
        "per_sequence_relative_tolerance",
        "aggregate_relative_tolerance",
        "maximum_per_sequence_absolute_drift",
        "maximum_per_sequence_relative_drift",
        "maximum_aggregate_relative_drift",
    }
    if (
        not isinstance(drift, Mapping)
        or set(drift) != drift_fields
        or drift.get("per_sequence_relative_tolerance")
        != CROSS_HARDWARE_SST_RTOL
        or drift.get("aggregate_relative_tolerance")
        != CROSS_HARDWARE_SST_SUM_RTOL
    ):
        raise Experiment5ReportRecoveryError(
            "report-recovery SST drift contract changed"
        )
    for field in (
        "maximum_per_sequence_absolute_drift",
        "maximum_per_sequence_relative_drift",
        "maximum_aggregate_relative_drift",
    ):
        value = drift[field]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            raise Experiment5ReportRecoveryError(
                f"report-recovery SST drift {field} is invalid"
            )
    if (
        float(drift["maximum_per_sequence_relative_drift"])
        > CROSS_HARDWARE_SST_RTOL
        or float(drift["maximum_aggregate_relative_drift"])
        > CROSS_HARDWARE_SST_SUM_RTOL
    ):
        raise Experiment5ReportRecoveryError(
            "report-recovery SST drift exceeds its sealed tolerance"
        )
    final = body["final"]
    if not isinstance(final, Mapping) or set(final) != {
        "path",
        "file_sha256",
        "final_sha256",
    }:
        raise Experiment5ReportRecoveryError("report-recovery final binding is invalid")
    final_path = _absolute_path(final["path"], "handoff.final.path")
    _sha256(final["file_sha256"], "handoff.final.file_sha256")
    _sha256(final["final_sha256"], "handoff.final.final_sha256")
    if plan is not None and (
        body["recovery_id"] != plan.recovery_id
        or body["report_recovery_plan_sha256"] != plan.plan_sha256
        or body["reporting_source_revision"] != plan.reporting_source_revision
        or body["evaluation_source_revision"] != plan.parent.evaluation_source_revision
        or body["evaluation_plan_file_sha256"]
        != plan.parent.evaluation_plan.file_sha256
        or body["evaluation_plan_sha256"]
        != plan.parent.evaluation_plan.scientific_sha256
        or body["evaluation_barrier_file_sha256"]
        != plan.parent.evaluation_barrier.file_sha256
        or body["evaluation_barrier_record_sha256"]
        != plan.parent.evaluation_barrier.scientific_sha256
        or body["budget_ledger_file_sha256"]
        != plan.parent.budget_ledger.file_sha256
        or body["budget_ledger_sha256"] != plan.parent.budget_ledger.scientific_sha256
        or final_path != plan.report_root / "final.json"
    ):
        raise Experiment5ReportRecoveryError(
            "report-recovery handoff differs from its plan"
        )
    return selected


def report_recovery_status(plan: ReportRecoveryPlan) -> dict[str, Any]:
    paths = _output_paths(plan)
    present = {
        key: path.is_file() and not path.is_symlink() for key, path in paths.items()
    }
    result: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "recovery_id": plan.recovery_id,
        "plan_sha256": plan.plan_sha256,
        "reporting_source_revision": plan.reporting_source_revision,
        "evaluation_source_revision": (plan.parent.evaluation_source_revision),
        "outputs": {
            key: {"path": str(paths[key]), "present": present[key]} for key in paths
        },
        "status": (
            "sealed"
            if all(present.values())
            else ("partial" if any(present.values()) else "pending")
        ),
    }
    if all(present.values()):
        handoff = verify_report_recovery(plan)
        result["handoff"] = handoff
    elif present["handoff"]:
        handoff = validate_report_recovery_handoff(
            _read_json(paths["handoff"], "report-recovery handoff"),
            plan=plan,
        )
        result["handoff"] = handoff
    else:
        result["handoff"] = None
    return result


__all__ = [
    "BASIS_AUDIT_SCHEMA_VERSION",
    "ENVELOPE_SCHEMA_VERSION",
    "FINAL_SCHEMA_VERSION",
    "HANDOFF_SCHEMA_VERSION",
    "PLAN_SCHEMA_VERSION",
    "Experiment5ReportRecoveryError",
    "ReportRecoveryPlan",
    "build_report_recovery_plan",
    "load_report_recovery_plan",
    "parse_report_recovery_plan",
    "report_recovery_status",
    "seal_report_recovery",
    "validate_report_recovery_handoff",
    "verify_report_recovery",
    "write_report_recovery_plan",
]
