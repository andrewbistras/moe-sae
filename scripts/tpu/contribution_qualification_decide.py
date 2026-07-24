#!/usr/bin/env python3
"""Seal one Experiment-2 qualification decision without launching work.

This helper runs on controller 0 after all four bounded ``launch-host``
processes have terminated.  It validates the immutable supervisor records and
either:

* seals a passing four-host receipt;
* permits the next, smaller declared batch only for an explicit OOM or a
  bounded static/persistence-headroom admission failure; or
* records an abort for every identity, transport, numerical, or unknown
  failure, including reaching the hard deadline.

It deliberately has no workload-launching command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from moe_sae.contribution_artifacts import (
    canonical_json_bytes,
    canonical_sha256,
    validate_recomposition_sidecar,
)
from moe_sae.contribution_maxtext import useful_flops_ledger
from moe_sae.contribution_performance import ContributionPerformanceError
from moe_sae.contribution_qualification_worker import (
    aggregate_supervised_host_evidence,
    validate_supervised_host_evidence,
)
from moe_sae.contribution_supervisor import (
    validate_execution_record,
    validate_supervision_result,
)
from moe_sae.runtime import V4_32_BF16_PEAK_FLOPS

PLAN_SCHEMA_VERSION = "gemma4-contribution-qualification-operator-plan-v1"
DECISION_SCHEMA_VERSION = "gemma4-contribution-qualification-operator-decision-v2"

PASS_EXIT = 0
FALLBACK_EXIT = 20
ABORT_EXIT = 30

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_OOM_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bRESOURCE_EXHAUSTED\b",
        r"\bout[ -]of[ -]memory\b",
        r"\bOOM\b",
        r"\bfailed to allocate\b",
        r"\ballocation (?:request )?failed\b",
    )
)
_FALLBACK_ADMISSION_MESSAGES = {
    "qualification observed an OOM": "explicit_oom",
    "writer backlog reached or exceeded its hard bound":
        "persistence_headroom",
    "persisted goodput is below 85% of device output goodput":
        "persistence_headroom",
    "auxiliary capture throughput is below 90% of writer-disabled":
        "persistence_headroom",
}
_LOW_MFU_ADMISSION_MESSAGE = "useful-prefix MFU is below the 10% hard gate"
_DECLARED_CANDIDATES = (
    {
        "candidate_id": "b128-ep8-fsdp2",
        "mesh": "ep8-fsdp2-tp1",
        "per_device_sequence_batch": 8,
        "global_sequence_batch": 128,
    },
    {
        "candidate_id": "b64-ep8-fsdp2",
        "mesh": "ep8-fsdp2-tp1",
        "per_device_sequence_batch": 4,
        "global_sequence_batch": 64,
    },
    {
        "candidate_id": "b32-ep8-fsdp2",
        "mesh": "ep8-fsdp2-tp1",
        "per_device_sequence_batch": 2,
        "global_sequence_batch": 32,
    },
)


class QualificationDecisionError(RuntimeError):
    """The operator decision could not be derived safely."""


def _read_json(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise QualificationDecisionError(f"cannot read {name}: {path}") from error
    if not isinstance(value, dict):
        raise QualificationDecisionError(f"{name} is not a JSON object: {path}")
    return value


def _sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
    except OSError as error:
        raise QualificationDecisionError(f"cannot hash {path}") from error
    return size, digest.hexdigest()


def _atomic_create_json(path: Path, value: Mapping[str, Any]) -> None:
    path = path.expanduser()
    if not path.is_absolute():
        raise QualificationDecisionError("output path must be absolute")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(dict(value))
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as error:
            raise QualificationDecisionError(
                f"immutable output already exists: {path}"
            ) from error
        directory = os.open(
            path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _self_hashed(
    body: Mapping[str, Any],
    *,
    hash_field: str,
) -> dict[str, Any]:
    unsigned = dict(body)
    return {
        **unsigned,
        hash_field: canonical_sha256(unsigned),
    }


def _parse_candidate(value: str) -> dict[str, Any]:
    parts = value.split(":")
    if len(parts) != 4:
        raise QualificationDecisionError(
            "candidate must be ID:MESH:PER_DEVICE_BATCH:GLOBAL_BATCH"
        )
    identifier, mesh, per_device, global_batch = parts
    if re.fullmatch(r"[a-z0-9][a-z0-9-]*", identifier) is None:
        raise QualificationDecisionError("candidate ID is invalid")
    if re.fullmatch(r"[a-z0-9][a-z0-9-]*", mesh) is None:
        raise QualificationDecisionError("candidate mesh is invalid")
    try:
        per_device_value = int(per_device)
        global_value = int(global_batch)
    except ValueError as error:
        raise QualificationDecisionError("candidate batches must be integers") from error
    if per_device_value <= 0 or global_value != per_device_value * 16:
        raise QualificationDecisionError(
            "candidate global batch must equal per-device batch times 16"
        )
    return {
        "candidate_id": identifier,
        "mesh": mesh,
        "per_device_sequence_batch": per_device_value,
        "global_sequence_batch": global_value,
    }


def initialize_plan(args: argparse.Namespace) -> int:
    commit = args.project_commit
    checkpoint_sha256 = args.checkpoint_sha256
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise QualificationDecisionError("project commit must be a full Git SHA")
    if _SHA256_RE.fullmatch(checkpoint_sha256) is None:
        raise QualificationDecisionError("checkpoint SHA-256 is invalid")
    config = Path(args.config).expanduser().resolve(strict=True)
    if not config.is_file():
        raise QualificationDecisionError("Experiment-2 config is not a file")
    candidates = [_parse_candidate(value) for value in args.candidate]
    if tuple(candidates) != _DECLARED_CANDIDATES:
        raise QualificationDecisionError(
            "operator plan candidate order differs from the frozen "
            "B128/B64/B32 sweep"
        )
    body = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "run_id": args.run_id,
        "project_commit": commit,
        "config_path": str(config),
        "config_sha256": _sha256_file(config)[1],
        "checkpoint_path": args.checkpoint,
        "checkpoint_sha256": checkpoint_sha256,
        "qualification_hard_deadline_seconds": 600,
        "candidate_order": candidates,
        "terminal_action": "stop_after_first_pass_no_workload_launch",
    }
    record = _self_hashed(body, hash_field="operator_plan_sha256")
    _atomic_create_json(Path(args.output), record)
    print(record["operator_plan_sha256"], flush=True)
    return PASS_EXIT


def _validated_host_records(
    paths: Sequence[Path],
) -> list[tuple[Path, dict[str, Any]]]:
    if len(paths) != 4:
        raise QualificationDecisionError(
            "candidate decision requires exactly four host-evidence paths"
        )
    checked = [
        (
            path,
            validate_supervised_host_evidence(
                _read_json(path, name="host evidence")
            ),
        )
        for path in paths
    ]
    checked.sort(key=lambda item: int(item[1]["controller_index"]))
    if [int(item[1]["controller_index"]) for item in checked] != [0, 1, 2, 3]:
        raise QualificationDecisionError(
            "host evidence controller indexes are incomplete or duplicated"
        )
    if all(host["status"] == "coordinator_ready" for _, host in checked):
        process_indices = [int(host["process_index"]) for _, host in checked]
        if sorted(process_indices) != [0, 1, 2, 3]:
            raise QualificationDecisionError(
                "host evidence JAX process indexes are incomplete or duplicated"
            )
    return checked


def _validated_supervisor_log(
    host_path: Path,
    host: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    execution = validate_execution_record(
        _read_json(host_path.parent / "execution.json", name="execution record")
    )
    result = validate_supervision_result(
        _read_json(
            host_path.parent / "supervision-result.json",
            name="supervision result",
        ),
        expected_execution_sha256=execution["execution_sha256"],
    )
    if (
        execution["execution_sha256"] != host["supervisor_execution_sha256"]
        or result["result_sha256"] != host["supervisor_result_sha256"]
        or result["status"] != host["supervisor_status"]
    ):
        raise QualificationDecisionError(
            "host evidence does not bind its exact supervisor records"
        )
    log_path = Path(execution["log_path"])
    if log_path != host_path.parent / "qualification.log":
        raise QualificationDecisionError(
            "supervisor log path is outside the candidate host directory"
        )
    size, digest = _sha256_file(log_path)
    if size != result["log_size_bytes"] or digest != result["log_sha256"]:
        raise QualificationDecisionError(
            "supervisor log differs from its terminal result hash"
        )
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        raise QualificationDecisionError("cannot inspect supervisor log") from error
    return result, text


def _terminal_failure_classification(
    checked: Sequence[tuple[Path, Mapping[str, Any]]],
) -> tuple[str, str] | None:
    failed = [(path, host) for path, host in checked if host["status"] != "coordinator_ready"]
    if not failed:
        return None

    oom_controllers: list[int] = []
    unknown: list[str] = []
    for path, host in failed:
        result, log = _validated_supervisor_log(path, host)
        status = result["status"]
        controller_index = int(host["controller_index"])
        if status in {"terminated_at_deadline", "killed_at_deadline"}:
            unknown.append(
                f"controller {controller_index}: hard deadline reached ({status})"
            )
        elif status == "failed" and any(pattern.search(log) for pattern in _OOM_PATTERNS):
            oom_controllers.append(controller_index)
        else:
            unknown.append(
                f"controller {controller_index}: supervisor_status={status}"
            )
    if unknown:
        raise QualificationDecisionError(
            "candidate has a deadline or unknown controller failure; explicit "
            "OOM evidence on another controller cannot authorize fallback; "
            + "; ".join(unknown)
        )
    if oom_controllers:
        return (
            "explicit_oom",
            (
                "a hash-verified supervisor log contains an explicit OOM marker on "
                f"controllers {oom_controllers}"
            ),
        )
    raise QualificationDecisionError(
        "candidate failed without an explicit OOM; fallback is not authorized"
    )


def _identical_recomposition_sidecar(
    checked: Sequence[tuple[Path, Mapping[str, Any]]],
) -> dict[str, Any]:
    sidecars = [
        validate_recomposition_sidecar(
            _read_json(
                path.parent / "recomposition-sidecar.json",
                name="recomposition sidecar",
            )
        )
        for path, _ in checked
    ]
    canonical_sidecars = [canonical_json_bytes(value) for value in sidecars]
    if any(value != canonical_sidecars[0] for value in canonical_sidecars[1:]):
        raise QualificationDecisionError(
            "controllers published non-identical recomposition sidecars"
        )
    selected = sidecars[0]
    for (_, host), value in zip(checked, sidecars, strict=True):
        if (
            host["recomposition_sidecar_sha256"] != value["sidecar_sha256"]
            or host["recomposition_sidecar_canonical_sha256"]
            != canonical_sha256(value)
        ):
            raise QualificationDecisionError(
                "host evidence does not bind its exact recomposition sidecar"
            )
    return selected


def _decision_record(
    *,
    args: argparse.Namespace,
    checked: Sequence[tuple[Path, Mapping[str, Any]]],
    outcome: str,
    reason_code: str,
    reason: str,
    coordinator_bundle_sha256: str | None,
    recomposition_sidecar_sha256: str | None,
    qualification_metrics: Mapping[str, Any] | None,
) -> dict[str, Any]:
    body = {
        "schema_version": DECISION_SCHEMA_VERSION,
        "candidate_id": args.candidate_id,
        "mesh": args.mesh,
        "per_device_sequence_batch": args.per_device_sequence_batch,
        "global_sequence_batch": args.batch_sequences,
        "outcome": outcome,
        "reason_code": reason_code,
        "reason": reason,
        "host_evidence_sha256": [
            host["host_evidence_sha256"] for _, host in checked
        ],
        "coordinator_bundle_sha256": coordinator_bundle_sha256,
        "recomposition_sidecar_sha256": recomposition_sidecar_sha256,
        "qualification_metrics": (
            None if qualification_metrics is None else dict(qualification_metrics)
        ),
        "next_action": (
            "stop_no_workload_launch"
            if outcome == "passed"
            else (
                "next_declared_smaller_batch"
                if outcome == "fallback_allowed"
                else "abort_operator_investigation_required"
            )
        ),
    }
    return _self_hashed(body, hash_field="operator_decision_sha256")


def _qualification_metrics(
    checked: Sequence[tuple[Path, Mapping[str, Any]]],
    *,
    batch_sequences: int,
    sequence_length: int,
) -> dict[str, Any]:
    """Recover the exact measured scalars after a clean admission rejection."""

    observations = [host["observation"] for _, host in checked]
    if len(observations) != 4 or not all(
        isinstance(value, Mapping) for value in observations
    ):
        raise QualificationDecisionError(
            "low-MFU decision lacks four validated host observations"
        )

    def common_int(name: str) -> int:
        values = [value.get(name) for value in observations]
        if (
            any(isinstance(value, bool) or not isinstance(value, int) for value in values)
            or len(set(values)) != 1
        ):
            raise QualificationDecisionError(
                f"low-MFU host observations disagree on {name}"
            )
        return int(values[0])

    def positive_max(name: str) -> float:
        values = [value.get(name) for value in observations]
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            for value in values
        ):
            raise QualificationDecisionError(
                f"low-MFU host observations contain invalid {name}"
            )
        return max(float(value) for value in values)

    measured_steps = common_int("synchronized_measurement_steps")
    global_output_tokens = common_int("global_output_tokens")
    writer_disabled_output_tokens = common_int(
        "writer_disabled_global_output_tokens"
    )
    local_persisted = [
        value.get("local_persisted_tokens") for value in observations
    ]
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in local_persisted
    ):
        raise QualificationDecisionError(
            "low-MFU host observations contain invalid persisted token counts"
        )
    measured_seconds = positive_max("measured_device_seconds")
    persisted_seconds = positive_max("persisted_window_seconds")
    writer_disabled_seconds = positive_max("writer_disabled_device_seconds")
    ledger = useful_flops_ledger(
        batch_sequences=batch_sequences,
        sequence_length=sequence_length,
        measured_steps=measured_steps,
    )
    useful_flops = int(ledger["useful_flops"])
    persisted_tokens = sum(int(value) for value in local_persisted)
    device_output_tps = global_output_tokens / measured_seconds
    persisted_tps = persisted_tokens / persisted_seconds
    writer_disabled_tps = (
        writer_disabled_output_tokens / writer_disabled_seconds
    )
    useful_mfu = useful_flops / (
        float(V4_32_BF16_PEAK_FLOPS) * measured_seconds
    )
    metrics = {
        "useful_prefix_mfu": useful_mfu,
        "minimum_useful_prefix_mfu": 0.10,
        "useful_flops": useful_flops,
        "measured_device_seconds": measured_seconds,
        "device_output_tokens_per_second": device_output_tps,
        "persisted_tokens_per_second": persisted_tps,
        "writer_disabled_tokens_per_second": writer_disabled_tps,
        "auxiliary_throughput_fraction": (
            device_output_tps / writer_disabled_tps
        ),
        "output_tokens": global_output_tokens,
        "persisted_tokens": persisted_tokens,
        "synchronized_measurement_steps": measured_steps,
    }
    if (
        not 0.0 < useful_mfu < 0.10
        or any(
            not math.isfinite(float(value))
            for value in metrics.values()
            if isinstance(value, float)
        )
    ):
        raise QualificationDecisionError(
            "low-MFU admission error is inconsistent with measured metrics"
        )
    return metrics


def decide_candidate(args: argparse.Namespace) -> int:
    selected_candidate = {
        "candidate_id": args.candidate_id,
        "mesh": args.mesh,
        "per_device_sequence_batch": args.per_device_sequence_batch,
        "global_sequence_batch": args.batch_sequences,
    }
    if selected_candidate not in _DECLARED_CANDIDATES:
        raise QualificationDecisionError(
            "candidate is outside the frozen B128/B64/B32 sweep"
        )
    paths = [Path(value).expanduser().resolve(strict=True) for value in args.host_evidence]
    checked = _validated_host_records(paths)
    terminal_failure = _terminal_failure_classification(checked)
    bundle: dict[str, Any] | None = None
    sidecar: dict[str, Any] | None = None
    outcome = "passed"
    reason_code = "qualification_passed"
    reason = "all four exact host observations passed aggregation"
    exit_code = PASS_EXIT
    qualification_metrics: dict[str, Any] | None = None

    if terminal_failure is not None:
        reason_code, reason = terminal_failure
        outcome = "fallback_allowed"
        exit_code = FALLBACK_EXIT
    else:
        sidecar = _identical_recomposition_sidecar(checked)
        try:
            bundle = aggregate_supervised_host_evidence(
                [host for _, host in checked],
                batch_sequences=args.batch_sequences,
                sequence_length=args.sequence_length,
            )
        except ContributionPerformanceError as error:
            reason = str(error)
            if reason == _LOW_MFU_ADMISSION_MESSAGE:
                reason_code = "useful_mfu_below_hard_gate"
                outcome = "aborted"
                exit_code = ABORT_EXIT
                qualification_metrics = _qualification_metrics(
                    checked,
                    batch_sequences=args.batch_sequences,
                    sequence_length=args.sequence_length,
                )
            else:
                reason_code = _FALLBACK_ADMISSION_MESSAGES.get(reason, "")
            if not reason_code:
                raise QualificationDecisionError(
                    "qualification admission failed for a non-headroom reason: "
                    f"{reason}"
                ) from error
            if reason != _LOW_MFU_ADMISSION_MESSAGE:
                outcome = "fallback_allowed"
                exit_code = FALLBACK_EXIT

    bundle_sha256: str | None = None
    if bundle is not None:
        bundle_sha256 = str(bundle["coordinator_bundle_sha256"])
        if (
            sidecar is None
            or bundle["recomposition_sidecar_sha256"]
            != sidecar["sidecar_sha256"]
            or bundle["recomposition_sidecar_canonical_sha256"]
            != canonical_sha256(sidecar)
        ):
            raise QualificationDecisionError(
                "coordinator bundle sidecar identity is inconsistent"
            )
        _atomic_create_json(Path(args.output), bundle)
        _atomic_create_json(
            Path(args.qualification_receipt),
            bundle["qualification_receipt"],
        )
        _atomic_create_json(Path(args.recomposition_sidecar), sidecar)
    decision = _decision_record(
        args=args,
        checked=checked,
        outcome=outcome,
        reason_code=reason_code,
        reason=reason,
        coordinator_bundle_sha256=bundle_sha256,
        recomposition_sidecar_sha256=(
            None if sidecar is None else str(sidecar["sidecar_sha256"])
        ),
        qualification_metrics=qualification_metrics,
    )
    _atomic_create_json(Path(args.decision), decision)
    print(json.dumps(decision, sort_keys=True), flush=True)
    return exit_code


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Decide a bounded Experiment-2 qualification candidate"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    initialize = commands.add_parser("init-plan")
    initialize.add_argument("--run-id", required=True)
    initialize.add_argument("--project-commit", required=True)
    initialize.add_argument("--config", required=True)
    initialize.add_argument("--checkpoint", required=True)
    initialize.add_argument("--checkpoint-sha256", required=True)
    initialize.add_argument("--candidate", action="append", required=True)
    initialize.add_argument("--output", required=True)

    decide = commands.add_parser("decide")
    decide.add_argument("--candidate-id", required=True)
    decide.add_argument("--mesh", required=True)
    decide.add_argument("--per-device-sequence-batch", type=int, required=True)
    decide.add_argument("--batch-sequences", type=int, required=True)
    decide.add_argument("--sequence-length", type=int, default=1_024)
    decide.add_argument("--host-evidence", action="append", required=True)
    decide.add_argument("--output", required=True)
    decide.add_argument("--decision", required=True)
    decide.add_argument("--qualification-receipt", required=True)
    decide.add_argument("--recomposition-sidecar", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "init-plan":
            return initialize_plan(args)
        if args.command == "decide":
            return decide_candidate(args)
        raise AssertionError(f"unhandled command {args.command}")
    except (
        QualificationDecisionError,
        ContributionPerformanceError,
        OSError,
        ValueError,
    ) as error:
        print(f"[qualification-decision] ABORT: {error}", file=sys.stderr, flush=True)
        return ABORT_EXIT


if __name__ == "__main__":
    raise SystemExit(main())
