"""JAX-free multi-process transaction wrapper for SAE full-shape smokes.

``SAESmokeRunner`` owns the durable, single-coordinator POSIX transaction and
``SAESmokeTPUAdapter`` owns numerical callbacks which every JAX process must
enter.  Calling ``SAESmokeRunner.run_attempt`` independently on every process
would deadlock: process 0 would hold the campaign lock while the other
processes waited on that same lock instead of entering the numerical
collective.  This module bridges those two contracts.

Only process zero acquires the campaign lock or mutates campaign records.  It
broadcasts the exact reserved attempt identity; every process independently
revalidates that identity, executes the same callback stage, and joins a
decision barrier before any process advances or raises.  The coordinator also
broadcasts the exact terminal record and every process revalidates it before
the lock is released.

The collective primitives are injected deliberately.  Importing this module
does not import JAX, Orbax, or cloud libraries.  A production hook must provide
an exact process-zero mapping broadcast and an all-process decision barrier;
if a collective primitive itself fails, distributed progress can no longer be
guaranteed by a Python protocol.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
import re
from typing import Any

from moe_sae.sae_smoke import (
    SMOKE_CONDITIONS,
    FinalizedCheckpoint,
    SAESmokeAttemptFailed,
    SAESmokeError,
    SAESmokeRunner,
    SmokeAttemptIdentity,
    SmokeAttemptPaths,
    SmokeCheckpointExpectation,
    SmokeNumericalCallbacks,
    SmokeOutcome,
    SmokeRestoreResult,
    SmokeUpdateResult,
    _attempt_from_mapping,
    _exclusive_campaign_lock,
    _read_json_object,
    canonical_sha256,
    new_smoke_attempt_id,
)


__all__ = [
    "CollectiveDecision",
    "SAESmokeCollectiveError",
    "SAESmokeCollectiveHooks",
    "SAESmokeCollectiveRunner",
    "single_process_collective_hooks",
]


COLLECTIVE_BOOTSTRAP_SCHEMA = "gemma4-sae-smoke-collective-bootstrap-v1"
COLLECTIVE_REPORT_SCHEMA = "gemma4-sae-smoke-collective-report-v1"
COLLECTIVE_TERMINAL_SCHEMA = "gemma4-sae-smoke-collective-terminal-v1"

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_ATTEMPT_ID_RE = re.compile(r"[0-9a-f]{32}")


class SAESmokeCollectiveError(SAESmokeError):
    """The multi-process smoke protocol cannot safely continue."""


@dataclass(frozen=True, slots=True)
class SAESmokeCollectiveHooks:
    """Injected process identity and exact collective operations.

    ``broadcast_mapping`` is called by every process with the same label and
    source index.  Only the source process supplies a mapping; other processes
    supply ``None``.  It must return the exact source mapping to every process.

    ``decision_barrier`` is an all-gather barrier.  It is called by every
    process with one self-hashed report and must return exactly one report for
    every process.  Order is irrelevant because explicit process indexes are
    checked and normalized here.  No caller may return from either primitive
    early on a peer's reported application error.
    """

    process_index: int
    process_count: int
    coordinator_index: int
    broadcast_mapping: Callable[
        [str, int, Mapping[str, Any] | None], Mapping[str, Any]
    ]
    decision_barrier: Callable[
        [str, Mapping[str, Any]], Sequence[Mapping[str, Any]]
    ]

    def __post_init__(self) -> None:
        for name in ("process_index", "process_count", "coordinator_index"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.process_count <= 0:
            raise ValueError("process_count must be positive")
        if not 0 <= self.process_index < self.process_count:
            raise ValueError("process_index is outside process_count")
        if self.coordinator_index != 0:
            raise ValueError("the SAE smoke coordinator must be process 0")
        if not callable(self.broadcast_mapping) or not callable(
            self.decision_barrier
        ):
            raise TypeError("collective hooks must be callable")


@dataclass(frozen=True, slots=True)
class CollectiveDecision:
    """Canonical result derived independently from all process reports."""

    stage: str
    attempt_sha256: str | None
    reports: tuple[Mapping[str, Any], ...]
    ok: bool
    value_sha256: str | None
    decision_sha256: str


@dataclass(frozen=True, slots=True)
class _FailureDetail:
    stage: str
    process_index: int
    error_type: str
    error_message: str

    @property
    def message(self) -> str:
        return (
            f"process {self.process_index} {self.error_type}: "
            f"{self.error_message}"
        )


class _CollectiveStageFailure(RuntimeError):
    def __init__(self, detail: _FailureDetail) -> None:
        super().__init__(detail.message)
        self.detail = detail


def _error_mapping(error: BaseException) -> dict[str, str]:
    return {
        "error_type": type(error).__name__,
        "error_message": str(error) or type(error).__name__,
    }


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise SAESmokeCollectiveError(f"{label} must be a lowercase SHA-256")
    return value


def _report(
    *,
    stage: str,
    process_index: int,
    attempt_sha256: str | None,
    value: Mapping[str, Any] | None = None,
    error: BaseException | None = None,
) -> dict[str, Any]:
    if (value is None) == (error is None):
        raise ValueError("a collective report must contain one value or one error")
    payload: dict[str, Any] = {
        "schema_version": COLLECTIVE_REPORT_SCHEMA,
        "stage": stage,
        "process_index": process_index,
        "attempt_sha256": attempt_sha256,
        "status": "passed" if error is None else "failed",
        "value_sha256": None if value is None else canonical_sha256(dict(value)),
        "error": None if error is None else _error_mapping(error),
    }
    payload["report_sha256"] = canonical_sha256(payload)
    return payload


def _validate_report(
    value: Mapping[str, Any],
    *,
    stage: str,
    process_count: int,
    attempt_sha256: str | None,
) -> dict[str, Any]:
    report = dict(value)
    expected_keys = {
        "schema_version",
        "stage",
        "process_index",
        "attempt_sha256",
        "status",
        "value_sha256",
        "error",
        "report_sha256",
    }
    if set(report) != expected_keys or report.get("schema_version") != (
        COLLECTIVE_REPORT_SCHEMA
    ):
        raise SAESmokeCollectiveError("collective report fields do not match schema")
    supplied_hash = report.get("report_sha256")
    if not isinstance(supplied_hash, str) or (
        canonical_sha256(
            {key: item for key, item in report.items() if key != "report_sha256"}
        )
        != supplied_hash
    ):
        raise SAESmokeCollectiveError("collective report self-hash mismatch")
    if report.get("stage") != stage:
        raise SAESmokeCollectiveError("collective report stage mismatch")
    process_index = report.get("process_index")
    if (
        isinstance(process_index, bool)
        or not isinstance(process_index, int)
        or not 0 <= process_index < process_count
    ):
        raise SAESmokeCollectiveError("collective report process index is invalid")
    if report.get("attempt_sha256") != attempt_sha256:
        raise SAESmokeCollectiveError("collective report attempt identity mismatch")
    status = report.get("status")
    if status == "passed":
        _require_sha256(report.get("value_sha256"), "collective value_sha256")
        if report.get("error") is not None:
            raise SAESmokeCollectiveError("passed collective report contains an error")
    elif status == "failed":
        if report.get("value_sha256") is not None:
            raise SAESmokeCollectiveError("failed collective report contains a value")
        error = report.get("error")
        if not isinstance(error, Mapping) or set(error) != {
            "error_type",
            "error_message",
        }:
            raise SAESmokeCollectiveError("failed collective report error is invalid")
        if not all(isinstance(item, str) and item for item in error.values()):
            raise SAESmokeCollectiveError("failed collective error detail is empty")
    else:
        raise SAESmokeCollectiveError("collective report status is invalid")
    return report


def _decision(
    *,
    hooks: SAESmokeCollectiveHooks,
    stage: str,
    attempt_sha256: str | None,
    value: Mapping[str, Any] | None,
    error: BaseException | None,
) -> CollectiveDecision:
    if error is not None:
        value = None
    local = _report(
        stage=stage,
        process_index=hooks.process_index,
        attempt_sha256=attempt_sha256,
        value=value,
        error=error,
    )
    gathered = tuple(hooks.decision_barrier(stage, local))
    if len(gathered) != hooks.process_count:
        raise SAESmokeCollectiveError(
            "collective decision returned the wrong process count"
        )
    by_index: dict[int, dict[str, Any]] = {}
    for item in gathered:
        if not isinstance(item, Mapping):
            raise SAESmokeCollectiveError(
                "collective decision returned a non-mapping report"
            )
        validated = _validate_report(
            item,
            stage=stage,
            process_count=hooks.process_count,
            attempt_sha256=attempt_sha256,
        )
        index = int(validated["process_index"])
        if index in by_index:
            raise SAESmokeCollectiveError(
                "collective decision contains duplicate process indexes"
            )
        by_index[index] = validated
    if tuple(sorted(by_index)) != tuple(range(hooks.process_count)):
        raise SAESmokeCollectiveError("collective decision omitted a process report")
    if by_index[hooks.process_index] != local:
        raise SAESmokeCollectiveError("collective decision changed the local report")
    ordered: tuple[Mapping[str, Any], ...] = tuple(
        by_index[index] for index in range(hooks.process_count)
    )
    passed = tuple(item for item in ordered if item["status"] == "passed")
    ok = len(passed) == hooks.process_count
    value_sha256: str | None = None
    if ok:
        values = {str(item["value_sha256"]) for item in passed}
        if len(values) != 1:
            ok = False
        else:
            value_sha256 = values.pop()
    deterministic = {
        "schema_version": "gemma4-sae-smoke-collective-decision-v1",
        "stage": stage,
        "attempt_sha256": attempt_sha256,
        "reports": [dict(item) for item in ordered],
        "ok": ok,
        "value_sha256": value_sha256,
    }
    return CollectiveDecision(
        stage=stage,
        attempt_sha256=attempt_sha256,
        reports=ordered,
        ok=ok,
        value_sha256=value_sha256,
        decision_sha256=canonical_sha256(deterministic),
    )


def _failure_detail(decision: CollectiveDecision) -> _FailureDetail:
    for report in decision.reports:
        if report["status"] == "failed":
            error = report["error"]
            assert isinstance(error, Mapping)
            return _FailureDetail(
                stage=decision.stage,
                process_index=int(report["process_index"]),
                error_type=str(error["error_type"]),
                error_message=str(error["error_message"]),
            )
    # All processes returned success, but their deterministic values differed.
    return _FailureDetail(
        stage=decision.stage,
        process_index=0,
        error_type="SAESmokeCollectiveError",
        error_message="processes returned different deterministic stage values",
    )


def _expectation_mapping(value: SmokeCheckpointExpectation) -> dict[str, Any]:
    return asdict(value)


def _checkpoint_mapping(value: FinalizedCheckpoint) -> dict[str, Any]:
    return asdict(value)


def _restore_mapping(value: SmokeRestoreResult) -> dict[str, Any]:
    return {
        "step": value.step,
        "tokens_seen": value.tokens_seen,
        "optimizer_updates": value.optimizer_updates,
        "attempt_sha256": value.attempt_sha256,
        "condition_shape_sha256": value.condition_shape_sha256,
        "state_sha256": value.state_sha256,
        "checkpoint_sha256": value.checkpoint_sha256,
    }


def _bootstrap_payload(
    runner: SAESmokeRunner,
    *,
    condition: str,
    requested_attempt_id: str | None,
) -> tuple[dict[str, Any], SmokeAttemptIdentity | None, SmokeAttemptPaths | None]:
    attempt: SmokeAttemptIdentity | None = None
    paths: SmokeAttemptPaths | None = None
    error: BaseException | None = None
    stage = "attempt-reservation"
    selected_id = requested_attempt_id or new_smoke_attempt_id()
    try:
        attempt, paths = runner._reserve_attempt(condition, selected_id)  # noqa: SLF001
        stage = "sequence-validation"
        runner._validate_sequence(condition)  # noqa: SLF001
    except BaseException as caught:  # preserve coordinator errors for all peers
        error = caught
    payload: dict[str, Any] = {
        "schema_version": COLLECTIVE_BOOTSTRAP_SCHEMA,
        "campaign_sha256": runner.campaign.sha256,
        "condition": condition,
        "attempt_id": selected_id,
        "attempt": None if attempt is None else attempt.to_mapping(),
        "setup_stage": stage,
        "setup_error": None if error is None else _error_mapping(error),
    }
    payload["bootstrap_sha256"] = canonical_sha256(payload)
    return payload, attempt, paths


def _bootstrap_error_payload(
    runner: SAESmokeRunner,
    *,
    condition: str,
    attempt_id: str,
    stage: str,
    error: BaseException,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": COLLECTIVE_BOOTSTRAP_SCHEMA,
        "campaign_sha256": runner.campaign.sha256,
        "condition": condition,
        "attempt_id": attempt_id,
        "attempt": None,
        "setup_stage": stage,
        "setup_error": _error_mapping(error),
    }
    payload["bootstrap_sha256"] = canonical_sha256(payload)
    return payload


def _validate_bootstrap(
    runner: SAESmokeRunner,
    payload_value: Mapping[str, Any],
    *,
    condition: str,
    requested_attempt_id: str | None,
    callbacks: object,
) -> tuple[SmokeAttemptIdentity | None, SmokeAttemptPaths | None]:
    payload = dict(payload_value)
    expected_keys = {
        "schema_version",
        "campaign_sha256",
        "condition",
        "attempt_id",
        "attempt",
        "setup_stage",
        "setup_error",
        "bootstrap_sha256",
    }
    if set(payload) != expected_keys or payload.get("schema_version") != (
        COLLECTIVE_BOOTSTRAP_SCHEMA
    ):
        raise SAESmokeCollectiveError("collective bootstrap fields do not match schema")
    if canonical_sha256(
        {key: value for key, value in payload.items() if key != "bootstrap_sha256"}
    ) != payload.get("bootstrap_sha256"):
        raise SAESmokeCollectiveError("collective bootstrap self-hash mismatch")
    if payload.get("campaign_sha256") != runner.campaign.sha256:
        raise SAESmokeCollectiveError("collective bootstrap campaign mismatch")
    if payload.get("condition") != condition:
        raise SAESmokeCollectiveError("collective bootstrap condition mismatch")
    attempt_id = payload.get("attempt_id")
    if not isinstance(attempt_id, str) or _ATTEMPT_ID_RE.fullmatch(attempt_id) is None:
        raise SAESmokeCollectiveError("collective bootstrap attempt ID is invalid")
    if requested_attempt_id is not None and requested_attempt_id != attempt_id:
        raise SAESmokeCollectiveError("processes requested different smoke attempt IDs")
    if not isinstance(callbacks, SmokeNumericalCallbacks):
        raise TypeError("callbacks must be SmokeNumericalCallbacks")
    attempt_value = payload.get("attempt")
    attempt: SmokeAttemptIdentity | None = None
    paths: SmokeAttemptPaths | None = None
    if attempt_value is not None:
        if not isinstance(attempt_value, Mapping):
            raise SAESmokeCollectiveError("collective attempt identity is not a mapping")
        attempt = _attempt_from_mapping(attempt_value)
        if attempt.attempt_id != attempt_id or attempt.condition != condition:
            raise SAESmokeCollectiveError("collective attempt identity mismatch")
        paths = runner.paths(
            attempt.condition,
            attempt.attempt_id,
            condition_attempt_number=attempt.condition_attempt_number,
        )
        visible, _record = runner._validate_attempt_record(  # noqa: SLF001
            paths.attempt_record,
            expected_attempt_id=attempt.attempt_id,
        )
        if visible != attempt:
            raise SAESmokeCollectiveError(
                "broadcast attempt differs from the durable attempt record"
            )
    setup_error = payload.get("setup_error")
    if setup_error is not None:
        if not isinstance(setup_error, Mapping) or set(setup_error) != {
            "error_type",
            "error_message",
        }:
            raise SAESmokeCollectiveError("collective setup error is invalid")
        raise SAESmokeCollectiveError(
            f"coordinator {setup_error['error_type']}: "
            f"{setup_error['error_message']}"
        )
    if attempt is None or paths is None:
        raise SAESmokeCollectiveError("coordinator reserved no smoke attempt")
    return attempt, paths


def _terminal_payload(
    *,
    status: str,
    attempt: SmokeAttemptIdentity,
    record: Mapping[str, Any] | None,
    publication_error: BaseException | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": COLLECTIVE_TERMINAL_SCHEMA,
        "status": status,
        "attempt_sha256": attempt.sha256,
        "record": None if record is None else dict(record),
        "publication_error": (
            None if publication_error is None else _error_mapping(publication_error)
        ),
    }
    payload["terminal_sha256"] = canonical_sha256(payload)
    return payload


def _validate_terminal_payload(
    runner: SAESmokeRunner,
    payload_value: Mapping[str, Any],
    *,
    attempt: SmokeAttemptIdentity,
    paths: SmokeAttemptPaths,
) -> tuple[SmokeOutcome | None, dict[str, Any]]:
    payload = dict(payload_value)
    expected_keys = {
        "schema_version",
        "status",
        "attempt_sha256",
        "record",
        "publication_error",
        "terminal_sha256",
    }
    if set(payload) != expected_keys or payload.get("schema_version") != (
        COLLECTIVE_TERMINAL_SCHEMA
    ):
        raise SAESmokeCollectiveError("collective terminal fields do not match schema")
    if canonical_sha256(
        {key: value for key, value in payload.items() if key != "terminal_sha256"}
    ) != payload.get("terminal_sha256"):
        raise SAESmokeCollectiveError("collective terminal self-hash mismatch")
    if payload.get("attempt_sha256") != attempt.sha256:
        raise SAESmokeCollectiveError("collective terminal attempt mismatch")
    publication_error = payload.get("publication_error")
    if publication_error is not None:
        if not isinstance(publication_error, Mapping):
            raise SAESmokeCollectiveError("terminal publication error is invalid")
        raise SAESmokeCollectiveError(
            f"coordinator publication failed: "
            f"{publication_error.get('error_type')}: "
            f"{publication_error.get('error_message')}"
        )
    record_value = payload.get("record")
    if not isinstance(record_value, Mapping):
        raise SAESmokeCollectiveError("collective terminal contains no record")
    status = payload.get("status")
    if status == "passed":
        visible = runner._validate_result_record(  # noqa: SLF001
            paths.result_record, attempt
        )
        if visible != dict(record_value):
            raise SAESmokeCollectiveError(
                "broadcast result differs from durable smoke result"
            )
        outcome = SmokeOutcome(
            attempt=attempt,
            paths=paths,
            checkpoint_sha256=str(visible["checkpoint_sha256"]),
            state_sha256=str(visible["state_sha256"]),
            outcome_sha256=str(visible["outcome_sha256"]),
            record_sha256=str(visible["record_sha256"]),
        )
        return outcome, {
            "status": "passed",
            "outcome_sha256": outcome.outcome_sha256,
            "record_sha256": outcome.record_sha256,
        }
    if status == "failed":
        visible = runner._validate_failure_record(  # noqa: SLF001
            paths.failure_record, attempt
        )
        if visible != dict(record_value):
            raise SAESmokeCollectiveError(
                "broadcast failure differs from durable smoke failure"
            )
        return None, {
            "status": "failed",
            "record_sha256": str(visible["record_sha256"]),
        }
    raise SAESmokeCollectiveError("collective terminal status is invalid")


class SAESmokeCollectiveRunner:
    """Run one durable smoke transaction collectively on every process."""

    def __init__(
        self,
        runner: SAESmokeRunner,
        hooks: SAESmokeCollectiveHooks,
    ) -> None:
        if not isinstance(runner, SAESmokeRunner):
            raise TypeError("runner must be a SAESmokeRunner")
        if not isinstance(hooks, SAESmokeCollectiveHooks):
            raise TypeError("hooks must be SAESmokeCollectiveHooks")
        self.runner = runner
        self.hooks = hooks

    @property
    def is_coordinator(self) -> bool:
        return self.hooks.process_index == self.hooks.coordinator_index

    def _seal_failure(
        self,
        *,
        attempt: SmokeAttemptIdentity,
        paths: SmokeAttemptPaths,
        detail: _FailureDetail,
        label: str,
    ) -> None:
        source: Mapping[str, Any] | None = None
        if self.is_coordinator:
            try:
                self.runner._publish_failure(  # noqa: SLF001
                    paths,
                    attempt,
                    stage=detail.stage,
                    error=_CollectiveStageFailure(detail),
                )
                record = _read_json_object(paths.failure_record, "smoke failure")
                source = _terminal_payload(
                    status="failed",
                    attempt=attempt,
                    record=record,
                    publication_error=None,
                )
            except BaseException as error:
                source = _terminal_payload(
                    status="publication-error",
                    attempt=attempt,
                    record=None,
                    publication_error=error,
                )
        terminal = self.hooks.broadcast_mapping(
            f"{label}:terminal", self.hooks.coordinator_index, source
        )
        local_value: Mapping[str, Any] | None = None
        local_error: BaseException | None = None
        try:
            _outcome, local_value = _validate_terminal_payload(
                self.runner, terminal, attempt=attempt, paths=paths
            )
        except BaseException as error:
            local_error = error
        terminal_decision = _decision(
            hooks=self.hooks,
            stage=f"{label}:terminal-revalidation",
            attempt_sha256=attempt.sha256,
            value=local_value,
            error=local_error,
        )
        if not terminal_decision.ok:
            terminal_detail = _failure_detail(terminal_decision)
            raise SAESmokeCollectiveError(
                f"smoke terminal publication/revalidation failed during "
                f"{terminal_detail.stage}: {terminal_detail.message}"
            )
        raise SAESmokeAttemptFailed(
            f"collective smoke {attempt.condition} attempt {attempt.attempt_id} "
            f"failed during {detail.stage}: {detail.message}",
            attempt=attempt,
            failure_path=paths.failure_record,
        )

    def _seal_success(
        self,
        *,
        attempt: SmokeAttemptIdentity,
        paths: SmokeAttemptPaths,
        checkpoint: FinalizedCheckpoint,
        label: str,
    ) -> SmokeOutcome:
        source: Mapping[str, Any] | None = None
        if self.is_coordinator:
            try:
                self.runner._publish_success(paths, attempt, checkpoint)  # noqa: SLF001
                record = _read_json_object(paths.result_record, "smoke result")
                source = _terminal_payload(
                    status="passed",
                    attempt=attempt,
                    record=record,
                    publication_error=None,
                )
            except BaseException as publication_error:
                # If the success record was never linked, preserve this attempt
                # as a durable failure.  Never create both terminal records.
                if not paths.result_record.exists():
                    try:
                        detail = _FailureDetail(
                            stage="result-publication",
                            process_index=self.hooks.coordinator_index,
                            error_type=type(publication_error).__name__,
                            error_message=(
                                str(publication_error)
                                or type(publication_error).__name__
                            ),
                        )
                        self.runner._publish_failure(  # noqa: SLF001
                            paths,
                            attempt,
                            stage=detail.stage,
                            error=_CollectiveStageFailure(detail),
                        )
                        record = _read_json_object(
                            paths.failure_record, "smoke failure"
                        )
                        source = _terminal_payload(
                            status="failed",
                            attempt=attempt,
                            record=record,
                            publication_error=None,
                        )
                    except BaseException as failure_error:
                        source = _terminal_payload(
                            status="publication-error",
                            attempt=attempt,
                            record=None,
                            publication_error=failure_error,
                        )
                else:
                    source = _terminal_payload(
                        status="publication-error",
                        attempt=attempt,
                        record=None,
                        publication_error=publication_error,
                    )
        terminal = self.hooks.broadcast_mapping(
            f"{label}:terminal", self.hooks.coordinator_index, source
        )
        outcome: SmokeOutcome | None = None
        local_value: Mapping[str, Any] | None = None
        local_error: BaseException | None = None
        try:
            outcome, local_value = _validate_terminal_payload(
                self.runner, terminal, attempt=attempt, paths=paths
            )
        except BaseException as error:
            local_error = error
        terminal_decision = _decision(
            hooks=self.hooks,
            stage=f"{label}:terminal-revalidation",
            attempt_sha256=attempt.sha256,
            value=local_value,
            error=local_error,
        )
        if not terminal_decision.ok:
            detail = _failure_detail(terminal_decision)
            raise SAESmokeCollectiveError(
                f"smoke terminal publication/revalidation failed during "
                f"{detail.stage}: {detail.message}"
            )
        if outcome is None:
            # Coordinator converted a result-publication failure into a durable
            # failed attempt and every peer revalidated it.
            failure = _read_json_object(paths.failure_record, "smoke failure")
            raise SAESmokeAttemptFailed(
                f"collective smoke {attempt.condition} attempt "
                f"{attempt.attempt_id} failed during result-publication: "
                f"{failure['error_message']}",
                attempt=attempt,
                failure_path=paths.failure_record,
            )
        return outcome

    def run_attempt(
        self,
        condition: str,
        *,
        callbacks: SmokeNumericalCallbacks,
        attempt_id: str | None = None,
    ) -> SmokeOutcome:
        """Run one attempt with process zero as the sole POSIX coordinator.

        Every process must call this method with the same condition and in the
        same collective order.  An explicit ``attempt_id`` is optional; when
        omitted, process zero generates it and broadcasts the immutable
        reserved identity.
        """

        if condition not in SMOKE_CONDITIONS:
            raise ValueError("smoke condition must be one of F, A, B")
        if attempt_id is not None and (
            not isinstance(attempt_id, str)
            or _ATTEMPT_ID_RE.fullmatch(attempt_id) is None
        ):
            raise ValueError("attempt_id must be 32 lowercase hexadecimal digits")
        lock = None
        lock_entered = False
        coordinator_attempt: SmokeAttemptIdentity | None = None
        coordinator_paths: SmokeAttemptPaths | None = None
        label = f"sae-smoke:{condition}"
        try:
            source: Mapping[str, Any] | None = None
            if self.is_coordinator:
                selected_attempt_id = attempt_id
                try:
                    selected_attempt_id = (
                        selected_attempt_id or new_smoke_attempt_id()
                    )
                    lock = _exclusive_campaign_lock(
                        self.runner.campaign_root / ".coordinator.lock"
                    )
                    lock.__enter__()
                    lock_entered = True
                    (
                        source,
                        coordinator_attempt,
                        coordinator_paths,
                    ) = _bootstrap_payload(
                        self.runner,
                        condition=condition,
                        requested_attempt_id=selected_attempt_id,
                    )
                except BaseException as error:
                    source = _bootstrap_error_payload(
                        self.runner,
                        condition=condition,
                        attempt_id=selected_attempt_id or ("0" * 32),
                        stage="campaign-lock",
                        error=error,
                    )
            bootstrap = self.hooks.broadcast_mapping(
                f"{label}:bootstrap", self.hooks.coordinator_index, source
            )
            attempt: SmokeAttemptIdentity | None = None
            paths: SmokeAttemptPaths | None = None
            setup_error: BaseException | None = None
            try:
                attempt, paths = _validate_bootstrap(
                    self.runner,
                    bootstrap,
                    condition=condition,
                    requested_attempt_id=attempt_id,
                    callbacks=callbacks,
                )
            except BaseException as error:
                setup_error = error
                # A reserved identity is still recoverable from a valid
                # bootstrap even when sequence validation failed.
                candidate = bootstrap.get("attempt")
                if isinstance(candidate, Mapping):
                    try:
                        attempt = _attempt_from_mapping(candidate)
                        paths = self.runner.paths(
                            attempt.condition,
                            attempt.attempt_id,
                            condition_attempt_number=(
                                attempt.condition_attempt_number
                            ),
                        )
                    except BaseException:
                        attempt = None
                        paths = None
            setup_value = (
                None
                if attempt is None
                else {
                    "attempt_sha256": attempt.sha256,
                    "bootstrap_sha256": bootstrap.get("bootstrap_sha256"),
                }
            )
            setup_decision = _decision(
                hooks=self.hooks,
                stage=f"{label}:setup",
                attempt_sha256=None if attempt is None else attempt.sha256,
                value=setup_value,
                error=setup_error,
            )
            if not setup_decision.ok:
                detail = _failure_detail(setup_decision)
                if attempt is None or paths is None:
                    raise SAESmokeCollectiveError(
                        f"smoke setup failed before a durable attempt was "
                        f"reserved: {detail.message}"
                    )
                # Preserve the coordinator's more precise reservation/sequence
                # stage when it was encoded in the bootstrap.
                source_stage = bootstrap.get("setup_stage")
                if isinstance(source_stage, str) and bootstrap.get(
                    "setup_error"
                ) is not None:
                    detail = _FailureDetail(
                        stage=source_stage,
                        process_index=detail.process_index,
                        error_type=detail.error_type,
                        error_message=detail.error_message,
                    )
                self._seal_failure(
                    attempt=attempt,
                    paths=paths,
                    detail=detail,
                    label=label,
                )
                raise AssertionError("unreachable")
            assert attempt is not None and paths is not None
            if self.is_coordinator and (
                coordinator_attempt != attempt or coordinator_paths != paths
            ):
                # This is a coordinator-local integrity failure, but it still
                # must cross a decision barrier before anybody raises.
                coordinator_mismatch: BaseException | None = (
                    SAESmokeCollectiveError(
                        "coordinator bootstrap state changed after broadcast"
                    )
                )
            else:
                coordinator_mismatch = None
            identity_decision = _decision(
                hooks=self.hooks,
                stage=f"{label}:identity-consensus",
                attempt_sha256=attempt.sha256,
                value={"attempt_sha256": attempt.sha256},
                error=coordinator_mismatch,
            )
            if not identity_decision.ok:
                self._seal_failure(
                    attempt=attempt,
                    paths=paths,
                    detail=_failure_detail(identity_decision),
                    label=label,
                )
                raise AssertionError("unreachable")

            update: SmokeUpdateResult | None = None
            expectation: SmokeCheckpointExpectation | None = None
            update_error: BaseException | None = None
            try:
                update = callbacks.update_once(attempt)
                expectation = self.runner._validate_update(  # noqa: SLF001
                    update, attempt
                )
            except BaseException as error:
                update_error = error
            update_decision = _decision(
                hooks=self.hooks,
                stage=f"{label}:optimizer-update",
                attempt_sha256=attempt.sha256,
                value=(
                    None
                    if expectation is None
                    else _expectation_mapping(expectation)
                ),
                error=update_error,
            )
            if not update_decision.ok:
                self._seal_failure(
                    attempt=attempt,
                    paths=paths,
                    detail=_failure_detail(update_decision),
                    label=label,
                )
                raise AssertionError("unreachable")
            assert update is not None and expectation is not None

            save_error: BaseException | None = None
            try:
                callbacks.save_orbax(update.state, expectation)
            except BaseException as error:
                save_error = error
            save_decision = _decision(
                hooks=self.hooks,
                stage=f"{label}:orbax-save",
                attempt_sha256=attempt.sha256,
                value={
                    "attempt_sha256": attempt.sha256,
                    "checkpoint_uri": expectation.checkpoint_uri,
                    "saved": True,
                },
                error=save_error,
            )
            if not save_decision.ok:
                self._seal_failure(
                    attempt=attempt,
                    paths=paths,
                    detail=_failure_detail(save_decision),
                    label=label,
                )
                raise AssertionError("unreachable")

            checkpoint: FinalizedCheckpoint | None = None
            inspection_error: BaseException | None = None
            try:
                checkpoint = callbacks.inspect_finalized_checkpoint(
                    expectation.checkpoint_uri
                )
                self.runner._validate_finalized_checkpoint(  # noqa: SLF001
                    checkpoint, expectation
                )
            except BaseException as error:
                inspection_error = error
            inspection_decision = _decision(
                hooks=self.hooks,
                stage=f"{label}:external-checkpoint-inspection",
                attempt_sha256=attempt.sha256,
                value=(
                    None if checkpoint is None else _checkpoint_mapping(checkpoint)
                ),
                error=inspection_error,
            )
            if not inspection_decision.ok:
                self._seal_failure(
                    attempt=attempt,
                    paths=paths,
                    detail=_failure_detail(inspection_decision),
                    label=label,
                )
                raise AssertionError("unreachable")
            assert checkpoint is not None

            restored: SmokeRestoreResult | None = None
            restore_error: BaseException | None = None
            try:
                restored = callbacks.restore_orbax(
                    expectation.checkpoint_uri, expectation
                )
                self.runner._validate_restore(  # noqa: SLF001
                    restored, checkpoint, expectation
                )
            except BaseException as error:
                restore_error = error
            restore_decision = _decision(
                hooks=self.hooks,
                stage=f"{label}:orbax-restore",
                attempt_sha256=attempt.sha256,
                value=None if restored is None else _restore_mapping(restored),
                error=restore_error,
            )
            if not restore_decision.ok:
                self._seal_failure(
                    attempt=attempt,
                    paths=paths,
                    detail=_failure_detail(restore_decision),
                    label=label,
                )
                raise AssertionError("unreachable")
            return self._seal_success(
                attempt=attempt,
                paths=paths,
                checkpoint=checkpoint,
                label=label,
            )
        finally:
            release_error: BaseException | None = None
            if lock is not None and lock_entered:
                try:
                    lock.__exit__(None, None, None)
                except BaseException as error:
                    release_error = error
            release_decision = _decision(
                hooks=self.hooks,
                stage=f"{label}:campaign-lock-release",
                attempt_sha256=None,
                value={
                    "campaign_sha256": self.runner.campaign.sha256,
                    "coordinator_index": self.hooks.coordinator_index,
                    "lock_released": True,
                },
                error=release_error,
            )
            if not release_decision.ok:
                detail = _failure_detail(release_decision)
                raise SAESmokeCollectiveError(
                    f"smoke campaign lock release failed: {detail.message}"
                )

    def run_suite(
        self,
        callbacks_by_condition: Mapping[str, SmokeNumericalCallbacks],
        *,
        attempt_ids: Mapping[str, str] | None = None,
    ) -> tuple[SmokeOutcome, SmokeOutcome, SmokeOutcome]:
        """Run the mandatory F -> A -> B suite in collective stage order."""

        supplied_ids = {} if attempt_ids is None else dict(attempt_ids)
        if set(supplied_ids) - set(SMOKE_CONDITIONS):
            raise ValueError("attempt_ids contains an unsupported condition")
        outcomes: list[SmokeOutcome] = []
        for condition in SMOKE_CONDITIONS:
            callbacks = callbacks_by_condition.get(condition)
            outcomes.append(
                self.run_attempt(
                    condition,
                    callbacks=callbacks,  # type: ignore[arg-type]
                    attempt_id=supplied_ids.get(condition),
                )
            )
        return outcomes[0], outcomes[1], outcomes[2]


def single_process_collective_hooks() -> SAESmokeCollectiveHooks:
    """Return strict in-process hooks for local tests and CPU smoke runs."""

    def broadcast(
        _label: str,
        source_index: int,
        value: Mapping[str, Any] | None,
    ) -> Mapping[str, Any]:
        if source_index != 0 or value is None:
            raise SAESmokeCollectiveError("single-process broadcast has no source")
        return dict(value)

    def decide(
        _label: str, value: Mapping[str, Any]
    ) -> Sequence[Mapping[str, Any]]:
        return (dict(value),)

    return SAESmokeCollectiveHooks(
        process_index=0,
        process_count=1,
        coordinator_index=0,
        broadcast_mapping=broadcast,
        decision_barrier=decide,
    )
