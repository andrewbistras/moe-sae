from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
import threading
from typing import Any

import pytest

from moe_sae.sae_smoke import (
    FinalizedCheckpoint,
    SAEFullShape,
    SAESmokeAttemptFailed,
    SAESmokeRunner,
    SmokeAttemptIdentity,
    SmokeCampaignIdentity,
    SmokeCheckpointExpectation,
    SmokeNumericalCallbacks,
    SmokeRestoreResult,
    SmokeUpdateResult,
)
from moe_sae.sae_smoke_collective import (
    SAESmokeCollectiveError,
    SAESmokeCollectiveHooks,
    SAESmokeCollectiveRunner,
    single_process_collective_hooks,
)


def digest(character: str) -> str:
    return character * 64


def campaign(tmp_path: Path) -> SmokeCampaignIdentity:
    return SmokeCampaignIdentity(
        config_sha256=digest("1"),
        repository_content_sha256=digest("2"),
        train_capture_manifest_sha256=digest("3"),
        eval_capture_manifest_sha256=digest("4"),
        consumer_environment_sha256=digest("5"),
        capture_compatibility_sha256=digest("6"),
        full_shape=SAEFullShape(
            hidden_dim=4,
            num_experts=4,
            bank_width=3,
            top_routed=2,
            topk_features=2,
            effective_batch_tokens=8,
        ),
        checkpoint_payload_root=tmp_path / "checkpoint-payloads",
    )


class ThreadCollectives:
    """Exact blocking broadcast/all-gather primitives for four fake hosts."""

    def __init__(
        self,
        count: int = 4,
        *,
        on_broadcast: Callable[[str, Mapping[str, Any]], None] | None = None,
    ) -> None:
        self.count = count
        self.on_broadcast = on_broadcast
        self.condition = threading.Condition()
        self.broadcasts: dict[str, dict[str, Any]] = {}
        self.decisions: dict[str, dict[str, Any]] = {}

    def broadcast(
        self,
        process_index: int,
        label: str,
        source_index: int,
        value: Mapping[str, Any] | None,
    ) -> Mapping[str, Any]:
        with self.condition:
            state = self.broadcasts.setdefault(
                label,
                {"arrivals": set(), "departures": 0, "value": None},
            )
            arrivals = state["arrivals"]
            assert isinstance(arrivals, set)
            assert process_index not in arrivals
            arrivals.add(process_index)
            if process_index == source_index:
                assert value is not None
                state["value"] = dict(value)
                if self.on_broadcast is not None:
                    self.on_broadcast(label, value)
            else:
                assert value is None
            self.condition.notify_all()
            assert self.condition.wait_for(
                lambda: len(arrivals) == self.count and state["value"] is not None,
                timeout=10,
            )
            result = dict(state["value"])
            state["departures"] += 1
            if state["departures"] == self.count:
                del self.broadcasts[label]
            self.condition.notify_all()
            return result

    def decide(
        self,
        process_index: int,
        label: str,
        report: Mapping[str, Any],
    ) -> Sequence[Mapping[str, Any]]:
        with self.condition:
            state = self.decisions.setdefault(
                label,
                {"reports": {}, "departures": 0},
            )
            reports = state["reports"]
            assert isinstance(reports, dict)
            assert process_index not in reports
            reports[process_index] = dict(report)
            self.condition.notify_all()
            assert self.condition.wait_for(
                lambda: len(reports) == self.count,
                timeout=10,
            )
            result = tuple(dict(reports[index]) for index in reversed(range(self.count)))
            state["departures"] += 1
            if state["departures"] == self.count:
                del self.decisions[label]
            self.condition.notify_all()
            return result

    def hooks(self, process_index: int) -> SAESmokeCollectiveHooks:
        return SAESmokeCollectiveHooks(
            process_index=process_index,
            process_count=self.count,
            coordinator_index=0,
            broadcast_mapping=lambda label, source, value: self.broadcast(
                process_index, label, source, value
            ),
            decision_barrier=lambda label, value: self.decide(
                process_index, label, value
            ),
        )


@dataclass
class FakeCallbacks:
    process_index: int
    condition: str
    fail_stage: str | None = None
    fail_process: int | None = None

    def __post_init__(self) -> None:
        self.events: list[str] = []
        self.attempt: SmokeAttemptIdentity | None = None
        self.state = object()
        self.expectation: SmokeCheckpointExpectation | None = None
        characters = {"F": ("a", "d"), "A": ("b", "e"), "B": ("c", "f")}
        self.state_sha256, self.checkpoint_sha256 = (
            digest(characters[self.condition][0]),
            digest(characters[self.condition][1]),
        )

    def _event(self, stage: str) -> None:
        self.events.append(stage)
        if self.fail_stage == stage and self.fail_process == self.process_index:
            raise RuntimeError(f"injected {stage} failure")

    def update(self, attempt: SmokeAttemptIdentity) -> SmokeUpdateResult:
        self._event("update")
        self.attempt = attempt
        return SmokeUpdateResult(
            state=self.state,
            initial_step=0,
            final_step=1,
            initial_tokens_seen=0,
            final_tokens_seen=attempt.expected_final_tokens_seen,
            optimizer_updates=1,
            attempt_sha256=attempt.sha256,
            condition_shape_sha256=attempt.condition_shape.sha256,
            state_sha256=self.state_sha256,
        )

    def save(self, state: object, expectation: SmokeCheckpointExpectation) -> None:
        self._event("save")
        assert state is self.state
        self.expectation = expectation

    def inspect(self, uri: str) -> FinalizedCheckpoint:
        self._event("inspect")
        assert self.attempt is not None and self.expectation is not None
        assert uri == self.expectation.checkpoint_uri
        return FinalizedCheckpoint(
            checkpoint_uri=uri,
            finalized=True,
            step=1,
            tokens_seen=self.attempt.expected_final_tokens_seen,
            optimizer_updates=1,
            attempt_sha256=self.attempt.sha256,
            condition_shape_sha256=self.attempt.condition_shape.sha256,
            state_sha256=self.state_sha256,
            checkpoint_sha256=self.checkpoint_sha256,
        )

    def restore(
        self, uri: str, expectation: SmokeCheckpointExpectation
    ) -> SmokeRestoreResult:
        self._event("restore")
        assert self.attempt is not None
        assert uri == expectation.checkpoint_uri
        return SmokeRestoreResult(
            state=object(),
            step=1,
            tokens_seen=self.attempt.expected_final_tokens_seen,
            optimizer_updates=1,
            attempt_sha256=self.attempt.sha256,
            condition_shape_sha256=self.attempt.condition_shape.sha256,
            state_sha256=self.state_sha256,
            checkpoint_sha256=self.checkpoint_sha256,
        )

    def callbacks(self) -> SmokeNumericalCallbacks:
        return SmokeNumericalCallbacks(
            update_once=self.update,
            save_orbax=self.save,
            inspect_finalized_checkpoint=self.inspect,
            restore_orbax=self.restore,
        )


def run_fake_hosts(
    tmp_path: Path,
    collectives: ThreadCollectives,
    *,
    conditions: tuple[str, ...] = ("F", "A", "B"),
    fail_stage: str | None = None,
    fail_process: int | None = None,
    mutate_coordinator: Callable[[SAESmokeRunner], None] | None = None,
) -> tuple[list[Any], list[BaseException | None], list[dict[str, FakeCallbacks]]]:
    selected_campaign = campaign(tmp_path)
    results: list[Any] = [None] * collectives.count
    errors: list[BaseException | None] = [None] * collectives.count
    numerics: list[dict[str, FakeCallbacks]] = []
    runners: list[SAESmokeRunner] = []
    for index in range(collectives.count):
        runners.append(
            SAESmokeRunner(
                tmp_path / "control",
                selected_campaign,
                smoke_cache_root=tmp_path / "host-cache",
            )
        )
        numerics.append(
            {
                condition: FakeCallbacks(
                    index,
                    condition,
                    fail_stage=fail_stage,
                    fail_process=fail_process,
                )
                for condition in conditions
            }
        )
    if mutate_coordinator is not None:
        mutate_coordinator(runners[0])

    def target(index: int) -> None:
        collective = SAESmokeCollectiveRunner(
            runners[index], collectives.hooks(index)
        )
        try:
            if conditions == ("F", "A", "B"):
                results[index] = collective.run_suite(
                    {
                        condition: numerics[index][condition].callbacks()
                        for condition in conditions
                    }
                )
            else:
                condition = conditions[0]
                results[index] = collective.run_attempt(
                    condition,
                    callbacks=numerics[index][condition].callbacks(),
                )
        except BaseException as error:
            errors[index] = error

    threads = [
        threading.Thread(target=target, args=(index,), daemon=True)
        for index in range(collectives.count)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15)
    assert not [thread for thread in threads if thread.is_alive()], (
        "a fake host was stranded in a collective"
    )
    return results, errors, numerics


def test_fake_four_hosts_run_exact_f_a_b_order_with_one_coordinator_lock(
    tmp_path: Path,
) -> None:
    results, errors, numerics = run_fake_hosts(tmp_path, ThreadCollectives())

    assert errors == [None] * 4
    for process_results in results:
        assert [item.attempt.condition for item in process_results] == ["F", "A", "B"]
    for condition_index, condition in enumerate(("F", "A", "B")):
        attempts = {
            process_results[condition_index].attempt.sha256
            for process_results in results
        }
        outcomes = {
            process_results[condition_index].outcome_sha256
            for process_results in results
        }
        assert len(attempts) == len(outcomes) == 1
        assert all(
            host[condition].events == ["update", "save", "inspect", "restore"]
            for host in numerics
        )
    suite = SAESmokeRunner(
        tmp_path / "control",
        campaign(tmp_path),
        smoke_cache_root=tmp_path / "host-cache",
    ).validate_passed_suite()
    assert [item.condition for item in suite.conditions] == ["F", "A", "B"]


def test_two_four_host_cohorts_serialize_on_one_campaign_lock(tmp_path: Path) -> None:
    cohorts: list[Any] = [None, None]

    def launch(index: int) -> None:
        cohorts[index] = run_fake_hosts(
            tmp_path,
            ThreadCollectives(),
            conditions=("F",),
        )

    launchers = [threading.Thread(target=launch, args=(index,)) for index in range(2)]
    for launcher in launchers:
        launcher.start()
    for launcher in launchers:
        launcher.join(timeout=20)
    assert not [launcher for launcher in launchers if launcher.is_alive()]

    successful = [cohort for cohort in cohorts if cohort[1] == [None] * 4]
    failed = [cohort for cohort in cohorts if cohort[1] != [None] * 4]
    assert len(successful) == len(failed) == 1
    assert all(isinstance(error, SAESmokeAttemptFailed) for error in failed[0][1])
    assert all(host["F"].events == [] for host in failed[0][2])
    attempts = list((tmp_path / "control").rglob("attempt.json"))
    results = list((tmp_path / "control").rglob("result.json"))
    failures = list((tmp_path / "control").rglob("failure.json"))
    assert (len(attempts), len(results), len(failures)) == (2, 1, 1)


def test_peer_update_failure_is_decided_before_advance_and_retained(
    tmp_path: Path,
) -> None:
    _results, errors, numerics = run_fake_hosts(
        tmp_path,
        ThreadCollectives(),
        conditions=("F",),
        fail_stage="update",
        fail_process=2,
    )

    assert all(isinstance(error, SAESmokeAttemptFailed) for error in errors)
    assert len({str(error) for error in errors}) == 1
    assert all(host["F"].events == ["update"] for host in numerics)
    failure_path = errors[0].failure_path  # type: ignore[union-attr]
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    assert failure["stage"].endswith("optimizer-update")
    assert "process 2 RuntimeError" in failure["error_message"]

    # The lock was released and the failed immutable attempt was preserved;
    # another four-host cohort can retry F successfully.
    results, retry_errors, _retry_numerics = run_fake_hosts(
        tmp_path, ThreadCollectives(), conditions=("F",)
    )
    assert retry_errors == [None] * 4
    assert len({result.attempt.sha256 for result in results}) == 1
    attempt_roots = list(failure_path.parent.parent.iterdir())
    assert len(attempt_roots) == 2


def test_coordinator_result_publication_error_reaches_every_peer_and_unlocks(
    tmp_path: Path,
) -> None:
    def break_publication(runner: SAESmokeRunner) -> None:
        def fail(*_args: Any, **_kwargs: Any) -> Any:
            raise OSError("injected result publication failure")

        runner._publish_success = fail  # type: ignore[method-assign]  # noqa: SLF001

    _results, errors, numerics = run_fake_hosts(
        tmp_path,
        ThreadCollectives(),
        conditions=("F",),
        mutate_coordinator=break_publication,
    )

    assert all(isinstance(error, SAESmokeAttemptFailed) for error in errors)
    assert all(
        host["F"].events == ["update", "save", "inspect", "restore"]
        for host in numerics
    )
    failure_path = errors[0].failure_path  # type: ignore[union-attr]
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    assert failure["stage"] == "result-publication"
    assert "injected result publication failure" in failure["error_message"]

    # A second cohort proves the campaign lock was not leaked by publication.
    results, retry_errors, _retry_numerics = run_fake_hosts(
        tmp_path, ThreadCollectives(), conditions=("F",)
    )
    assert retry_errors == [None] * 4
    assert all(result.attempt.condition == "F" for result in results)


def test_coordinator_failure_publication_error_never_strands_peers(
    tmp_path: Path,
) -> None:
    def break_failure_publication(runner: SAESmokeRunner) -> None:
        def fail(*_args: Any, **_kwargs: Any) -> Any:
            raise OSError("injected failure publication failure")

        runner._publish_failure = fail  # type: ignore[method-assign]  # noqa: SLF001

    _results, errors, numerics = run_fake_hosts(
        tmp_path,
        ThreadCollectives(),
        conditions=("F",),
        fail_stage="update",
        fail_process=1,
        mutate_coordinator=break_failure_publication,
    )

    assert all(isinstance(error, SAESmokeCollectiveError) for error in errors)
    assert len({str(error) for error in errors}) == 1
    assert "publication" in str(errors[0])
    assert all(host["F"].events == ["update"] for host in numerics)

    # The first attempt is intentionally left unterminated/fail-closed because
    # the durable failure write itself failed, but its campaign lock is free.
    results, retry_errors, _retry_numerics = run_fake_hosts(
        tmp_path, ThreadCollectives(), conditions=("F",)
    )
    assert retry_errors == [None] * 4
    assert all(result.attempt.condition == "F" for result in results)


def test_tampered_published_result_fails_collectively_without_stranding(
    tmp_path: Path,
) -> None:
    tampered = False

    def tamper_terminal(label: str, payload: Mapping[str, Any]) -> None:
        nonlocal tampered
        if not label.endswith(":terminal") or payload.get("status") != "passed":
            return
        record = payload["record"]
        assert isinstance(record, Mapping)
        attempt_value = record["attempt_sha256"]
        candidates = list((tmp_path / "control").rglob("result.json"))
        assert len(candidates) == 1
        value = json.loads(candidates[0].read_text(encoding="utf-8"))
        assert value["attempt_sha256"] == attempt_value
        value["state_sha256"] = digest("9")
        candidates[0].write_text(json.dumps(value), encoding="utf-8")
        tampered = True

    _results, errors, _numerics = run_fake_hosts(
        tmp_path,
        ThreadCollectives(on_broadcast=tamper_terminal),
        conditions=("F",),
    )

    assert tampered
    assert all(isinstance(error, SAESmokeCollectiveError) for error in errors)
    assert len({str(error) for error in errors}) == 1
    assert "terminal-revalidation" in str(errors[0])


def test_single_process_hooks_and_hook_identity_validation(tmp_path: Path) -> None:
    selected_runner = SAESmokeRunner(
        tmp_path / "control",
        campaign(tmp_path),
        smoke_cache_root=tmp_path / "host-cache",
    )
    numerics = FakeCallbacks(0, "F")
    outcome = SAESmokeCollectiveRunner(
        selected_runner, single_process_collective_hooks()
    ).run_attempt("F", callbacks=numerics.callbacks(), attempt_id="1" * 32)
    assert outcome.attempt.attempt_id == "1" * 32

    with pytest.raises(ValueError, match="process 0"):
        SAESmokeCollectiveHooks(
            process_index=0,
            process_count=2,
            coordinator_index=1,
            broadcast_mapping=lambda *_args: {},
            decision_barrier=lambda *_args: (),
        )


def test_module_import_is_jax_free() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import moe_sae.sae_smoke_collective; "
            "assert 'jax' not in sys.modules; assert 'orbax' not in sys.modules",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
