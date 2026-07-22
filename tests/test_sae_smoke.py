from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import pytest

from moe_sae.sae_smoke import (
    FinalizedCheckpoint,
    SAEFullShape,
    SAESmokeAttemptExists,
    SAESmokeAttemptFailed,
    SAESmokeIntegrityError,
    SAESmokePathError,
    SAESmokeRunner,
    SmokeAttemptIdentity,
    SmokeCampaignIdentity,
    SmokeCheckpointExpectation,
    SmokeNumericalCallbacks,
    SmokeRestoreResult,
    SmokeSuiteAttestation,
    SmokeUpdateResult,
    canonical_sha256,
)


def digest(character: str) -> str:
    return character * 64


def campaign(tmp_path: Path, *, payload_root: str | Path | None = None):
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
        checkpoint_payload_root=(
            tmp_path / "checkpoint-payloads"
            if payload_root is None
            else payload_root
        ),
    )


def runner_for(
    tmp_path: Path,
    selected_campaign: SmokeCampaignIdentity | None = None,
) -> SAESmokeRunner:
    return SAESmokeRunner(
        tmp_path / "artifacts",
        campaign(tmp_path) if selected_campaign is None else selected_campaign,
        smoke_cache_root=tmp_path / "host-cache",
    )


class FakeNumerics:
    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []
        self.states: dict[str, object] = {}
        self.expectations: dict[str, SmokeCheckpointExpectation] = {}
        self.checkpoint_digest = digest("c")
        self.update_transform = lambda value: value
        self.checkpoint_transform = lambda value: value
        self.restore_transform = lambda value: value

    def update(self, attempt: SmokeAttemptIdentity) -> SmokeUpdateResult:
        self.events.append(("update", attempt.condition))
        state: object = {
            "condition": attempt.condition,
            "parameters": [1.0, 2.0],
            "optimizer": {"count": 1},
            "step": 1,
            "tokens_seen": attempt.expected_final_tokens_seen,
        }
        self.states[attempt.checkpoint_uri] = state
        value = SmokeUpdateResult(
            state=state,
            initial_step=0,
            final_step=1,
            initial_tokens_seen=0,
            final_tokens_seen=attempt.expected_final_tokens_seen,
            optimizer_updates=1,
            attempt_sha256=attempt.sha256,
            condition_shape_sha256=attempt.condition_shape.sha256,
            state_sha256=digest("b"),
        )
        return self.update_transform(value)

    def save(self, state: object, expected: SmokeCheckpointExpectation) -> None:
        self.events.append(("save", str(state)))
        assert state is self.states[expected.checkpoint_uri]
        self.expectations[expected.checkpoint_uri] = expected

    def inspect(self, uri: str) -> FinalizedCheckpoint:
        self.events.append(("inspect", uri))
        expected = self.expectations[uri]
        value = FinalizedCheckpoint(
            checkpoint_uri=uri,
            finalized=True,
            step=expected.step,
            tokens_seen=expected.tokens_seen,
            optimizer_updates=expected.optimizer_updates,
            attempt_sha256=expected.attempt_sha256,
            condition_shape_sha256=expected.condition_shape_sha256,
            state_sha256=expected.state_sha256,
            checkpoint_sha256=self.checkpoint_digest,
        )
        return self.checkpoint_transform(value)

    def restore(
        self, uri: str, expected: SmokeCheckpointExpectation
    ) -> SmokeRestoreResult:
        self.events.append(("restore", uri))
        value = SmokeRestoreResult(
            state=self.states[uri],
            step=expected.step,
            tokens_seen=expected.tokens_seen,
            optimizer_updates=expected.optimizer_updates,
            attempt_sha256=expected.attempt_sha256,
            condition_shape_sha256=expected.condition_shape_sha256,
            state_sha256=expected.state_sha256,
            checkpoint_sha256=self.checkpoint_digest,
        )
        return self.restore_transform(value)

    def callbacks(self) -> SmokeNumericalCallbacks:
        return SmokeNumericalCallbacks(
            update_once=self.update,
            save_orbax=self.save,
            inspect_finalized_checkpoint=self.inspect,
            restore_orbax=self.restore,
        )


def attempt_id(index: int) -> str:
    return f"{index:032x}"


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_production_shape_identity_matches_planned_a_b_f_allocations() -> None:
    shape = SAEFullShape.gemma4_26b_a4b()
    a = shape.for_condition("A")
    b = shape.for_condition("B")
    f = shape.for_condition("F")

    assert (a.total_features, b.total_features, f.total_features) == (
        131_072,
        8_192,
        131_072,
    )
    assert (a.candidates_per_token, b.candidates_per_token, f.candidates_per_token) == (
        8_192,
        8_192,
        131_072,
    )
    assert a.parameter_count == f.parameter_count == 738_328_576
    assert a.topk_features == b.topk_features == f.topk_features == 64


def test_campaign_identity_binds_pre_attestation_inputs_directly(
    tmp_path: Path,
) -> None:
    selected = campaign(tmp_path)
    mapping = selected.to_mapping()

    assert mapping["train_capture_manifest_sha256"] == digest("3")
    assert mapping["eval_capture_manifest_sha256"] == digest("4")
    assert mapping["consumer_environment_sha256"] == digest("5")
    assert mapping["capture_compatibility_sha256"] == digest("6")
    assert "sae_storage_attestation_sha256" not in mapping
    changed = replace(selected, train_capture_manifest_sha256=digest("7"))
    assert changed.sha256 != selected.sha256


def test_f_then_a_then_b_runs_one_update_and_full_checkpoint_roundtrip(
    tmp_path: Path,
) -> None:
    selected_campaign = campaign(tmp_path)
    runner = runner_for(tmp_path, selected_campaign)
    numerics = FakeNumerics()

    outcomes = [
        runner.run_attempt(
            condition,
            attempt_id=attempt_id(index),
            callbacks=numerics.callbacks(),
        )
        for index, condition in enumerate(("F", "A", "B"), start=1)
    ]

    assert [outcome.attempt.condition for outcome in outcomes] == ["F", "A", "B"]
    assert len({outcome.outcome_sha256 for outcome in outcomes}) == 3
    assert all(outcome.checkpoint_sha256 == digest("c") for outcome in outcomes)
    assert [event[0] for event in numerics.events] == [
        "update",
        "save",
        "inspect",
        "restore",
    ] * 3
    for outcome in outcomes:
        result = read(outcome.paths.result_record)
        assert result["step"] == 1
        assert result["tokens_seen"] == 8
        assert result["optimizer_updates"] == 1
        assert result["checkpoint_sha256"] == digest("c")
        assert not outcome.paths.failure_record.exists()
        assert "/sae-smoke/" in outcome.paths.checkpoint_uri


def test_passed_suite_cryptographically_binds_exact_f_a_b_results(
    tmp_path: Path,
) -> None:
    selected_runner = runner_for(tmp_path)
    numerics = FakeNumerics()
    outcomes = [
        selected_runner.run_attempt(
            condition,
            attempt_id=attempt_id(index),
            callbacks=numerics.callbacks(),
        )
        for index, condition in enumerate(("F", "A", "B"), start=1)
    ]

    suite = selected_runner.validate_passed_suite()

    assert isinstance(suite, SmokeSuiteAttestation)
    assert suite.campaign_sha256 == selected_runner.campaign.sha256
    assert [item.condition for item in suite.conditions] == ["F", "A", "B"]
    assert [item.attempt_sha256 for item in suite.conditions] == [
        outcome.attempt.sha256 for outcome in outcomes
    ]
    assert [item.outcome_sha256 for item in suite.conditions] == [
        outcome.outcome_sha256 for outcome in outcomes
    ]
    assert [item.checkpoint_sha256 for item in suite.conditions] == [
        outcome.checkpoint_sha256 for outcome in outcomes
    ]
    assert [item.state_sha256 for item in suite.conditions] == [
        outcome.state_sha256 for outcome in outcomes
    ]
    assert suite.sha256 == canonical_sha256(suite.to_mapping())
    assert selected_runner.validate_passed_suite() == suite


def test_suite_rejects_incomplete_tampered_and_out_of_order_campaigns(
    tmp_path: Path,
) -> None:
    incomplete = runner_for(tmp_path / "incomplete")
    incomplete.run_attempt(
        "F", attempt_id=attempt_id(1), callbacks=FakeNumerics().callbacks()
    )
    with pytest.raises(SAESmokeIntegrityError, match="condition A"):
        incomplete.validate_passed_suite()

    tampered = runner_for(tmp_path / "tampered")
    broken = FakeNumerics()
    broken.update_transform = lambda value: replace(value, final_step=0)
    with pytest.raises(SAESmokeAttemptFailed):
        tampered.run_attempt(
            "F", attempt_id=attempt_id(1), callbacks=broken.callbacks()
        )
    numerics = FakeNumerics()
    for index, condition in enumerate(("F", "A", "B"), start=2):
        tampered.run_attempt(
            condition,
            attempt_id=attempt_id(index),
            callbacks=numerics.callbacks(),
        )
    failure_path = tampered.paths(
        "F", attempt_id(1), condition_attempt_number=1
    ).failure_record
    failure = read(failure_path)
    failure["error_message"] = "rewritten"
    failure_path.write_text(json.dumps(failure), encoding="utf-8")
    with pytest.raises(SAESmokeIntegrityError, match="failure self-hash"):
        tampered.validate_passed_suite()

    out_of_order = runner_for(tmp_path / "out-of-order")
    outcomes = [
        out_of_order.run_attempt(
            condition,
            attempt_id=attempt_id(index),
            callbacks=FakeNumerics().callbacks(),
        )
        for index, condition in enumerate(("F", "A", "B"), start=1)
    ]
    b_attempt_path = outcomes[-1].paths.attempt_record
    b_attempt = read(b_attempt_path)
    f_result = read(outcomes[0].paths.result_record)
    b_attempt["reserved_at"] = f_result["completed_at"]
    b_attempt["record_sha256"] = canonical_sha256(
        {key: value for key, value in b_attempt.items() if key != "record_sha256"}
    )
    b_attempt_path.write_text(json.dumps(b_attempt), encoding="utf-8")
    with pytest.raises(SAESmokeIntegrityError, match="mandatory F, A, B order"):
        out_of_order.validate_passed_suite()


def test_suite_rejects_a_second_later_success_even_when_self_hashes_are_valid(
    tmp_path: Path,
) -> None:
    selected_runner = runner_for(tmp_path)
    numerics = FakeNumerics()
    outcomes = [
        selected_runner.run_attempt(
            condition,
            attempt_id=attempt_id(index),
            callbacks=numerics.callbacks(),
        )
        for index, condition in enumerate(("F", "A", "B"), start=1)
    ]
    with pytest.raises(SAESmokeAttemptFailed) as caught:
        selected_runner.run_attempt(
            "B", attempt_id=attempt_id(4), callbacks=numerics.callbacks()
        )
    second = caught.value.attempt
    second_paths = selected_runner.paths(
        "B", second.attempt_id, condition_attempt_number=2
    )
    failure = read(second_paths.failure_record)
    second_paths.failure_record.unlink()
    deterministic = {
        "status": "passed",
        "attempt_sha256": second.sha256,
        "campaign_sha256": second.campaign_sha256,
        "condition": "B",
        "condition_shape_sha256": second.condition_shape.sha256,
        "checkpoint_uri": second.checkpoint_uri,
        "checkpoint_sha256": outcomes[-1].checkpoint_sha256,
        "state_sha256": outcomes[-1].state_sha256,
        "step": 1,
        "tokens_seen": second.expected_final_tokens_seen,
        "optimizer_updates": 1,
    }
    result = {
        "schema_version": "gemma4-sae-smoke-result-v1",
        **deterministic,
        "completed_at": failure["failed_at"],
        "outcome_sha256": canonical_sha256(deterministic),
    }
    result["record_sha256"] = canonical_sha256(result)
    second_paths.result_record.write_text(json.dumps(result), encoding="utf-8")

    with pytest.raises(SAESmokeIntegrityError, match="exactly one passed"):
        selected_runner.validate_passed_suite()


def test_out_of_order_attempt_is_retained_and_does_not_satisfy_prerequisite(
    tmp_path: Path,
) -> None:
    runner = runner_for(tmp_path)
    numerics = FakeNumerics()

    with pytest.raises(SAESmokeAttemptFailed, match="requires exactly one passed F") as caught:
        runner.run_attempt("A", attempt_id=attempt_id(1), callbacks=numerics.callbacks())

    failure_path = caught.value.failure_path
    before = failure_path.read_bytes()
    assert read(failure_path)["stage"] == "sequence-validation"
    assert caught.value.attempt.condition_attempt_number == 1
    assert numerics.events == []

    runner.run_attempt("F", attempt_id=attempt_id(2), callbacks=numerics.callbacks())
    outcome = runner.run_attempt("A", attempt_id=attempt_id(3), callbacks=numerics.callbacks())
    assert outcome.attempt.condition_attempt_number == 2
    assert failure_path.read_bytes() == before


def test_exact_update_step_tokens_and_shape_are_hard_requirements(tmp_path: Path) -> None:
    runner = runner_for(tmp_path)
    numerics = FakeNumerics()
    numerics.update_transform = lambda value: replace(
        value,
        final_tokens_seen=value.final_tokens_seen - 1,
        condition_shape_sha256=digest("d"),
    )

    with pytest.raises(SAESmokeAttemptFailed, match="optimizer update") as caught:
        runner.run_attempt("F", attempt_id=attempt_id(1), callbacks=numerics.callbacks())

    failure = read(caught.value.failure_path)
    assert failure["stage"] == "optimizer-update"
    assert [event[0] for event in numerics.events] == ["update"]
    assert not caught.value.failure_path.with_name("result.json").exists()


@pytest.mark.parametrize(
    "transform, message",
    [
        (lambda value: replace(value, finalized=False), "not finalized"),
        (lambda value: replace(value, step=2), "step mismatch"),
        (lambda value: replace(value, tokens_seen=7), "tokens_seen mismatch"),
        (
            lambda value: replace(value, checkpoint_uri=value.checkpoint_uri + "-other"),
            "checkpoint_uri mismatch",
        ),
        (lambda value: replace(value, checkpoint_sha256="INVALID"), "lowercase SHA-256"),
    ],
)
def test_external_finalized_checkpoint_is_independently_validated(
    tmp_path: Path,
    transform,
    message: str,
) -> None:
    runner = runner_for(tmp_path)
    numerics = FakeNumerics()
    numerics.checkpoint_transform = transform

    with pytest.raises(SAESmokeAttemptFailed, match=message) as caught:
        runner.run_attempt("F", attempt_id=attempt_id(1), callbacks=numerics.callbacks())

    assert read(caught.value.failure_path)["stage"] == "external-checkpoint-inspection"
    assert [event[0] for event in numerics.events] == ["update", "save", "inspect"]


@pytest.mark.parametrize(
    "transform, message",
    [
        (lambda value: replace(value, step=0), "step mismatch"),
        (lambda value: replace(value, tokens_seen=0), "tokens_seen mismatch"),
        (lambda value: replace(value, state_sha256=digest("d")), "state_sha256 mismatch"),
        (
            lambda value: replace(value, checkpoint_sha256=digest("e")),
            "checkpoint_sha256 mismatch",
        ),
    ],
)
def test_restored_full_state_must_match_progress_state_and_external_digest(
    tmp_path: Path,
    transform,
    message: str,
) -> None:
    runner = runner_for(tmp_path)
    numerics = FakeNumerics()
    numerics.restore_transform = transform

    with pytest.raises(SAESmokeAttemptFailed, match=message) as caught:
        runner.run_attempt("F", attempt_id=attempt_id(1), callbacks=numerics.callbacks())

    assert read(caught.value.failure_path)["stage"] == "orbax-restore"
    assert [event[0] for event in numerics.events] == [
        "update",
        "save",
        "inspect",
        "restore",
    ]


def test_callback_exception_is_retained_and_same_attempt_cannot_overwrite(
    tmp_path: Path,
) -> None:
    runner = runner_for(tmp_path)
    numerics = FakeNumerics()

    def broken_save(state: object, expected: SmokeCheckpointExpectation) -> None:
        del state, expected
        raise RuntimeError("writer lost quorum")

    callbacks = replace(numerics.callbacks(), save_orbax=broken_save)
    with pytest.raises(SAESmokeAttemptFailed, match="writer lost quorum") as caught:
        runner.run_attempt("F", attempt_id=attempt_id(1), callbacks=callbacks)

    attempt_before = caught.value.attempt
    attempt_bytes = caught.value.failure_path.with_name("attempt.json").read_bytes()
    failure_bytes = caught.value.failure_path.read_bytes()
    with pytest.raises(SAESmokeAttemptExists):
        runner.run_attempt("F", attempt_id=attempt_id(1), callbacks=numerics.callbacks())
    assert caught.value.failure_path.with_name("attempt.json").read_bytes() == attempt_bytes
    assert caught.value.failure_path.read_bytes() == failure_bytes
    assert attempt_before.attempt_id == attempt_id(1)


def test_tampered_prior_result_fails_closed_and_retains_new_attempt(tmp_path: Path) -> None:
    runner = runner_for(tmp_path)
    numerics = FakeNumerics()
    first = runner.run_attempt("F", attempt_id=attempt_id(1), callbacks=numerics.callbacks())
    result = read(first.paths.result_record)
    result["tokens_seen"] = 7
    first.paths.result_record.write_text(json.dumps(result), encoding="utf-8")

    with pytest.raises(SAESmokeAttemptFailed, match="self-hash mismatch") as caught:
        runner.run_attempt("A", attempt_id=attempt_id(2), callbacks=numerics.callbacks())

    assert caught.value.failure_path.exists()
    assert read(caught.value.failure_path)["stage"] == "sequence-validation"


def test_smoke_paths_are_isolated_and_production_roots_are_rejected(
    tmp_path: Path,
) -> None:
    selected_campaign = campaign(tmp_path, payload_root="gs://bucket/checkpoints")
    runner = runner_for(tmp_path, selected_campaign)
    outcome = runner.run_attempt(
        "F", attempt_id=attempt_id(1), callbacks=FakeNumerics().callbacks()
    )

    relative_parts = outcome.paths.attempt_root.relative_to(tmp_path).parts
    assert "sae-smoke" in relative_parts
    assert not {"training", "runs", "capacity", "eval"}.intersection(relative_parts)
    assert outcome.paths.checkpoint_uri.startswith("gs://bucket/checkpoints/sae-smoke/")
    assert "/sae-smoke/" in str(outcome.paths.cache_path)
    assert not outcome.paths.cache_path.exists()
    created = {
        path.name
        for path in (tmp_path / "artifacts").rglob("*")
        if path.is_dir()
    }
    assert not {"training", "runs", "capacity", "eval"}.intersection(created)

    with pytest.raises(SAESmokePathError, match="training"):
        SAESmokeRunner(
            tmp_path / "training",
            selected_campaign,
            smoke_cache_root=tmp_path / "host-cache",
        )
    with pytest.raises(SAESmokePathError, match="eval"):
        SAESmokeRunner(
            tmp_path / "artifacts-2",
            selected_campaign,
            smoke_cache_root=tmp_path / "eval",
        )
    with pytest.raises(SAESmokePathError, match="runs"):
        campaign(tmp_path, payload_root="gs://bucket/runs/checkpoints")
    with pytest.raises(SAESmokePathError, match="capacity"):
        campaign(tmp_path, payload_root=tmp_path / "capacity" / "checkpoints")


def test_failed_attempt_does_not_block_clean_retry_but_passed_condition_does(
    tmp_path: Path,
) -> None:
    runner = runner_for(tmp_path)
    broken = FakeNumerics()
    broken.update_transform = lambda value: replace(value, final_step=0)
    with pytest.raises(SAESmokeAttemptFailed):
        runner.run_attempt("F", attempt_id=attempt_id(1), callbacks=broken.callbacks())

    good = FakeNumerics()
    passed = runner.run_attempt("F", attempt_id=attempt_id(2), callbacks=good.callbacks())
    assert passed.attempt.condition_attempt_number == 2

    with pytest.raises(SAESmokeAttemptFailed, match="already has a passed") as caught:
        runner.run_attempt("F", attempt_id=attempt_id(3), callbacks=good.callbacks())
    assert caught.value.failure_path.exists()
    assert read(caught.value.failure_path)["stage"] == "sequence-validation"
