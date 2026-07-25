from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from moe_sae import contribution_training_worker as training_worker_module
from moe_sae.contribution_config import load_contribution_config
from moe_sae.contribution_performance import (
    MINIMUM_USEFUL_PREFIX_MFU,
    ExactPerformanceIdentity,
)
from moe_sae.contribution_training_performance import (
    SAEArithmetic,
    SAEQualificationObservation,
)
from moe_sae.contribution_training_worker import (
    ContributionTrainingWorkerError,
    PreparedTrainingExecutable,
    RestoredTrainingState,
    TrainingLoopDependencies,
    TrainingWorkerPlan,
    _parser,
    _prove_prefetch_for_plan,
    chosen_mode_evidence,
    prefetch_process_lane,
    qualify_training_modes,
    run_qualified_training,
    validate_training_mode_selection,
)
from moe_sae.object_store import ObjectIdentity

CONFIG = Path(__file__).parents[1] / "configs" / "gemma4_contribution_v1.yaml"


def identity(label: str) -> ExactPerformanceIdentity:
    digit = {"resident": "a", "rematerialized": "b", "drift": "c"}[label]
    return ExactPerformanceIdentity(digit * 64, "d" * 64, "e" * 64)


class FakeClock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def observation(
    mode: str,
    *,
    seconds: float,
) -> SAEQualificationObservation:
    return SAEQualificationObservation(
        identity=identity(mode),
        qualification_wall_seconds=10.0,
        measured_device_seconds=seconds,
        scientific_tokens=8,
        warmup_steps=1,
        measured_steps=2,
        compile_count=1,
        recompilations_after_warmup=0,
        oom_events=0,
        nonfinite_events=0,
        capacity_overflow_events=0,
        exact_l0=True,
        state_progress_exact=True,
        parameters_resident_before_timing=True,
        inputs_resident_before_timing=True,
        synchronized_device_timing=True,
        transition_output_sha256="f" * 64,
        replay_output_sha256="f" * 64,
    )


def selection(clock: FakeClock | None = None) -> dict[str, Any]:
    selected_clock = clock or FakeClock()

    def probe(mode):
        selected_clock.advance(2.0)
        return observation(
            mode,
            seconds=1.0 if mode == "resident" else 2.0,
        )

    return qualify_training_modes(
        probe,
        SAEArithmetic(
            workload="routed",
            condition="A",
            hidden_dim=2,
            candidates_per_token=4,
            selected_features_per_token=1,
        ),
        peak_flops_per_second=1_000,
        clock=selected_clock,
    )


def test_phase2_seed0_plan_supports_all_five_layouts() -> None:
    config = load_contribution_config(CONFIG)
    for condition in ("A", "R", "B", "F", "Q"):
        plan = TrainingWorkerPlan.from_config(
            config,
            phase="phase2_mechanism_pilot",
            condition=condition,
            seed=0,
        )
        assert plan.total_tokens == 2_000_000
        assert plan.planned_updates == 123
        assert plan.warmup_updates == 12
        assert plan.effective_batch_tokens == 16_384
        assert plan.expected_tokens_after(123) == 2_000_000


def test_standalone_cli_exposes_supervised_parent_and_hidden_child_commands() -> None:
    parser = _parser()
    common = [
        "--condition",
        "A",
        "--seed",
        "0",
        "--capture-control-root",
        "/capture",
        "--object-prefix",
        "gs://bucket/contributions",
        "--resident-cache-root",
        "/cache",
        "--resident-cache-max-bytes",
        "1000000",
        "--prefetch-receipt-path",
        "/cache/prefetch.json",
        "--source-revision",
        "a" * 40,
    ]
    qualify = parser.parse_args(
        [
            "qualify",
            *common,
            "--supervision-dir",
            "/control/supervision",
            "--host-id",
            "worker-0",
            "--working-directory",
            "/repo",
            "--selection-path",
            "/control/selection.json",
        ]
    )
    assert qualify.command == "qualify"
    assert qualify.warmup_steps == 1
    assert qualify.measured_steps == 3

    train = parser.parse_args(
        [
            "train",
            *common,
            "--supervision-dir",
            "/control/supervision",
            "--host-id",
            "worker-0",
            "--working-directory",
            "/repo",
            "--selection-path",
            "/control/selection.json",
            "--checkpoint-control-root",
            "/control/checkpoints",
            "--checkpoint-payload-root",
            "gs://bucket/checkpoints",
            "--maximum-runtime-seconds",
            "3600",
            "--result-path",
            "/control/result.json",
        ]
    )
    assert train.command == "train"
    assert train.maximum_runtime_seconds == 3_600

    prefetch = parser.parse_args(
        [
            "prefetch",
            "--phase",
            "phase2_mechanism_pilot",
            "--capture-control-root",
            "/capture",
            "--object-prefix",
            "gs://bucket/contributions",
            "--resident-cache-root",
            "/cache",
            "--resident-cache-max-bytes",
            "1000000",
            "--process-count",
            "4",
            "--process-index",
            "2",
            "--prefetch-receipt-path",
            "/cache/prefetch.json",
            "--source-revision",
            "a" * 40,
            "--supervision-dir",
            "/control/prefetch-supervision",
            "--host-id",
            "worker-0",
            "--working-directory",
            "/repo",
        ]
    )
    assert prefetch.command == "prefetch"
    assert prefetch.process_index == 2
    assert prefetch.maximum_runtime_seconds == 600

    prefetch_child = parser.parse_args(
        [
            "prefetch-child",
            "--phase",
            "phase2_mechanism_pilot",
            "--capture-control-root",
            "/capture",
            "--object-prefix",
            "gs://bucket/contributions",
            "--resident-cache-root",
            "/cache",
            "--resident-cache-max-bytes",
            "1000000",
            "--process-count",
            "4",
            "--process-index",
            "2",
            "--prefetch-receipt-path",
            "/cache/prefetch.json",
            "--source-revision",
            "a" * 40,
        ]
    )
    assert prefetch_child.command == "prefetch-child"


def test_prefetch_parent_is_hard_bounded_and_launches_only_its_hidden_child(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = _parser()
    arguments = [
        "prefetch",
        "--phase",
        "phase2_mechanism_pilot",
        "--capture-control-root",
        "/capture",
        "--object-prefix",
        "gs://bucket/contributions",
        "--resident-cache-root",
        "/cache",
        "--resident-cache-max-bytes",
        "1000000",
        "--process-count",
        "4",
        "--process-index",
        "2",
        "--prefetch-receipt-path",
        "/cache/prefetch.json",
        "--source-revision",
        "a" * 40,
        "--supervision-dir",
        "/control/prefetch-supervision",
        "--host-id",
        "controller-c00001-p00002",
        "--working-directory",
        "/repo",
    ]
    too_long = parser.parse_args(
        [*arguments, "--maximum-runtime-seconds", "601"]
    )
    with pytest.raises(
        ContributionTrainingWorkerError,
        match="cannot exceed 600",
    ):
        training_worker_module._prefetch_command(too_long)

    args = parser.parse_args(
        [*arguments, "--maximum-runtime-seconds", "600"]
    )
    captured: dict[str, Any] = {}

    def supervise(_args, **kwargs):
        captured.update(kwargs)
        return {"result_sha256": "b" * 64}

    receipt = {
        "receipt_sha256": "c" * 64,
        "process_partition": {"process_index": 2},
        "object_count": 31,
        "object_bytes": 123,
        "elapsed_seconds": 4.0,
    }
    monkeypatch.setattr(training_worker_module, "_supervise", supervise)
    monkeypatch.setattr(
        training_worker_module,
        "_read_json",
        lambda *_args, **_kwargs: receipt,
    )
    monkeypatch.setattr(
        training_worker_module,
        "validate_prefetch_receipt",
        lambda value: value,
    )
    assert training_worker_module._prefetch_command(args) == 0
    assert captured["maximum_seconds"] == 600
    assert captured["qualification_receipt"] is None
    assert captured["operation"] == "prefetch"
    child = captured["child_command"]
    assert child[:4] == [
        child[0],
        "-m",
        "moe_sae.contribution_training_worker",
        "prefetch-child",
    ]
    assert "qualify-child" not in child
    assert "train-child" not in child
    assert json.loads(capsys.readouterr().out)["receipt_sha256"] == "c" * 64


def test_process_lane_prefetch_is_near_one_copy_and_missing_cache_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from moe_sae import contribution_storage

    total_tokens = 2_000_000
    shard_tokens = 16_384
    entries: list[tuple[int, int, ObjectIdentity]] = []
    payloads: dict[str, bytes] = {}
    for index, start in enumerate(range(0, total_tokens, shard_tokens)):
        end = min(total_tokens, start + shard_tokens)
        payload = hashlib.sha256(str(index).encode("ascii")).digest() * 2
        digest = hashlib.sha256(payload).hexdigest()
        uri = f"gs://bucket/contributions/object-{index:04d}"
        identity = ObjectIdentity(
            uri=uri,
            generation=index + 1,
            size_bytes=len(payload),
            crc32c="AAAAAA==",
            sha256=digest,
        )
        entries.append((start, end, identity))
        payloads[uri] = payload

    class Store:
        def __init__(self) -> None:
            self.downloads: list[str] = []

        def download(self, identity: ObjectIdentity, destination: Path) -> None:
            self.downloads.append(identity.uri)
            destination.write_bytes(payloads[identity.uri])

    class Dataset:
        manifest_sha256 = "1" * 64

        def __init__(self, _root, *, object_prefix, cache, recomposition_sidecar):
            assert object_prefix == "gs://bucket/contributions"
            assert recomposition_sidecar is None
            self.cache = cache

        @staticmethod
        def token_range_object_identities(start: int, end: int):
            return tuple(
                identity
                for left, right, identity in entries
                if max(start, left) < min(end, right)
            )

        def prefetch_token_range(self, start: int, end: int):
            identities = self.token_range_object_identities(start, end)
            paths = self.cache.materialize_many(identities, max_workers=4)
            resident = self.cache.require_resident_many(identities)
            assert paths == resident
            return {
                "objects": len(paths),
                "bytes": sum(path.stat().st_size for path in paths),
            }

        def require_resident_token_range(self, start: int, end: int):
            identities = self.token_range_object_identities(start, end)
            paths = self.cache.require_resident_many(identities)
            return {
                "objects": len(paths),
                "bytes": sum(path.stat().st_size for path in paths),
            }

    Dataset.total_tokens = total_tokens
    monkeypatch.setattr(contribution_storage, "ContributionDataset", Dataset)
    config = load_contribution_config(CONFIG)
    store = Store()
    receipts = []
    for process_index in range(4):
        receipts.append(
            prefetch_process_lane(
                config=config,
                phase="phase2_mechanism_pilot",
                source_revision="a" * 40,
                capture_control_root=tmp_path / "control",
                object_prefix="gs://bucket/contributions",
                cache_root=tmp_path / f"cache-{process_index}",
                cache_max_bytes=1_000_000,
                process_count=4,
                process_index=process_index,
                recomposition_sidecar=None,
                object_store=store,
            )
        )

    ranges = [
        (
            receipt["process_partition"]["range_start"],
            receipt["process_partition"]["range_end"],
        )
        for receipt in receipts
    ]
    assert ranges == [
        (0, 500_000),
        (500_000, 1_000_000),
        (1_000_000, 1_500_000),
        (1_500_000, 2_000_000),
    ]
    unique_bytes = sum(identity.size_bytes for _, _, identity in entries)
    aggregate_bytes = sum(receipt["object_bytes"] for receipt in receipts)
    assert 1.0 <= aggregate_bytes / unique_bytes < 1.04

    first = receipts[0]
    first_identity = Dataset.token_range_object_identities(0, 500_000)[0]
    (tmp_path / "cache-0" / f"{first_identity.sha256}.safetensors").unlink()
    plan = TrainingWorkerPlan.from_config(
        config,
        phase="phase2_mechanism_pilot",
        condition="A",
        seed=0,
    )
    with pytest.raises(
        ContributionTrainingWorkerError,
        match="no longer resident",
    ):
        _prove_prefetch_for_plan(
            first,
            config=config,
            plan=plan,
            source_revision="a" * 40,
            capture_control_root=tmp_path / "control",
            object_prefix="gs://bucket/contributions",
            cache_root=tmp_path / "cache-0",
            cache_max_bytes=1_000_000,
            recomposition_sidecar=None,
        )


def test_qualifier_benchmarks_both_modes_and_binds_faster_receipt() -> None:
    result = selection()
    checked = validate_training_mode_selection(result)
    assert checked["candidate_order"] == ["resident", "rematerialized"]
    assert checked["chosen_mode"] == "resident"
    mode, selected_identity, receipt = chosen_mode_evidence(checked)
    assert mode == "resident"
    assert selected_identity == identity("resident")
    assert (
        receipt["performance"]["useful_prefix_mfu"]
        >= MINIMUM_USEFUL_PREFIX_MFU
    )
    assert receipt["evidence"]["qualification_wall_seconds"] == 4.0


def test_mode_selection_binds_resident_v2_workload_context() -> None:
    clock = FakeClock()
    context = {
        "capture_manifest_sha256": "1" * 64,
        "resident_cache_policy": "resident_only_no_refill",
        "bank_capacity": 7,
    }

    def probe(mode):
        clock.advance(1.0)
        return observation(mode, seconds=1.0)

    result = qualify_training_modes(
        probe,
        SAEArithmetic(
            workload="routed",
            condition="A",
            hidden_dim=2,
            candidates_per_token=4,
            selected_features_per_token=1,
        ),
        peak_flops_per_second=1_000,
        workload_context=context,
        clock=clock,
    )
    assert validate_training_mode_selection(result)["workload_context"] == context
    result["workload_context"]["bank_capacity"] = 8
    with pytest.raises(ContributionTrainingWorkerError, match="self-hash"):
        validate_training_mode_selection(result)


def test_qualifier_total_has_one_nonwaivable_600_second_budget() -> None:
    clock = FakeClock()

    def probe(mode):
        clock.advance(301.0)
        return observation(mode, seconds=1.0)

    with pytest.raises(ContributionTrainingWorkerError, match="600-second total"):
        qualify_training_modes(
            probe,
            SAEArithmetic(
                workload="routed",
                condition="B",
                hidden_dim=2,
                candidates_per_token=4,
                selected_features_per_token=1,
            ),
            peak_flops_per_second=1_000,
            clock=clock,
        )


def test_one_failed_mode_can_select_the_other_but_both_attempts_are_retained() -> None:
    clock = FakeClock()

    def probe(mode):
        clock.advance(1.0)
        if mode == "resident":
            raise RuntimeError("resident OOM")
        return observation(mode, seconds=1.0)

    result = qualify_training_modes(
        probe,
        SAEArithmetic(
            workload="shared_q",
            condition="Q",
            hidden_dim=2,
            candidates_per_token=4,
            selected_features_per_token=1,
        ),
        peak_flops_per_second=1_000,
        clock=clock,
    )
    assert result["chosen_mode"] == "rematerialized"
    assert result["candidates"][0]["status"] == "failed"
    assert result["candidates"][0]["error_message"] == "resident OOM"


@dataclass(frozen=True)
class State:
    step: int
    tokens: int
    value: int


class Source:
    total_tokens = 10
    planned_updates = 3

    def iter_updates(self):
        yield 4
        yield 4
        yield 2


class Checkpoints:
    def __init__(self) -> None:
        self.restored: tuple[State, dict[str, Any]] | None = None
        self.saves: list[tuple[State, bool]] = []

    def restore(self, _template, run_identity):
        if self.restored is None:
            return None
        state, metadata = self.restored
        return RestoredTrainingState(state, metadata)

    def save(self, state, run_identity, _timing, final):
        metadata = {"run_identity_sha256": run_identity["run_identity_sha256"]}
        self.restored = (state, metadata)
        self.saves.append((state, final))
        return {"step": state.step, "tokens": state.tokens, "final": final}


def plan() -> TrainingWorkerPlan:
    return TrainingWorkerPlan(
        phase_name="tiny",
        condition="A",
        seed=0,
        total_tokens=10,
        planned_updates=3,
        warmup_updates=1,
        global_microbatch_tokens=2,
        microbatches_per_update=2,
        checkpoint_tokens=(4, 10),
        randomization_key=0,
    )


def dependencies(
    checkpoints: Checkpoints,
    *,
    fail_at_step: int | None = None,
    lowered_identity: ExactPerformanceIdentity | None = None,
):
    clock = FakeClock()

    def lower(_mode, _state, _batch, _mean):
        return PreparedTrainingExecutable(
            handle=object(),
            identity=lowered_identity or identity("resident"),
            graph_manifest={"stablehlo_sha256": "1" * 64},
            hardware_manifest={"global_chip_count": 16},
            execution_config={"mode": "resident"},
        )

    def step(_prepared, state, batch, _mean):
        if fail_at_step is not None and state.step == fail_at_step:
            raise RuntimeError("simulated preemption")
        clock.advance(0.25)
        new = State(state.step + 1, state.tokens + batch, state.value * 3 + batch)
        return new, {
            "loss": 1.0 / new.step,
            "sse": 1.0,
            "sst": 2.0,
            "l0": 2.0,
            "valid_tokens": batch,
            "topk_valid": True,
            "grad_norm": 0.5,
            "learning_rate": 1e-3,
        }

    return TrainingLoopDependencies(
        initialize_state=lambda: State(0, 0, 1),
        lower=lower,
        compile=lambda _prepared: 0.5,
        step=step,
        synchronize=lambda _state: None,
        progress=lambda state: (state.step, state.tokens),
        restore=checkpoints.restore,
        save=checkpoints.save,
        clock=clock,
    )


def test_extended_training_requires_and_uses_exact_passing_receipt() -> None:
    checkpoints = Checkpoints()
    result = run_qualified_training(
        plan(),
        Source(),
        mean=None,
        selection=selection(),
        dependencies=dependencies(checkpoints),
        expected_l0=2,
        maximum_runtime_seconds=3_600,
    )
    assert result["status"] == "completed"
    assert result["steps"] == 3
    assert result["tokens"] == 10
    assert result["authorization_basis"] == "qualified_exact_identity"
    assert result["chip_seconds"] > 0
    assert checkpoints.saves[-1][1] is True


def test_bounded_direct_training_uses_resident_graph_without_selection() -> None:
    checkpoints = Checkpoints()
    result = run_qualified_training(
        plan(),
        Source(),
        mean=None,
        selection=None,
        dependencies=dependencies(checkpoints),
        expected_l0=2,
        maximum_runtime_seconds=600,
        direct_bounded_resident=True,
    )
    assert result["status"] == "completed"
    assert result["input_mode"] == "resident"
    assert result["steps"] == 3
    assert result["tokens"] == 10
    assert result["authorization_basis"] == "bounded_ungated_workload"
    assert result["qualification_receipt_sha256"] is None
    assert result["run_identity"]["admission"] == {
        "basis": "bounded_ungated_resident",
        "evidence_sha256": None,
    }
    assert checkpoints.saves[-1][1] is True


def test_bounded_direct_training_rejects_an_extended_deadline() -> None:
    with pytest.raises(
        ContributionTrainingWorkerError,
        match="cannot exceed 600 seconds",
    ):
        run_qualified_training(
            plan(),
            Source(),
            mean=None,
            selection=None,
            dependencies=dependencies(Checkpoints()),
            expected_l0=2,
            maximum_runtime_seconds=601,
            direct_bounded_resident=True,
        )


def test_bounded_direct_training_resumes_from_exact_lowered_identity() -> None:
    store = Checkpoints()
    with pytest.raises(RuntimeError, match="simulated preemption"):
        run_qualified_training(
            plan(),
            Source(),
            mean=None,
            selection=None,
            dependencies=dependencies(store, fail_at_step=1),
            expected_l0=2,
            maximum_runtime_seconds=600,
            direct_bounded_resident=True,
        )
    assert store.restored is not None
    assert store.restored[0].step == 1

    result = run_qualified_training(
        plan(),
        Source(),
        mean=None,
        selection=None,
        dependencies=dependencies(store),
        expected_l0=2,
        maximum_runtime_seconds=600,
        direct_bounded_resident=True,
    )
    assert result["status"] == "completed"
    assert result["tokens"] == 10
    assert store.restored[0] == State(3, 10, 77)


def test_lowered_identity_drift_is_rejected_before_compile_or_steps() -> None:
    with pytest.raises(
        ContributionTrainingWorkerError,
        match="differs from the chosen qualification",
    ):
        run_qualified_training(
            plan(),
            Source(),
            mean=None,
            selection=selection(),
            dependencies=dependencies(
                Checkpoints(),
                lowered_identity=identity("drift"),
            ),
            expected_l0=2,
            maximum_runtime_seconds=3_600,
        )


def test_interrupted_resume_matches_uninterrupted_fixed_order() -> None:
    uninterrupted_store = Checkpoints()
    uninterrupted = run_qualified_training(
        plan(),
        Source(),
        mean=None,
        selection=selection(),
        dependencies=dependencies(uninterrupted_store),
        expected_l0=2,
        maximum_runtime_seconds=3_600,
    )
    expected_state = uninterrupted_store.restored[0]

    resumed_store = Checkpoints()
    with pytest.raises(RuntimeError, match="simulated preemption"):
        run_qualified_training(
            plan(),
            Source(),
            mean=None,
            selection=selection(),
            dependencies=dependencies(resumed_store, fail_at_step=1),
            expected_l0=2,
            maximum_runtime_seconds=3_600,
        )
    assert resumed_store.restored[0].step == 1

    resumed = run_qualified_training(
        plan(),
        Source(),
        mean=None,
        selection=selection(),
        dependencies=dependencies(resumed_store),
        expected_l0=2,
        maximum_runtime_seconds=3_600,
    )
    assert resumed["tokens"] == uninterrupted["tokens"] == 10
    assert resumed_store.restored[0] == expected_state


def test_resume_rejects_state_off_the_exact_token_boundary() -> None:
    store = Checkpoints()
    store.restored = (
        State(1, 3, 9),
        {"run_identity_sha256": "will-be-replaced"},
    )

    def forged_restore(_template, run_identity):
        return RestoredTrainingState(
            store.restored[0],
            {"run_identity_sha256": run_identity["run_identity_sha256"]},
        )

    deps = dependencies(store)
    deps = TrainingLoopDependencies(
        initialize_state=deps.initialize_state,
        lower=deps.lower,
        compile=deps.compile,
        step=deps.step,
        synchronize=deps.synchronize,
        progress=deps.progress,
        restore=forged_restore,
        save=deps.save,
        clock=deps.clock,
    )
    with pytest.raises(ContributionTrainingWorkerError, match="fixed-order boundary"):
        run_qualified_training(
            plan(),
            Source(),
            mean=None,
            selection=selection(),
            dependencies=deps,
            expected_l0=2,
            maximum_runtime_seconds=3_600,
        )
