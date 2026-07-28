from __future__ import annotations

import hashlib
import json
from pathlib import Path
import threading

import pytest

from moe_sae.contribution_artifacts import canonical_sha256
from moe_sae.horizon_supervisor import (
    CommandOutcome,
    CompletionSpec,
    FileFingerprint,
    HorizonPlanError,
    HorizonStateError,
    HorizonSupervisor,
    ResourceBinding,
    classify_failure,
    inspect_completion,
    parse_horizon_plan,
)


MODELS = ("sae", "transcoder")
CONDITIONS = ("A", "R", "B", "F")
MILESTONES = (2_000_000, 5_000_000, 10_000_000)


def _completion(path: Path) -> dict:
    return {"path": str(path), "format": "file"}


def _plan(tmp_path: Path) -> dict:
    working = tmp_path / "repo"
    working.mkdir(parents=True)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    resources = []
    tasks = [
        {
            "id": "capture-train",
            "kind": "capture",
            "depends_on": [],
            "command": ["fake", "--task", "{task_id}"],
            "completion": _completion(artifacts / "capture-train"),
            "role": {"split": "train"},
        },
        {
            "id": "capture-eval",
            "kind": "capture",
            "depends_on": [],
            "command": ["fake", "--task", "{task_id}"],
            "completion": _completion(artifacts / "capture-eval"),
            "role": {"split": "eval"},
        },
        {
            "id": "overlay-train",
            "kind": "overlay",
            "depends_on": ["capture-train"],
            "command": ["fake", "--task", "{task_id}"],
            "completion": _completion(artifacts / "overlay-train"),
            "role": {"split": "train"},
        },
        {
            "id": "overlay-eval",
            "kind": "overlay",
            "depends_on": ["capture-eval"],
            "command": ["fake", "--task", "{task_id}"],
            "completion": _completion(artifacts / "overlay-eval"),
            "role": {"split": "eval"},
        },
        {
            "id": "mirror-us",
            "kind": "mirror",
            "depends_on": ["overlay-train", "overlay-eval"],
            "command": ["fake", "--task", "{task_id}"],
            "completion": _completion(artifacts / "mirror-us"),
            "role": {"region": "us-east1"},
        },
        {
            "id": "mirror-eu",
            "kind": "mirror",
            "depends_on": ["overlay-train", "overlay-eval"],
            "command": ["fake", "--task", "{task_id}"],
            "completion": _completion(artifacts / "mirror-eu"),
            "role": {"region": "europe-west4"},
        },
    ]
    eval_ids: dict[str, list[str]] = {model: [] for model in MODELS}
    for index, (model, condition) in enumerate(
        (item for model in MODELS for item in ((model, c) for c in CONDITIONS))
    ):
        resource = f"v6e-{model}-{condition.lower()}"
        resources.append(
            {
                "id": resource,
                "allocation": "queued_spot",
                "project": "neural-dragnet",
                "zone": "us-east1-d" if model == "transcoder" else "europe-west4-a",
                "node_name_prefix": f"horizon-{index}-node",
                "queued_resource_prefix": f"horizon-{index}-queue",
                "accelerator_type": "v6e-16",
                "runtime_version": "v6e-ubuntu-2404",
                "service_account": (
                    "655343590563-compute@developer.gserviceaccount.com"
                ),
                "network": "default",
                "subnetwork": "default",
                "process_count": 4,
                "labels": {"purpose": "gemma-horizon"},
                "formation_timeout_seconds": 3600,
                "poll_seconds": 10,
                "cleanup_on_finish": True,
            }
        )
        train_id = f"train-{model}-{condition.lower()}"
        train_path = artifacts / train_id
        tasks.append(
            {
                "id": train_id,
                "kind": "train",
                "depends_on": [
                    "mirror-us" if model == "transcoder" else "mirror-eu"
                ],
                "command": [
                    "fake",
                    "--task",
                    "{task_id}",
                    "--attempt",
                    "{attempt_id}",
                    "--seconds",
                    "{segment_seconds}",
                ],
                "completion": _completion(train_path),
                "role": {
                    "scientific_model": model,
                    "condition": condition,
                    "seed": 1,
                },
                "resource_id": resource,
                "segment": {
                    "seconds": 3500,
                    "max_segments": 8,
                    "checkpoint": _completion(artifacts / f"{train_id}-checkpoint"),
                    "boundary_exit_codes": [0, 124, 137, 143],
                },
            }
        )
        for milestone in MILESTONES:
            eval_id = f"eval-{model}-{condition.lower()}-{milestone}"
            eval_ids[model].append(eval_id)
            tasks.append(
                {
                    "id": eval_id,
                    "kind": "eval",
                    "depends_on": [train_id],
                    "command": ["fake", "--task", "{task_id}", "--tpu", "{tpu_name}"],
                    "completion": _completion(artifacts / eval_id),
                    "role": {
                        "scientific_model": model,
                        "condition": condition,
                        "seed": 1,
                        "milestone_tokens": milestone,
                    },
                    "resource_id": resource,
                }
            )
    for model in MODELS:
        tasks.append(
            {
                "id": f"report-{model}",
                "kind": "report",
                "depends_on": eval_ids[model],
                "command": ["fake", "--task", "{task_id}"],
                "completion": _completion(artifacts / f"report-{model}"),
                "role": {"scientific_model": model},
            }
        )
    return {
        "schema_version": "gemma4-horizon-launch-plan-v1",
        "campaign_id": "horizon-test",
        "state_root": str(tmp_path / "state"),
        "working_directory": str(working),
        "inherit_environment": [],
        "environment": {"PYTHONUNBUFFERED": "1"},
        "poll_seconds": 0.001,
        "max_parallel_tasks": 8,
        "retry_policy": {
            "max_transient_attempts": 4,
            "initial_backoff_seconds": 0.001,
            "maximum_backoff_seconds": 0.002,
            "transport_grace_seconds": 5,
        },
        "resources": resources,
        "tasks": tasks,
    }


def test_exact_plan_covers_eight_trajectories_and_24_evaluations(tmp_path):
    plan = parse_horizon_plan(_plan(tmp_path))

    assert len([task for task in plan.tasks if task.kind == "train"]) == 8
    assert len([task for task in plan.tasks if task.kind == "eval"]) == 24
    assert len(plan.resources) == 8
    assert len(plan.plan_sha256) == 64


def test_controller_forwards_explicit_tpu_ssh_user(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("TPU_SSH_USER", "abistras")
    plan = parse_horizon_plan(_plan(tmp_path))
    supervisor = HorizonSupervisor(
        plan,
        resource_manager=FakeResources(tmp_path / "resource-state"),
    )
    assert supervisor._base_environment(plan.tasks[0])["TPU_SSH_USER"] == (
        "abistras"
    )


def test_plan_rejects_a_missing_milestone(tmp_path):
    raw = _plan(tmp_path)
    raw["tasks"] = [
        task
        for task in raw["tasks"]
        if task["id"] != "eval-sae-a-2000000"
    ]
    next(task for task in raw["tasks"] if task["id"] == "report-sae")[
        "depends_on"
    ].remove("eval-sae-a-2000000")

    with pytest.raises(HorizonPlanError, match="every SAE/transcoder"):
        parse_horizon_plan(raw)


def test_plan_rejects_secret_values_and_unbounded_training(tmp_path):
    raw = _plan(tmp_path)
    raw["environment"]["HF_TOKEN"] = "must-not-be-persisted"
    with pytest.raises(HorizonPlanError, match="secret-like"):
        parse_horizon_plan(raw)

    raw = _plan(tmp_path / "second")
    train = next(task for task in raw["tasks"] if task["kind"] == "train")
    train["segment"]["seconds"] = 3601
    with pytest.raises(HorizonPlanError, match="must not exceed 3600"):
        parse_horizon_plan(raw)


def test_numerical_and_corruption_failures_outrank_ssh_words():
    assert (
        classify_failure(
            returncode=255,
            output="connection reset after non-finite gradient",
            timed_out=False,
            completion_present=False,
            checkpoint_before=None,
            checkpoint_after=None,
            segment=None,
        )
        == "numerical_failure"
    )
    assert (
        classify_failure(
            returncode=255,
            output="connection reset; manifest hash mismatch",
            timed_out=False,
            completion_present=False,
            checkpoint_before=None,
            checkpoint_after=None,
            segment=None,
        )
        == "scientific_corruption"
    )


def test_segment_boundary_requires_checkpoint_progress(tmp_path):
    plan = parse_horizon_plan(_plan(tmp_path))
    segment = next(task.segment for task in plan.tasks if task.kind == "train")
    old = FileFingerprint("/checkpoint", 10, "a" * 64)
    new = FileFingerprint("/checkpoint", 11, "b" * 64)
    assert (
        classify_failure(
            returncode=124,
            output="bounded worker ended",
            timed_out=False,
            completion_present=False,
            checkpoint_before=old,
            checkpoint_after=new,
            segment=segment,
        )
        == "segment_boundary"
    )
    assert (
        classify_failure(
            returncode=124,
            output="bounded worker ended",
            timed_out=False,
            completion_present=False,
            checkpoint_before=old,
            checkpoint_after=old,
            segment=segment,
        )
        == "unclassified_failure"
    )


def test_json_completion_is_self_hashed_and_then_immutable(tmp_path):
    path = tmp_path / "done.json"
    unsigned = {"schema_version": "done-v1", "status": "complete"}
    path.write_text(
        json.dumps({**unsigned, "record_sha256": canonical_sha256(unsigned)})
    )
    spec = CompletionSpec.parse(
        {
            "path": str(path),
            "format": "json",
            "expected": {"status": "complete"},
            "self_hash_field": "record_sha256",
        },
        "completion",
    )
    observed = inspect_completion(spec)
    assert observed is not None
    path.write_text("{}")
    with pytest.raises(HorizonStateError):
        inspect_completion(spec)


class FakeResources:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.bindings: dict[str, ResourceBinding] = {}
        self.cleaned = False

    def ensure(self, spec, *, attempt_id, stop_event):
        binding = self.bindings.get(spec.resource_id)
        if binding is None:
            binding = ResourceBinding(
                spec.resource_id,
                attempt_id,
                spec.allocation,
                spec.project,
                spec.zone,
                f"{spec.node_name_prefix}-{attempt_id}",
                f"{spec.queued_resource_prefix}-{attempt_id}",
                self.root / f"{spec.resource_id}.json",
            )
            self.bindings[spec.resource_id] = binding
        return binding

    def invalidate(self, binding, *, reason):
        self.bindings.pop(binding.resource_id, None)

    def cleanup(self):
        self.cleaned = True
        self.bindings.clear()


class FakeCommands:
    def __init__(self, plan) -> None:
        self.tasks = plan.task_by_id
        self.calls: dict[str, int] = {}

    def __call__(
        self,
        command,
        *,
        environment,
        working_directory,
        timeout_seconds,
        stdout_path,
        stderr_path,
        stop_event,
    ):
        task_id = command[command.index("--task") + 1]
        task = self.tasks[task_id]
        count = self.calls.get(task_id, 0) + 1
        self.calls[task_id] = count
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text(f"{task_id} call {count}\n")
        stderr_path.write_text("")
        if task.kind == "train" and count == 1:
            assert task.segment is not None
            task.segment.checkpoint.path.write_text(f"checkpoint {task_id}")
        else:
            task.completion.path.write_text(f"complete {task_id}")
        return CommandOutcome(
            returncode=0,
            timed_out=False,
            stopped=False,
            elapsed_seconds=0.01,
            stdout=FileFingerprint(
                str(stdout_path),
                stdout_path.stat().st_size,
                hashlib.sha256(stdout_path.read_bytes()).hexdigest(),
            ),
            stderr=FileFingerprint(
                str(stderr_path),
                0,
                hashlib.sha256(b"").hexdigest(),
            ),
            output_tail="",
        )


def test_full_dag_runs_concurrently_and_rotates_training_segments(tmp_path):
    plan = parse_horizon_plan(_plan(tmp_path))
    commands = FakeCommands(plan)
    resources = FakeResources(tmp_path / "resource-state")
    supervisor = HorizonSupervisor(
        plan,
        command_runner=commands,
        resource_manager=resources,
        stop_event=threading.Event(),
        sleeper=lambda _seconds: None,
    )

    result = supervisor.run()

    assert result["status"] == "completed"
    assert len(result["completed_tasks"]) == 40
    assert all(
        commands.calls[task.task_id] == 2
        for task in plan.tasks
        if task.kind == "train"
    )
    assert resources.cleaned
    attempts = list((plan.state_root / "attempts").iterdir())
    numbers = sorted(int(path.name.split("-")[1]) for path in attempts)
    assert numbers == list(range(1, len(numbers) + 1))


def test_witness_detects_completion_mutation_on_restart(tmp_path):
    plan = parse_horizon_plan(_plan(tmp_path))
    commands = FakeCommands(plan)
    resources = FakeResources(tmp_path / "resource-state")
    first = HorizonSupervisor(
        plan,
        command_runner=commands,
        resource_manager=resources,
        sleeper=lambda _seconds: None,
    )
    assert first.run()["status"] == "completed"
    task = plan.tasks[0]
    task.completion.path.write_text("changed")

    with pytest.raises(HorizonStateError, match="immutable completion changed"):
        HorizonSupervisor(
            plan,
            command_runner=commands,
            resource_manager=FakeResources(tmp_path / "resource-state-2"),
        ).run()
