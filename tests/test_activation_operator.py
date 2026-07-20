from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import subprocess
import sys

import pytest

from moe_sae import cli
from moe_sae.activation_operator import (
    ACTIVATION_PHASES,
    ActivationControl,
    ActivationOperatorError,
    ActivationRunSpec,
    ActivationSupervisorLeaseHeld,
)


def spec(suffix: str = "") -> ActivationRunSpec:
    return ActivationRunSpec(
        config_sha256="a" * 64,
        checkpoint=f"/checkpoint{suffix}",
        checkpoint_sha256="b" * 64,
        grain_train="train-*",
        grain_eval="eval-*",
        dataset_train_manifest="train.json",
        dataset_eval_manifest="eval.json",
    )


def test_state_is_provenance_bound_self_hashed_and_resumable(tmp_path: Path) -> None:
    control = ActivationControl(tmp_path)
    created = control.begin(spec())
    assert control.begin(spec())["run_id"] == created["run_id"]
    with pytest.raises(ActivationOperatorError, match="different checkpoint"):
        control.begin(spec("-changed"))

    control.mark_running("benchmark")
    control.mark_running("benchmark")
    assert control.load_state()["phases"]["benchmark"]["attempts"] == 1
    control.mark_passed("benchmark", {"gate": True})
    assert control.load_state()["phases"]["benchmark"]["status"] == "passed"

    payload = json.loads(control.state_path.read_text(encoding="utf-8"))
    payload["phases"]["benchmark"]["status"] = "failed"
    control.state_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ActivationOperatorError, match="self-hash"):
        control.load_state()


def test_abort_remains_latched_for_every_controller_until_clear(
    tmp_path: Path,
) -> None:
    control = ActivationControl(tmp_path)
    control.begin(spec())
    request = control.request_abort("operator stop")

    first = control.acknowledge_abort("canary")
    second = control.acknowledge_abort("canary")
    assert first["last_abort_sha256"] == request["abort_sha256"]
    assert second["last_abort_sha256"] == request["abort_sha256"]
    assert control.pending_abort() == request
    assert control.clear_abort()
    assert control.pending_abort() is None
    assert not control.clear_abort()


def test_abort_cannot_be_cleared_while_a_phase_is_running(tmp_path: Path) -> None:
    control = ActivationControl(tmp_path)
    control.begin(spec())
    control.mark_running("train")
    control.request_abort("stop")
    with pytest.raises(ActivationOperatorError, match="phase is running"):
        control.clear_abort()


def test_late_success_cannot_overwrite_failure_and_retry_has_new_attempt(
    tmp_path: Path,
) -> None:
    control = ActivationControl(tmp_path)
    control.begin(spec())
    running = control.mark_running("canary")
    first_attempt = running["phases"]["canary"]["attempt_id"]
    failed = control.mark_failed(
        "canary", RuntimeError("peer failed"), attempt_id=first_attempt
    )
    assert failed["phases"]["canary"]["status"] == "failed"
    late = control.mark_passed("canary", attempt_id=first_attempt)
    assert late["phases"]["canary"]["status"] == "failed"

    resumed = control.mark_running("canary")
    second_attempt = resumed["phases"]["canary"]["attempt_id"]
    assert second_attempt != first_attempt
    with pytest.raises(ActivationOperatorError, match="stale completion"):
        control.mark_passed("canary", attempt_id=first_attempt)
    passed = control.mark_passed("canary", attempt_id=second_attempt)
    assert passed["phases"]["canary"]["status"] == "passed"


def test_lifetime_controller_lane_rejects_duplicate_supervisor(
    tmp_path: Path,
) -> None:
    control = ActivationControl(tmp_path)
    control.begin(spec())
    with control.supervisor_lease("controller-0"):
        with pytest.raises(ActivationSupervisorLeaseHeld, match="owns this controller"):
            with control.supervisor_lease("controller-0"):
                pass
        with control.supervisor_lease("controller-1"):
            pass
    with control.supervisor_lease("controller-0"):
        pass


def test_clear_refuses_an_acknowledged_abort_while_any_lane_is_leased(
    tmp_path: Path,
) -> None:
    control = ActivationControl(tmp_path)
    control.begin(spec())
    control.request_abort("stop")
    control.acknowledge_abort("train")
    with control.supervisor_lease("controller-0"):
        with pytest.raises(ActivationOperatorError, match="lifetime lease"):
            control.clear_abort()
    assert control.clear_abort()


def test_inherited_descriptor_keeps_lane_locked_after_parent_closes(
    tmp_path: Path,
) -> None:
    control = ActivationControl(tmp_path)
    control.begin(spec())
    with control.supervisor_lease("controller-0") as lease:
        child = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(0.4)"],
            pass_fds=(lease.descriptor,),
        )
    try:
        with pytest.raises(ActivationSupervisorLeaseHeld):
            with control.supervisor_lease("controller-0"):
                pass
    finally:
        child.wait(timeout=5)
    with control.supervisor_lease("controller-0"):
        pass


def test_four_lane_rendezvous_and_completion_releases_are_attempt_exact(
    tmp_path: Path,
) -> None:
    control = ActivationControl(tmp_path)
    control.begin(spec())
    operation = "c" * 64
    with ExitStack() as stack:
        leases = [
            stack.enter_context(
                control.supervisor_lease(
                    f"controller-{index}", operation_sha256=operation
                )
            )
            for index in range(4)
        ]
        for lease in leases:
            control.supervisor_heartbeat(lease, "ready", step="preflight")
        assert len(
            control.supervisor_rendezvous(
                leases[0], "preflight", timeout_seconds=0.1
            )
        ) == 4

        # A lagging controller may consume the immutable start release after a
        # peer has already advanced from ready to running.
        control.supervisor_heartbeat(leases[0], "running", step="preflight")
        control.supervisor_rendezvous(
            leases[1], "preflight", timeout_seconds=0.1
        )

        for lease in leases:
            control.supervisor_heartbeat(lease, "succeeded", step="preflight")
        assert (
            control.supervisor_wait_completion(
                leases[0],
                "preflight",
                local_state="succeeded",
                timeout_seconds=0.1,
            )
            == "succeeded"
        )
        control.supervisor_heartbeat(leases[0], "ready", step="benchmark")
        assert (
            control.supervisor_wait_completion(
                leases[1],
                "preflight",
                local_state="succeeded",
                timeout_seconds=0.1,
            )
            == "succeeded"
        )

    # Reusing the same operation/step cannot consume the prior invocation's
    # release because all four lease IDs change.
    with ExitStack() as stack:
        resumed = [
            stack.enter_context(
                control.supervisor_lease(
                    f"controller-{index}", operation_sha256=operation
                )
            )
            for index in range(4)
        ]
        for lease in resumed:
            control.supervisor_heartbeat(lease, "ready", step="preflight")
        ready = control.supervisor_rendezvous(
            resumed[0], "preflight", timeout_seconds=0.1
        )
        assert {item["lease_id"] for item in ready} == {
            lease.lease_id for lease in resumed
        }


def test_fresh_ready_json_without_a_held_matching_lane_lock_is_ignored(
    tmp_path: Path,
) -> None:
    control = ActivationControl(tmp_path)
    control.begin(spec())
    operation = "d" * 64
    with control.supervisor_lease(
        "old-controller", operation_sha256=operation
    ) as old:
        stale_ready = control.supervisor_heartbeat(
            old, "ready", step="benchmark"
        )
    # Restore a cryptographically valid and fresh-looking ready record after
    # its owning lease has closed. It must not count toward rendezvous.
    old.status_path.write_text(json.dumps(stale_ready), encoding="utf-8")
    with control.supervisor_lease(
        "current-controller", operation_sha256=operation
    ) as current:
        control.supervisor_heartbeat(current, "ready", step="benchmark")
        records = control._matching_supervisors(current, step="benchmark")
        assert [item["lease_id"] for item in records] == [current.lease_id]


def test_supervisor_status_is_visible_and_identical_heartbeats_are_throttled(
    tmp_path: Path,
) -> None:
    control = ActivationControl(tmp_path)
    control.begin(spec())
    with control.supervisor_lease("controller-0") as lease:
        first = control.supervisor_heartbeat(lease, "running", step="canary")
        second = control.supervisor_heartbeat(lease, "running", step="canary")
        assert second == first
        status = control.status()["supervisors"]
        assert len(status) == 1
        assert status[0]["state"] == "running"
        assert status[0]["step"] == "canary"
        assert status[0]["active"] is True
        assert status[0]["heartbeat_age_seconds"] >= 0.0


def test_startup_abort_retires_the_actual_running_phase(tmp_path: Path) -> None:
    control = ActivationControl(tmp_path)
    control.begin(spec())
    control.mark_running("train")
    control.request_abort("stop after parent crash")
    with pytest.raises(RuntimeError, match="train boundary"):
        cli._raise_if_activation_aborted(control, "benchmark")
    state = control.load_state()
    assert state["phases"]["train"]["status"] == "aborted"
    assert state["phases"]["benchmark"]["status"] == "pending"


def test_activation_lease_descriptor_is_validated_for_benchmark_descendants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with (tmp_path / "lease.lock").open("a+b", buffering=0) as handle:
        monkeypatch.setenv(
            "MOE_SAE_ACTIVATION_LEASE_FD", str(handle.fileno())
        )
        assert cli._inherited_activation_lease_fds() == (handle.fileno(),)
    monkeypatch.setenv("MOE_SAE_ACTIVATION_LEASE_FD", "not-an-fd")
    with pytest.raises(RuntimeError, match="descriptor is invalid"):
        cli._inherited_activation_lease_fds()


class _Config:
    config_hash = "a" * 64
    benchmark_steps = 200

    def validate_production(self) -> None:
        return None


def farm_args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        config=str(tmp_path / "config.yaml"),
        artifact_root=str(tmp_path / "artifacts"),
        checkpoint="/checkpoint",
        checkpoint_sha256="b" * 64,
        grain_train="train-*",
        grain_eval="eval-*",
        dataset_train_manifest="train.json",
        dataset_eval_manifest="eval.json",
    )


def test_activation_farm_uses_fresh_preflight_phase_and_gate_children(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = farm_args(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(cli, "_config", lambda _args: _Config())
    monkeypatch.setattr(cli, "_require_production_artifact_root", lambda _root: None)

    def run_child(
        _command,
        *,
        label,
        control=None,
        phase=None,
        lease_fd=None,
        supervisor=None,
    ):
        del control, phase, lease_fd, supervisor
        calls.append(label)

    monkeypatch.setattr(cli, "_run_activation_subprocess", run_child)
    assert cli._command_activation_farm(args) == 0
    assert calls == [
        "preflight",
        "benchmark",
        "benchmark completion gate",
        "canary",
        "canary completion gate",
        "train",
        "train completion gate",
        "eval",
        "eval completion gate",
    ]
    state = ActivationControl(args.artifact_root).load_state()
    assert [state["phases"][phase]["status"] for phase in ACTIVATION_PHASES] == [
        "passed"
    ] * 4


def test_activation_farm_gates_and_skips_existing_final_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = farm_args(tmp_path)
    root = Path(args.artifact_root)
    for phase in ACTIVATION_PHASES:
        path = cli._activation_final_record(root, phase)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    calls: list[str] = []
    monkeypatch.setattr(cli, "_config", lambda _args: _Config())
    monkeypatch.setattr(cli, "_require_production_artifact_root", lambda _root: None)
    monkeypatch.setattr(
        cli,
        "_run_activation_subprocess",
        lambda _command, *, label, **_kwargs: calls.append(label),
    )
    assert cli._command_activation_farm(args) == 0
    assert calls == [
        "preflight",
        "existing benchmark gate",
        "existing canary gate",
        "existing train gate",
        "existing eval gate",
    ]


def test_public_activation_commands_are_parseable() -> None:
    parser = cli.build_parser()
    status = parser.parse_args(["activation-status"])
    assert status.handler is cli._command_activation_status
    abort = parser.parse_args(["activation-abort", "--reason", "stop safely"])
    assert abort.handler is cli._command_activation_abort
    farm = parser.parse_args(
        [
            "activation-farm",
            "--checkpoint",
            "/checkpoint",
            "--checkpoint-sha256",
            "b" * 64,
            "--grain-train",
            "train-*",
            "--grain-eval",
            "eval-*",
            "--dataset-train-manifest",
            "train.json",
            "--dataset-eval-manifest",
            "eval.json",
        ]
    )
    assert farm.handler is cli._command_activation_farm
