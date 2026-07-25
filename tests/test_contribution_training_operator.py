from __future__ import annotations

import argparse
import importlib.util
import json
import os
import py_compile
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from moe_sae.contribution_artifacts import (
    HIDDEN_SIZE,
    build_recomposition_sidecar,
    canonical_sha256,
)
from moe_sae.contribution_config import MAXTEXT_FORK_COMMIT
from moe_sae.contribution_performance import (
    ExactPerformanceIdentity,
    QualificationEvidence,
    build_qualification_receipt,
)
from moe_sae.contribution_recomposition import (
    SUPPORTED_ACCUMULATION_ORDER,
)

ROOT = Path(__file__).resolve().parents[1]
OPERATOR_PATH = ROOT / "scripts" / "tpu" / "train_contribution.py"
CONFIG_PATH = ROOT / "configs" / "gemma4_contribution_v1.yaml"


def _load(path: Path, name: str):
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def operator():
    return _load(OPERATOR_PATH, "_test_contribution_training_operator")


def _identity() -> ExactPerformanceIdentity:
    return ExactPerformanceIdentity("a" * 64, "b" * 64, "c" * 64)


def _qualification_receipt() -> dict:
    identity = _identity()
    return build_qualification_receipt(
        QualificationEvidence(
            identity=identity,
            source_evidence_sha256="d" * 64,
            useful_flops_ledger_sha256="e" * 64,
            qualification_wall_seconds=10.0,
            measured_device_seconds=1.0,
            persisted_window_seconds=1.0,
            useful_flops=200,
            peak_flops_per_second=1_000,
            warmup_steps=1,
            synchronized_measurement_steps=3,
            measured_graph_compile_count=1,
            auxiliary_validation_compile_count=1,
            total_compile_count=2,
            recompilations_after_warmup=0,
            oom_events=0,
            nonfinite_events=0,
            capacity_overflow_events=0,
            dropped_routed_tokens=0,
            output_tokens=100,
            persisted_tokens=100,
            max_writer_backlog_batches=0,
            writer_backlog_limit_batches=2,
            writer_failures=0,
            writer_queue_drained=True,
            static_shapes=True,
            parameters_resident_before_timing=True,
            inputs_resident_before_timing=True,
            synchronized_device_timing=True,
            instrumentation_disabled_output_sha256="f" * 64,
            instrumentation_enabled_output_sha256="f" * 64,
            writer_disabled_output_tokens=100,
            writer_disabled_device_seconds=1.0,
        )
    )


def _sidecar() -> dict:
    scale = np.full((HIDDEN_SIZE,), 0x3F80, dtype=np.uint16)
    return build_recomposition_sidecar(
        checkpoint_sha256="1" * 64,
        maxtext_fork_commit=MAXTEXT_FORK_COMMIT,
        source_fingerprints={"environment": "2" * 64},
        routed_rmsnorm_scale=scale,
        shared_rmsnorm_scale=scale,
        outer_ffn_rmsnorm_scale=scale,
        layer_scalar=np.asarray([0x3F80], dtype=np.uint16),
        accumulation_order=SUPPORTED_ACCUMULATION_ORDER,
    )


def _bundle(operator, sidecar: dict) -> dict:
    receipt = _qualification_receipt()
    identity = _identity()
    unsigned = {
        "schema_version": operator.COORDINATOR_BUNDLE_SCHEMA_VERSION,
        "identity_relationship": (
            "receipt-binds-observed-executable-identity-not-"
            "authorization-contract-identity"
        ),
        "authorization_contract_sha256": "3" * 64,
        "host_evidence_sha256": [
            f"{index + 4:x}" * 64 for index in range(4)
        ],
        "controller_to_jax_process_index": [
            {"controller_index": 0, "process_index": 2},
            {"controller_index": 1, "process_index": 0},
            {"controller_index": 2, "process_index": 3},
            {"controller_index": 3, "process_index": 1},
        ],
        "observed_executable_identity": {
            **identity.to_dict(),
            "identity_sha256": identity.identity_sha256,
        },
        "recomposition_sidecar_sha256": sidecar["sidecar_sha256"],
        "recomposition_sidecar_canonical_sha256": canonical_sha256(sidecar),
        "source_evidence": {"schema_version": "test-source"},
        "useful_flops_ledger": {"schema_version": "test-ledger"},
        "qualification_receipt": receipt,
    }
    return {
        **unsigned,
        "coordinator_bundle_sha256": canonical_sha256(unsigned),
    }


def _write_inputs(operator, root: Path) -> dict[str, Path]:
    sidecar = _sidecar()
    sidecar_path = root / "recomposition-sidecar.json"
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    bundle_path = root / "coordinator-bundle.json"
    bundle_path.write_text(
        json.dumps(_bundle(operator, sidecar)),
        encoding="utf-8",
    )
    capture = root / "capture"
    capture.mkdir()
    manifest_unsigned = {
        "schema_version": "gemma4-expert-contribution-v2",
        "sealed": True,
    }
    (capture / "manifest.json").write_text(
        json.dumps(
            {
                **manifest_unsigned,
                "manifest_sha256": canonical_sha256(manifest_unsigned),
            }
        ),
        encoding="utf-8",
    )
    return {
        "sidecar": sidecar_path,
        "bundle": bundle_path,
        "capture": capture,
    }


def _settings(
    operator,
    tmp_path: Path,
    *,
    action: str = "qualify",
    maximum_runtime_seconds: int | None = None,
):
    return operator.Settings(
        gcp_project="neural-dragnet",
        tpu_name="gemma4-v4-32",
        tpu_zone="us-central2-b",
        project_commit="9" * 40,
        project_root="/home/researcher/src/moe-sae",
        remote_uv="/home/researcher/.local/bin/uv",
        capture_control_root="/nfs/contributions/phase2/train",
        object_prefix="gs://bucket/contributions/train",
        recomposition_sidecar="/nfs/qualification/recomposition-sidecar.json",
        qualification_bundle="/nfs/qualification/coordinator-bundle.json",
        cache_root="/dev/shm/contribution-sae",
        cache_max_bytes=40_000_000_000,
        sae_control_root="/nfs/experiment2/sae",
        checkpoint_payload_root="gs://bucket/sae-checkpoints",
        ssh_key_file=None,
        action=action,
        phase="phase2_mechanism_pilot",
        condition="A",
        seed=0,
        run_id="phase2-A-seed0",
        attempt_id=f"{action}-001",
        maximum_runtime_seconds=(
            maximum_runtime_seconds
            if maximum_runtime_seconds is not None
            else (600 if action == "qualify" else 7_200)
        ),
        local_log_root=tmp_path / f"{action}-logs",
    )


def _init_args(
    operator,
    tmp_path: Path,
    inputs: dict[str, Path],
    *,
    action: str,
    attempt_id: str,
    maximum_runtime_seconds: int | None = None,
) -> argparse.Namespace:
    run_root = tmp_path / "sae" / "runs" / "phase2-A-seed0"
    attempt_root = (
        run_root / "operator-attempts" / f"{action}-{attempt_id}"
    )
    return argparse.Namespace(
        action=action,
        run_id="phase2-A-seed0",
        attempt_id=attempt_id,
        project_commit="9" * 40,
        config=str(CONFIG_PATH),
        phase="phase2_mechanism_pilot",
        condition="A",
        seed=0,
        capture_control_root=str(inputs["capture"]),
        object_prefix="gs://bucket/contributions/train",
        recomposition_sidecar=str(inputs["sidecar"]),
        qualification_bundle=str(inputs["bundle"]),
        cache_root="/dev/shm/contribution-sae",
        cache_max_bytes=40_000_000_000,
        sae_control_root=str(tmp_path / "sae"),
        run_root=str(run_root),
        attempt_root=str(attempt_root),
        checkpoint_payload_root="gs://bucket/sae-checkpoints",
        maximum_runtime_seconds=(
            maximum_runtime_seconds
            if maximum_runtime_seconds is not None
            else (600 if action == "qualify" else 7_200)
        ),
    )


def test_operator_is_executable_and_compiles(tmp_path: Path) -> None:
    assert os.access(OPERATOR_PATH, os.X_OK)
    py_compile.compile(
        str(OPERATOR_PATH),
        cfile=str(tmp_path / "train-contribution.pyc"),
        doraise=True,
    )


def test_lineage_accepts_nonidentity_permutation_and_rejects_tampering(
    operator,
    tmp_path: Path,
) -> None:
    inputs = _write_inputs(operator, tmp_path)
    lineage = operator._qualification_lineage(
        inputs["bundle"],
        inputs["sidecar"],
    )
    assert lineage["controller_to_jax_process_index"] == [
        {"controller_index": 0, "process_index": 2},
        {"controller_index": 1, "process_index": 0},
        {"controller_index": 2, "process_index": 3},
        {"controller_index": 3, "process_index": 1},
    ]
    assert (
        lineage["qualification_receipt_sha256"]
        == _qualification_receipt()["receipt_sha256"]
    )

    bundle = json.loads(inputs["bundle"].read_text(encoding="utf-8"))
    bundle["controller_to_jax_process_index"][3]["process_index"] = 3
    unsigned = dict(bundle)
    unsigned.pop("coordinator_bundle_sha256")
    bundle["coordinator_bundle_sha256"] = canonical_sha256(unsigned)
    inputs["bundle"].write_text(json.dumps(bundle), encoding="utf-8")
    with pytest.raises(operator.OperatorError, match="not a permutation"):
        operator._qualification_lineage(
            inputs["bundle"],
            inputs["sidecar"],
        )


def test_attempt_contract_is_create_only_and_train_requires_qualification(
    operator,
    tmp_path: Path,
) -> None:
    inputs = _write_inputs(operator, tmp_path)
    qualify_args = _init_args(
        operator,
        tmp_path,
        inputs,
        action="qualify",
        attempt_id="q001",
    )
    plan = operator.initialize_attempt(qualify_args)
    assert plan["controller_to_jax_process_index"][0] == {
        "controller_index": 0,
        "process_index": 2,
    }
    assert plan["prefetch_hard_deadline_seconds"] == 600
    with pytest.raises(operator.OperatorError, match="already exists"):
        operator.initialize_attempt(qualify_args)

    train_args = _init_args(
        operator,
        tmp_path,
        inputs,
        action="train",
        attempt_id="t001",
    )
    with pytest.raises(operator.OperatorError, match="mode-selection"):
        operator.initialize_attempt(train_args)

    run_root = Path(train_args.run_root)
    (run_root / "mode-selection.json").write_text("{}", encoding="utf-8")
    receipt_root = run_root / "qualified-prefetch"
    receipt_root.mkdir()
    for item in plan["controller_to_jax_process_index"]:
        (receipt_root / (
            f"controller-c{item['controller_index']:05d}-"
            f"p{item['process_index']:05d}.json"
        )).write_text("{}", encoding="utf-8")
    trained = operator.initialize_attempt(train_args)
    assert trained["action"] == "train"
    assert trained["maximum_runtime_seconds"] == 7_200


def test_bounded_train_creates_direct_resident_contract_without_selection(
    operator,
    tmp_path: Path,
) -> None:
    inputs = _write_inputs(operator, tmp_path)
    train_args = _init_args(
        operator,
        tmp_path,
        inputs,
        action="train",
        attempt_id="direct001",
        maximum_runtime_seconds=3_600,
    )
    plan = operator.initialize_attempt(train_args)
    assert plan["action"] == "train"
    assert plan["maximum_runtime_seconds"] == 3_600
    assert plan["admission_mode"] == operator.DIRECT_BOUNDED_ADMISSION
    run_root = Path(train_args.run_root)
    assert not (run_root / "mode-selection.json").exists()
    contract = json.loads(
        (run_root / "run-contract.json").read_text(encoding="utf-8")
    )
    assert contract["admission_mode"] == operator.DIRECT_BOUNDED_ADMISSION


def test_commands_keep_controller_transport_and_jax_lane_distinct(
    operator,
    tmp_path: Path,
) -> None:
    settings = _settings(operator, tmp_path)
    prefetch = operator._prefetch_command(
        settings,
        controller_index=1,
        process_index=3,
    )
    joined = " ".join(prefetch)
    assert "--worker=1" in prefetch
    assert "moe_sae.contribution_training_worker" in joined
    assert "prefetch" in joined
    assert "--process-index 3" in joined
    assert "/process-p00003" in joined
    assert "--maximum-runtime-seconds 600" in joined
    assert "--supervision-dir" in joined

    qualify = operator._action_command(
        settings,
        controller_index=1,
        process_index=3,
    )
    joined = " ".join(qualify)
    assert "--worker=1" in qualify
    assert "qualify" in joined
    assert "/process-p00003" in joined
    assert "controller-c00001-p00003" in joined
    assert "--maximum-runtime-seconds" not in joined

    training = operator._action_command(
        _settings(operator, tmp_path, action="train"),
        controller_index=1,
        process_index=3,
    )
    joined = " ".join(training)
    assert "--maximum-runtime-seconds 7200" in joined
    assert "--checkpoint-control-root" in joined
    assert "--metric-path" in joined
    assert "--direct-bounded-resident" not in joined

    direct = operator._action_command(
        _settings(
            operator,
            tmp_path,
            action="train",
            maximum_runtime_seconds=3_600,
        ),
        controller_index=1,
        process_index=3,
    )
    joined = " ".join(direct)
    assert "--maximum-runtime-seconds 3600" in joined
    assert "--direct-bounded-resident" in joined


def test_qualification_runtime_is_exactly_600(operator, tmp_path: Path) -> None:
    settings = _settings(operator, tmp_path)
    assert settings.maximum_runtime_seconds == 600
    environment = {
        "GCP_PROJECT": settings.gcp_project,
        "TPU_NAME": settings.tpu_name,
        "TPU_ZONE": settings.tpu_zone,
        "PROJECT_COMMIT": settings.project_commit,
        "TPU_PROJECT_ROOT": settings.project_root,
        "REMOTE_UV": settings.remote_uv,
        "CONTRIBUTION_TRAIN_CAPTURE_ROOT": settings.capture_control_root,
        "CONTRIBUTION_OBJECT_PREFIX": settings.object_prefix,
        "CONTRIBUTION_RECOMPOSITION_SIDECAR": (
            settings.recomposition_sidecar
        ),
        "CONTRIBUTION_QUALIFICATION_BUNDLE": settings.qualification_bundle,
        "CONTRIBUTION_SAE_CACHE_ROOT": settings.cache_root,
        "CONTRIBUTION_SAE_CACHE_MAX_BYTES": str(settings.cache_max_bytes),
        "CONTRIBUTION_SAE_CONTROL_ROOT": settings.sae_control_root,
        "CONTRIBUTION_SAE_CHECKPOINT_ROOT": (
            settings.checkpoint_payload_root
        ),
    }
    parser = operator._public_parser()
    args = parser.parse_args(
        [
            "--action",
            "qualify",
            "--phase",
            settings.phase,
            "--condition",
            settings.condition,
            "--seed",
            "0",
            "--run-id",
            settings.run_id,
            "--attempt-id",
            settings.attempt_id,
            "--maximum-runtime-seconds",
            "599",
        ]
    )
    with pytest.raises(operator.OperatorError, match="fixed at 600"):
        operator._settings(environment, args)


def test_dry_run_is_read_only_and_declares_the_four_mapped_stages(
    operator,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = _settings(operator, tmp_path)
    assert operator._dry_run(settings) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["qualification_hard_deadline_seconds"] == 600
    assert payload["prefetch_hard_deadline_seconds"] == 600
    assert len(payload["fanout_controllers"]) == 4
    assert {
        tuple(item["stages"]) for item in payload["fanout_controllers"]
    } == {("prefetch", "qualify")}
    assert all(
        item["process_index"] == "<from-sealed-v4-bundle>"
        for item in payload["fanout_controllers"]
    )


def test_prefetch_failure_never_launches_accelerator_action(
    operator,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(operator, tmp_path)
    plan_body = {
        "schema_version": operator.ATTEMPT_PLAN_SCHEMA_VERSION,
        "action": "qualify",
        "attempt_id": settings.attempt_id,
        "attempt_root": settings.attempt_root,
        "run_root": settings.run_root,
        "run_contract_sha256": "a" * 64,
        "admission_mode": operator.QUALIFIED_ADMISSION,
        "controller_to_jax_process_index": [
            {"controller_index": 0, "process_index": 2},
            {"controller_index": 1, "process_index": 0},
            {"controller_index": 2, "process_index": 3},
            {"controller_index": 3, "process_index": 1},
        ],
        "maximum_runtime_seconds": 600,
        "qualification_hard_deadline_seconds": 600,
        "prefetch_hard_deadline_seconds": 600,
        "terminal_policy": (
            "four_mapped_prefetches_then_four_concurrent_supervised_workers"
        ),
    }
    plan = {
        **plan_body,
        "attempt_plan_sha256": canonical_sha256(plan_body),
    }
    completed = subprocess.CompletedProcess(
        args=["gcloud"],
        returncode=0,
        stdout=(json.dumps(plan) + "\n").encode("utf-8"),
    )
    monkeypatch.setattr(operator.shutil, "which", lambda _name: "/bin/gcloud")
    monkeypatch.setattr(operator, "_verify_local_checkout", lambda _settings: None)
    monkeypatch.setattr(operator, "_run_capture", lambda *_args: completed)
    stages: list[str] = []

    def fanout(_commands, logs, *, maximum_wait_seconds, stage):
        del maximum_wait_seconds
        stages.append(stage)
        assert stage == "prefetch"
        statuses = [0, 1, 0, 0]
        mapping = [(0, 2), (1, 0), (2, 3), (3, 1)]
        for log, (controller, process), status in zip(
            logs,
            mapping,
            statuses,
            strict=True,
        ):
            log.write_text(
                f"terminal prefetch controller={controller} "
                f"process={process} status={status}\n",
                encoding="utf-8",
            )
        return [1, 1, 1, 1]

    monkeypatch.setattr(operator, "_fanout", fanout)
    with pytest.raises(operator.OperatorError, match="prefetches failed"):
        operator.run(settings)
    assert stages == ["prefetch"]


def test_nonzero_gcloud_transport_accepts_exact_bound_worker_markers(
    operator,
    tmp_path: Path,
) -> None:
    settings = _settings(operator, tmp_path, action="train")
    mapping = {0: 2, 1: 0, 2: 3, 3: 1}
    prefetch_logs = []
    action_logs = []
    for controller, process in mapping.items():
        prefetch = tmp_path / f"prefetch-{controller}.log"
        prefetch.write_text(
            f"transport diagnostic\nterminal prefetch "
            f"controller={controller} process={process} status=0\n",
            encoding="utf-8",
        )
        prefetch_logs.append(prefetch)
        action = tmp_path / f"action-{controller}.log"
        action.write_text(
            f"terminal action=train controller={controller} "
            f"process={process} status=0\n",
            encoding="utf-8",
        )
        action_logs.append(action)

    assert operator._validated_fanout_completion(
        settings=settings,
        mapping=mapping,
        stage="prefetch",
        returncodes=[1, 1, 1, 1],
        log_paths=prefetch_logs,
    ) == [0, 0, 0, 0]
    assert operator._validated_fanout_completion(
        settings=settings,
        mapping=mapping,
        stage="action",
        returncodes=[1, 1, 1, 1],
        log_paths=action_logs,
    ) == [0, 0, 0, 0]


def test_zero_gcloud_transport_still_requires_every_terminal_marker(
    operator,
    tmp_path: Path,
) -> None:
    settings = _settings(operator, tmp_path, action="train")
    mapping = {0: 2, 1: 0, 2: 3, 3: 1}
    logs = []
    for controller, process in mapping.items():
        log = tmp_path / f"prefetch-{controller}.log"
        log.write_text(
            (
                f"terminal prefetch controller={controller} "
                f"process={process} status=0\n"
                if controller != 3
                else "gcloud transport exited zero without remote marker\n"
            ),
            encoding="utf-8",
        )
        logs.append(log)

    with pytest.raises(operator.OperatorError, match="exactly one"):
        operator._validated_fanout_completion(
            settings=settings,
            mapping=mapping,
            stage="prefetch",
            returncodes=[0, 0, 0, 0],
            log_paths=logs,
        )


def test_worker_terminal_marker_is_exactly_bound_and_unique(
    operator,
    tmp_path: Path,
) -> None:
    settings = _settings(operator, tmp_path, action="train")
    log = tmp_path / "worker.log"
    log.write_text(
        "terminal prefetch controller=0 process=2 status=0\n"
        "terminal prefetch controller=0 process=2 status=0\n",
        encoding="utf-8",
    )
    with pytest.raises(operator.OperatorError, match="exactly one"):
        operator._terminal_worker_status(
            log,
            settings=settings,
            stage="prefetch",
            controller_index=0,
            process_index=2,
        )

    log.write_text(
        "terminal action=train controller=0 process=3 status=0\n",
        encoding="utf-8",
    )
    with pytest.raises(operator.OperatorError, match="does not match"):
        operator._terminal_worker_status(
            log,
            settings=settings,
            stage="action",
            controller_index=0,
            process_index=2,
        )


@pytest.mark.parametrize("status", ["", "-1", "00", "256", "0 trailing"])
def test_worker_terminal_marker_rejects_noncanonical_status(
    operator,
    tmp_path: Path,
    status: str,
) -> None:
    settings = _settings(operator, tmp_path, action="train")
    log = tmp_path / "worker.log"
    log.write_text(
        f"terminal action=train controller=0 process=2 status={status}\n",
        encoding="utf-8",
    )
    with pytest.raises(operator.OperatorError, match="worker status|range"):
        operator._terminal_worker_status(
            log,
            settings=settings,
            stage="action",
            controller_index=0,
            process_index=2,
        )


@pytest.mark.parametrize(
    ("action_statuses", "failure"),
    [
        ([0, 0, 0, 0], None),
        ([0, 7, 0, 0], "train workers failed"),
    ],
)
def test_run_uses_exact_remote_status_after_transport_teardown(
    operator,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    action_statuses: list[int],
    failure: str | None,
) -> None:
    settings = _settings(
        operator,
        tmp_path,
        action="train",
        maximum_runtime_seconds=600,
    )
    mapping = {0: 2, 1: 0, 2: 3, 3: 1}
    plan_body = {
        "schema_version": operator.ATTEMPT_PLAN_SCHEMA_VERSION,
        "action": settings.action,
        "attempt_id": settings.attempt_id,
        "attempt_root": settings.attempt_root,
        "run_root": settings.run_root,
        "run_contract_sha256": "a" * 64,
        "admission_mode": operator.DIRECT_BOUNDED_ADMISSION,
        "controller_to_jax_process_index": [
            {"controller_index": controller, "process_index": process}
            for controller, process in mapping.items()
        ],
        "maximum_runtime_seconds": 600,
        "qualification_hard_deadline_seconds": 600,
        "prefetch_hard_deadline_seconds": 600,
        "terminal_policy": (
            "four_mapped_prefetches_then_four_concurrent_supervised_workers"
        ),
    }
    plan = {
        **plan_body,
        "attempt_plan_sha256": canonical_sha256(plan_body),
    }
    completed = subprocess.CompletedProcess(
        args=["gcloud"],
        returncode=0,
        stdout=(json.dumps(plan) + "\n").encode("utf-8"),
    )
    monkeypatch.setattr(operator.shutil, "which", lambda _name: "/bin/gcloud")
    monkeypatch.setattr(operator, "_verify_local_checkout", lambda _settings: None)
    monkeypatch.setattr(operator, "_run_capture", lambda *_args: completed)
    stages = []

    def fanout(_commands, logs, *, maximum_wait_seconds, stage):
        del maximum_wait_seconds
        stages.append(stage)
        marker_stage = "prefetch" if stage == "prefetch" else "action=train"
        statuses = [0, 0, 0, 0] if stage == "prefetch" else action_statuses
        for log, (controller, process), status in zip(
            logs,
            mapping.items(),
            statuses,
            strict=True,
        ):
            log.write_text(
                f"terminal {marker_stage} controller={controller} "
                f"process={process} status={status}\n",
                encoding="utf-8",
            )
        return [1, 1, 1, 1]

    monkeypatch.setattr(operator, "_fanout", fanout)
    if failure is None:
        assert operator.run(settings) == 0
    else:
        with pytest.raises(operator.OperatorError, match=failure):
            operator.run(settings)
    assert stages == ["prefetch", "train"]
