from __future__ import annotations

import argparse
import importlib.util
import json
import os
import py_compile
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
OPERATOR_PATH = ROOT / "scripts" / "tpu" / "harvest_contribution.py"


def _load(path: Path, name: str):
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def operator():
    return _load(OPERATOR_PATH, "_test_contribution_harvest_operator")


def _settings(operator, tmp_path: Path):
    return operator.Settings(
        gcp_project="neural-dragnet",
        tpu_name="gemma4-v4-32",
        tpu_zone="us-central2-b",
        project_commit="a" * 40,
        project_root="/home/researcher/src/moe-sae",
        remote_uv="/home/researcher/.local/bin/uv",
        local_buffer_root="/mnt/local",
        model_checkpoint="gs://bucket/model/0/items",
        checkpoint_sha256="b" * 64,
        grain_files="gs://bucket/data/train/*.array_record",
        staged_manifest="gs://bucket/data/train/manifest.json",
        source_v1_root="/control/captures/train",
        source_cache_max_bytes=1 << 30,
        control_root="/control/experiment2/phase1_smoke/train",
        object_prefix="gs://bucket/experiment2/contributions/train",
        recomposition_sidecar="/control/qualification/recomposition-sidecar.json",
        qualification_receipt="/control/qualification/qualification-receipt.json",
        qualification_bundle="/control/qualification/coordinator-bundle.json",
        qualification_decision="/control/qualification/operator-decision.json",
        maxtext_root=None,
        ssh_key_file=None,
        phase="phase1_smoke",
        split="train",
        attempt_id="phase1-train-001",
        mesh="ep8-fsdp2-tp1",
        per_device_sequence_batch=8,
        maximum_runtime_seconds=3_600,
        prior_control_root=None,
        prior_phase_evidence=None,
        local_log_root=tmp_path / "operator-logs",
    )


def _self_hash(operator, value: dict, field: str) -> dict:
    return {**value, field: operator._canonical_sha256(value)}


def _attempt_args(operator, tmp_path: Path, *, batch: int = 8):
    config = tmp_path / "gemma4_contribution_v1.yaml"
    config.write_text("protocol: test\n", encoding="utf-8")
    source = tmp_path / "source-v1"
    source.mkdir()
    (source / "manifest.json").write_text("{}", encoding="utf-8")
    qualification = tmp_path / "qualification"
    qualification.mkdir()

    receipt = _self_hash(
        operator,
        {
            "schema_version": operator.QUALIFICATION_RECEIPT_SCHEMA_VERSION,
            "status": "passed",
            "identity": {"graph_sha256": "1" * 64},
        },
        "receipt_sha256",
    )
    sidecar = {
        "schema_version": "test-sidecar-v1",
        "sidecar_sha256": "2" * 64,
    }
    bundle = _self_hash(
        operator,
        {
            "schema_version": operator.COORDINATOR_BUNDLE_SCHEMA_VERSION,
            "qualification_receipt": receipt,
            "controller_to_jax_process_index": [
                {"controller_index": 0, "process_index": 0},
                {"controller_index": 1, "process_index": 2},
                {"controller_index": 2, "process_index": 3},
                {"controller_index": 3, "process_index": 1},
            ],
            "recomposition_sidecar_sha256": sidecar["sidecar_sha256"],
            "recomposition_sidecar_canonical_sha256": (
                operator._canonical_sha256(sidecar)
            ),
        },
        "coordinator_bundle_sha256",
    )
    decision = _self_hash(
        operator,
        {
            "schema_version": operator.OPERATOR_DECISION_SCHEMA_VERSION,
            "outcome": "passed",
            "next_action": "stop_no_workload_launch",
            "mesh": "ep8-fsdp2-tp1",
            "per_device_sequence_batch": batch,
            "global_sequence_batch": batch * 16,
            "coordinator_bundle_sha256": bundle["coordinator_bundle_sha256"],
            "recomposition_sidecar_sha256": sidecar["sidecar_sha256"],
        },
        "operator_decision_sha256",
    )
    paths = {}
    for name, value in (
        ("qualification_receipt", receipt),
        ("recomposition_sidecar", sidecar),
        ("qualification_bundle", bundle),
        ("qualification_decision", decision),
    ):
        path = qualification / f"{name}.json"
        path.write_text(
            json.dumps(value, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        paths[name] = str(path)

    control = tmp_path / "phase1-control"
    return argparse.Namespace(
        attempt_id="attempt-001",
        attempt_root=str(control / "attempts" / "attempt-001"),
        project_commit="a" * 40,
        config=str(config),
        phase="phase1_smoke",
        split="train",
        checkpoint="gs://bucket/model/0/items",
        checkpoint_sha256="b" * 64,
        grain_files="gs://bucket/data/train/*.array_record",
        staged_manifest="gs://bucket/data/train/manifest.json",
        source_v1_root=str(source),
        source_cache_max_bytes=1 << 30,
        control_root=str(control),
        object_prefix="gs://bucket/experiment2/contributions/train",
        recomposition_sidecar=paths["recomposition_sidecar"],
        qualification_receipt=paths["qualification_receipt"],
        qualification_bundle=paths["qualification_bundle"],
        qualification_decision=paths["qualification_decision"],
        mesh="ep8-fsdp2-tp1",
        per_device_sequence_batch=8,
        maximum_runtime_seconds=3_600,
        prior_control_root=None,
        prior_phase_evidence=None,
        maxtext_root=None,
        phase_evidence=str(control / "phase-evidence.json"),
    )


def test_operator_is_executable_and_compiles(tmp_path: Path) -> None:
    assert os.access(OPERATOR_PATH, os.X_OK)
    py_compile.compile(
        str(OPERATOR_PATH),
        cfile=str(tmp_path / "harvest-operator.pyc"),
        doraise=True,
    )


def test_settings_uses_attempt_scoped_default_log_path(
    operator,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    environment = {
        "GCP_PROJECT": "neural-dragnet",
        "TPU_NAME": "gemma4-v4-32",
        "TPU_ZONE": "us-central2-b",
        "PROJECT_COMMIT": "a" * 40,
        "CHECKPOINT_SHA256": "b" * 64,
        "TPU_PROJECT_ROOT": "/home/researcher/src/moe-sae",
        "REMOTE_UV": "/home/researcher/.local/bin/uv",
        "LOCAL_BUFFER_ROOT": "/mnt/local",
        "MODEL_CHECKPOINT": "gs://bucket/model/0/items",
        "CONTRIBUTION_GRAIN_FILES": "gs://bucket/data/train/*.array_record",
        "CONTRIBUTION_STAGED_MANIFEST": ("gs://bucket/data/train/manifest.json"),
        "CONTRIBUTION_SOURCE_V1_ROOT": "/control/captures/train",
        "CONTRIBUTION_SOURCE_CACHE_MAX_BYTES": str(1 << 30),
        "CONTRIBUTION_CONTROL_ROOT": ("/control/experiment2/phase1_smoke/train"),
        "CONTRIBUTION_OBJECT_PREFIX": ("gs://bucket/experiment2/contributions/train"),
        "CONTRIBUTION_RECOMPOSITION_SIDECAR": (
            "/control/qualification/recomposition-sidecar.json"
        ),
        "CONTRIBUTION_QUALIFICATION_RECEIPT": (
            "/control/qualification/qualification-receipt.json"
        ),
        "CONTRIBUTION_QUALIFICATION_BUNDLE": (
            "/control/qualification/coordinator-bundle.json"
        ),
        "CONTRIBUTION_QUALIFICATION_DECISION": (
            "/control/qualification/operator-decision.json"
        ),
    }
    args = argparse.Namespace(
        phase="phase1_smoke",
        split="train",
        attempt_id="attempt-001",
        mesh="ep8-fsdp2-tp1",
        per_device_sequence_batch=8,
        maximum_runtime_seconds=3_600,
        prior_control_root=None,
        prior_phase_evidence=None,
    )
    settings = operator._settings(environment, args)
    assert settings.local_log_root == (
        tmp_path / "artifacts/gemma4_v1/operator-results/"
        "experiment2-harvest/attempt-001"
    )


def test_four_host_commands_share_identity_but_use_private_paths(
    operator,
    tmp_path: Path,
) -> None:
    settings = _settings(operator, tmp_path)
    built = [
        operator._host_command(settings, controller_index)
        for controller_index in range(4)
    ]
    assert [
        next(value for value in command if value.startswith("--worker="))
        for command, _ in built
    ] == ["--worker=0", "--worker=1", "--worker=2", "--worker=3"]
    joined = [" ".join(command) for command, _ in built]
    for command in joined:
        assert "moe_sae.contribution_harvest_worker" in command
        assert "launch-host" in command
        assert "--mesh ep8-fsdp2-tp1" in command
        assert "--per-device-sequence-batch 8" in command
        assert "--maximum-runtime-seconds 3600" in command
        assert settings.qualification_receipt in command
        assert settings.attempt_root in command
    for index, command in enumerate(joined):
        assert f"--controller-index {index}" in command
        assert "--process-index" not in command
        assert f"source-cache/train/controller-c{index:05d}" in command
        assert f"controller-c{index:05d}/buffer" in command
        for other in set(range(4)) - {index}:
            assert f"source-cache/train/controller-c{other:05d}" not in command
    publish = " ".join(
        operator._publish_command(settings, [evidence for _, evidence in built])
    )
    assert "--worker=0" in publish
    assert " publish " in publish
    assert publish.count("--host-evidence") == 4


def test_attempt_plan_binds_qualification_and_is_create_only(
    operator,
    tmp_path: Path,
) -> None:
    args = _attempt_args(operator, tmp_path)
    assert operator.initialize_attempt(args) == 0
    plan_path = Path(args.attempt_root) / "operator-plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    claimed = plan.pop("operator_plan_sha256")
    assert claimed == operator._canonical_sha256(plan)
    assert plan["mesh"] == "ep8-fsdp2-tp1"
    assert plan["per_device_sequence_batch"] == 8
    assert plan["global_sequence_batch"] == 128
    assert plan["process_count"] == 4
    assert plan["progress_authority"].endswith("never_bucket_listing")
    with pytest.raises(operator.OperatorError, match="already exists"):
        operator.initialize_attempt(args)


def test_attempt_rejects_explicit_batch_different_from_decision(
    operator,
    tmp_path: Path,
) -> None:
    args = _attempt_args(operator, tmp_path, batch=4)
    with pytest.raises(operator.OperatorError, match="mesh/batch"):
        operator.initialize_attempt(args)
    assert not Path(args.attempt_root).exists()


@pytest.mark.parametrize("invalid_index", (False, "0"))
def test_attempt_rejects_non_integer_qualification_topology_indices(
    operator,
    tmp_path: Path,
    invalid_index: object,
) -> None:
    args = _attempt_args(operator, tmp_path)
    bundle_path = Path(args.qualification_bundle)
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    bundle.pop("coordinator_bundle_sha256")
    bundle["controller_to_jax_process_index"][0]["controller_index"] = invalid_index
    bundle = _self_hash(
        operator,
        bundle,
        "coordinator_bundle_sha256",
    )
    bundle_path.write_text(
        json.dumps(bundle, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )

    decision_path = Path(args.qualification_decision)
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    decision.pop("operator_decision_sha256")
    decision["coordinator_bundle_sha256"] = bundle["coordinator_bundle_sha256"]
    decision = _self_hash(
        operator,
        decision,
        "operator_decision_sha256",
    )
    decision_path.write_text(
        json.dumps(decision, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )

    with pytest.raises(operator.OperatorError, match="coordinator bundle"):
        operator.initialize_attempt(args)
    assert not Path(args.attempt_root).exists()


def test_noncomplete_fanout_cannot_reach_publish(
    operator,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(operator, tmp_path)
    logged: list[list[str]] = []

    monkeypatch.setattr(operator.shutil, "which", lambda _name: "/usr/bin/gcloud")
    monkeypatch.setattr(operator, "_verify_local_checkout", lambda _settings: None)

    def fake_logged(command, _path):
        logged.append(list(command))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(operator, "_run_logged", fake_logged)
    monkeypatch.setattr(
        operator,
        "_fanout",
        lambda *_args, **_kwargs: [0, 17, 0, 0],
    )
    with pytest.raises(operator.OperatorError, match="publication was not attempted"):
        operator.run(settings)
    assert len(logged) == 1
    assert "_init-attempt" in " ".join(logged[0])


def test_operator_has_no_bucket_listing_progress_path() -> None:
    source = OPERATOR_PATH.read_text(encoding="utf-8")
    forbidden = ("gcloud storage ls", "gsutil ls", "list_blobs", "objects list")
    assert not any(value in source for value in forbidden)
    assert "publish_only_after_four_complete_immutable_supervised_records" in source
