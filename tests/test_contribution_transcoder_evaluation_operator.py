from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest

from moe_sae.contribution_artifacts import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "tpu" / "evaluate_transcoder.py"


@pytest.fixture()
def operator():
    spec = importlib.util.spec_from_file_location(
        "evaluate_transcoder_operator",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _environment(tmp_path: Path) -> dict[str, str]:
    bundle = tmp_path / "qualification.json"
    bundle.write_text(
        json.dumps(
            {
                "controller_to_jax_process_index": [
                    {"controller_index": index, "process_index": 3 - index}
                    for index in range(4)
                ]
            }
        )
    )
    artifact = tmp_path / "artifacts"
    return {
        "GCP_PROJECT": "neural-dragnet",
        "TPU_NAME": "gemma4-exp3",
        "TPU_ZONE": "us-central2-b",
        "PROJECT_COMMIT": "a" * 40,
        "TRAINING_PROJECT_COMMIT": "c" * 40,
        "TPU_PROJECT_ROOT": "/home/user/moe-sae",
        "REMOTE_UV": "/home/user/.local/bin/uv",
        "CONTRIBUTION_QUALIFICATION_BUNDLE": str(bundle),
        "CONTRIBUTION_RECOMPOSITION_SIDECAR": "/nfs/sidecar.json",
        "CONTRIBUTION_TRAIN_CAPTURE_ROOT": "/nfs/target/train",
        "CONTRIBUTION_EVAL_CAPTURE_ROOT": "/nfs/target/eval",
        "CONTRIBUTION_OBJECT_PREFIX": "gs://bucket/target/train",
        "CONTRIBUTION_EVAL_OBJECT_PREFIX": "gs://bucket/target/eval",
        "CONTRIBUTION_SOURCE_OVERLAY_ROOT": "/nfs/source/train",
        "CONTRIBUTION_EVAL_SOURCE_OVERLAY_ROOT": "/nfs/source/eval",
        "CONTRIBUTION_SOURCE_OVERLAY_PREFIX": "gs://bucket/source/train",
        "CONTRIBUTION_EVAL_SOURCE_OVERLAY_PREFIX": "gs://bucket/source/eval",
        "CONTRIBUTION_SAE_CACHE_ROOT": "/mnt/cache",
        "CONTRIBUTION_SAE_CACHE_MAX_BYTES": "200000000000",
        "CONTRIBUTION_SAE_CONTROL_ROOT": "/nfs/sae",
        "CONTRIBUTION_SAE_CHECKPOINT_ROOT": "gs://bucket/checkpoints",
        "ARTIFACT_ROOT": str(artifact),
        "BULK_GCS_ROOT": "gs://bucket/root",
        "TRANSCODER_HARVEST_RECEIPT_SHA256": "b" * 64,
        "TRANSCODER_HARVEST_CHIP_SECONDS": "123.5",
    }


def test_settings_and_worker_bind_all_four_runs_and_both_manifests(
    operator,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    environment = _environment(tmp_path)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    settings = operator.Settings.from_environment()
    worker = operator._worker(settings, 2)

    assert settings.run_roots == {
        "A": "/nfs/sae/runs/experiment3-a-seed0",
        "R": "/nfs/sae/runs/experiment3-r-seed0",
        "B": "/nfs/sae/runs/experiment3-b-seed0",
        "F": "/nfs/sae/runs/experiment3-f-seed0",
    }
    assert "moe_sae.contribution_transcoder_evaluation_worker" in worker
    assert settings.commit == "a" * 40
    assert settings.training_commit == "c" * 40
    for flag, value in (
        ("--train-target-root", "/nfs/target/train"),
        ("--eval-target-root", "/nfs/target/eval"),
        ("--train-source-root", "/nfs/source/train"),
        ("--eval-source-root", "/nfs/source/eval"),
        ("--run-root-a", settings.run_roots["A"]),
        ("--run-root-f", settings.run_roots["F"]),
        ("--shared-harvest-receipt-sha256", "b" * 64),
        ("--source-revision", "a" * 40),
        ("--training-source-revision", "c" * 40),
    ):
        index = worker.index(flag)
        assert worker[index + 1] == value
    assert worker[worker.index("--process-index") + 1] == "2"


def test_internal_ip_transport_is_explicit(
    operator,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    environment = _environment(tmp_path)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    settings = operator.Settings.from_environment()
    external = operator._command(settings, 0, 3)
    internal_settings = replace(settings, internal_ip=True)
    internal = operator._command(internal_settings, 0, 3)
    assert "--internal-ip" not in external
    assert "--internal-ip" in internal


def test_explicit_tpu_ssh_user_is_rendered(
    operator,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    environment = _environment(tmp_path)
    environment["TPU_SSH_USER"] = "abistras"
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    settings = operator.Settings.from_environment()
    command = operator._command(settings, 0, 3)
    assert command[command.index("ssh") + 1] == "abistras@gemma4-exp3"


def test_training_revision_is_required(
    operator,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    environment = _environment(tmp_path)
    environment.pop("TRAINING_PROJECT_COMMIT")
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("TRAINING_PROJECT_COMMIT", raising=False)

    with pytest.raises(operator.OperatorError, match="TRAINING_PROJECT_COMMIT"):
        operator.Settings.from_environment()


def test_public_horizon_selection_reaches_worker(
    operator,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    for name, value in _environment(tmp_path).items():
        monkeypatch.setenv(name, value)
    settings = operator._apply_public_arguments(
        operator.Settings.from_environment(),
        operator._parser().parse_args(
            [
                "--config",
                "/nfs/contracts/transcoder-horizon-final.yaml",
                "--phase",
                "learning_horizon_10m",
                "--checkpoint-tokens",
                "2000000",
                "--result-root",
                "/nfs/evaluation/transcoder-2m",
                "--run-root-a",
                "/nfs/runs/transcoder-horizon-a",
            ]
        ),
    )
    worker = operator._worker(settings, 1)
    assert worker[worker.index("--config") + 1] == (
        "/nfs/contracts/transcoder-horizon-final.yaml"
    )
    assert worker[worker.index("--phase") + 1] == "learning_horizon_10m"
    assert worker[worker.index("--checkpoint-tokens") + 1] == "2000000"
    assert worker[worker.index("--run-root-a") + 1] == (
        "/nfs/runs/transcoder-horizon-a"
    )
    assert settings.result_root == "/nfs/evaluation/transcoder-2m"


def test_v5litepod_worker_receives_exact_observed_device_kind(
    operator,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    for name, value in _environment(tmp_path).items():
        monkeypatch.setenv(name, value)
    settings = replace(
        operator.Settings.from_environment(),
        accelerator_type="v5litepod-16",
        peak_flops_per_second=3_152_000_000_000_000,
        v5litepod_device_kind="TPU v5 observed",
    )
    worker = operator._worker(settings, 1)
    assert worker[:2] == [
        "env",
        "MOE_SAE_V5LITEPOD_DEVICE_KIND=TPU v5 observed",
    ]
    assert worker[worker.index("--accelerator-type") + 1] == "v5litepod-16"
    assert worker[worker.index("--peak-flops-per-second") + 1] == (
        "3152000000000000"
    )


def test_mapping_and_terminal_marker_are_exact(
    operator,
    tmp_path: Path,
) -> None:
    environment = _environment(tmp_path)
    mapping = operator._mapping_payload(
        Path(environment["CONTRIBUTION_QUALIFICATION_BUNDLE"]).read_bytes()
    )
    assert mapping == {0: 3, 1: 2, 2: 1, 3: 0}
    log = tmp_path / "worker.log"
    log.write_text(
        "noise\n"
        "terminal action=evaluate-transcoder controller=1 process=2 status=0\n"
    )
    assert operator._terminal_status(log, controller=1, process=2) == 0
    with pytest.raises(operator.OperatorError, match="terminal"):
        operator._terminal_status(log, controller=1, process=3)


def test_mapping_is_fetched_by_mac_from_remote_worker_zero(
    operator,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    environment = _environment(tmp_path)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    settings = operator.Settings.from_environment()
    bundle = Path(
        environment["CONTRIBUTION_QUALIFICATION_BUNDLE"]
    ).read_bytes()
    calls = []

    class Completed:
        returncode = 0
        stdout = bundle
        stderr = b""

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return Completed()

    monkeypatch.setattr(operator.subprocess, "run", run)
    assert operator._mapping(settings) == {0: 3, 1: 2, 2: 1, 3: 0}
    command = calls[0][0]
    assert "--worker=0" in command
    assert command[-1].endswith(
        environment["CONTRIBUTION_QUALIFICATION_BUNDLE"]
    )


def test_operator_result_validation_rejects_tampering(
    operator,
    tmp_path: Path,
) -> None:
    body = {
        "schema_version": operator.RESULT_SCHEMA_VERSION,
        "protocol_id": operator.PROTOCOL_ID,
        "status": "transfer_passed",
        "source_revision": "a" * 40,
        "training_source_revision": "c" * 40,
        "parents": {
            "run_contract_sha256": {
                condition: canonical_sha256({"condition": condition})
                for condition in ("A", "R", "B", "F")
            }
        },
    }
    result = {**body, "result_sha256": canonical_sha256(body)}
    path = tmp_path / "scientific-result.json"
    path.write_text(json.dumps(result))
    assert (
        operator._validate_result(
            path,
            expected_source_revision="a" * 40,
            expected_training_source_revision="c" * 40,
        )["status"]
        == "transfer_passed"
    )
    result["status"] = "changed"
    path.write_text(json.dumps(result))
    with pytest.raises(operator.OperatorError, match="self-hash"):
        operator._validate_result(
            path,
            expected_source_revision="a" * 40,
            expected_training_source_revision="c" * 40,
        )

    wrong_protocol = {
        **body,
        "protocol_id": "gemma4-contribution-transcoder-v1-20260727",
    }
    path.write_text(
        json.dumps(
            {
                **wrong_protocol,
                "result_sha256": canonical_sha256(wrong_protocol),
            }
        )
    )
    with pytest.raises(operator.OperatorError, match="identity"):
        operator._validate_result(
            path,
            expected_source_revision="a" * 40,
            expected_training_source_revision="c" * 40,
        )

    valid = {**body, "result_sha256": canonical_sha256(body)}
    path.write_text(json.dumps(valid))
    with pytest.raises(operator.OperatorError, match="identity"):
        operator._validate_result(
            path,
            expected_source_revision="d" * 40,
            expected_training_source_revision="c" * 40,
        )
    with pytest.raises(operator.OperatorError, match="identity"):
        operator._validate_result(
            path,
            expected_source_revision="a" * 40,
            expected_training_source_revision="d" * 40,
        )


def test_operator_accepts_horizon_result_identity(
    operator,
    tmp_path: Path,
) -> None:
    body = {
        "schema_version": operator.LEARNING_HORIZON_RESULT_SCHEMA_VERSION,
        "protocol_id": operator.LEARNING_HORIZON_PROTOCOL_ID,
        "status": "curve_point_measured",
        "source_revision": "a" * 40,
        "training_source_revision": "c" * 40,
        "parents": {
            "run_contract_sha256": {
                condition: canonical_sha256({"condition": condition})
                for condition in ("A", "R", "B", "F")
            }
        },
    }
    path = tmp_path / "scientific-result.json"
    path.write_text(
        json.dumps(
            {
                **body,
                "result_sha256": canonical_sha256(body),
            }
        )
    )
    assert (
        operator._validate_result(
            path,
            expected_source_revision="a" * 40,
            expected_training_source_revision="c" * 40,
        )["status"]
        == "curve_point_measured"
    )
