from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
OPERATOR_PATH = ROOT / "scripts" / "tpu" / "evaluate_contribution.py"


def _load():
    name = "_test_contribution_evaluation_operator"
    specification = importlib.util.spec_from_file_location(name, OPERATOR_PATH)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


@pytest.fixture
def operator():
    return _load()


def _settings(operator, tmp_path: Path):
    return operator.Settings(
        project="neural-dragnet",
        tpu="gemma4-v4-32",
        zone="us-central2-b",
        commit="a" * 40,
        root="/home/researcher/moe-sae",
        uv="/home/researcher/.local/bin/uv",
        bundle="/nfs/coordinator-bundle.json",
        sidecar="/nfs/recomposition-sidecar.json",
        train_capture="/nfs/contributions/train",
        eval_capture="/nfs/contributions/eval",
        train_objects="gs://bucket/contributions/train",
        eval_objects="gs://bucket/contributions/eval",
        cache_root="/dev/shm/contribution-sae",
        cache_bytes=64_000_000_000,
        sae_control="/nfs/sae/phase2-test",
        checkpoint_root="gs://bucket/checkpoints/phase2-test",
        result_root=str(tmp_path / "result"),
        log_root=tmp_path / "logs",
        ssh_key=None,
    )


def _write_result(operator, root: Path) -> None:
    body = {
        "schema_version": operator.RESULT_SCHEMA_VERSION,
        "status": "complete",
    }
    root.mkdir(parents=True)
    (root / "scientific-result.json").write_text(
        json.dumps(
            {
                **body,
                "result_sha256": operator._canonical_sha256(body),
            }
        ),
        encoding="ascii",
    )


def test_internal_ip_transport_is_explicit(
    operator,
    tmp_path: Path,
) -> None:
    settings = _settings(operator, tmp_path)
    external = operator._command(settings, 0, 2)
    internal = operator._command(
        replace(settings, internal_ip=True),
        0,
        2,
    )
    assert "--internal-ip" not in external
    assert "--internal-ip" in internal


def test_main_accepts_remote_success_after_transport_teardown(
    operator,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(operator, tmp_path)
    mapping = {0: 2, 1: 0, 2: 3, 3: 1}
    _write_result(operator, Path(settings.result_root))
    monkeypatch.setattr(
        operator.Settings,
        "from_environment",
        classmethod(lambda cls: settings),
    )
    monkeypatch.setattr(operator, "_mapping", lambda _path: mapping)

    def run(_command, log):
        controller = int(log.name.split("-c", 1)[1].split("-", 1)[0])
        process = mapping[controller]
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(
            f"terminal action=evaluate controller={controller} "
            f"process={process} status=0\n",
            encoding="ascii",
        )
        return 1

    monkeypatch.setattr(operator, "_run", run)
    assert operator.main() == 0


def test_zero_transport_without_marker_fails_closed(
    operator,
    tmp_path: Path,
) -> None:
    log = tmp_path / "evaluation.log"
    log.write_text("gcloud exited zero\n", encoding="ascii")
    with pytest.raises(operator.OperatorError, match="exactly one"):
        operator._terminal_evaluation_status(
            log,
            controller=0,
            process=2,
        )


def test_public_horizon_selection_reaches_worker(
    operator,
    tmp_path: Path,
) -> None:
    settings = operator._apply_public_arguments(
        _settings(operator, tmp_path),
        operator._parser().parse_args(
            [
                "--config",
                "/nfs/contracts/contribution-horizon-final.yaml",
                "--phase",
                "learning_horizon_10m",
                "--checkpoint-tokens",
                "5000000",
                "--result-root",
                "/nfs/evaluation/contribution-5m",
                "--run-root-a",
                "/nfs/runs/horizon-a",
                "--run-root-r",
                "/nfs/runs/horizon-r",
                "--run-root-b",
                "/nfs/runs/horizon-b",
                "--run-root-f",
                "/nfs/runs/horizon-f",
            ]
        ),
    )
    worker = operator._worker(settings, 2)
    assert worker[worker.index("--config") + 1] == (
        "/nfs/contracts/contribution-horizon-final.yaml"
    )
    assert worker[worker.index("--phase") + 1] == "learning_horizon_10m"
    assert worker[worker.index("--checkpoint-tokens") + 1] == "5000000"
    assert worker[worker.index("--run-root-a") + 1] == "/nfs/runs/horizon-a"
    assert settings.result_root == "/nfs/evaluation/contribution-5m"


def test_nonzero_remote_evaluation_status_is_preserved(
    operator,
    tmp_path: Path,
) -> None:
    log = tmp_path / "evaluation.log"
    log.write_text(
        "terminal action=evaluate controller=0 process=2 status=7\n",
        encoding="ascii",
    )
    assert (
        operator._terminal_evaluation_status(
            log,
            controller=0,
            process=2,
        )
        == 7
    )


def test_scientific_result_must_be_self_hashed(
    operator,
    tmp_path: Path,
) -> None:
    path = tmp_path / "scientific-result.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": operator.RESULT_SCHEMA_VERSION,
                "result_sha256": "0" * 64,
            }
        ),
        encoding="ascii",
    )
    with pytest.raises(operator.OperatorError, match="self-hash"):
        operator._validate_scientific_result(path)


def test_horizon_scientific_result_identity_is_accepted(
    operator,
    tmp_path: Path,
) -> None:
    body = {
        "schema_version": operator.LEARNING_HORIZON_RESULT_SCHEMA_VERSION,
        "protocol_id": operator.LEARNING_HORIZON_PROTOCOL_ID,
        "status": "curve_point_measured",
    }
    path = tmp_path / "scientific-result.json"
    path.write_text(
        json.dumps(
            {
                **body,
                "result_sha256": operator._canonical_sha256(body),
            }
        ),
        encoding="ascii",
    )
    assert operator._validate_scientific_result(path)["status"] == (
        "curve_point_measured"
    )

    mismatched = {**body, "protocol_id": "wrong-protocol"}
    path.write_text(
        json.dumps(
            {
                **mismatched,
                "result_sha256": operator._canonical_sha256(mismatched),
            }
        ),
        encoding="ascii",
    )
    with pytest.raises(operator.OperatorError, match="identity"):
        operator._validate_scientific_result(path)
