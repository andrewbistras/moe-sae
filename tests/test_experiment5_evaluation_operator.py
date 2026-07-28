from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from moe_sae.experiment5_evaluation_worker import _parser as _worker_parser

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "tpu" / "evaluate_experiment5.py"


@pytest.fixture()
def operator():
    spec = importlib.util.spec_from_file_location(
        "experiment5_evaluation_operator",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _environment(tmp_path: Path, *, panel: str) -> dict[str, str]:
    result = {
        "GCP_PROJECT": "test-project",
        "TPU_NAME": "experiment5-slice",
        "TPU_ZONE": "us-test1-a",
        "PROJECT_COMMIT": "a" * 40,
        "TPU_PROJECT_ROOT": "/remote/moe-sae",
        "REMOTE_UV": "/remote/bin/uv",
        "TPU_ACCELERATOR_TYPE": "v6e-16",
        "TPU_USE_INTERNAL_IP": "1",
        "TPU_SSH_USER": "researcher",
        "CONTRIBUTION_FRESH_CONTROLLER_MAPPING": str(tmp_path / "mapping.json"),
        "CONTRIBUTION_REGIONAL_MIRROR_SEAL": "/remote/mirror.json",
        "CONTRIBUTION_RECOMPOSITION_SIDECAR": "/remote/sidecar.json",
        "CONTRIBUTION_TRAIN_CAPTURE_ROOT": "/remote/target/train",
        "CONTRIBUTION_EVAL_CAPTURE_ROOT": "/remote/target/eval",
        "CONTRIBUTION_OBJECT_PREFIX": "gs://bucket/target/train",
        "CONTRIBUTION_EVAL_OBJECT_PREFIX": "gs://bucket/target/eval",
        "CONTRIBUTION_SAE_CACHE_ROOT": "/local/cache",
        "CONTRIBUTION_SAE_CACHE_MAX_BYTES": str(160 * 1024**3),
        "CONTRIBUTION_SAE_CONTROL_ROOT": "/remote/control",
        "CONTRIBUTION_SAE_CHECKPOINT_ROOT": "gs://bucket/checkpoints",
        "EXPERIMENT5_RESULT_PATH_CONTROLLER": str(
            tmp_path / "published" / "scientific-result.json"
        ),
        "EXPERIMENT5_PAIRED_SCORES_PATH_CONTROLLER": str(
            tmp_path / "published" / "paired-scores.json"
        ),
    }
    if panel == "sae":
        result["CONTRIBUTION_EVALUATION_LOG_ROOT"] = str(tmp_path / "logs")
    else:
        result.update(
            {
                "TRANSCODER_EVALUATION_LOG_ROOT": str(tmp_path / "logs"),
                "CONTRIBUTION_SOURCE_OVERLAY_ROOT": "/remote/source/train",
                "CONTRIBUTION_EVAL_SOURCE_OVERLAY_ROOT": "/remote/source/eval",
                "CONTRIBUTION_SOURCE_OVERLAY_PREFIX": "gs://bucket/source/train",
                "CONTRIBUTION_EVAL_SOURCE_OVERLAY_PREFIX": "gs://bucket/source/eval",
            }
        )
    return result


def _arguments(panel: str, *, dry_run: bool = False) -> list[str]:
    result = [
        "--panel",
        panel,
        "--config",
        f"/remote/configs/gemma4_hybrid_{panel}_v1.yaml",
        "--phase",
        "routed_shared_hybrid_10m",
        "--seed",
        "1",
        "--checkpoint-tokens",
        "5000000",
        "--result-root",
        "/remote/results/sae/seed1/5000000",
    ]
    for condition in ("h", "r", "b", "f"):
        result.extend(
            (
                f"--run-root-{condition}",
                f"/remote/control/runs/experiment5-{panel}-{condition}-seed1",
            )
        )
    if dry_run:
        result.append("--dry-run")
    return result


def test_worker_uses_one_process_private_cache_lane_and_no_sae_overlay(
    operator,
    tmp_path: Path,
) -> None:
    args = operator._parser().parse_args(_arguments("sae"))
    settings = operator._settings(args, _environment(tmp_path, panel="sae"))

    command = operator._worker(settings, 3)

    assert command[command.index("--cache-root") + 1] == ("/local/cache/process-p00003")
    assert command[command.index("--panel") + 1] == "sae"
    assert "--train-source-root" not in command
    assert "--eval-source-root" not in command
    assert command.count("--run-root-h") == 1
    assert command.count("--run-root-f") == 1


def test_transcoder_worker_requires_and_passes_all_four_overlay_inputs(
    operator,
    tmp_path: Path,
) -> None:
    environment = _environment(tmp_path, panel="transcoder")
    args = operator._parser().parse_args(_arguments("transcoder"))
    settings = operator._settings(args, environment)
    command = operator._worker(settings, 0)

    for name in (
        "--train-source-root",
        "--eval-source-root",
        "--train-source-object-prefix",
        "--eval-source-object-prefix",
    ):
        assert command.count(name) == 1

    del environment["CONTRIBUTION_EVAL_SOURCE_OVERLAY_PREFIX"]
    with pytest.raises(operator.OperatorError, match="all source-overlay"):
        operator._settings(args, environment)


@pytest.mark.parametrize("panel", ("sae", "transcoder"))
def test_operator_command_exactly_matches_unified_worker_cli(
    operator,
    tmp_path: Path,
    panel: str,
) -> None:
    public = operator._parser().parse_args(_arguments(panel))
    settings = operator._settings(public, _environment(tmp_path, panel=panel))
    command = operator._worker(settings, 2)
    module_index = command.index("moe_sae.experiment5_evaluation_worker")

    parsed = _worker_parser().parse_args(command[module_index + 1 :])

    assert parsed.panel == panel
    assert parsed.process_index == 2
    assert parsed.cache_root == "/local/cache/process-p00002"
    assert parsed.checkpoint_tokens == 5_000_000
    assert (parsed.train_source_root is None) == (panel == "sae")


def test_sae_rejects_even_one_overlay_input(operator, tmp_path: Path) -> None:
    environment = _environment(tmp_path, panel="sae")
    environment["CONTRIBUTION_SOURCE_OVERLAY_ROOT"] = "/remote/wrong"
    args = operator._parser().parse_args(_arguments("sae"))

    with pytest.raises(operator.OperatorError, match="prohibits source-overlay"):
        operator._settings(args, environment)


def test_main_fans_all_four_controllers_fetches_validates_and_publishes_pair(
    operator,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    environment = _environment(tmp_path, panel="transcoder")
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    mapping = {0: 2, 1: 0, 2: 3, 3: 1}
    monkeypatch.setattr(operator, "_mapping", lambda _settings: mapping)
    commands: list[list[str]] = []

    def fake_run(command, log):
        match = re.fullmatch(
            r"evaluation-c([0-9]{5})-p([0-9]{5})[.]log",
            log.name,
        )
        assert match is not None
        controller, process = (int(value) for value in match.groups())
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(
            "terminal action=experiment5-evaluate "
            f"controller={controller} process={process} status=0\n",
            encoding="ascii",
        )
        commands.append(command)
        return 0

    fetched: list[str] = []

    def fake_remote(_settings, *, controller, remote_path):
        assert controller == 1
        fetched.append(remote_path)
        return b"{}\n"

    result = SimpleNamespace(
        panel="transcoder",
        seed=1,
        milestone=5_000_000,
        body={"source_revision": "a" * 40},
        result_sha256="b" * 64,
    )
    paired = SimpleNamespace(
        panel="transcoder",
        seed=1,
        milestone=5_000_000,
        artifact_sha256="c" * 64,
    )
    monkeypatch.setattr(operator, "_run", fake_run)
    monkeypatch.setattr(operator, "_remote_bytes", fake_remote)
    monkeypatch.setattr(
        operator,
        "validate_experiment5_result",
        lambda value: result if value == {} else None,
    )
    monkeypatch.setattr(
        operator,
        "validate_experiment5_paired_scores",
        lambda value, *, result: paired if value == {} and result is not None else None,
    )

    assert operator.main(_arguments("transcoder")) == 0
    assert len(commands) == 4
    assert sorted(fetched) == [
        "/remote/results/sae/seed1/5000000/paired-scores.json",
        "/remote/results/sae/seed1/5000000/scientific-result.json",
    ]
    assert (
        Path(environment["EXPERIMENT5_RESULT_PATH_CONTROLLER"]).read_bytes() == b"{}\n"
    )
    assert (
        Path(environment["EXPERIMENT5_PAIRED_SCORES_PATH_CONTROLLER"]).read_bytes()
        == b"{}\n"
    )
