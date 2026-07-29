from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from moe_sae.experiment5_routed_ladder_evaluation_worker import (
    _parser as _worker_parser,
    load_process_evidence,
    merge_process_evidence,
    seal_merged_result,
    seal_process_evidence,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "tpu" / "evaluate_experiment5_routed_ladder.py"
SOURCE_REVISION = "a" * 40
H_TRAINING_REVISION = "b" * 40


@pytest.fixture()
def operator():
    specification = importlib.util.spec_from_file_location(
        "experiment5_routed_ladder_evaluation_operator",
        SCRIPT,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _environment(tmp_path: Path, *, panel: str = "sae") -> dict[str, str]:
    environment = {
        "GCP_PROJECT": "test-project",
        "TPU_NAME": "activation-ladder-slice",
        "TPU_ZONE": "us-test1-a",
        "PROJECT_COMMIT": SOURCE_REVISION,
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
        "CONTRIBUTION_SAE_CHECKPOINT_ROOT": "gs://bucket/checkpoints",
        "EXPERIMENT5_ROUTED_ACTIVATION_LADDER_RESULT_ROOT_CONTROLLER": str(
            tmp_path / "published"
        ),
    }
    if panel == "sae":
        environment["CONTRIBUTION_EVALUATION_LOG_ROOT"] = str(tmp_path / "logs")
    else:
        environment.update(
            {
                "TRANSCODER_EVALUATION_LOG_ROOT": str(tmp_path / "logs"),
                "CONTRIBUTION_SOURCE_OVERLAY_ROOT": "/remote/source/train",
                "CONTRIBUTION_EVAL_SOURCE_OVERLAY_ROOT": "/remote/source/eval",
                "CONTRIBUTION_SOURCE_OVERLAY_PREFIX": "gs://bucket/source/train",
                "CONTRIBUTION_EVAL_SOURCE_OVERLAY_PREFIX": "gs://bucket/source/eval",
            }
        )
    return environment


def _arguments(
    *,
    panel: str = "sae",
    condition: str = "A512ReLU",
    dry_run: bool = False,
) -> list[str]:
    arguments = [
        "--panel",
        panel,
        "--config",
        f"/remote/configs/gemma4_experiment5_routed_ladder_{panel}_v1.yaml",
        "--condition",
        condition,
        "--seed",
        "0",
        "--checkpoint-tokens",
        "2000000",
        "--source-revision",
        SOURCE_REVISION,
        "--training-source-revision",
        H_TRAINING_REVISION if condition == "H" else SOURCE_REVISION,
        "--run-root",
        f"/remote/runs/{panel}/{condition}/seed0",
        "--result-root",
        f"/remote/results/{panel}/{condition}/seed0/2000000",
    ]
    if condition == "H":
        arguments.extend(
            (
                "--h-config",
                f"/remote/configs/gemma4_hybrid_{panel}_v1.yaml",
            )
        )
    if dry_run:
        arguments.append("--dry-run")
    return arguments


@pytest.mark.parametrize(
    ("panel", "condition"),
    (
        ("sae", "A512ReLU"),
        ("transcoder", "A512Softplus"),
        ("sae", "H"),
        ("transcoder", "H"),
    ),
)
def test_a_and_h_commands_exactly_match_worker_contract(
    operator,
    tmp_path: Path,
    panel: str,
    condition: str,
) -> None:
    public = operator._parser().parse_args(
        _arguments(panel=panel, condition=condition)
    )
    settings = operator._settings(
        public,
        _environment(tmp_path, panel=panel),
    )
    command = operator._worker(settings, 3)
    module_index = command.index(
        "moe_sae.experiment5_routed_ladder_evaluation_worker"
    )

    parsed = _worker_parser().parse_args(command[module_index + 1 :])

    assert parsed.panel == panel
    assert parsed.condition == condition
    assert parsed.process_index == 3
    assert parsed.cache_root == "/local/cache/process-p00003"
    assert (parsed.h_config is not None) == (condition == "H")
    assert parsed.training_source_revision == (
        H_TRAINING_REVISION if condition == "H" else SOURCE_REVISION
    )
    assert (parsed.train_source_root is None) == (panel == "sae")


def test_h_config_is_required_only_for_h_and_a_revision_is_current(
    operator,
    tmp_path: Path,
) -> None:
    environment = _environment(tmp_path)
    h_arguments = _arguments(condition="H")
    h_config = h_arguments.index("--h-config")
    del h_arguments[h_config : h_config + 2]
    with pytest.raises(operator.OperatorError, match="requires --h-config"):
        operator._settings(operator._parser().parse_args(h_arguments), environment)

    a_arguments = _arguments()
    a_arguments[a_arguments.index("--training-source-revision") + 1] = (
        H_TRAINING_REVISION
    )
    with pytest.raises(operator.OperatorError, match="match its training revision"):
        operator._settings(operator._parser().parse_args(a_arguments), environment)

    a_arguments = _arguments()
    a_arguments.extend(("--h-config", "/remote/configs/wrong.yaml"))
    with pytest.raises(operator.OperatorError, match="valid only for H"):
        operator._settings(operator._parser().parse_args(a_arguments), environment)


@pytest.mark.parametrize("condition", ("A512ReLU", "H"))
def test_dry_run_emits_exact_worker_without_starting_fanout(
    operator,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    condition: str,
) -> None:
    for name, value in _environment(tmp_path).items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(
        operator,
        "_mapping",
        lambda _settings: pytest.fail("dry run tried to contact controllers"),
    )

    assert operator.main(_arguments(condition=condition, dry_run=True)) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["condition"] == condition
    parsed = _worker_parser().parse_args(
        payload["worker_process_zero"][
            payload["worker_process_zero"].index(
                "moe_sae.experiment5_routed_ladder_evaluation_worker"
            )
            + 1 :
        ]
    )
    assert parsed.condition == condition
    assert (parsed.h_config is not None) == (condition == "H")


def _sealed_result(root: Path) -> tuple[bytes, str, bytes]:
    scores = SimpleNamespace(
        valid_mask=np.ones((2,), dtype=np.bool_),
        component_sse=np.asarray([2.0, 3.0]),
        component_sst=np.asarray([10.0, 12.0]),
        routed_sse=np.asarray([1.5, 2.5]),
        routed_sst=np.asarray([9.0, 11.0]),
        oracle_shared_sse=np.asarray([1.0, 2.0]),
        oracle_shared_sst=np.asarray([8.0, 10.0]),
        learned_full_ffn_sse=None,
        learned_full_ffn_sst=None,
    )
    lane = root / "lane"
    seal_process_evidence(
        lane,
        panel="sae",
        condition="A512ReLU",
        seed=0,
        checkpoint_tokens=2_000_000,
        checkpoint_actual_tokens=2_015_232,
        process_index=0,
        process_count=1,
        source_revision=SOURCE_REVISION,
        training_source_revision=SOURCE_REVISION,
        config_sha256="1" * 64,
        config_file_sha256="2" * 64,
        h_config_sha256=None,
        h_config_file_sha256=None,
        training_result_sha256="3" * 64,
        training_contract_sha256="4" * 64,
        checkpoint_sha256="5" * 64,
        target_manifest_sha256="6" * 64,
        source_manifest_sha256=None,
        recomposition_sidecar_sha256="7" * 64,
        centering_receipt_sha256="8" * 64,
        input_receipt_sha256="9" * 64,
        learned_input_receipt_sha256=None,
        evaluation_wall_seconds=1.0,
        sequence_id=np.asarray([10, 11], dtype=np.int64),
        position=np.asarray([0, 0], dtype=np.uint16),
        scores=scores,
    )
    evidence = load_process_evidence(lane / "process-evidence.json")
    summary, tensors = merge_process_evidence([evidence])
    final_root = root / "final"
    result = seal_merged_result(
        final_root,
        summary=summary,
        tensors=tensors,
        expected_tokens=2,
    )
    relative = result["score_data_object"]
    return (
        (final_root / "evaluation-result.json").read_bytes(),
        relative,
        (final_root / relative).read_bytes(),
    )


def test_main_atomically_publishes_validated_result_and_score_object(
    operator,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    environment = _environment(tmp_path)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    result_payload, relative, score_payload = _sealed_result(tmp_path / "remote")
    mapping = {0: 2, 1: 0, 2: 3, 3: 1}
    monkeypatch.setattr(operator, "_mapping", lambda _settings: mapping)

    def fake_run(_command, log):
        match = re.fullmatch(
            r"evaluation-c([0-9]{5})-p([0-9]{5})[.]log",
            log.name,
        )
        assert match is not None
        controller, process = (int(value) for value in match.groups())
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(
            "terminal action=activation-ladder-evaluate "
            f"controller={controller} process={process} status=0\n",
            encoding="ascii",
        )
        return 0

    def fake_remote(_settings, *, controller, remote_path):
        assert controller == 1
        if remote_path.endswith("/evaluation-result.json"):
            return result_payload
        assert remote_path.endswith("/" + relative)
        return score_payload

    rename_observation: list[tuple[bool, bool]] = []
    real_rename = operator.os.rename

    def observing_rename(source, destination):
        staging = Path(source)
        rename_observation.append(
            (
                (staging / "evaluation-result.json").is_file(),
                (staging / relative).is_file(),
            )
        )
        return real_rename(source, destination)

    monkeypatch.setattr(operator, "_run", fake_run)
    monkeypatch.setattr(operator, "_remote_bytes", fake_remote)
    monkeypatch.setattr(operator.os, "rename", observing_rename)

    assert operator.main(_arguments()) == 0

    destination = Path(
        environment[
            "EXPERIMENT5_ROUTED_ACTIVATION_LADDER_RESULT_ROOT_CONTROLLER"
        ]
    )
    assert rename_observation == [(True, True)]
    assert (destination / "evaluation-result.json").read_bytes() == result_payload
    assert (destination / relative).read_bytes() == score_payload


def test_publish_is_idempotent_but_rejects_partial_or_changed_destination(
    operator,
    tmp_path: Path,
) -> None:
    result_payload, relative, score_payload = _sealed_result(tmp_path / "remote")
    settings = operator._settings(
        operator._parser().parse_args(_arguments()),
        _environment(tmp_path),
    )

    first = operator._publish(
        settings,
        result_payload=result_payload,
        relative=relative,
        score_payload=score_payload,
    )
    second = operator._publish(
        settings,
        result_payload=result_payload,
        relative=relative,
        score_payload=score_payload,
    )
    assert first == second

    (settings.result_root_controller / relative).write_bytes(b"changed")
    with pytest.raises(operator.OperatorError, match="partial or changed"):
        operator._publish(
            settings,
            result_payload=result_payload,
            relative=relative,
            score_payload=score_payload,
        )
