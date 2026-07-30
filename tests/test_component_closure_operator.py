from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _operator_module():
    path = ROOT / "scripts" / "tpu" / "evaluate_component_closure.py"
    spec = importlib.util.spec_from_file_location(
        "_test_component_closure_operator",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_run_all_operators_queues_six_bundles_over_five_resources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _operator_module()
    resources = ("sae-a", "sae-a", "sae-b", "tc-v5", "tc-v4a", "tc-v4b")
    evaluation = {
        "state_root": str(tmp_path),
        "bundles": [
            {
                "panel": "sae" if index < 3 else "transcoder",
                "seed": index % 3,
                "resource_id": resource_id,
            }
            for index, resource_id in enumerate(resources)
        ],
    }
    calls: list[tuple[str, int]] = []
    lock = threading.Lock()

    def run_one(
        _evaluation: dict,
        _training: dict,
        *,
        bundle_index: int,
        attempt_id: str,
        log_root: Path,
        timeout_seconds: int,
        dry_run: bool,
    ) -> int:
        del attempt_id, log_root, timeout_seconds, dry_run
        with lock:
            calls.append(
                (
                    _evaluation["bundles"][bundle_index]["resource_id"],
                    bundle_index,
                )
            )
        return 0

    monkeypatch.setattr(module, "_run_bundle_operator", run_one)
    monkeypatch.setattr(
        module,
        "parse_training_campaign",
        lambda _training: SimpleNamespace(
            resource=lambda resource_id: SimpleNamespace(resource_id=resource_id)
        ),
    )
    assert (
        module._run_all_operators(
            evaluation,
            {},
            maximum_wall_seconds=60,
            dry_run=True,
        )
        == 0
    )
    assert sorted(index for _resource, index in calls) == list(range(6))
    assert [index for resource, index in calls if resource == "sae-a"] == [0, 1]


def test_remote_bundle_command_fans_out_process_identity() -> None:
    module = _operator_module()

    class Routed:
        @staticmethod
        def _gcloud_ssh(
            _settings: object,
            *,
            controller: int,
            remote_command: str,
        ) -> list[str]:
            return ["ssh", str(controller), remote_command]

    training = SimpleNamespace(
        remote_uv="/remote/.venv/bin/uv",
        remote_project_root="/remote/moe-sae",
        project_commit="a" * 40,
    )
    resource = SimpleNamespace(
        environment_dict=lambda: {},
    )
    command = module._remote_command(
        Routed,
        SimpleNamespace(),
        training=training,
        resource=resource,
        bundle={"schema_version": "bundle", "cells": []},
        controller_index=2,
        process_index=3,
        attempt_id="closure-eval-test",
    )
    joined = " ".join(command)
    assert "evaluate_component_closure.py" in joined
    assert "run-bundle-json" in joined
    assert "--process-index" in joined
    assert " 3 " in joined
    assert "controller=%s process=%s" in joined


def _operator_case(
    module,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    markers: dict[int, str],
    transports: dict[int, int],
) -> tuple[dict, dict]:
    mapping = {controller: controller for controller in range(4)}
    cell = {
        "logical_condition": "B",
        "source_protocol_id": "component-closure-sae-b",
        "source_condition": "B",
        "checkpoint_tokens": 10_000_000,
        "parent_training_result_sha256": "a" * 64,
        "parent_checkpoint_pointer_sha256": "b" * 64,
        "result_root": "/remote/component-closure/sae-b-seed0",
        "result_root_controller": str(tmp_path / "published"),
    }
    bundle = {
        "bundle_sha256": "c" * 64,
        "resource_id": "sae-a",
        "panel": "sae",
        "seed": 0,
        "source_revision": "d" * 40,
        "cells": [cell],
    }
    evaluation = {
        "training_plan_sha256": "e" * 64,
        "plan_sha256": "f" * 64,
        "state_root": str(tmp_path / "state"),
        "bundles": [bundle],
    }
    training = {"plan_sha256": evaluation["training_plan_sha256"]}
    resource = SimpleNamespace(
        resource_id="sae-a",
        environment_dict=lambda: {},
    )
    parsed_training = SimpleNamespace(resource=lambda _resource_id: resource)
    monkeypatch.setattr(
        module,
        "_operator_binding",
        lambda _evaluation, _training, *, bundle_index: (
            _evaluation["bundles"][bundle_index],
            parsed_training,
            SimpleNamespace(),
        ),
    )

    class Routed:
        @staticmethod
        def _mapping(_settings: object) -> dict[int, int]:
            return mapping

        @staticmethod
        def _subprocess_environment() -> dict[str, str]:
            return {}

        @staticmethod
        def _remote_bytes(
            _settings: object,
            *,
            controller: int,
            remote_path: str,
        ) -> bytes:
            assert controller == 0
            if remote_path.endswith("/evaluation-result.json"):
                return result_payload
            assert remote_path.endswith("/objects/score.bin")
            return score_payload

    monkeypatch.setattr(module, "_load_routed_operator", lambda: Routed)
    monkeypatch.setattr(
        module,
        "_remote_command",
        lambda *_args, controller_index, **_kwargs: [
            "gcloud",
            str(controller_index),
        ],
    )

    def run_remote(
        _command: list[str],
        log: Path,
        *,
        timeout_seconds: int,
        environment: dict[str, str],
    ) -> int:
        del timeout_seconds, environment
        controller = int(log.name.split("-c", 1)[1].split("-", 1)[0])
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(markers[controller], encoding="ascii")
        return transports[controller]

    monkeypatch.setattr(module, "_run_remote", run_remote)
    score_payload = b"component-closure-score"
    result_body = {
        "protocol_id": cell["source_protocol_id"],
        "status": "cell_evaluated",
        "panel": bundle["panel"],
        "condition": cell["source_condition"],
        "seed": bundle["seed"],
        "checkpoint_tokens": cell["checkpoint_tokens"],
        "source_revision": bundle["source_revision"],
        "parent_training_result_sha256": cell[
            "parent_training_result_sha256"
        ],
        "parent_checkpoint_pointer_sha256": cell[
            "parent_checkpoint_pointer_sha256"
        ],
        "score_data_object": "objects/score.bin",
        "score_data_sha256": hashlib.sha256(score_payload).hexdigest(),
        "score_data_bytes": len(score_payload),
    }
    result_payload = json.dumps(
        {
            **result_body,
            "result_sha256": module.canonical_sha256(result_body),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return evaluation, training


def test_nonzero_transports_with_four_exact_success_markers_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _operator_module()
    markers = {
        controller: (
            "gcloud transport diagnostic\n"
            "terminal action=component-closure-evaluate-bundle "
            f"controller={controller} process={controller} status=0\n"
        )
        for controller in range(4)
    }
    evaluation, training = _operator_case(
        module,
        tmp_path,
        monkeypatch,
        markers=markers,
        transports={controller: 1 for controller in range(4)},
    )

    assert (
        module._run_bundle_operator(
            evaluation,
            training,
            bundle_index=0,
            attempt_id="transport-teardown",
            log_root=tmp_path / "logs",
            timeout_seconds=60,
            dry_run=False,
        )
        == 0
    )
    assert (tmp_path / "published/evaluation-result.json").is_file()
    completion = json.loads(
        (
            tmp_path / "state/bundles/sae-seed0/completion.json"
        ).read_text(encoding="ascii")
    )
    assert completion["status"] == "bundle_published"
    assert completion["cells"][0]["logical_condition"] == "B"
    assert (
        "transports exited nonzero after exact remote success"
        in capsys.readouterr().err
    )


@pytest.mark.parametrize(
    "bad_marker",
    [
        (
            "terminal action=component-closure-evaluate-bundle "
            "controller=2 process=2 status=7\n"
        ),
        "gcloud transport diagnostic without a terminal marker\n",
        (
            "terminal action=component-closure-evaluate-bundle "
            "controller=2 process=2 status=unknown\n"
        ),
    ],
    ids=("terminal-nonzero", "terminal-missing", "terminal-malformed"),
)
def test_nonzero_transport_does_not_mask_invalid_terminal_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bad_marker: str,
) -> None:
    module = _operator_module()
    markers = {
        controller: (
            "terminal action=component-closure-evaluate-bundle "
            f"controller={controller} process={controller} status=0\n"
        )
        for controller in range(4)
    }
    markers[2] = bad_marker
    evaluation, training = _operator_case(
        module,
        tmp_path,
        monkeypatch,
        markers=markers,
        transports={controller: 1 for controller in range(4)},
    )

    with pytest.raises(module.ComponentClosureEvaluationOperatorError):
        module._run_bundle_operator(
            evaluation,
            training,
            bundle_index=0,
            attempt_id="transport-failure",
            log_root=tmp_path / "logs",
            timeout_seconds=60,
            dry_run=False,
        )
    assert not (tmp_path / "published").exists()
    assert not (
        tmp_path / "state/bundles/sae-seed0/completion.json"
    ).exists()
