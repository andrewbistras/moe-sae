from __future__ import annotations

import importlib.util
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
