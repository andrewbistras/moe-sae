from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "tpu" / "attest_accelerator.py"


@pytest.fixture()
def operator():
    name = "_test_attest_accelerator"
    specification = importlib.util.spec_from_file_location(name, SCRIPT)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def arguments(
    operator,
    tmp_path: Path,
    *,
    accelerator_type: str | None = None,
):
    argv = [
        "--run-id",
        "experiment4",
        "--scientific-model",
        "sae",
        "--condition",
        "F",
        "--project",
        "neural-dragnet",
        "--tpu-name",
        "experiment4-sae-f",
        "--zone",
        "europe-west4-a",
        "--output-root",
        str(tmp_path),
    ]
    if accelerator_type is not None:
        argv.extend(("--accelerator-type", accelerator_type))
    return operator._parser().parse_args(argv)


def test_attestation_freezes_live_create_time_and_spot_shape(
    operator,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    description = {
        "name": (
            "projects/neural-dragnet/locations/europe-west4-a/nodes/"
            "experiment4-sae-f"
        ),
        "acceleratorType": "v6e-16",
        "state": "READY",
        "createTime": "2026-07-28T10:11:12.123Z",
        "schedulingConfig": {"spot": True, "preemptible": True},
    }

    class Completed:
        returncode = 0
        stdout = json.dumps(description)
        stderr = ""

    monkeypatch.setattr(operator.subprocess, "run", lambda *_a, **_k: Completed())
    result = operator.attest(arguments(operator, tmp_path))
    assert result["contract"]["resource_create_time"] == description["createTime"]
    assert result["attestation"]["spot"] is True
    assert (tmp_path / "accelerator-contract.json").is_file()


def test_describe_command_is_read_only(operator, tmp_path: Path) -> None:
    command = operator._describe_command(arguments(operator, tmp_path))
    assert command[:6] == [
        "gcloud",
        "compute",
        "tpus",
        "tpu-vm",
        "describe",
        "experiment4-sae-f",
    ]
    assert not {"create", "delete", "update"}.intersection(command)


def test_attestation_accepts_exact_v5litepod_type(
    operator,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    description = {
        "name": (
            "projects/neural-dragnet/locations/europe-west4-b/nodes/"
            "experiment4-sae-f"
        ),
        "acceleratorType": "v5litepod-16",
        "state": "READY",
        "createTime": "2026-07-28T12:00:00.000Z",
        "schedulingConfig": {"spot": True, "preemptible": True},
    }

    class Completed:
        returncode = 0
        stdout = json.dumps(description)
        stderr = ""

    monkeypatch.setattr(operator.subprocess, "run", lambda *_a, **_k: Completed())
    args = arguments(
        operator,
        tmp_path,
        accelerator_type="v5litepod-16",
    )
    args.zone = "europe-west4-b"
    result = operator.attest(args)
    assert result["contract"]["accelerator_type"] == "v5litepod-16"
