from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "tpu" / "harvest_transcoder_overlay.py"


@pytest.fixture()
def operator():
    spec = importlib.util.spec_from_file_location(
        "harvest_transcoder_overlay_operator",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_frozen_v2_requires_explicit_new_overlay_acknowledgement(operator):
    with pytest.raises(operator.OperatorError, match="reuses its sealed"):
        operator.run(argparse.Namespace(new_overlay=False))


def test_new_overlay_command_rejects_an_already_sealed_control_root(operator):
    args = argparse.Namespace(
        remote_uv="uv",
        local_buffer_root="/dev/shm/moe-sae",
        split="train",
        project_commit="a" * 40,
        mesh="ep8-fsdp2-tp1",
        per_device_sequence_batch=2,
        project_root="/home/user/moe-sae",
        checkpoint="/nfs/checkpoint",
        checkpoint_sha256="b" * 64,
        grain_files="gs://bucket/grain",
        staged_manifest="/nfs/staged.json",
        source_v1_root="/nfs/source-v1",
        source_cache_max_bytes=1_000,
        target_control_root="/nfs/target",
        control_root="/nfs/new-overlay",
        object_prefix="gs://bucket/new-overlay",
        maximum_runtime_seconds=600,
        tpu_name="gemma4",
        gcp_project="project",
        tpu_zone="us-central2-b",
        ssh_key_file=None,
    )

    command = operator._worker_command(args, 0)
    joined = " ".join(command)
    assert "configs/gemma4_transcoder_v2.yaml" in joined
    assert "sealed overlay already exists" in joined
    assert "/nfs/new-overlay/manifest.json" in joined


def _operator_args(**overrides):
    values = {
        "remote_uv": "uv",
        "local_buffer_root": "/dev/shm/moe-sae",
        "split": "train",
        "project_commit": "a" * 40,
        "mesh": "ep8-fsdp2-tp1",
        "per_device_sequence_batch": 2,
        "project_root": "/home/user/moe-sae",
        "checkpoint": "/nfs/checkpoint",
        "checkpoint_sha256": "b" * 64,
        "grain_files": "gs://bucket/grain",
        "staged_manifest": "/nfs/staged.json",
        "source_v1_root": "/nfs/source-v1",
        "source_cache_max_bytes": 1_000,
        "target_control_root": "/nfs/target",
        "control_root": "/nfs/new-overlay",
        "object_prefix": "gs://bucket/new-overlay",
        "maximum_runtime_seconds": 600,
        "tpu_name": "gemma4",
        "gcp_project": "project",
        "tpu_zone": "us-central2-b",
        "ssh_key_file": None,
        "local_log_root": "/tmp/logs",
        "new_overlay": True,
        "dry_run": True,
        "config": "configs/gemma4_transcoder_v2.yaml",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _horizon_capture_config(tmp_path: Path) -> Path:
    value = yaml.safe_load(
        (ROOT / "configs/gemma4_transcoder_horizon_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    value["artifacts"]["train_target_manifest_sha256"] = "c" * 64
    path = tmp_path / "target-bound-horizon.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def test_frozen_v2_cannot_inherit_the_longer_horizon_runtime(operator):
    with pytest.raises(operator.OperatorError, match=r"\[1,600\]"):
        operator.run(_operator_args(maximum_runtime_seconds=601))


def test_target_bound_horizon_gets_explicit_train_only_one_hour_bound(
    operator,
    tmp_path: Path,
    capsys,
):
    config = _horizon_capture_config(tmp_path)
    args = _operator_args(
        config=str(config),
        maximum_runtime_seconds=3_600,
    )

    assert operator.run(args) == 0
    rendered = capsys.readouterr().out
    assert f"--config {config}" in rendered
    assert "timeout --signal=TERM --kill-after=30s 3600s" in rendered

    with pytest.raises(operator.OperatorError, match=r"\[1,3600\]"):
        operator.run(
            _operator_args(
                config=str(config),
                maximum_runtime_seconds=3_601,
            )
        )
    with pytest.raises(operator.OperatorError, match="reuses the sealed 250k"):
        operator.run(
            _operator_args(
                config=str(config),
                split="eval",
            )
        )
