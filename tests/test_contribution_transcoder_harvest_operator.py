from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

import pytest


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
