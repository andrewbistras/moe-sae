from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from moe_sae.contribution_artifacts import HIDDEN_SIZE
from moe_sae.contribution_transcoder_harvest import (
    _routed_input_capture_chunks,
    routed_input_capture_to_wire,
)
from moe_sae.contribution_transcoder_harvest_worker import (
    RoutedInputHarvestWorkerError,
    _parent_dataset_token_target,
    _parser as worker_parser,
)


def _operator():
    path = (
        Path(__file__).parents[1]
        / "scripts"
        / "tpu"
        / "harvest_transcoder_overlay.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_test_transcoder_overlay_operator", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_wire_conversion_bitcasts_bfloat16_and_rejects_extra_shape():
    ml_dtypes = pytest.importorskip("ml_dtypes")
    routed = np.arange(2 * HIDDEN_SIZE, dtype=np.float32).reshape(2, HIDDEN_SIZE)
    routed = routed.astype(ml_dtypes.bfloat16)
    batch = routed_input_capture_to_wire(
        routed,
        np.asarray([10, 11], dtype=np.int32),
        np.asarray([0, 1], dtype=np.int32),
    )

    assert batch.routed_input.dtype == np.uint16
    np.testing.assert_array_equal(batch.routed_input, routed.view(np.uint16))
    assert batch.sequence_id.dtype == np.int64
    assert batch.position.dtype == np.uint16

    with pytest.raises(Exception):
        routed_input_capture_to_wire(
            routed[:, :-1],
            np.asarray([10, 11]),
            np.asarray([0, 1]),
        )


def test_execution_batch_is_split_into_artifact_sized_wire_chunks():
    routed = np.arange(5 * HIDDEN_SIZE, dtype=np.float32).reshape(
        5, HIDDEN_SIZE
    )
    chunks = tuple(
        _routed_input_capture_chunks(
            routed,
            np.asarray([10, 11, 12, 13, 14], dtype=np.int64),
            np.asarray([0, 1, 2, 3, 4], dtype=np.uint16),
            max_tokens=2,
        )
    )

    assert [chunk.tokens for chunk in chunks] == [2, 2, 1]
    np.testing.assert_array_equal(
        np.concatenate([chunk.sequence_id for chunk in chunks]),
        np.asarray([10, 11, 12, 13, 14], dtype=np.int64),
    )


def test_worker_cli_freezes_a_maximum_600_second_bound():
    parser = worker_parser()
    required = [
        "--config",
        "/repo/config.yaml",
        "--split",
        "train",
        "--checkpoint",
        "gs://bucket/checkpoint",
        "--checkpoint-sha256",
        "a" * 64,
        "--grain-files",
        "gs://bucket/train/*.array_record",
        "--staged-manifest",
        "gs://bucket/train/manifest.json",
        "--source-v1-root",
        "/nfs/source",
        "--source-cache-root",
        "/tmp/cache",
        "--source-cache-max-bytes",
        "1000",
        "--target-control-root",
        "/nfs/target",
        "--control-root",
        "/nfs/overlay",
        "--local-buffer-root",
        "/tmp/buffer",
        "--object-prefix",
        "gs://bucket/overlay",
        "--repo-root",
        "/repo",
        "--jax-cache-dir",
        "/tmp/jax",
        "--mesh",
        "ep16-fsdp1-tp1",
        "--per-device-sequence-batch",
        "8",
    ]
    assert parser.parse_args(required).maximum_runtime_seconds == 600


def test_staged_dataset_target_comes_from_immutable_source_v1_parent():
    target = {
        "total_tokens": 2_000_000,
        "source_v1_manifest": {"total_tokens": 50_000_000},
    }
    assert _parent_dataset_token_target(target) == 50_000_000

    with pytest.raises(
        RoutedInputHarvestWorkerError,
        match="incompatible with its source-v1 parent",
    ):
        _parent_dataset_token_target(
            {
                "total_tokens": 50_000_001,
                "source_v1_manifest": {"total_tokens": 50_000_000},
            }
        )


def test_operator_builds_four_concurrent_worker_commands_with_external_timeout():
    module = _operator()
    args = SimpleNamespace(
        gcp_project="project",
        tpu_name="tpu",
        tpu_zone="us-central2-b",
        project_commit="a" * 40,
        project_root="/home/user/moe-sae",
        remote_uv="/home/user/.local/bin/uv",
        split="eval",
        checkpoint="gs://bucket/model/items",
        checkpoint_sha256="b" * 64,
        grain_files="gs://bucket/eval/*.array_record",
        staged_manifest="gs://bucket/eval/manifest.json",
        source_v1_root="/nfs/source-eval",
        source_cache_max_bytes=1 << 30,
        target_control_root="/nfs/target-eval",
        control_root="/nfs/overlay-eval",
        local_buffer_root="/tmp/moe-sae",
        object_prefix="gs://bucket/overlay-eval",
        mesh="ep16-fsdp1-tp1",
        per_device_sequence_batch=8,
        maximum_runtime_seconds=600,
        local_log_root="/tmp/logs",
        ssh_key_file=None,
        dry_run=True,
    )

    commands = [module._worker_command(args, index) for index in range(4)]
    rendered = [" ".join(command) for command in commands]

    assert all("timeout --signal=TERM --kill-after=30s 600s" in item for item in rendered)
    assert all("contribution_transcoder_harvest_worker" in item for item in rendered)
    assert [next(part for part in command if part.startswith("--worker=")) for command in commands] == [
        "--worker=0",
        "--worker=1",
        "--worker=2",
        "--worker=3",
    ]
