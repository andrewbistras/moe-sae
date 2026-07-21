from __future__ import annotations

import argparse
from contextlib import nullcontext
import itertools
import json

import jax.numpy as jnp
import numpy as np
import pytest

from moe_sae.config import Gemma4Config
from moe_sae.model import CaptureBatch
import moe_sae.profile_worker as profile_worker


def _capture(rows: int) -> CaptureBatch:
    ids = np.tile(np.arange(8, dtype=np.uint8), (rows, 1))
    return CaptureBatch(
        activation=jnp.zeros((rows, 2816), dtype=jnp.bfloat16),
        expert_ids=jnp.asarray(ids),
        router_weights=jnp.full((rows, 8), 0.125, dtype=jnp.bfloat16),
        sequence_id=np.arange(rows, dtype=np.int64),
        position=jnp.arange(rows, dtype=jnp.uint16) % 1024,
    )


def test_profile_destination_is_explicitly_diagnostic(tmp_path):
    profile_worker._validate_diagnostic_destination(
        tmp_path / "diagnostics" / "attempt-1"
    )
    with pytest.raises(ValueError, match="named diagnostics"):
        profile_worker._validate_diagnostic_destination(tmp_path / "benchmarks")
    with pytest.raises(ValueError, match="directory, not a file"):
        profile_worker._validate_diagnostic_destination(
            tmp_path / "diagnostics" / "profile.json"
        )


def test_capture_validation_stays_device_side_and_fails_duplicate_routes():
    config = Gemma4Config()
    summary = profile_worker._validate_capture_on_device(
        _capture(4), config, expected_rows=4
    )
    assert summary["unique_experts_per_token"] is True
    assert summary["max_router_sum_error"] == 0.0

    duplicate = _capture(4)._replace(
        expert_ids=jnp.zeros((4, 8), dtype=jnp.uint8)
    )
    with pytest.raises(RuntimeError, match="duplicate routed experts"):
        profile_worker._validate_capture_on_device(
            duplicate, config, expected_rows=4
        )


def test_short_profile_writes_nonselectable_schema(monkeypatch, tmp_path):
    config = Gemma4Config()
    raw = {
        "inputs": jnp.ones((1, 1024), dtype=jnp.int32),
        "inputs_position": jnp.arange(1024, dtype=jnp.int32)[None, :],
        "inputs_segmentation": jnp.ones((1, 1024), dtype=jnp.int32),
    }
    capture = _capture(1024)

    class Mesh:
        shape = {"data": 1}
        axis_names = ("data",)

    class Runtime:
        class Config:
            use_ragged_sort = True
            ragged_buffer_factor = 1.1
            megablox = True
            sparse_matmul = False
            use_gmm_v2 = False
            use_tokamax_gmm = False
            attention = "dot_product"
            model_call_mode = "train"
            prefuse_moe_weights = False

        config = Config()
        model = object()
        mesh = Mesh()
        train_iterator = itertools.repeat(raw)
        eval_iterator = None

    runtime = Runtime()
    closed: list[object] = []
    monkeypatch.setattr(profile_worker, "load_config", lambda path: config)
    monkeypatch.setattr(profile_worker, "require_v4_32_runtime", lambda: None)
    monkeypatch.setattr(profile_worker, "_process_context", lambda: (0, 1, 1))
    monkeypatch.setattr(
        profile_worker,
        "environment_fingerprint",
        lambda root: {"sha256": "e" * 64},
    )
    monkeypatch.setattr(
        profile_worker,
        "_verify_checkpoint",
        lambda *args, **kwargs: ("a" * 64, 0.25),
    )
    monkeypatch.setattr(profile_worker, "build_maxtext_argv", lambda **kwargs: [])
    monkeypatch.setattr(profile_worker, "create_maxtext_runtime", lambda argv: runtime)
    monkeypatch.setattr(
        profile_worker.Gemma4PrefixAdapter,
        "from_maxtext",
        lambda model, config: object(),
    )
    monkeypatch.setattr(
        profile_worker, "_capture_output_shardings", lambda mesh: None
    )
    monkeypatch.setattr(
        profile_worker, "compile_adapter_step", lambda adapter, **kwargs: lambda *x: capture
    )
    monkeypatch.setattr(
        profile_worker, "maxtext_mesh_context", lambda runtime: nullcontext()
    )
    monkeypatch.setattr(
        profile_worker,
        "_parameter_sharding_summary",
        lambda model: {"arrays": 1, "elements": 1, "bytes": 2, "groups": []},
    )
    monkeypatch.setattr(profile_worker, "_jit_cache_size", lambda step: 1)
    monkeypatch.setattr(
        profile_worker, "_close_maxtext_runtime", lambda value: closed.append(value)
    )

    output = tmp_path / "diagnostics" / "attempt-1"
    args = argparse.Namespace(
        config="unused",
        checkpoint="checkpoint",
        checkpoint_sha256="a" * 64,
        output=str(output),
        mesh="ep16-fsdp1-tp1",
        per_device_sequence_batch=1,
        warmup_steps=1,
        measured_steps=1,
        trace_root=None,
        trace_steps=1,
    )
    assert profile_worker._profile(args) == 0

    payload = json.loads((output / "profile.json").read_text())
    assert payload["schema_version"] == profile_worker.PROFILE_SCHEMA
    assert payload["diagnostic_only"] is True
    assert payload["eligible_for_benchmark_selection"] is False
    assert payload["production_hard_gate_evaluated"] is False
    assert payload["measured_steps"] == 1
    assert payload["effective_kernel_config"]["use_ragged_sort"] is True
    assert payload["effective_kernel_config"]["ragged_buffer_factor"] == 1.1
    assert not (output / "selected.json").exists()
    assert closed == [runtime]


def test_command_caps_diagnostic_steps_before_loading_tpu():
    with pytest.raises(SystemExit, match="measured-steps must lie"):
        profile_worker.main(
            [
                "--config",
                "unused",
                "--checkpoint",
                "unused",
                "--checkpoint-sha256",
                "a" * 64,
                "--output",
                "/tmp/diagnostics/attempt",
                "--mesh",
                "ep16-fsdp1-tp1",
                "--per-device-sequence-batch",
                "1",
                "--measured-steps",
                "200",
            ]
        )
