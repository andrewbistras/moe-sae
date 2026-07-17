from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import moe_sae.distributed as distributed_module
from moe_sae.checkpointing import (
    checkpoint_payload_uri,
    final_pointer_from_latest,
    make_pointer,
    publish_pointer,
)
from moe_sae.distributed import (
    DistributedConfig,
    create_mesh,
    evaluate_distributed,
    init_distributed_state,
    precompute_distributed_a_capacity,
    restore_distributed_checkpoint,
    run_distributed_training,
    save_distributed_checkpoint,
    train_distributed,
)
from moe_sae.pipeline import ActivationBatch, TrainingConfig
from moe_sae.sae import SAEConfig


def test_default_mesh_is_process_by_local_devices() -> None:
    mesh = create_mesh()
    assert mesh.shape["data"] == jax.process_count()
    assert mesh.shape["feature"] == jax.local_device_count()
    assert np.asarray(mesh.devices).shape == (
        jax.process_count(),
        jax.local_device_count(),
    )


def test_capacity_scan_uses_static_local_microbatches_and_margin() -> None:
    sae = SAEConfig(
        "A",
        hidden_dim=2,
        num_experts=4,
        top_routed=2,
        bank_width=2,
        k=1,
        parameter_dtype="float32",
        warmup_steps=1,
        total_steps=2,
    )
    training = TrainingConfig(
        sae,
        seed=1,
        total_tokens=6,
        effective_batch_tokens=4,
        microbatch_tokens=2,
        checkpoint_every_tokens=4,
        health_gate_tokens=4,
    )
    activation = np.ones((6, 2), np.float32)
    routes = np.tile(np.asarray([[0, 1]], np.int32), (6, 1))
    capacity = precompute_distributed_a_capacity(
        [dict(activation=activation, expert_ids=routes)], training
    )
    # Each two-row micro gives experts 0/1 a load of two; 10% then round to 8.
    assert capacity == 8


def test_two_dimensional_train_eval_and_a_b_parity_in_subprocess() -> None:
    # Device count must be configured before JAX initializes, so exercise the
    # real 2D collectives in a fresh interpreter.
    script = r"""
import jax
import jax.numpy as jnp
import numpy as np

from moe_sae.distributed import (
    DistributedConfig,
    create_mesh,
    distributed_evaluate,
    run_distributed_training,
    shard_train_state,
)
from moe_sae.pipeline import ActivationBatch, TrainingConfig, train
from moe_sae.sae import SAEConfig, init_train_state

x = np.asarray([
    [1.0, 0.0, 0.0, 0.5],
    [0.0, 1.0, 0.0, -0.5],
    [0.0, 0.0, 1.0, 0.25],
    [1.0, 1.0, 0.0, 0.0],
], np.float32)
routes = np.asarray([[0, 1], [2, 3], [0, 2], [1, 3]], np.int32)
distributed = DistributedConfig(
    data_axis_size=2, feature_axis_size=2, microbatches_per_update=2
)
mesh = create_mesh(distributed)
assert mesh.shape == {"data": 2, "feature": 2}

for condition in ("B", "A"):
    sae = SAEConfig(
        condition,
        hidden_dim=4,
        num_experts=4,
        top_routed=2,
        bank_width=2,
        k=1,
        parameter_dtype="float32",
        warmup_steps=1,
        total_steps=1,
    )
    training = TrainingConfig(
        sae,
        seed=0,
        total_tokens=4,
        effective_batch_tokens=4,
        microbatch_tokens=2,
        checkpoint_every_tokens=4,
        health_gate_tokens=4,
        live_feature_fraction=0,
        loss_ratio_threshold=10,
        a_capacity=4 if condition == "A" else None,
    )
    state = init_train_state(jax.random.PRNGKey(0), sae)
    state = state._replace(
        params=state.params._replace(bias=jnp.ones_like(state.params.bias))
    )
    source = ActivationBatch(x, routes if condition == "A" else None)
    reference = train(
        training, [source], np.zeros(4, np.float32), state=state
    ).state
    sharded = shard_train_state(state, mesh, sae, distributed)
    result = run_distributed_training(
        training,
        [source],
        np.zeros(4, np.float32),
        mesh,
        state=sharded,
        distributed=distributed,
        a_capacity=2 if condition == "A" else None,
        require_health_pass=False,
    )
    for actual, expected in zip(
        jax.tree_util.tree_leaves(result.state),
        jax.tree_util.tree_leaves(reference),
        strict=True,
    ):
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=2e-7)
    assert result.timing.compile_seconds > 0
    assert result.timing.optimizer_wall_seconds > 0
    assert result.timing.checkpoint_seconds == 0
    assert result.timing.global_chip_count == 4
    assert result.timing.chip_seconds == (
        result.timing.optimizer_wall_seconds * 4
    )
    assert result.timing.tokens == 4
    evaluation = distributed_evaluate(
        result.state.params,
        [source],
        np.zeros(4, np.float32),
        sae,
        mesh,
        max_tokens=4,
        microbatch_tokens=2,
        a_capacity=2 if condition == "A" else None,
        distributed=distributed,
    )
    assert evaluation.tokens == 4
    assert evaluation.l0 == 1
    assert evaluation.topk_valid
"""
    environment = dict(os.environ)
    environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    completed = subprocess.run(
        [sys.executable, "-c", script],
        env=environment,
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_final_checkpoint_boundary_does_not_reserialize_latest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sae = SAEConfig(
        "B",
        hidden_dim=2,
        num_experts=2,
        top_routed=1,
        bank_width=2,
        k=1,
        parameter_dtype="float32",
        warmup_steps=1,
        total_steps=1,
    )
    training = TrainingConfig(
        sae,
        seed=7,
        total_tokens=2,
        effective_batch_tokens=2,
        microbatch_tokens=1,
        checkpoint_every_tokens=2,
        health_gate_tokens=2,
        live_feature_fraction=0.0,
        loss_ratio_threshold=2.0,
    )
    distributed = DistributedConfig(microbatches_per_update=2)
    mesh = create_mesh(distributed)
    state = init_distributed_state(jax.random.PRNGKey(7), mesh, sae, distributed)
    state = state._replace(
        params=state.params._replace(bias=10.0 * jnp.ones_like(state.params.bias))
    )
    saved: list[str] = []
    timing_sidecars: list[dict[str, object]] = []

    def record_save(path, state, mesh, training_config, metadata=None):
        del state, mesh, training_config, metadata
        saved.append(Path(path).name)

    def record_timing(checkpoint_dir, **values):
        del checkpoint_dir
        timing_sidecars.append(values)

    monkeypatch.setattr(distributed_module, "save_distributed_checkpoint", record_save)
    monkeypatch.setattr(distributed_module, "_publish_checkpoint_timing", record_timing)
    payload_root = tmp_path / "bulk-checkpoints"
    result = run_distributed_training(
        training,
        [ActivationBatch(np.ones((2, 2), np.float32))],
        np.zeros(2, np.float32),
        mesh,
        state=state,
        distributed=distributed,
        checkpoint_dir=tmp_path,
        checkpoint_payload_root=payload_root,
        require_health_pass=False,
    )

    assert len(saved) == 1
    assert saved[0].startswith("step-00000001-tokens-000000000002-attempt-")
    assert len(timing_sidecars) == 2
    assert all(item["tokens_seen"] == 2 for item in timing_sidecars)
    assert result.latest_checkpoint == str(tmp_path / "latest.json")
    assert result.final_checkpoint == str(tmp_path / "final.json")
    latest = json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))
    final = json.loads((tmp_path / "final.json").read_text(encoding="utf-8"))
    assert latest["payload_uri"] == final["payload_uri"]
    assert Path(latest["payload_uri"]).is_relative_to(payload_root)


def test_orbax_sharded_checkpoint_roundtrip_when_tpu_extra_installed(
    tmp_path: Path,
) -> None:
    pytest.importorskip("orbax.checkpoint")
    sae = SAEConfig(
        "B",
        hidden_dim=2,
        num_experts=2,
        top_routed=1,
        bank_width=2,
        k=1,
        parameter_dtype="float32",
        warmup_steps=1,
        total_steps=1,
    )
    training = TrainingConfig(
        sae,
        seed=0,
        total_tokens=2,
        effective_batch_tokens=2,
        microbatch_tokens=1,
        checkpoint_every_tokens=2,
        health_gate_tokens=2,
    )
    distributed = DistributedConfig(microbatches_per_update=2)
    mesh = create_mesh(distributed)
    state = init_distributed_state(jax.random.PRNGKey(0), mesh, sae, distributed)
    payload_root = tmp_path / "payloads"
    attempt_id = "c" * 32
    destination = checkpoint_payload_uri(
        payload_root,
        {},
        step=0,
        tokens_seen=0,
        attempt_id=attempt_id,
    )
    save_distributed_checkpoint(
        destination, state, mesh, training, {"purpose": "roundtrip"}
    )
    with pytest.raises(ValueError, match="latest.json or final.json"):
        restore_distributed_checkpoint(destination, state, mesh, training)
    latest = make_pointer(
        kind="latest",
        payload_uri=destination,
        step=0,
        tokens_seen=0,
        attempt_id=attempt_id,
        run_identity={},
    )
    publish_pointer(tmp_path / "control", latest)
    restored, metadata = restore_distributed_checkpoint(
        tmp_path / "control" / "latest.json",
        state,
        mesh,
        training,
        checkpoint_payload_root=payload_root,
    )
    assert metadata["purpose"] == "roundtrip"
    assert metadata["checkpoint_payload_uri"] == destination
    assert metadata["tokens_seen"] == 0
    assert metadata["steps"] == 0
    for actual, expected in zip(
        jax.tree_util.tree_leaves(restored),
        jax.tree_util.tree_leaves(state),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)
    publish_pointer(tmp_path / "control", final_pointer_from_latest(latest))
    with pytest.raises(RuntimeError, match="not final"):
        evaluate_distributed(
            tmp_path / "control" / "final.json",
            training,
            [ActivationBatch(np.ones((2, 2), np.float32))],
            np.zeros(2, np.float32),
            max_tokens=2,
            checkpoint_payload_root=payload_root,
        )


def test_orbax_interrupted_training_matches_uninterrupted_on_same_mesh(
    tmp_path: Path,
) -> None:
    pytest.importorskip("orbax.checkpoint")
    sae = SAEConfig(
        "B",
        hidden_dim=2,
        num_experts=2,
        top_routed=1,
        bank_width=2,
        k=1,
        parameter_dtype="float32",
        warmup_steps=1,
        total_steps=2,
    )
    training = TrainingConfig(
        sae,
        seed=7,
        total_tokens=4,
        effective_batch_tokens=2,
        microbatch_tokens=1,
        checkpoint_every_tokens=2,
        health_gate_tokens=2,
        live_feature_fraction=0.0,
        loss_ratio_threshold=2.0,
    )
    distributed = DistributedConfig(microbatches_per_update=2)
    mesh = create_mesh(distributed)
    state = init_distributed_state(jax.random.PRNGKey(7), mesh, sae, distributed)
    state = state._replace(
        params=state.params._replace(bias=10.0 * jnp.ones_like(state.params.bias))
    )
    source = ActivationBatch(
        np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 1.0]], np.float32)
    )
    uninterrupted = run_distributed_training(
        training,
        [source],
        np.zeros(2, np.float32),
        mesh,
        state=state,
        distributed=distributed,
        require_health_pass=False,
    )
    identity = {"capture_manifest_sha256": "a" * 64}
    partial = run_distributed_training(
        training,
        [source],
        np.zeros(2, np.float32),
        mesh,
        state=state,
        distributed=distributed,
        checkpoint_dir=tmp_path,
        stop_after_tokens=2,
        require_health_pass=False,
        checkpoint_identity=identity,
    )
    template = init_distributed_state(jax.random.PRNGKey(7), mesh, sae, distributed)
    restored, timing = restore_distributed_checkpoint(
        tmp_path / "latest.json",
        template,
        mesh,
        training,
        checkpoint_identity=identity,
    )
    assert timing["capture_manifest_sha256"] == "a" * 64

    resumed = run_distributed_training(
        training,
        [source],
        np.zeros(2, np.float32),
        mesh,
        state=restored,
        distributed=distributed,
        checkpoint_dir=tmp_path,
        require_health_pass=False,
        prior_optimizer_wall_seconds=partial.optimizer_wall_seconds,
        prior_compile_seconds=partial.compile_seconds,
        prior_checkpoint_seconds=partial.checkpoint_seconds,
        checkpoint_identity=identity,
    )
    for actual, expected in zip(
        jax.tree_util.tree_leaves(resumed.state),
        jax.tree_util.tree_leaves(uninterrupted.state),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)
    payload_directories = list(
        (tmp_path / "checkpoint_payloads" / "runs").glob("*/*")
    )
    assert len(payload_directories) == 1
    assert "tokens-000000000004" in payload_directories[0].name

    # A valid latest pointer is authoritative even when the stage-level
    # run.json was not updated before a preemption.
    stale_run_json_resume = train_distributed(
        training,
        [source],
        np.zeros(2, np.float32),
        checkpoint_dir=tmp_path,
        checkpoint_identity=identity,
        prior_run_timing={"tokens_seen": 0},
    )
    for actual, expected in zip(
        jax.tree_util.tree_leaves(stale_run_json_resume.state),
        jax.tree_util.tree_leaves(uninterrupted.state),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)
