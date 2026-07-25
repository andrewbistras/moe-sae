from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

import moe_sae.distributed as distributed_module
from moe_sae.checkpointing import (
    checkpoint_payload_uri,
    final_pointer_from_latest,
    make_pointer,
    publish_pointer,
)
from moe_sae.distributed import (
    DistributedConfig,
    _validate_exact_distributed_l0,
    create_mesh,
    evaluate_distributed,
    init_distributed_state,
    precompute_distributed_a_capacity,
    replicate_process_local_array,
    restore_distributed_checkpoint,
    run_distributed_training,
    save_distributed_checkpoint,
    train_distributed,
)
from moe_sae.pipeline import ActivationBatch, TrainingConfig
from moe_sae.metrics_journal import JournalIdentity, MetricsJournal
from moe_sae.sae import SAEConfig, init_train_state


def test_default_mesh_is_process_by_local_devices() -> None:
    mesh = create_mesh()
    assert mesh.shape["data"] == jax.process_count()
    assert mesh.shape["feature"] == jax.local_device_count()
    assert np.asarray(mesh.devices).shape == (
        jax.process_count(),
        jax.local_device_count(),
    )


def test_replicated_array_uses_process_local_global_assembly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mesh = create_mesh()
    real_assemble = jax.make_array_from_process_local_data
    calls: list[tuple[NamedSharding, tuple[int, ...], tuple[int, ...] | None]] = []

    def record_assemble(sharding, local_data, global_shape=None):
        assert isinstance(local_data, np.ndarray)
        calls.append((sharding, tuple(local_data.shape), global_shape))
        return real_assemble(
            sharding,
            local_data,
            global_shape=global_shape,
        )

    monkeypatch.setattr(
        jax,
        "make_array_from_process_local_data",
        record_assemble,
    )
    value = np.asarray([1.25, -0.5, 3.0], np.float32)
    actual = replicate_process_local_array(value, mesh)

    assert calls == [
        (
            NamedSharding(mesh, P()),
            value.shape,
            value.shape,
        )
    ]
    assert actual.sharding == NamedSharding(mesh, P())
    assert actual.is_fully_replicated
    np.testing.assert_array_equal(np.asarray(actual), value)


@pytest.mark.parametrize("typed_key", [False, True])
def test_distributed_initialization_assembles_a_replicated_process_local_key(
    monkeypatch: pytest.MonkeyPatch,
    typed_key: bool,
) -> None:
    mesh = create_mesh()
    config = SAEConfig(
        "B",
        hidden_dim=3,
        num_experts=2,
        top_routed=1,
        bank_width=2,
        k=1,
        parameter_dtype="float32",
        warmup_steps=0,
        total_steps=1,
    )
    key = jax.random.key(7) if typed_key else jax.random.PRNGKey(7)
    observed: list[np.ndarray] = []
    real_replicate = distributed_module.replicate_process_local_array

    def record_replicate(value, selected_mesh):
        observed.append(value)
        return real_replicate(value, selected_mesh)

    monkeypatch.setattr(
        distributed_module,
        "replicate_process_local_array",
        record_replicate,
    )
    actual = init_distributed_state(key, mesh, config)
    repeated = init_distributed_state(key, mesh, config)
    expected = init_train_state(key, config)

    assert len(observed) == 2
    np.testing.assert_array_equal(observed[0], jax.random.key_data(key))
    np.testing.assert_array_equal(observed[1], jax.random.key_data(key))
    assert observed[0].dtype == np.uint32
    assert observed[1].dtype == np.uint32
    for actual_leaf, repeated_leaf, expected_leaf in zip(
        jax.tree_util.tree_leaves(actual),
        jax.tree_util.tree_leaves(repeated),
        jax.tree_util.tree_leaves(expected),
        strict=True,
    ):
        np.testing.assert_array_equal(
            np.asarray(actual_leaf),
            np.asarray(repeated_leaf),
        )
        np.testing.assert_allclose(
            np.asarray(actual_leaf),
            np.asarray(expected_leaf),
            rtol=2e-7,
            atol=1e-7,
            equal_nan=True,
        )


def test_legacy_and_typed_keys_have_identical_distributed_initialization() -> None:
    mesh = create_mesh()
    config = SAEConfig(
        "B",
        hidden_dim=3,
        num_experts=2,
        top_routed=1,
        bank_width=2,
        k=1,
        parameter_dtype="float32",
        warmup_steps=0,
        total_steps=1,
    )
    legacy = init_distributed_state(jax.random.PRNGKey(11), mesh, config)
    typed = init_distributed_state(jax.random.key(11), mesh, config)

    for legacy_leaf, typed_leaf in zip(
        jax.tree_util.tree_leaves(legacy),
        jax.tree_util.tree_leaves(typed),
        strict=True,
    ):
        np.testing.assert_array_equal(
            np.asarray(legacy_leaf),
            np.asarray(typed_leaf),
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


def test_exact_distributed_l0_uses_integer_counts_not_tpu_float_ratio() -> None:
    # This was the scalar result observed on the v4-32 final production batch.
    # It lies outside the former 1e-6 tolerance despite exact integer counts.
    observed_tpu_ratio = 63.999996185302734
    assert not math.isclose(observed_tpu_ratio, 64.0, rel_tol=0.0, abs_tol=1e-6)

    _validate_exact_distributed_l0(
        l0_sum=794_624,
        valid_tokens=12_416,
        k=64,
    )


@pytest.mark.parametrize(
    ("l0_sum", "valid_tokens", "message"),
    [
        (794_623, 12_416, "selected=794623"),
        (0, 0, "zero scientific tokens"),
    ],
)
def test_exact_distributed_l0_rejects_real_count_mismatches(
    l0_sum: int, valid_tokens: int, message: str
) -> None:
    with pytest.raises(RuntimeError, match=message):
        _validate_exact_distributed_l0(
            l0_sum=l0_sum,
            valid_tokens=valid_tokens,
            k=64,
        )


def test_checkpoint_control_visibility_retries_enoent_and_stale_contents() -> None:
    expected = {"tokens_seen": 8, "kind": "latest"}
    observations: list[object] = [
        FileNotFoundError("negative lookup"),
        json.dumps({"tokens_seen": 4, "kind": "latest"}),
        json.dumps(expected),
    ]
    sleeps: list[float] = []

    class DelayedControlPath:
        def read_text(self, *, encoding: str) -> str:
            assert encoding == "utf-8"
            observation = observations.pop(0)
            if isinstance(observation, Exception):
                raise observation
            return str(observation)

        def __str__(self) -> str:
            return "/shared/latest.json"

    distributed_module._load_exact_checkpoint_control_with_retry(
        DelayedControlPath(),  # type: ignore[arg-type]
        expected,
        timeout_seconds=1.0,
        poll_seconds=0.1,
        clock=lambda: 0.0,
        sleep=sleeps.append,
    )
    assert not observations
    assert sleeps == [0.1, 0.1]


def test_pointer_publish_error_reaches_collective_decision_before_raise(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pointer = make_pointer(
        kind="latest",
        payload_uri=str(tmp_path / "payloads" / "runs" / "ignored"),
        step=1,
        tokens_seen=2,
        attempt_id="a" * 32,
        run_identity={},
    )
    decisions: list[bool] = []

    monkeypatch.setattr(jax, "process_index", lambda: 0)
    monkeypatch.setattr(
        distributed_module,
        "_checkpoint_collective_failed",
        lambda failed: decisions.append(failed) or failed,
    )
    monkeypatch.setattr(
        distributed_module,
        "publish_pointer",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("write failed")),
    )

    with pytest.raises(RuntimeError, match="pointer publication"):
        distributed_module._publish_checkpoint_pointer(tmp_path, pointer)
    # Preparation succeeded; publication failed.  The local exception was
    # converted into a collective decision before it was raised.
    assert decisions == [False, True]


def test_pointer_peer_uses_remote_failure_decision_without_visibility_branch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pointer = make_pointer(
        kind="latest",
        payload_uri=str(tmp_path / "payloads" / "runs" / "ignored"),
        step=1,
        tokens_seen=2,
        attempt_id="b" * 32,
        run_identity={},
    )
    remote_decisions = iter((False, True))
    visibility_called = False

    monkeypatch.setattr(jax, "process_index", lambda: 1)
    monkeypatch.setattr(
        distributed_module,
        "_checkpoint_collective_failed",
        lambda _failed: next(remote_decisions),
    )

    def visibility(*_args, **_kwargs):
        nonlocal visibility_called
        visibility_called = True

    monkeypatch.setattr(
        distributed_module, "_load_exact_checkpoint_control_with_retry", visibility
    )
    with pytest.raises(RuntimeError, match="pointer publication"):
        distributed_module._publish_checkpoint_pointer(tmp_path, pointer)
    assert not visibility_called


def test_metrics_journal_peer_receives_coordinator_failure_before_next_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    callback_called = False
    monkeypatch.setattr(jax, "process_index", lambda: 1)
    monkeypatch.setattr(
        distributed_module,
        "_checkpoint_collective_failed",
        lambda _failed: True,
    )

    def callback(_metric):
        nonlocal callback_called
        callback_called = True

    with pytest.raises(RuntimeError, match="metrics journal append failed"):
        distributed_module._append_training_journal_metric(
            None,
            object(),  # type: ignore[arg-type]
            step=1,
            tokens_seen=2,
            optimizer_wall_seconds=1.0,
            metric_callback=callback,
        )
    assert not callback_called


def test_pointer_visibility_error_is_collectively_decided_before_raise(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pointer = make_pointer(
        kind="latest",
        payload_uri=str(tmp_path / "payloads" / "runs" / "ignored"),
        step=1,
        tokens_seen=2,
        attempt_id="c" * 32,
        run_identity={},
    )
    decisions: list[bool] = []
    monkeypatch.setattr(jax, "process_index", lambda: 1)
    monkeypatch.setattr(
        distributed_module,
        "_checkpoint_collective_failed",
        lambda failed: decisions.append(failed) or failed,
    )
    monkeypatch.setattr(
        distributed_module,
        "_load_exact_checkpoint_control_with_retry",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("visibility timed out")
        ),
    )

    with pytest.raises(RuntimeError, match="pointer visibility"):
        distributed_module._publish_checkpoint_pointer(tmp_path, pointer)
    assert decisions == [False, False, True]


def test_timing_publish_error_reaches_collective_decision_before_raise(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    decisions: list[bool] = []
    monkeypatch.setattr(jax, "process_index", lambda: 0)
    monkeypatch.setattr(
        distributed_module,
        "_checkpoint_collective_failed",
        lambda failed: decisions.append(failed) or failed,
    )
    monkeypatch.setattr(
        distributed_module,
        "atomic_write_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("write failed")),
    )

    with pytest.raises(RuntimeError, match="timing publication"):
        distributed_module._publish_checkpoint_timing(
            tmp_path,
            checkpoint_identity={"run": "one"},
            tokens_seen=2,
            steps=1,
            optimizer_wall_seconds=1.0,
            compile_seconds=2.0,
            checkpoint_seconds=3.0,
        )
    assert decisions == [False, True]


def test_noncoordinator_checkpoint_read_uses_broadcast_absence_not_local_nfs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from jax.experimental import multihost_utils

    monkeypatch.setattr(jax, "process_index", lambda: 2)
    monkeypatch.setattr(jax, "process_count", lambda: 4)
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda *_args, **_kwargs: pytest.fail("noncoordinator read shared NFS"),
    )
    monkeypatch.setattr(
        distributed_module,
        "_checkpoint_phase_decision",
        lambda _operation, error: pytest.fail(str(error)) if error else None,
    )

    def broadcast(value, *, is_source: bool):
        assert not is_source
        assert np.asarray(value).shape == (1,)
        return np.asarray([-1], dtype=np.int32)

    monkeypatch.setattr(multihost_utils, "broadcast_one_to_all", broadcast)
    assert (
        distributed_module._coordinator_read_checkpoint_json_if_exists(
            tmp_path / "latest.json", operation="latest pointer"
        )
        is None
    )


def test_checkpoint_timing_resume_uses_coordinator_value_without_exists_branch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = {
        "run": "one",
        "tokens_seen": 8,
        "optimizer_wall_seconds": 1.0,
        "compile_seconds": 2.0,
        "checkpoint_seconds": 3.0,
    }
    reads: list[Path] = []

    def read(path: Path, *, operation: str):
        assert operation == "checkpoint timing"
        reads.append(path)
        return payload

    monkeypatch.setattr(
        distributed_module, "_coordinator_read_checkpoint_json_if_exists", read
    )
    assert distributed_module._load_checkpoint_timing(
        tmp_path,
        checkpoint_identity={"run": "one"},
        restored_tokens=8,
    ) == payload
    assert reads == [tmp_path / "checkpoint_timing.json"]


def test_evaluation_restores_training_capacity_then_uses_eval_capacity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sae = SAEConfig(
        "A",
        hidden_dim=2,
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
        seed=1,
        total_tokens=2,
        effective_batch_tokens=2,
        microbatch_tokens=1,
        checkpoint_every_tokens=2,
        health_gate_tokens=2,
        a_capacity=8,
    )
    observed: dict[str, int] = {}

    def restore(pointer, template, mesh, restored_config, **kwargs):
        del pointer, mesh, kwargs
        observed["restore"] = restored_config.a_capacity
        return template._replace(tokens_seen=jnp.asarray(2), step=jnp.asarray(1)), {}

    def evaluate(params, source, mean, sae_config, mesh, **kwargs):
        del params, source, mean, sae_config, mesh
        observed["evaluate"] = kwargs["a_capacity"]
        return SimpleNamespace(tokens=2)

    monkeypatch.setattr(distributed_module, "restore_distributed_checkpoint", restore)
    monkeypatch.setattr(distributed_module, "distributed_evaluate", evaluate)
    result = evaluate_distributed(
        tmp_path / "final.json",
        training,
        [ActivationBatch(np.ones((2, 2), np.float32), np.zeros((2, 2), np.int32))],
        np.zeros(2, np.float32),
        max_tokens=2,
        evaluation_a_capacity=24,
    )
    assert result.tokens == 2
    assert observed == {"restore": 8, "evaluate": 24}


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


def test_two_dimensional_partial_final_batch_preserves_exact_l0_in_subprocess(
    tmp_path: Path,
) -> None:
    """The masked production tail must not fail the exact-L0 invariant."""

    script = r"""
import jax
import jax.numpy as jnp
import numpy as np
import sys
from pathlib import Path

import moe_sae.distributed as distributed_module
from moe_sae.distributed import (
    DistributedConfig,
    create_mesh,
    run_distributed_training,
    shard_train_state,
)
from moe_sae.metrics_journal import JournalIdentity, MetricsJournal
from moe_sae.pipeline import ActivationBatch, TrainingConfig
from moe_sae.sae import SAEConfig, init_train_state


def save_checkpoint_without_orbax(path, *_args, **_kwargs):
    # The base test environment deliberately excludes the TPU/Orbax extra.  We
    # still exercise the complete journal -> latest -> final control-plane path.
    payload = Path(str(path))
    payload.mkdir(parents=True)
    (payload / "commit_success.txt").write_text("ok", encoding="utf-8")


distributed_module.save_distributed_checkpoint = save_checkpoint_without_orbax

total_tokens = 16_384 + 12_416
root = Path(sys.argv[1])
x = np.ones((total_tokens, 4), np.float32)
route_patterns = np.asarray([[0, 1], [2, 3]], np.int32)
routes = route_patterns[np.arange(total_tokens) % 2]
distributed = DistributedConfig(
    data_axis_size=2, feature_axis_size=2, microbatches_per_update=4
)
mesh = create_mesh(distributed)
sae = SAEConfig(
    "A",
    hidden_dim=4,
    num_experts=4,
    top_routed=2,
    bank_width=64,
    k=64,
    parameter_dtype="float32",
    warmup_steps=1,
    total_steps=2,
)
training = TrainingConfig(
    sae,
    seed=0,
    total_tokens=total_tokens,
    effective_batch_tokens=16_384,
    microbatch_tokens=4_096,
    checkpoint_every_tokens=total_tokens,
    health_gate_tokens=16_384,
    live_feature_fraction=0,
    loss_ratio_threshold=10,
    a_capacity=4_096,
)
state = init_train_state(jax.random.PRNGKey(0), sae)
state = state._replace(
    params=state.params._replace(bias=jnp.ones_like(state.params.bias))
)
result = run_distributed_training(
    training,
    [ActivationBatch(x, routes)],
    np.zeros(4, np.float32),
    mesh,
    state=shard_train_state(state, mesh, sae, distributed),
    distributed=distributed,
    a_capacity=4_096,
    checkpoint_dir=root / "checkpoints",
    checkpoint_identity={"capture_manifest_sha256": "a" * 64},
    metrics_journal_path=root / "metrics.jsonl",
    metrics_journal_identity=JournalIdentity("b" * 32, "A", 0),
    require_health_pass=False,
)
assert int(result.state.step) == 2
assert int(result.state.tokens_seen) == total_tokens
assert [int(metric.valid_tokens) for metric in result.metrics] == [16_384, 12_416]
assert [float(metric.l0) for metric in result.metrics] == [64.0, 64.0]
assert result.metrics_journal is not None
assert result.metrics_journal["complete_prefix_exact_l0"] is True
assert result.metrics_journal["complete_prefix_topk_valid"] is True
assert result.metrics_journal["binding"]["last_tokens_seen"] == total_tokens
assert result.final_checkpoint is not None
assert Path(result.final_checkpoint).is_file()
records = MetricsJournal(
    root / "metrics.jsonl", JournalIdentity("b" * 32, "A", 0)
).records()
assert records[-1]["valid_tokens"] == 12_416
assert records[-1]["l0"] == 64.0
"""
    environment = dict(os.environ)
    environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
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
    journal_path = tmp_path / "authoritative-metrics.jsonl"
    journal_identity = JournalIdentity("b" * 32, "B", 7)
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
        metrics_journal_path=journal_path,
        metrics_journal_identity=journal_identity,
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
    # Model the crash window in which an update row was durable but its
    # checkpoint pointer was not. Resume must discard only this ahead suffix.
    MetricsJournal(journal_path, journal_identity).append(
        {
            "step": 2,
            "tokens_seen": 4,
            "valid_tokens": 2,
            "l0": 1.0,
            "topk_valid": True,
            "optimizer_wall_seconds": partial.optimizer_wall_seconds + 999.0,
        }
    )

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
        metrics_journal_path=journal_path,
        metrics_journal_identity=journal_identity,
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
    assert resumed.metrics_journal is not None
    assert resumed.metrics_journal["binding"]["row_count"] == 2
    assert resumed.metrics_journal["binding"]["last_step"] == 2
    assert resumed.metrics_journal["binding"]["last_tokens_seen"] == 4
    journal_records = MetricsJournal(journal_path, journal_identity).records()
    assert [
        (record["step"], record["tokens_seen"])
        for record in journal_records
    ] == [(1, 2), (2, 4)]
    assert {
        "loss",
        "sse",
        "sst",
        "l0",
        "valid_tokens",
        "topk_valid",
        "grad_norm",
        "clip_coefficient",
        "learning_rate",
        "live_fraction",
        "optimizer_wall_seconds",
    } <= journal_records[-1].keys()

    # A valid latest pointer is authoritative even when the stage-level
    # run.json was not updated before a preemption.
    stale_run_json_resume = train_distributed(
        training,
        [source],
        np.zeros(2, np.float32),
        checkpoint_dir=tmp_path,
        checkpoint_identity=identity,
        prior_run_timing={"tokens_seen": 0},
        metrics_journal_path=journal_path,
        metrics_journal_identity=journal_identity,
    )
    for actual, expected in zip(
        jax.tree_util.tree_leaves(stale_run_json_resume.state),
        jax.tree_util.tree_leaves(uninterrupted.state),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)


def test_metrics_journal_append_failure_prevents_checkpoint_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sae = SAEConfig(
        "B",
        hidden_dim=2,
        num_experts=2,
        top_routed=1,
        bank_width=2,
        k=1,
        parameter_dtype="float32",
        warmup_steps=0,
        total_steps=1,
    )
    training = TrainingConfig(
        sae,
        seed=3,
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
    state = init_distributed_state(jax.random.PRNGKey(3), mesh, sae, distributed)
    state = state._replace(
        params=state.params._replace(bias=10.0 * jnp.ones_like(state.params.bias))
    )
    journal_path = tmp_path / "training.jsonl"
    journal_identity = JournalIdentity("c" * 32, "B", 3)

    def fail_append(_self, _record):
        raise OSError("injected journal failure")

    monkeypatch.setattr(MetricsJournal, "append", fail_append)
    with pytest.raises(RuntimeError, match="metrics journal append failed"):
        run_distributed_training(
            training,
            [ActivationBatch(np.ones((2, 2), np.float32))],
            np.zeros(2, np.float32),
            mesh,
            state=state,
            distributed=distributed,
            checkpoint_dir=tmp_path / "checkpoints",
            checkpoint_identity={"capture_manifest_sha256": "d" * 64},
            metrics_journal_path=journal_path,
            metrics_journal_identity=journal_identity,
            require_health_pass=False,
        )

    assert not (tmp_path / "checkpoints" / "latest.json").exists()
    assert MetricsJournal(journal_path, journal_identity).snapshot().row_count == 0


def test_resumed_health_uses_the_entire_metrics_journal_prefix(
    tmp_path: Path,
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
        total_steps=2,
    )
    training = TrainingConfig(
        sae,
        seed=4,
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
    state = init_distributed_state(jax.random.PRNGKey(4), mesh, sae, distributed)
    state = state._replace(
        params=state.params._replace(bias=10.0 * jnp.ones_like(state.params.bias))
    )
    source = ActivationBatch(
        np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 1.0]], np.float32)
    )
    partial = run_distributed_training(
        training,
        [source],
        np.zeros(2, np.float32),
        mesh,
        state=state,
        distributed=distributed,
        stop_after_tokens=2,
        require_health_pass=False,
    )
    journal_path = tmp_path / "training.jsonl"
    journal_identity = JournalIdentity("e" * 32, "B", 4)
    journal = MetricsJournal(journal_path, journal_identity)
    journal.append(
        {
            "step": 1,
            "tokens_seen": 2,
            "valid_tokens": 2,
            "l0": 1.0,
            "topk_valid": False,
            "optimizer_wall_seconds": partial.optimizer_wall_seconds / 2.0,
        }
    )

    resumed = run_distributed_training(
        training,
        [source],
        np.zeros(2, np.float32),
        mesh,
        state=partial.state,
        distributed=distributed,
        prior_optimizer_wall_seconds=partial.optimizer_wall_seconds,
        metrics_journal_path=journal_path,
        metrics_journal_identity=journal_identity,
        require_health_pass=False,
    )

    assert resumed.metrics_journal is not None
    assert resumed.metrics_journal["complete_prefix_topk_valid"] is False
    assert resumed.health is not None
    assert resumed.health.topk_valid is False
    assert resumed.health.passed is False
