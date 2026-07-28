import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

import moe_sae.data as data_module
from moe_sae.runtime import (
    HARVEST_MESHES,
    BenchmarkObservation,
    _staged_path_accessible,
    build_maxtext_argv,
    choose_harvest_benchmark,
    compact_capture,
    create_maxtext_runtime,
    environment_fingerprint,
    estimate_useful_prefix_flops,
    immutable_range_id,
    scan_token_shaped_secrets,
    scientific_token_mask,
    token_quota,
    updates_for_tokens,
    accelerator_runtime_status,
    require_accelerator_runtime,
    RuntimeDependencyError,
)


def _observation(mesh_index: int, *, mfu: float, tokens: int) -> BenchmarkObservation:
    elapsed = 2.0
    peak = 4.4e15
    return BenchmarkObservation(
        mesh=HARVEST_MESHES[mesh_index],
        per_device_sequence_batch=1,
        synchronized_steps=200,
        elapsed_seconds=elapsed,
        device_elapsed_seconds=elapsed,
        persistence_seconds=0.0,
        useful_flops=mfu * peak * elapsed,
        output_tokens=tokens,
        persisted_tokens=tokens,
        max_writer_backlog=1,
    )


def test_benchmark_selection_requires_gate_and_maximizes_persisted_goodput():
    too_low = _observation(0, mfu=0.49, tokens=99_000)
    slower = _observation(1, mfu=0.55, tokens=100_000)
    faster = _observation(2, mfu=0.51, tokens=120_000)
    assert choose_harvest_benchmark([too_low, slower, faster]) == faster
    with pytest.raises(RuntimeError, match="no harvester benchmark"):
        choose_harvest_benchmark([too_low])


def test_v6e_16_runtime_requires_exact_four_by_four_topology(
    monkeypatch,
) -> None:
    devices = [
        SimpleNamespace(
            platform="tpu",
            device_kind="TPU v6 lite",
            process_index=index // 4,
            id=index,
        )
        for index in range(16)
    ]
    jax = ModuleType("jax")
    jax.devices = lambda: devices
    jax.process_count = lambda: 4
    jax.local_device_count = lambda: 4
    jax.distributed = SimpleNamespace(
        is_initialized=lambda: True,
        initialize=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "jax", jax)

    passed, detail = accelerator_runtime_status("v6e-16")
    assert passed is True
    assert "process_count=4" in detail
    require_accelerator_runtime(
        "v6e-16",
        peak_flops_per_second=14_688_000_000_000_000,
    )

    jax.process_count = lambda: 2
    passed, _ = accelerator_runtime_status("v6e-16")
    assert passed is False


def test_v6e_runtime_rejects_v4_peak_before_authoritative_work(
    monkeypatch,
) -> None:
    jax = ModuleType("jax")
    jax.distributed = SimpleNamespace(
        is_initialized=lambda: True,
        initialize=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "jax", jax)

    with pytest.raises(RuntimeDependencyError, match="14688000000000000"):
        require_accelerator_runtime(
            "v6e-16",
            peak_flops_per_second=4_400_000_000_000_000,
        )


def test_benchmark_uses_device_time_for_mfu_and_total_time_for_persistence():
    observation = BenchmarkObservation(
        mesh=HARVEST_MESHES[0],
        per_device_sequence_batch=1,
        synchronized_steps=200,
        elapsed_seconds=10.0,
        device_elapsed_seconds=2.0,
        persistence_seconds=7.0,
        useful_flops=4.4e15,
        output_tokens=1_000,
        persisted_tokens=1_000,
    )
    assert observation.useful_mfu == pytest.approx(0.5)
    assert observation.device_tokens_per_second == pytest.approx(500.0)
    assert observation.persisted_tokens_per_second == pytest.approx(100.0)
    assert observation.persistence_fraction == pytest.approx(0.7)


def test_prefix_flops_are_explicit_and_scale_with_batch():
    one = estimate_useful_prefix_flops(batch_sequences=1)
    four = estimate_useful_prefix_flops(batch_sequences=4)
    assert one > 0
    assert four == 4 * one


def test_maxtext_argv_locks_scientific_capture_settings():
    argv = build_maxtext_argv(
        checkpoint_path="gs://bucket/checkpoint",
        grain_train_files="gs://bucket/c4/train-*",
        grain_eval_files="gs://bucket/c4/validation-*",
        mesh=HARVEST_MESHES[0],
        per_device_sequence_batch=2,
        base_config_path="base.yml",
    )
    joined = "\n".join(argv)
    assert argv[0] == "base.yml"
    assert "model_name=gemma4-26b" in joined
    assert "tokenizer_path=google/gemma-4-26B-A4B" in joined
    assert "max_target_length=1024" in joined
    assert "scan_layers=false" in joined
    assert "use_multimodal=false" in joined
    assert "load_balance_loss_weight=0" in joined
    assert "use_ragged_sort=true" in joined
    assert "eval_per_device_batch_size=2" in joined
    assert "ici_expert_parallelism=16" in joined


def test_packed_runtime_forwards_exact_qualified_b128_contract(
    monkeypatch,
) -> None:
    config = SimpleNamespace(
        mesh_axes=("expert", "fsdp", "tensor"),
        per_device_batch_size=8,
        eval_per_device_batch_size=8,
        global_batch_size_to_load=128,
        global_batch_size_to_load_eval=128,
    )
    mesh = object()
    model = object()

    jax = ModuleType("jax")
    jax.sharding = SimpleNamespace(Mesh=lambda _devices, _axes: mesh)
    common_types = ModuleType("maxtext.common.common_types")
    common_types.MODEL_MODE_TRAIN = "train"
    pyconfig = SimpleNamespace(initialize=lambda _argv: config)
    configs = ModuleType("maxtext.configs")
    configs.pyconfig = pyconfig
    input_interface = SimpleNamespace(
        create_data_iterator=lambda _config, _mesh: (None, None)
    )
    input_pipeline = ModuleType("maxtext.input_pipeline")
    input_pipeline.input_pipeline_interface = input_interface
    maxtext_utils = SimpleNamespace(create_device_mesh=lambda _config: object())
    model_creation_utils = SimpleNamespace(
        from_pretrained=lambda _config, **_kwargs: model
    )
    utils = ModuleType("maxtext.utils")
    utils.maxtext_utils = maxtext_utils
    utils.model_creation_utils = model_creation_utils
    packages = {
        "jax": jax,
        "maxtext": ModuleType("maxtext"),
        "maxtext.common": ModuleType("maxtext.common"),
        "maxtext.common.common_types": common_types,
        "maxtext.configs": configs,
        "maxtext.input_pipeline": input_pipeline,
        "maxtext.utils": utils,
    }
    for name, value in packages.items():
        monkeypatch.setitem(sys.modules, name, value)

    calls = []

    def build(files, **kwargs):
        calls.append((tuple(files), kwargs))
        return object()

    monkeypatch.setattr(data_module, "build_maxtext_packed_iterator", build)
    runtime = create_maxtext_runtime(
        ("base.yml",),
        packed_train_files=("train-0",),
        packed_eval_files=("eval-0",),
        packed_lane_prefix_replay_records=4,
    )

    assert runtime.model is model
    assert [call[0] for call in calls] == [("train-0",), ("eval-0",)]
    assert [call[1]["global_batch_size"] for call in calls] == [128, 128]
    assert [
        call[1]["qualified_per_device_sequence_batch"] for call in calls
    ] == [8, 8]
    assert [call[1]["replay_lane_prefix_records"] for call in calls] == [4, 4]


def test_exact_host_quota_and_immutable_ids():
    quotas = [token_quota(10, 3, index) for index in range(3)]
    assert quotas == [4, 3, 3]
    assert sum(quotas) == 10
    assert immutable_range_id("train", 2, 7) == "train-p00002-b000000000007"


def test_scientific_mask_and_common_compaction():
    token_ids = np.array([[2, 10, 11, 1], [0, 12, 13, 14]], dtype=np.int32)
    segmentation = np.array([[1, 1, 1, 1], [0, 2, 2, 0]], dtype=np.int32)
    mask = scientific_token_mask(token_ids, segmentation, excluded_token_ids=(0, 1, 2))
    assert mask.tolist() == [[False, True, True, False], [False, True, True, False]]
    activation = np.arange(8 * 3).reshape(2, 4, 3)
    expert_ids = np.arange(8 * 8).reshape(2, 4, 8)
    compact = compact_capture(
        {"activation": activation, "expert_ids": expert_ids}, mask, limit=3
    )
    assert compact["activation"].shape == (3, 3)
    assert compact["expert_ids"].shape == (3, 8)
    np.testing.assert_array_equal(
        compact["activation"], activation.reshape(-1, 3)[[1, 2, 5]]
    )


def test_token_shaped_secret_scan_reports_path_only(tmp_path: Path):
    clean = tmp_path / "clean.txt"
    clean.write_text("no credentials", encoding="utf-8")
    assert scan_token_shaped_secrets(tmp_path) == []
    suspect = tmp_path / "suspect.txt"
    suspect.write_text("hf_" + "a" * 24, encoding="utf-8")
    assert scan_token_shaped_secrets(tmp_path) == [Path("suspect.txt")]


def test_staged_local_path_probe_requires_a_real_match(tmp_path: Path):
    shard = tmp_path / "train-000.array_record"
    shard.write_bytes(b"staged")
    assert _staged_path_accessible(str(tmp_path / "*.array_record"))[0]
    passed, detail = _staged_path_accessible(str(tmp_path / "missing-*"))
    assert not passed and "matched no files" in detail


def test_update_plan_masks_only_the_final_padding():
    assert updates_for_tokens(50_000_000, 16_384) == (3_052, 3_968)


def test_environment_fingerprint_is_secret_free_and_hashed(tmp_path: Path):
    fingerprint = environment_fingerprint(tmp_path)
    assert len(fingerprint["sha256"]) == 64
    assert "packages" in fingerprint and "devices" in fingerprint
    assert "environment" not in fingerprint
