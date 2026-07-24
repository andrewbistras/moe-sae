from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import jax
import ml_dtypes
import numpy as np
import pytest

from moe_sae.r1.maxtext_prefix_runtime import (
    CompiledPrefix,
    MaxTextPrefixRuntimeError,
    PREFIX_MESH_CANDIDATES,
    PrefixInputBatch,
    PrefixMeshCandidate,
    PrefixRuntimeSettings,
    _expected_partition_specs,
    assert_real_v6e64_runtime,
    compile_prefix,
    create_device_synthetic_prefix_leaves,
    create_synthetic_prefix_batch,
    initialize_v6e_distributed_runtime,
    prune_output_only_parameters,
    split_restored_prefix_leaves,
    validate_initialized_config,
)
from moe_sae.r1.jax_prefix_restore import LeafSharding
from moe_sae.r1.prefix_batchsplit_contract import (
    build_pinned_prefix_performance_tree_plan,
)


def _config(mesh: PrefixMeshCandidate) -> SimpleNamespace:
    return SimpleNamespace(
        num_decoder_layers=30,
        first_num_dense_layers=3,
        max_target_length=1_024,
        batch_split_factor=2,
        scan_layers=True,
        pure_nnx_decoder=True,
        use_batch_split_schedule=True,
        enable_dropout=False,
        dropout_rate=0.0,
        load_balance_loss_weight=0.0,
        routed_bias_update_rate=0.0,
        num_experts=256,
        num_experts_per_tok=8,
        n_routing_groups=8,
        topk_routing_group=4,
        shared_experts=1,
        routed_scaling_factor=2.5,
        routed_score_func="sigmoid",
        routed_bias=True,
        float32_gate_logits=True,
        sparse_matmul=True,
        mlp_bias=False,
        prefuse_moe_weights=False,
        shard_exp_on_fsdp=False,
        use_2d_fsdp_sharding=False,
        use_ring_of_experts=False,
        use_ragged_sort=True,
        ragged_buffer_factor=-1.0,
        num_moe_emb_chunks=0,
        num_moe_token_chunks=1,
        use_random_routing=False,
        float32_weight_sum=True,
        record_internal_nn_metrics=0,
        check_vma=False,
        capacity_factor=-1.0,
        use_pathways=False,
        using_pipeline_parallelism=False,
        logits_via_embedding=True,
        allow_split_physical_axes=mesh.allow_split_physical_axes,
        dtype=SimpleNamespace(name="bfloat16"),
        weight_dtype=SimpleNamespace(name="bfloat16"),
        quantization="",
        use_qwix_quantization=False,
        ici_data_parallelism=1,
        ici_fsdp_parallelism=mesh.fsdp,
        ici_expert_parallelism=mesh.expert,
        ici_context_parallelism=1,
        ici_tensor_parallelism=1,
        ici_pipeline_parallelism=1,
        global_batch_size_to_load=128,
    )


def test_mesh_ladder_is_exact_and_ordered() -> None:
    assert [
        (item.name, item.expert, item.fsdp) for item in PREFIX_MESH_CANDIDATES
    ] == [
        ("EP16/FSDP4", 16, 4),
        ("EP32/FSDP2", 32, 2),
        ("EP8/FSDP8", 8, 8),
    ]
    with pytest.raises(MaxTextPrefixRuntimeError, match="64"):
        PrefixMeshCandidate("EP8/FSDP4", fsdp=4, expert=8)
    assert [item.allow_split_physical_axes for item in PREFIX_MESH_CANDIDATES] == [
        True,
        True,
        False,
    ]


def test_overrides_define_one_forward_only_b128_program(tmp_path: Path) -> None:
    settings = PrefixRuntimeSettings(
        maxtext_base_yml=tmp_path / "base.yml",
        output_directory=tmp_path / "out",
    )
    values = settings.maxtext_overrides()
    assert values["base_num_decoder_layers"] == 30
    assert values["per_device_batch_size"] == 2.0
    assert values["batch_split_factor"] == 2
    assert values["weight_dtype"] == "bfloat16"
    assert values["enable_dropout"] is False
    assert values["float32_gate_logits"] is True
    assert values["sparse_matmul"] is True
    assert values["mlp_bias"] is False
    assert values["prefuse_moe_weights"] is False
    assert values["shard_exp_on_fsdp"] is False
    assert values["use_2d_fsdp_sharding"] is False
    assert values["use_ring_of_experts"] is False
    assert values["use_ragged_sort"] is True
    assert values["ragged_buffer_factor"] == -1.0
    assert values["num_moe_emb_chunks"] == 0
    assert values["num_moe_token_chunks"] == 1
    assert values["use_random_routing"] is False
    assert values["float32_weight_sum"] is True
    assert values["record_internal_nn_metrics"] == 0
    assert values["check_vma"] is False
    assert values["capacity_factor"] == -1.0
    assert values["use_pathways"] is False
    assert values["ici_expert_parallelism"] == 16
    assert values["ici_fsdp_parallelism"] == 4
    assert values["pipeline_parallel_layers"] == 0
    assert values["enable_checkpointing"] is False
    assert values["allow_split_physical_axes"] is True


def test_ep8_keeps_physical_axes_unsplit(tmp_path: Path) -> None:
    settings = PrefixRuntimeSettings(
        maxtext_base_yml=tmp_path / "base.yml",
        output_directory=tmp_path / "out",
        mesh=PREFIX_MESH_CANDIDATES[2],
    )
    assert settings.maxtext_overrides()["allow_split_physical_axes"] is False


@pytest.mark.parametrize("mesh", PREFIX_MESH_CANDIDATES)
def test_initialized_config_accepts_each_fixed_mesh(
    mesh: PrefixMeshCandidate,
) -> None:
    validate_initialized_config(_config(mesh), mesh)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("num_decoder_layers", 31, "num_decoder_layers"),
        ("global_batch_size_to_load", 64, "global batch"),
        ("dropout_rate", 0.1, "dropout_rate"),
        ("routed_bias_update_rate", 0.001, "routed_bias_update_rate"),
        ("float32_gate_logits", False, "float32_gate_logits"),
        ("sparse_matmul", False, "sparse_matmul"),
        ("mlp_bias", True, "mlp_bias"),
        ("prefuse_moe_weights", True, "prefuse_moe_weights"),
        ("shard_exp_on_fsdp", True, "shard_exp_on_fsdp"),
        ("use_2d_fsdp_sharding", True, "use_2d_fsdp_sharding"),
        ("use_ring_of_experts", True, "use_ring_of_experts"),
        ("use_ragged_sort", False, "use_ragged_sort"),
        ("ragged_buffer_factor", 1.1, "ragged_buffer_factor"),
        ("num_moe_emb_chunks", 2, "num_moe_emb_chunks"),
        ("num_moe_token_chunks", 2, "num_moe_token_chunks"),
        ("use_random_routing", True, "use_random_routing"),
        ("float32_weight_sum", False, "float32_weight_sum"),
        ("record_internal_nn_metrics", 1, "record_internal_nn_metrics"),
        ("check_vma", True, "check_vma"),
        ("capacity_factor", 1.25, "capacity_factor"),
        ("weight_dtype", SimpleNamespace(name="float32"), "weight dtype"),
        ("use_pathways", True, "use_pathways"),
        ("allow_split_physical_axes", False, "allow_split_physical_axes"),
    ],
)
def test_initialized_config_rejects_execution_drift(
    field: str,
    value: object,
    message: str,
) -> None:
    mesh = PREFIX_MESH_CANDIDATES[0]
    config = _config(mesh)
    setattr(config, field, value)
    with pytest.raises(MaxTextPrefixRuntimeError, match=message):
        validate_initialized_config(config, mesh)


def test_pruning_removes_only_final_norm_and_rejects_logits() -> None:
    norm = object()
    model = SimpleNamespace(decoder=SimpleNamespace(decoder_norm=norm))
    assert prune_output_only_parameters(model) is model
    assert not hasattr(model.decoder, "decoder_norm")

    with pytest.raises(MaxTextPrefixRuntimeError, match="logits"):
        prune_output_only_parameters(
            SimpleNamespace(
                decoder=SimpleNamespace(decoder_norm=norm, logits_dense=object())
            )
        )
    with pytest.raises(MaxTextPrefixRuntimeError, match="final decoder norm"):
        prune_output_only_parameters(SimpleNamespace(decoder=SimpleNamespace()))


def _fake_restored_leaves() -> dict[str, np.ndarray]:
    dtype_by_name = {
        "bfloat16": ml_dtypes.bfloat16,
        "float32": np.float32,
    }
    return {
        leaf.leaf_key: np.broadcast_to(
            np.asarray(0, dtype=dtype_by_name[leaf.dtype]),
            leaf.logical_shape,
        )
        for leaf in build_pinned_prefix_performance_tree_plan().leaves
    }


def test_exact_restore_split_matches_nnx_and_target_paths() -> None:
    model, target = split_restored_prefix_leaves(_fake_restored_leaves())
    assert len(model) == 30
    assert model["token_embedder/embedding"].shape == (129_280, 7_168)
    assert (
        model[
            "decoder/moe_layers/DeepSeekMoeBlock_0/MoeBlock_0/gate/bias"
        ].dtype
        == np.float32
    )
    assert target["pre_self_attention_layer_norm"]["scale"].shape == (7_168,)
    assert target["DeepSeekMoeBlock_0"]["MoeBlock_0"]["gate"]["kernel"].shape == (
        7_168,
        256,
    )
    assert "shared_experts" not in target["DeepSeekMoeBlock_0"]


def test_exact_44_leaf_partition_table_matches_target_gather() -> None:
    specs = _expected_partition_specs()
    assert len(specs) == 44
    assert specs["params-token_embedder-embedding"] == (None, "fsdp")
    assert specs[
        "params-decoder-moe_layers-DeepSeekMoeBlock_0-MoeBlock_0-wi_0"
    ] == ("expert", None, "fsdp", None)
    assert specs[
        "target-routed-only-DeepSeekMoeBlock_0-MoeBlock_0-wi_0"
    ] == ("expert", "fsdp", None)
    assert specs[
        "target-routed-only-self_attention-out-kernel"
    ] == (None, None, "fsdp")
    assert specs[
        "target-routed-only-DeepSeekMoeBlock_0-MoeBlock_0-gate-bias"
    ] == (None,)


def test_synthetic_allocator_creates_runtime_arrays_without_host_tree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    leaves = tuple(
        SimpleNamespace(
            leaf_key=f"leaf-{index:02d}",
            logical_shape=(2,),
            dtype="bfloat16" if index != 2 else "float32",
            section="embedding" if index == 0 else "scanned-completed-moe",
            family="router-gate-kernel" if index == 1 else f"family-{index}",
        )
        for index in range(44)
    )
    monkeypatch.setattr(
        "moe_sae.r1.maxtext_prefix_runtime."
        "build_pinned_prefix_performance_tree_plan",
        lambda: SimpleNamespace(leaves=leaves),
    )
    mesh = jax.sharding.Mesh(
        np.asarray(jax.devices()[:1]),
        ("data",),
        axis_types=(jax.sharding.AxisType.Explicit,),
    )
    bindings = tuple(
        LeafSharding(
            leaf.leaf_key,
            jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None)),
        )
        for leaf in leaves
    )
    arrays = create_device_synthetic_prefix_leaves(bindings, seed=7)
    assert tuple(arrays) == tuple(leaf.leaf_key for leaf in leaves)
    assert np.any(np.asarray(arrays["leaf-00"]) != 0)
    assert np.any(np.asarray(arrays["leaf-01"]) != 0)
    assert np.all(np.asarray(arrays["leaf-02"]) == 0)
    assert arrays["leaf-02"].dtype == np.float32


def test_restore_split_rejects_missing_shape_and_dtype_drift() -> None:
    leaves = _fake_restored_leaves()
    missing = dict(leaves)
    missing.pop(next(iter(missing)))
    with pytest.raises(MaxTextPrefixRuntimeError, match="exact 44"):
        split_restored_prefix_leaves(missing)

    bad_shape = dict(leaves)
    key = "target-routed-only-post_self_attention_layer_norm-scale"
    bad_shape[key] = np.zeros((1,), dtype=ml_dtypes.bfloat16)
    with pytest.raises(MaxTextPrefixRuntimeError, match="shape"):
        split_restored_prefix_leaves(bad_shape)

    bad_dtype = dict(leaves)
    bad_dtype[key] = np.zeros((7_168,), dtype=np.float32)
    with pytest.raises(MaxTextPrefixRuntimeError, match="dtype"):
        split_restored_prefix_leaves(bad_dtype)


def test_compiled_prefix_keeps_parameters_dynamic() -> None:
    calls: list[tuple[object, ...]] = []

    def executable(*args: object) -> str:
        calls.append(args)
        return "output"

    compiled = CompiledPrefix(
        executable=executable,
        model="model-parameters",
        target_parameters="target-parameters",
    )
    batch = PrefixInputBatch("tokens", "positions", "segments")
    assert compiled(batch) == "output"
    assert calls == [
        (
            "model-parameters",
            "target-parameters",
            "tokens",
            "positions",
            "segments",
        )
    ]
    with pytest.raises(MaxTextPrefixRuntimeError, match="wrong type"):
        compiled(("tokens", "positions", "segments"))  # type: ignore[arg-type]


def test_prefix_compile_disables_nnx_graph_updates() -> None:
    source = inspect.getsource(compile_prefix)
    assert "nnx.jit(forward, graph_updates=False)" in source


def test_real_topology_and_synthetic_batch_fail_closed_on_cpu() -> None:
    with pytest.raises(MaxTextPrefixRuntimeError, match="real v6e-64"):
        assert_real_v6e64_runtime()
    with pytest.raises(MaxTextPrefixRuntimeError, match="global B128"):
        create_synthetic_prefix_batch(object())


def test_distributed_initialization_is_explicit_and_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []
    monkeypatch.setattr(jax.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(
        jax.distributed,
        "initialize",
        lambda *, initialization_timeout: calls.append(initialization_timeout),
    )
    initialize_v6e_distributed_runtime(initialization_timeout_seconds=900)
    assert calls == [900]

    with pytest.raises(MaxTextPrefixRuntimeError, match=r"\[300, 1800\]"):
        initialize_v6e_distributed_runtime(initialization_timeout_seconds=299)
    monkeypatch.setattr(jax.distributed, "is_initialized", lambda: True)
    with pytest.raises(MaxTextPrefixRuntimeError, match="before"):
        initialize_v6e_distributed_runtime()
