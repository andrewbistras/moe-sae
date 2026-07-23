from __future__ import annotations

import ast
from collections import Counter
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

import moe_sae.r1.prefix_batchsplit_contract as batchsplit_contract
from moe_sae.r1.checkpoint_inventory import (
    PINNED_GEOMETRY,
    expected_prefix_mapping_descriptors,
)
from moe_sae.r1.prefix_batchsplit_contract import (
    BatchSplitMeshLayoutV1,
    PINNED_LAYOUT_LADDER,
    PINNED_MAXTEXT_REVISION,
    PINNED_PREFIX_BATCHSPLIT_CONTRACT,
    PINNED_PREFIX_BATCHSPLIT_CONTRACT_SHA256,
    PINNED_PREFIX_MAPPING_DESCRIPTOR_SHA256,
    PINNED_UNBOUND_PREFIX_PERFORMANCE_TREE_PLAN_SHA256,
    PrefixBatchSplitContractError,
    PrefixBatchSplitContractV1,
    PrefixPerformanceTreePlanV1,
    build_pinned_prefix_performance_tree_plan,
)
from moe_sae.r1.protocol import canonical_sha256


@pytest.fixture(scope="module")
def performance_plan() -> PrefixPerformanceTreePlanV1:
    return build_pinned_prefix_performance_tree_plan()


def test_first_candidate_contract_is_exact_and_immutable() -> None:
    contract = PINNED_PREFIX_BATCHSPLIT_CONTRACT
    assert contract == PrefixBatchSplitContractV1()
    assert contract.maxtext_revision == ("c03f8bd307d18049fa814fc8c104bd9ec8c292aa")
    assert contract.maxtext_revision == PINNED_MAXTEXT_REVISION
    assert contract.checkpoint_geometry_sha256 == PINNED_GEOMETRY.digest
    assert contract.sequence_length == 1_024
    assert contract.global_batch_size == 128
    assert contract.batch_split_factor == 2
    assert contract.host_count == 16
    assert contract.chips_per_host == 4
    assert contract.device_count == 64
    assert contract.activation_batch_axes == ("data", "fsdp", "expert")
    assert contract.parameter_scan_axis == 1
    assert (
        contract.dense_scan_count,
        contract.completed_moe_scan_count,
        contract.partial_target_count,
        contract.target_layer_zero_indexed,
    ) == (3, 27, 1, 30)
    assert (
        contract.source_entry_count,
        contract.source_mapping_count,
        contract.source_destination_count,
        contract.performance_leaf_count,
    ) == (43_679, 21_930, 510, 44)
    assert contract.router_bias_storage_dtype == "float32"
    assert contract.production_sealable is False
    assert contract.production_admission_enabled is False
    assert contract.digest == PINNED_PREFIX_BATCHSPLIT_CONTRACT_SHA256
    with pytest.raises(FrozenInstanceError):
        contract.global_batch_size = 64  # type: ignore[misc]


def test_layout_ladder_shards_batch_over_all_devices_without_replica_axes() -> None:
    assert tuple(layout.name for layout in PINNED_LAYOUT_LADDER) == (
        "EP16/FSDP4",
        "EP32/FSDP2",
        "EP8/FSDP8",
    )
    assert tuple(
        (layout.expert_parallelism, layout.fsdp_parallelism)
        for layout in PINNED_LAYOUT_LADDER
    ) == ((16, 4), (32, 2), (8, 8))
    for layout in PINNED_LAYOUT_LADDER:
        assert layout.data_parallelism == 1
        assert layout.context_parallelism == 1
        assert layout.tensor_parallelism == 1
        assert layout.activation_shard_product == 64
        assert layout.device_count == 64

    with pytest.raises(PrefixBatchSplitContractError, match="replica-only context/TP"):
        BatchSplitMeshLayoutV1(
            "EP16/FSDP2",
            data_parallelism=1,
            fsdp_parallelism=2,
            expert_parallelism=16,
            context_parallelism=2,
        )
    with pytest.raises(PrefixBatchSplitContractError, match="replica-only context/TP"):
        BatchSplitMeshLayoutV1(
            "EP16/FSDP2",
            data_parallelism=1,
            fsdp_parallelism=2,
            expert_parallelism=16,
            tensor_parallelism=2,
        )
    with pytest.raises(PrefixBatchSplitContractError, match="exactly 64"):
        BatchSplitMeshLayoutV1("EP16/FSDP2", 1, 2, 16)

    valid_but_not_in_ladder = BatchSplitMeshLayoutV1("EP64/FSDP1", 1, 1, 64)
    with pytest.raises(PrefixBatchSplitContractError, match="layout ladder"):
        replace(
            PINNED_PREFIX_BATCHSPLIT_CONTRACT,
            layout_ladder=(
                valid_but_not_in_ladder,
                *PINNED_LAYOUT_LADDER[1:],
            ),
        )


def test_b64_and_batch_split_geometry_fail_closed() -> None:
    with pytest.raises(PrefixBatchSplitContractError, match="B64 is forbidden"):
        replace(PINNED_PREFIX_BATCHSPLIT_CONTRACT, global_batch_size=64)
    with pytest.raises(
        PrefixBatchSplitContractError, match="batch_split_factor must be exactly 2"
    ):
        replace(PINNED_PREFIX_BATCHSPLIT_CONTRACT, batch_split_factor=1)
    with pytest.raises(
        PrefixBatchSplitContractError, match="first batch-split candidate.*B128"
    ):
        replace(PINNED_PREFIX_BATCHSPLIT_CONTRACT, global_batch_size=256)
    with pytest.raises(
        PrefixBatchSplitContractError, match="sequence_length must be exactly 1024"
    ):
        replace(PINNED_PREFIX_BATCHSPLIT_CONTRACT, sequence_length=2_048)
    assert PINNED_PREFIX_BATCHSPLIT_CONTRACT.sequences_per_host == 8
    assert PINNED_PREFIX_BATCHSPLIT_CONTRACT.sequences_per_chip == 2
    assert PINNED_PREFIX_BATCHSPLIT_CONTRACT.sequences_per_chip_split_microbatch == 1


def test_capture_output_abi_and_b128_byte_arithmetic_are_exact() -> None:
    contract = PINNED_PREFIX_BATCHSPLIT_CONTRACT
    outputs = {output.name: output for output in contract.capture_outputs}
    assert tuple(outputs) == (
        "routed_activation",
        "global_expert_ids",
        "scaled_router_weights",
    )
    activation = outputs["routed_activation"]
    assert activation.shape == (128, 1_024, 7_168)
    assert (activation.device_dtype, activation.durable_dtype) == (
        "bfloat16",
        "bfloat16",
    )
    expert_ids = outputs["global_expert_ids"]
    assert expert_ids.shape == (128, 1_024, 8)
    assert (expert_ids.device_dtype, expert_ids.durable_dtype) == (
        "int32",
        "uint16",
    )
    weights = outputs["scaled_router_weights"]
    assert weights.shape == (128, 1_024, 8)
    assert (weights.device_dtype, weights.durable_dtype) == (
        "float32",
        "bfloat16",
    )

    metadata = {field.name: field for field in contract.durable_metadata}
    assert tuple(metadata) == ("sequence_id", "position")
    assert metadata["sequence_id"].shape == (128, 1_024)
    assert metadata["sequence_id"].durable_dtype == "int64"
    assert metadata["position"].shape == (128, 1_024)
    assert metadata["position"].durable_dtype == "uint16"
    with pytest.raises(FrozenInstanceError):
        metadata["sequence_id"].durable_dtype = "uint16"  # type: ignore[misc]

    arithmetic = contract.capture_byte_arithmetic()
    assert arithmetic["durable"] == {
        "scope": "durable-model-outputs-plus-host-metadata",
        "per_model_output_per_step": {
            "routed_activation": 1_879_048_192,
            "global_expert_ids": 2_097_152,
            "scaled_router_weights": 2_097_152,
        },
        "per_metadata_field_per_step": {
            "sequence_id": 1_048_576,
            "position": 262_144,
        },
        "per_token": 14_378,
        "per_step": 1_884_553_216,
        "per_host": 117_784_576,
        "per_chip": 29_446_144,
        "per_split_microbatch": 942_276_608,
        "per_chip_split_microbatch": 14_723_072,
    }
    assert arithmetic["device"] == {
        "scope": "three-device-model-outputs-only",
        "per_model_output_per_step": {
            "routed_activation": 1_879_048_192,
            "global_expert_ids": 4_194_304,
            "scaled_router_weights": 4_194_304,
        },
        "per_token": 14_400,
        "per_step": 1_887_436_800,
        "per_host": 117_964_800,
        "per_chip": 29_491_200,
        "per_split_microbatch": 943_718_400,
        "per_chip_split_microbatch": 14_745_600,
    }


def test_plan_is_derived_from_canonical_inventory_descriptors_exactly_once(
    performance_plan: PrefixPerformanceTreePlanV1,
) -> None:
    canonical = expected_prefix_mapping_descriptors()
    flattened = tuple(
        sorted(
            (
                mapping
                for destination in performance_plan.destinations
                for mapping in destination.mappings
            ),
            key=lambda mapping: mapping.source_key,
        )
    )
    assert flattened == canonical
    assert expected_prefix_mapping_descriptors() is canonical
    assert len(flattened) == 21_930
    assert len(performance_plan.destinations) == 510
    assert len(performance_plan.leaves) == 44
    assert performance_plan.source_mapping_count == 21_930
    assert performance_plan.scale_mapping_count == 21_749
    assert performance_plan.source_entry_count == 43_679
    assert (
        performance_plan.source_entry_count,
        performance_plan.source_mapping_count,
        len(performance_plan.destinations),
        len(performance_plan.leaves),
    ) == (
        PINNED_PREFIX_BATCHSPLIT_CONTRACT.source_entry_count,
        PINNED_PREFIX_BATCHSPLIT_CONTRACT.source_mapping_count,
        PINNED_PREFIX_BATCHSPLIT_CONTRACT.source_destination_count,
        PINNED_PREFIX_BATCHSPLIT_CONTRACT.performance_leaf_count,
    )
    assert performance_plan.digest == PINNED_UNBOUND_PREFIX_PERFORMANCE_TREE_PLAN_SHA256
    assert (
        canonical_sha256(
            {
                "schema": "deepseek-r1-prefix-mapping-descriptors-v1",
                "mappings": [mapping.as_dict() for mapping in canonical],
            }
        )
        == PINNED_PREFIX_MAPPING_DESCRIPTOR_SHA256
    )

    source_keys = tuple(mapping.source_key for mapping in flattened)
    scale_keys = tuple(
        mapping.scale_key for mapping in flattened if mapping.scale_key is not None
    )
    assert len(source_keys) == len(set(source_keys))
    assert len(scale_keys) == len(set(scale_keys))
    assert set(source_keys).isdisjoint(scale_keys)
    assert Counter(
        len(destination.mappings) for destination in performance_plan.destinations
    ) == Counter({1: 426, 256: 84})
    for destination in performance_plan.destinations:
        if len(destination.mappings) == 256:
            assert tuple(
                mapping.destination_slice_index for mapping in destination.mappings
            ) == tuple(range(256))


def test_510_destinations_map_bijectively_to_the_44_leaf_tree(
    performance_plan: PrefixPerformanceTreePlanV1,
) -> None:
    assert Counter(leaf.section for leaf in performance_plan.leaves) == Counter(
        {
            "embedding": 1,
            "scanned-dense": 12,
            "scanned-completed-moe": 17,
            "target-routed-only": 14,
        }
    )
    assert Counter(
        destination.performance_section for destination in performance_plan.destinations
    ) == Counter(
        {
            "embedding": 1,
            "scanned-dense": 36,
            "scanned-completed-moe": 459,
            "target-routed-only": 14,
        }
    )
    expected_destination_count = {
        "embedding": 1,
        "scanned-dense": 3,
        "scanned-completed-moe": 27,
        "target-routed-only": 1,
    }
    consumed = []
    for leaf in performance_plan.leaves:
        assert (
            len(leaf.source_destination_keys)
            == expected_destination_count[leaf.section]
        )
        consumed.extend(leaf.source_destination_keys)
    planned = [
        destination.destination_key for destination in performance_plan.destinations
    ]
    assert len(consumed) == len(set(consumed)) == 510
    assert set(consumed) == set(planned)


def test_scan_axis_shapes_and_fp32_router_bias_are_explicit(
    performance_plan: PrefixPerformanceTreePlanV1,
) -> None:
    leaves = {leaf.leaf_key: leaf for leaf in performance_plan.leaves}
    assert leaves["params-token_embedder-embedding"].logical_shape == (
        129_280,
        7_168,
    )
    assert leaves[
        "params-decoder-dense_layers-self_attention-wq_a-kernel"
    ].logical_shape == (7_168, 3, 1_536)
    assert leaves[
        "params-decoder-moe_layers-DeepSeekMoeBlock_0-MoeBlock_0-wi_0"
    ].logical_shape == (256, 27, 7_168, 2_048)
    completed_bias = leaves[
        "params-decoder-moe_layers-DeepSeekMoeBlock_0-MoeBlock_0-gate-bias"
    ]
    assert completed_bias.logical_shape == (256, 27)
    assert completed_bias.dtype == "float32"
    target_bias = leaves["target-routed-only-DeepSeekMoeBlock_0-MoeBlock_0-gate-bias"]
    assert target_bias.logical_shape == (256,)
    assert target_bias.dtype == "float32"
    assert leaves[
        "target-routed-only-DeepSeekMoeBlock_0-MoeBlock_0-wi_0"
    ].logical_shape == (256, 7_168, 2_048)

    bias_destinations = [
        destination
        for destination in performance_plan.destinations
        if destination.destination_dtype == "float32"
    ]
    assert len(bias_destinations) == 28
    assert all(
        destination.source_keys[0].endswith(".mlp.gate.e_score_correction_bias")
        for destination in bias_destinations
    )


def test_target_consumes_only_layer30_attention_router_and_routed_experts(
    performance_plan: PrefixPerformanceTreePlanV1,
) -> None:
    target = [
        destination
        for destination in performance_plan.destinations
        if destination.performance_section == "target-routed-only"
    ]
    target_sources = {
        source for destination in target for source in destination.source_keys
    }
    assert len(target) == 14
    assert all(source.startswith("model.layers.30.") for source in target_sources)
    assert all(".shared_experts." not in source for source in target_sources)
    assert all(
        not source.startswith(("model.norm.", "lm_head.")) for source in target_sources
    )
    all_sources = {
        mapping.source_key
        for destination in performance_plan.destinations
        for mapping in destination.mappings
    }
    assert all(not source.startswith("model.layers.61.") for source in all_sources)
    assert all(
        not source.startswith(f"model.layers.{layer}.")
        for source in all_sources
        for layer in range(31, 61)
    )
    completed_families = {
        leaf.family
        for leaf in performance_plan.leaves
        if leaf.section == "scanned-completed-moe"
    }
    target_families = {
        leaf.family
        for leaf in performance_plan.leaves
        if leaf.section == "target-routed-only"
    }
    assert completed_families - target_families == {
        "DeepSeekMoeBlock_0-shared_experts-wi_0-kernel",
        "DeepSeekMoeBlock_0-shared_experts-wi_1-kernel",
        "DeepSeekMoeBlock_0-shared_experts-wo-kernel",
    }


def test_plan_and_contract_cannot_claim_production_or_accept_forged_mapping(
    performance_plan: PrefixPerformanceTreePlanV1,
) -> None:
    assert performance_plan.inventory_bound is False
    assert performance_plan.inventory_selection_sha256 is None
    assert performance_plan.production_sealable is False
    assert performance_plan.production_admission_enabled is False
    with pytest.raises(PrefixBatchSplitContractError, match="cannot seal"):
        replace(performance_plan, production_sealable=True)
    with pytest.raises(PrefixBatchSplitContractError, match="cannot seal"):
        replace(performance_plan, production_admission_enabled=True)
    with pytest.raises(PrefixBatchSplitContractError, match="remain unbound"):
        replace(
            performance_plan,
            inventory_selection_sha256="a" * 64,
        )
    with pytest.raises(PrefixBatchSplitContractError, match="remain unbound"):
        replace(
            performance_plan,
            inventory_bound=True,
            inventory_selection_sha256="a" * 64,
        )
    with pytest.raises(TypeError):
        build_pinned_prefix_performance_tree_plan(object())  # type: ignore[arg-type]

    destination = performance_plan.destinations[0]
    mapping = destination.mappings[0]
    forged_mapping = replace(
        mapping,
        destination_key=mapping.destination_key + "-forged",
    )
    forged_destination = replace(
        destination,
        destination_key=forged_mapping.destination_key,
        mappings=(forged_mapping,),
    )
    forged_destinations = (
        forged_destination,
        *performance_plan.destinations[1:],
    )
    with pytest.raises(
        PrefixBatchSplitContractError,
        match=(
            "destination route differs|unknown canonical|"
            "canonical mappings exactly once|unique and sorted"
        ),
    ):
        replace(performance_plan, destinations=forged_destinations)

    with pytest.raises(PrefixBatchSplitContractError, match="cannot seal"):
        replace(PINNED_PREFIX_BATCHSPLIT_CONTRACT, production_sealable=True)
    with pytest.raises(PrefixBatchSplitContractError, match="cannot seal"):
        replace(
            PINNED_PREFIX_BATCHSPLIT_CONTRACT,
            production_admission_enabled=True,
        )


def test_dense_wi_routes_cannot_be_swapped_between_canonical_leaves(
    performance_plan: PrefixPerformanceTreePlanV1,
) -> None:
    wi_0 = "params-decoder-dense_layers-mlp-wi_0-kernel"
    wi_1 = "params-decoder-dense_layers-mlp-wi_1-kernel"
    swapped_destinations = tuple(
        replace(
            destination,
            performance_leaf_key=(
                wi_1
                if destination.performance_leaf_key == wi_0
                else wi_0
                if destination.performance_leaf_key == wi_1
                else destination.performance_leaf_key
            ),
        )
        for destination in performance_plan.destinations
    )
    source_keys = {
        leaf.leaf_key: leaf.source_destination_keys for leaf in performance_plan.leaves
    }
    swapped_leaves = tuple(
        replace(
            leaf,
            source_destination_keys=(
                source_keys[wi_1]
                if leaf.leaf_key == wi_0
                else source_keys[wi_0]
                if leaf.leaf_key == wi_1
                else leaf.source_destination_keys
            ),
        )
        for leaf in performance_plan.leaves
    )
    with pytest.raises(
        PrefixBatchSplitContractError,
        match="destination route differs from the canonical route",
    ):
        replace(
            performance_plan,
            destinations=swapped_destinations,
            leaves=swapped_leaves,
        )


def test_dense_wi_leaf_families_cannot_be_swapped(
    performance_plan: PrefixPerformanceTreePlanV1,
) -> None:
    wi_0 = "params-decoder-dense_layers-mlp-wi_0-kernel"
    wi_1 = "params-decoder-dense_layers-mlp-wi_1-kernel"
    families = {leaf.leaf_key: leaf.family for leaf in performance_plan.leaves}
    swapped_leaves = tuple(
        replace(
            leaf,
            family=(
                families[wi_1]
                if leaf.leaf_key == wi_0
                else families[wi_0]
                if leaf.leaf_key == wi_1
                else leaf.family
            ),
        )
        for leaf in performance_plan.leaves
    )
    with pytest.raises(
        PrefixBatchSplitContractError,
        match="performance leaf family differs from its canonical route",
    ):
        replace(performance_plan, leaves=swapped_leaves)


def test_plan_construction_enforces_the_pinned_plan_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        batchsplit_contract,
        "PINNED_UNBOUND_PREFIX_PERFORMANCE_TREE_PLAN_SHA256",
        "0" * 64,
    )
    with pytest.raises(
        PrefixBatchSplitContractError,
        match="unbound performance plan digest differs from its pin",
    ):
        build_pinned_prefix_performance_tree_plan()


def test_contract_module_has_no_jax_numerical_or_cloud_runtime_imports() -> None:
    source = Path(batchsplit_contract.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            roots.add(node.module.split(".")[0])
    assert roots.isdisjoint(
        {
            "jax",
            "numpy",
            "torch",
            "google",
            "orbax",
            "safetensors",
        }
    )
