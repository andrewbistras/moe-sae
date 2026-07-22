from __future__ import annotations

import ast
from collections import defaultdict
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import struct

import pytest

import moe_sae.r1.checkpoint_inventory as checkpoint_inventory
from moe_sae.r1.checkpoint_inventory import (
    AuditedAggregateTermV1,
    CheckpointInventoryError,
    DeepSeekR1CheckpointGeometryV1,
    F32_DTYPE,
    FP8_DTYPE,
    PINNED_PREFIX_TOTALS,
    audited_summary_cross_check,
    build_synthetic_checkpoint_inventory_selection,
    decode_safetensor_header_bytes,
    decode_strict_json_bytes,
    decode_strict_json_bytes_with_receipt,
    expected_full_source_specs,
    expected_prefix_logical_specs,
    select_block31_prefix,
    validate_prefix_source_keys,
)


TEST_INDEX_SHA256 = "a" * 64


@pytest.fixture(scope="module")
def tiny_geometry() -> DeepSeekR1CheckpointGeometryV1:
    return DeepSeekR1CheckpointGeometryV1(
        vocab_size=11,
        hidden_dim=6,
        main_layers=4,
        first_dense_layers=1,
        target_layer_zero_indexed=2,
        attention_heads=2,
        q_lora_rank=3,
        kv_lora_rank=2,
        qk_nope_head_dim=2,
        qk_rope_head_dim=1,
        value_head_dim=2,
        dense_mlp_dim=8,
        routed_mlp_dim=5,
        routed_experts=2,
        shared_experts=1,
        fp8_block_size=4,
    )


def _source_shard(
    key: str,
    geometry: DeepSeekR1CheckpointGeometryV1,
) -> str:
    logical_key = key.removesuffix("_scale_inv")
    if logical_key == "model.embed_tokens.weight":
        number = 1
    elif logical_key.startswith(f"model.layers.{geometry.target_layer_zero_indexed}."):
        number = 2
    elif logical_key.startswith("model.layers."):
        layer = int(logical_key.split(".")[2])
        number = 1 if layer < geometry.target_layer_zero_indexed else 3
    else:
        number = 3
    return f"model-{number:05d}-of-000003.safetensors"


def _wire_metadata(
    geometry: DeepSeekR1CheckpointGeometryV1,
    *,
    omit: frozenset[str] = frozenset(),
    add: tuple[tuple[str, str, tuple[int, ...], str], ...] = (),
    overrides: dict[str, tuple[str, tuple[int, ...]]] | None = None,
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    specs = [
        spec for spec in expected_full_source_specs(geometry) if spec.key not in omit
    ]
    raw_specs = [
        (spec.key, spec.dtype, spec.shape, _source_shard(spec.key, geometry))
        for spec in specs
    ]
    raw_specs.extend(add)
    overrides = overrides or {}

    by_shard: dict[str, list[tuple[str, str, tuple[int, ...]]]] = defaultdict(list)
    for key, dtype, shape, shard in raw_specs:
        dtype, shape = overrides.get(key, (dtype, shape))
        by_shard[shard].append((key, dtype, shape))

    headers: dict[str, dict[str, object]] = {}
    weight_map: dict[str, str] = {}
    source_numel = 0
    bytes_per_dtype = {FP8_DTYPE: 1, "BF16": 2, F32_DTYPE: 4}
    for shard in sorted(by_shard):
        header: dict[str, object] = {"__metadata__": {"format": "pt"}}
        cursor = 0
        for key, dtype, shape in sorted(by_shard[shard]):
            numel = 1
            for dimension in shape:
                numel *= dimension
            size = numel * bytes_per_dtype[dtype]
            header[key] = {
                "dtype": dtype,
                "shape": list(shape),
                "data_offsets": [cursor, cursor + size],
            }
            cursor += size
            source_numel += numel
            weight_map[key] = shard
        headers[shard] = header
    index: dict[str, object] = {
        "metadata": {"total_size": 2 * source_numel},
        "weight_map": weight_map,
    }
    return index, headers


def _build_tiny(
    geometry: DeepSeekR1CheckpointGeometryV1,
):
    index, headers = _wire_metadata(geometry)
    return _build_wire(index, headers, geometry)


def _descriptor_manifest(
    headers: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    descriptors: dict[str, dict[str, object]] = {}
    for shard, header in headers.items():
        encoded = json.dumps(
            header,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        raw_header = encoded + b" " * ((-len(encoded)) % 8)
        payload_end = max(
            value["data_offsets"][1]
            for key, value in header.items()
            if key != "__metadata__"
        )
        blob = struct.pack("<Q", len(raw_header)) + raw_header + bytes(payload_end)
        decoded, descriptor = decode_safetensor_header_bytes(
            raw_header,
            shard=shard,
            expected_lfs_sha256=hashlib.sha256(blob).hexdigest(),
            expected_blob_size_bytes=len(blob),
        )
        assert decoded == header
        descriptors[shard] = descriptor
    return descriptors


def _build_wire(
    index: dict[str, object],
    headers: dict[str, dict[str, object]],
    geometry: DeepSeekR1CheckpointGeometryV1,
):
    raw_index = json.dumps(
        index,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    index_sha256 = hashlib.sha256(raw_index).hexdigest()
    decoded_index, index_receipt = decode_strict_json_bytes_with_receipt(
        raw_index,
        expected_sha256=index_sha256,
        maximum_bytes=len(raw_index),
    )
    assert decoded_index == index
    return build_synthetic_checkpoint_inventory_selection(
        index,
        headers,
        verified_index_sha256=index_sha256,
        index_decode_receipt=index_receipt,
        shard_descriptors=_descriptor_manifest(headers),
        geometry=geometry,
    )


def test_enumerated_full_and_prefix_inventories_are_canonical_and_addressed(
    tiny_geometry: DeepSeekR1CheckpointGeometryV1,
) -> None:
    first_index, first_headers = _wire_metadata(tiny_geometry)
    first = _build_wire(first_index, first_headers, tiny_geometry)

    reverse_index = {
        "weight_map": dict(reversed(list(first_index["weight_map"].items()))),
        "metadata": {"total_size": first_index["metadata"]["total_size"]},
    }
    reverse_headers = {
        shard: dict(reversed(list(header.items())))
        for shard, header in reversed(list(first_headers.items()))
    }
    second = _build_wire(reverse_index, reverse_headers, tiny_geometry)

    expected_full = expected_full_source_specs(tiny_geometry)
    expected_prefix = expected_prefix_logical_specs(tiny_geometry)
    assert first.full.metadata_authenticated is True
    assert first.selected.metadata_authenticated is True
    assert first.metadata_authenticated is True
    # Synthetic geometry is useful for contract tests but can never authorize
    # the real R1 conversion, even with internally complete fake receipts.
    assert first.full.inventory_sealable is False
    assert first.selected.inventory_sealable is False
    assert first.inventory_sealable is False
    assert first.conversion_sealable is False
    assert first.full.totals.source_entry_count == len(expected_full)
    assert first.selected.totals.logical_tensor_count == len(expected_prefix)
    assert first.selected.totals.scale_tensor_count == sum(
        spec.dtype == FP8_DTYPE for spec in expected_prefix
    )
    assert first.full.required_shards == (
        "model-00001-of-000003.safetensors",
        "model-00002-of-000003.safetensors",
        "model-00003-of-000003.safetensors",
    )
    assert first.selected.required_shards == (
        "model-00001-of-000003.safetensors",
        "model-00002-of-000003.safetensors",
    )
    assert first.full.digest == second.full.digest
    assert first.selected.digest == second.selected.digest
    assert first.digest == second.digest


def test_mapping_descriptors_cover_every_selected_family_and_target_is_routed_only(
    tiny_geometry: DeepSeekR1CheckpointGeometryV1,
) -> None:
    selection = _build_tiny(tiny_geometry)
    expected = expected_prefix_logical_specs(tiny_geometry)
    assert {mapping.source_key for mapping in selection.mappings} == {
        spec.key for spec in expected
    }

    by_source = {mapping.source_key: mapping for mapping in selection.mappings}
    embedding = by_source["model.embed_tokens.weight"]
    assert embedding.destination_key == "params-token_embedder-embedding"
    assert embedding.transform == "identity-v1"

    dense_gate = by_source["model.layers.0.mlp.gate_proj.weight"]
    assert dense_gate.destination_key == "params-decoder-dense_layers_0-mlp-wi_0-kernel"
    assert dense_gate.destination_shape == (6, 8)
    assert dense_gate.scale_key == dense_gate.source_key + "_scale_inv"

    target = tiny_geometry.target_layer_zero_indexed
    q_b = by_source[f"model.layers.{target}.self_attn.q_b_proj.weight"]
    assert q_b.destination_key == (
        "params-decoder-moe_layers_1-self_attention-wq_b-kernel"
    )
    assert q_b.destination_shape == (3, 2, 3)
    assert q_b.prefix_semantics == "target-routed-only-before-shared-expert"

    expert = by_source[f"model.layers.{target}.mlp.experts.1.gate_proj.weight"]
    assert expert.destination_key == (
        "params-decoder-moe_layers_1-DeepSeekMoeBlock_0-MoeBlock_0-wi_0"
    )
    assert expert.destination_shape == (2, 6, 5)
    assert expert.destination_slice_axis == 0
    assert expert.destination_slice_index == 1
    assert expert.prefix_semantics == "target-routed-only-before-shared-expert"

    router_bias = by_source[f"model.layers.{target}.mlp.gate.e_score_correction_bias"]
    assert router_bias.destination_dtype == "float32"
    assert router_bias.transform == "identity-v1"

    target_sources = {
        source for source in by_source if source.startswith(f"model.layers.{target}.")
    }
    assert all("shared_experts" not in source for source in target_sources)
    assert any(
        "shared_experts" in source
        for source in by_source
        if source.startswith("model.layers.1.")
    )


def test_mapping_descriptors_are_recomputed_not_merely_structurally_accepted(
    tiny_geometry: DeepSeekR1CheckpointGeometryV1,
) -> None:
    selection = _build_tiny(tiny_geometry)
    first = selection.mappings[0]
    forged_destination = replace(first, destination_key="attacker-destination")
    with pytest.raises(CheckpointInventoryError, match="canonical HF-to-MaxText"):
        replace(selection, mappings=(forged_destination, *selection.mappings[1:]))

    with pytest.raises(CheckpointInventoryError, match="unknown mapping transform"):
        replace(first, transform="arbitrary-code-v999")


def test_digest_bearing_records_reject_mutable_nested_containers(
    tiny_geometry: DeepSeekR1CheckpointGeometryV1,
) -> None:
    selection = _build_tiny(tiny_geometry)
    with pytest.raises(CheckpointInventoryError, match="entries must be"):
        replace(selection.full, entries=list(selection.full.entries))
    with pytest.raises(CheckpointInventoryError, match="selection mappings must be"):
        replace(selection, mappings=list(selection.mappings))
    with pytest.raises(CheckpointInventoryError, match="immutable tuples"):
        replace(
            selection.mappings[0], source_shape=list(selection.mappings[0].source_shape)
        )


def test_synthetic_geometry_is_explicit_and_cannot_masquerade_as_block31(
    tiny_geometry: DeepSeekR1CheckpointGeometryV1,
) -> None:
    selection = _build_tiny(tiny_geometry)
    assert selection.full.source_kind == "synthetic-test-fixture"
    assert selection.full.scope == "full-synthetic-snapshot"
    assert selection.selected.scope == "synthetic-routed-prefix"
    assert selection.full.config_sha256 == "0" * 64
    assert selection.inventory_sealable is False
    with pytest.raises(CheckpointInventoryError, match="exact pinned R1-0528"):
        select_block31_prefix(selection.full)


def test_receipts_are_all_or_nothing_and_bind_decoded_headers(
    tiny_geometry: DeepSeekR1CheckpointGeometryV1,
) -> None:
    index, headers = _wire_metadata(tiny_geometry)
    descriptors = _descriptor_manifest(headers)
    without_receipts = build_synthetic_checkpoint_inventory_selection(
        index,
        headers,
        verified_index_sha256=TEST_INDEX_SHA256,
        index_decode_receipt=None,
        shard_descriptors=None,
        geometry=tiny_geometry,
    )
    assert without_receipts.metadata_authenticated is False
    assert without_receipts.inventory_sealable is False
    assert without_receipts.conversion_sealable is False

    header_receipts_only = build_synthetic_checkpoint_inventory_selection(
        index,
        headers,
        verified_index_sha256=TEST_INDEX_SHA256,
        index_decode_receipt=None,
        shard_descriptors=descriptors,
        geometry=tiny_geometry,
    )
    assert header_receipts_only.full.shard_descriptors
    assert header_receipts_only.metadata_authenticated is False
    assert header_receipts_only.inventory_sealable is False

    partial = dict(descriptors)
    partial.pop(next(iter(partial)))
    with pytest.raises(CheckpointInventoryError, match="cover parsed shards exactly"):
        build_synthetic_checkpoint_inventory_selection(
            index,
            headers,
            verified_index_sha256=without_receipts.full.verified_index_sha256,
            index_decode_receipt=None,
            shard_descriptors=partial,
            geometry=tiny_geometry,
        )

    corrupted = deepcopy(descriptors)
    shard = next(iter(corrupted))
    corrupted[shard]["decoded_header_sha256"] = "f" * 64
    with pytest.raises(
        CheckpointInventoryError, match="decoded header digest mismatch"
    ):
        build_synthetic_checkpoint_inventory_selection(
            index,
            headers,
            verified_index_sha256=without_receipts.full.verified_index_sha256,
            index_decode_receipt=None,
            shard_descriptors=corrupted,
            geometry=tiny_geometry,
        )


def test_strict_raw_decoder_binds_hash_size_and_duplicate_rejection() -> None:
    raw = b'{"outer":{"x":1},"ok":true}'
    digest = hashlib.sha256(raw).hexdigest()
    assert decode_strict_json_bytes(
        raw, expected_sha256=digest, maximum_bytes=len(raw)
    ) == {"outer": {"x": 1}, "ok": True}

    duplicated = b'{"outer":{"x":1,"x":2}}'
    with pytest.raises(CheckpointInventoryError, match="duplicate JSON field"):
        decode_strict_json_bytes(
            duplicated,
            expected_sha256=hashlib.sha256(duplicated).hexdigest(),
            maximum_bytes=len(duplicated),
        )
    with pytest.raises(CheckpointInventoryError, match="SHA-256 mismatch"):
        decode_strict_json_bytes(raw, expected_sha256="f" * 64, maximum_bytes=len(raw))
    with pytest.raises(CheckpointInventoryError, match="byte length"):
        decode_strict_json_bytes(
            raw, expected_sha256=digest, maximum_bytes=len(raw) - 1
        )


@pytest.mark.parametrize(
    ("bad_key", "reason"),
    (
        ("model.layers.2.mlp.shared_experts.gate_proj.weight", "target shared expert"),
        ("model.layers.3.input_layernorm.weight", "suffix layer"),
        ("model.norm.weight", "final normalization or LM head"),
        ("lm_head.weight", "final normalization or LM head"),
        ("model.layers.4.embed_tokens.weight", "MTP block"),
        ("model.layers.4.mlp.experts.0.up_proj.weight_scale_inv", "MTP block"),
    ),
)
def test_prefix_leakage_categories_are_explicitly_rejected(
    tiny_geometry: DeepSeekR1CheckpointGeometryV1,
    bad_key: str,
    reason: str,
) -> None:
    with pytest.raises(CheckpointInventoryError, match=reason):
        validate_prefix_source_keys((bad_key,), tiny_geometry)


def test_fp8_scale_shape_uses_ceiling_blocks(
    tiny_geometry: DeepSeekR1CheckpointGeometryV1,
) -> None:
    index, headers = _wire_metadata(tiny_geometry)
    key = "model.layers.0.self_attn.q_a_proj.weight_scale_inv"
    shard = index["weight_map"][key]
    assert headers[shard][key]["shape"] == [1, 2]  # ceil(3/4), ceil(6/4)
    _build_tiny(tiny_geometry)


def test_missing_and_orphan_scales_fail_before_exact_key_comparison(
    tiny_geometry: DeepSeekR1CheckpointGeometryV1,
) -> None:
    weight = "model.layers.0.self_attn.q_a_proj.weight"
    missing_scale = weight + "_scale_inv"
    index, headers = _wire_metadata(tiny_geometry, omit=frozenset({missing_scale}))
    with pytest.raises(CheckpointInventoryError, match="missing FP8 inverse scale"):
        _build_wire(index, headers, tiny_geometry)

    orphan = "model.layers.0.not_a_weight_scale_inv"
    index, headers = _wire_metadata(
        tiny_geometry,
        add=((orphan, F32_DTYPE, (1, 1), "model-00001-of-000003.safetensors"),),
    )
    with pytest.raises(CheckpointInventoryError, match="orphan FP8 inverse scale"):
        _build_wire(index, headers, tiny_geometry)


def test_wrong_scale_shape_dtype_and_logical_shape_are_rejected(
    tiny_geometry: DeepSeekR1CheckpointGeometryV1,
) -> None:
    weight = "model.layers.0.self_attn.q_a_proj.weight"
    scale = weight + "_scale_inv"
    index, headers = _wire_metadata(
        tiny_geometry, overrides={scale: (F32_DTYPE, (1, 1))}
    )
    with pytest.raises(CheckpointInventoryError, match="inverse scale shape mismatch"):
        _build_wire(index, headers, tiny_geometry)

    index, headers = _wire_metadata(tiny_geometry, overrides={scale: ("BF16", (1, 2))})
    with pytest.raises(CheckpointInventoryError, match="must have dtype F32"):
        _build_wire(index, headers, tiny_geometry)

    norm = "model.layers.0.input_layernorm.weight"
    index, headers = _wire_metadata(tiny_geometry, overrides={norm: ("BF16", (7,))})
    with pytest.raises(CheckpointInventoryError, match="source shape mismatch"):
        _build_wire(index, headers, tiny_geometry)

    index, headers = _wire_metadata(tiny_geometry, overrides={norm: (F32_DTYPE, (6,))})
    with pytest.raises(CheckpointInventoryError, match="source dtype mismatch"):
        _build_wire(index, headers, tiny_geometry)


def test_duplicate_source_key_across_headers_is_rejected(
    tiny_geometry: DeepSeekR1CheckpointGeometryV1,
) -> None:
    index, headers = _wire_metadata(tiny_geometry)
    key = "model.embed_tokens.weight"
    duplicate = deepcopy(headers["model-00001-of-000003.safetensors"][key])
    duplicate["data_offsets"] = [
        headers["model-00002-of-000003.safetensors"][
            next(
                item
                for item in headers["model-00002-of-000003.safetensors"]
                if item != "__metadata__"
            )
        ]["data_offsets"][0],
        0,
    ]
    # Duplicate detection occurs before parsing the duplicate's offsets or
    # checking that the index assigns it to this second shard.
    headers["model-00002-of-000003.safetensors"][key] = duplicate
    with pytest.raises(CheckpointInventoryError, match="duplicate source tensor key"):
        _build_wire(index, headers, tiny_geometry)


def test_index_header_fields_offsets_and_payload_bytes_are_strict(
    tiny_geometry: DeepSeekR1CheckpointGeometryV1,
) -> None:
    index, headers = _wire_metadata(tiny_geometry)
    bad_index = deepcopy(index)
    bad_index["unexpected"] = True
    with pytest.raises(CheckpointInventoryError, match="unknown fields"):
        _build_wire(bad_index, headers, tiny_geometry)

    index, headers = _wire_metadata(tiny_geometry)
    shard = sorted(headers)[0]
    key = next(item for item in headers[shard] if item != "__metadata__")
    headers[shard][key]["unexpected"] = 1
    with pytest.raises(CheckpointInventoryError, match="unknown fields"):
        _build_wire(index, headers, tiny_geometry)

    index, headers = _wire_metadata(tiny_geometry)
    shard = sorted(headers)[0]
    first_key = min(
        (key for key in headers[shard] if key != "__metadata__"),
        key=lambda key: headers[shard][key]["data_offsets"][0],
    )
    start, end = headers[shard][first_key]["data_offsets"]
    headers[shard][first_key]["data_offsets"] = [start + 1, end + 1]
    with pytest.raises(CheckpointInventoryError, match="non-contiguous"):
        _build_wire(index, headers, tiny_geometry)

    index, headers = _wire_metadata(tiny_geometry)
    shard = sorted(headers)[0]
    key = next(item for item in headers[shard] if item != "__metadata__")
    headers[shard][key]["data_offsets"][1] += 1
    with pytest.raises(CheckpointInventoryError, match="payload byte mismatch"):
        _build_wire(index, headers, tiny_geometry)


def test_header_index_disagreement_and_exact_source_key_set_fail_closed(
    tiny_geometry: DeepSeekR1CheckpointGeometryV1,
) -> None:
    index, headers = _wire_metadata(tiny_geometry)
    key = "model.embed_tokens.weight"
    index["weight_map"][key] = "model-00002-of-000003.safetensors"
    with pytest.raises(
        CheckpointInventoryError, match="header/index shard disagreement"
    ):
        _build_wire(index, headers, tiny_geometry)

    unknown = "model.layers.0.unknown.weight"
    index, headers = _wire_metadata(
        tiny_geometry,
        add=((unknown, "BF16", (1,), "model-00001-of-000003.safetensors"),),
    )
    with pytest.raises(CheckpointInventoryError, match="full source key set mismatch"):
        _build_wire(index, headers, tiny_geometry)


def test_pinned_compact_audit_summary_matches_source_backed_prefix_totals() -> None:
    # These seven terms are the rows in docs/R1_CHECKPOINT_CONVERSION.md.  They
    # cross-check arithmetic without constructing 43,679 fake source entries.
    terms = (
        AuditedAggregateTermV1("token embedding", 1, 0, 1, 0, 0, 926_679_040, 0, 0),
        AuditedAggregateTermV1(
            "dense layers 0-2",
            36,
            24,
            12,
            0,
            1_750_401_024,
            49_152,
            0,
            106_920,
        ),
        AuditedAggregateTermV1(
            "complete MoE layers 3-29",
            21_114,
            20_952,
            135,
            27,
            310_646_734_848,
            49_987_584,
            6_912,
            18_961_128,
        ),
        AuditedAggregateTermV1(
            "target routed experts",
            768,
            768,
            0,
            0,
            11_274_289_152,
            0,
            0,
            688_128,
        ),
        AuditedAggregateTermV1(
            "target attention and input norm",
            8,
            5,
            3,
            0,
            187_105_280,
            9_216,
            0,
            11_448,
        ),
        AuditedAggregateTermV1(
            "target post-attention norm", 1, 0, 1, 0, 0, 7_168, 0, 0
        ),
        AuditedAggregateTermV1("target router", 2, 0, 1, 1, 0, 1_835_008, 256, 0),
    )
    summary = audited_summary_cross_check(terms)
    assert summary.logical_tensor_count == 21_930
    assert summary.scale_tensor_count == 21_749
    assert summary.logical_numel == 324_837_104_640
    assert summary.payload_bytes == 325_894_763_808
    assert summary.scale_numel == PINNED_PREFIX_TOTALS.scale_numel
    assert summary.metadata_authenticated is False
    assert summary.inventory_sealable is False
    assert summary.conversion_sealable is False
    assert not hasattr(summary, "digest")
    assert not hasattr(summary, "canonical_bytes")


def test_inventory_module_has_no_numerical_or_cloud_runtime_imports() -> None:
    source = Path(checkpoint_inventory.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            roots.add(node.module.split(".")[0])
    assert roots.isdisjoint({"torch", "jax", "numpy", "google", "safetensors"})
