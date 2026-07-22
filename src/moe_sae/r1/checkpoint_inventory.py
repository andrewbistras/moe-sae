"""Fail-closed DeepSeek-R1-0528 checkpoint inventory and prefix selection.

This module deliberately stops at metadata.  It does not open tensor payloads,
dequantize weights, import a numerical runtime, or talk to object storage.  Its
job is to turn an independently authenticated Hugging Face safetensor index and
the corresponding safetensor headers into two canonical inventories:

* the complete released R1-0528 source inventory; and
* the exact source entries needed for the one-indexed block-31 routed prefix.

Only the exact pinned source with every enumerated entry and a complete,
authenticated shard/header receipt manifest is inventory-sealable. It is
still never conversion-sealable here. The ``AuditedSummaryCrossCheckV1`` type
near the end exists solely for cheap arithmetic checks and has no digest.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, ClassVar


HF_REPOSITORY = "deepseek-ai/DeepSeek-R1-0528"
HF_REVISION = "4236a6af538feda4548eca9ab308586007567f52"
MAXTEXT_REVISION = "c03f8bd307d18049fa814fc8c104bd9ec8c292aa"
HF_CONFIG_SHA256 = "79ddea672a62e95d3f0be27be434375538e1988975971cffcf87c96c1de84a65"
HF_INDEX_SHA256 = "a2c5d238ab1e8301a117336e9797e0d3551cb8f6dc9dc787318b23d8f54c9248"
HF_MODELING_SHA256 = "eb6d535742061cf9da7a4232ecfe2f089d5ce20080afc5d2181e9d70ec4af8a4"
HF_INDEX_DECLARED_TOTAL_SIZE = 1_369_062_772_000

# Canonical JSON SHA-256 of all 163 exact-revision Hugging Face API LFS
# records, each represented as {"path", "sha256", "size"} and sorted by
# path.  Source endpoint (queried with ``blobs=true``) and arithmetic were
# independently checked on 2026-07-22.  The digest pins every entry without a
# 163-row constant table in this runtime module.
PINNED_LFS_MANIFEST_SHA256 = (
    "22bc591ff31ed8d79c7ef1780e6bc342d9024388b4bf2b7cca5519ab3d8d895b"
)
PINNED_LFS_MANIFEST_ENTRY_COUNT = 163
PINNED_LFS_TOTAL_BLOB_BYTES = 688_586_727_753

FP8_DTYPE = "F8_E4M3"
BF16_DTYPE = "BF16"
F32_DTYPE = "F32"
_DTYPE_BYTES = {FP8_DTYPE: 1, BF16_DTYPE: 2, F32_DTYPE: 4}
_SCALE_SUFFIX = "_scale_inv"

# Parsing limits are intentionally just above the pinned snapshot.  A mapping
# from an untrusted source cannot make this metadata-only pass grow without
# bound before the exact key-set comparison runs.
MAX_SOURCE_ENTRIES = 100_000
MAX_SHARDS = 200
MAX_KEY_LENGTH = 512
MAX_SHARD_NAME_LENGTH = 128
MAX_HEADER_METADATA_FIELDS = 32
MAX_RANK = 4
MAX_DIMENSION = 1_000_000
MAX_TOTAL_PAYLOAD_BYTES = 2_000_000_000_000
MAX_RAW_HEADER_BYTES = 64_000_000
MAX_RAW_INDEX_BYTES = 32_000_000

_SHA256 = re.compile(r"[0-9a-f]{64}")
_SHARD_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\.safetensors")
_LAYER_KEY = re.compile(r"model\.layers\.(\d+)\.(.+)")


class CheckpointInventoryError(ValueError):
    """Checkpoint metadata is incomplete, ambiguous, or incompatible."""


def _reject_json_constant(value: str) -> None:
    raise CheckpointInventoryError(f"non-finite JSON number is forbidden: {value}")


def _object_without_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CheckpointInventoryError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def decode_strict_json_bytes(
    raw: bytes,
    *,
    expected_sha256: str,
    maximum_bytes: int,
) -> Mapping[str, Any]:
    """Decode authenticated JSON bytes with duplicate and size rejection."""

    if type(raw) is not bytes:
        raise CheckpointInventoryError("strict JSON input must be bytes")
    if _SHA256.fullmatch(expected_sha256) is None:
        raise CheckpointInventoryError("expected_sha256 must be lowercase SHA-256")
    _integer(maximum_bytes, "maximum_bytes", minimum=1, maximum=MAX_RAW_HEADER_BYTES)
    if not raw or len(raw) > maximum_bytes:
        raise CheckpointInventoryError(
            f"strict JSON byte length must be in [1, {maximum_bytes}]"
        )
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_sha256 != expected_sha256:
        raise CheckpointInventoryError(
            f"strict JSON SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise CheckpointInventoryError("strict JSON must be UTF-8") from error
    try:
        value = json.loads(
            text,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except CheckpointInventoryError:
        raise
    except (TypeError, json.JSONDecodeError) as error:
        raise CheckpointInventoryError("invalid strict JSON") from error
    return _mapping(value, "strict JSON root")


@dataclass(frozen=True, slots=True)
class StrictJSONReceiptV1:
    """Receipt binding raw authenticated bytes to one strict decoded mapping."""

    raw_sha256: str
    raw_size_bytes: int
    decoded_sha256: str
    decoder_contract: str

    def __post_init__(self) -> None:
        for name in ("raw_sha256", "decoded_sha256"):
            if _SHA256.fullmatch(getattr(self, name)) is None:
                raise CheckpointInventoryError(f"{name} must be lowercase SHA-256")
        _integer(
            self.raw_size_bytes,
            "strict JSON raw_size_bytes",
            minimum=1,
            maximum=MAX_RAW_HEADER_BYTES,
        )
        if self.decoder_contract != "strict-json-no-duplicates-v1":
            raise CheckpointInventoryError("unsupported strict JSON decoder contract")

    def as_dict(self) -> dict[str, Any]:
        return {
            "raw_sha256": self.raw_sha256,
            "raw_size_bytes": self.raw_size_bytes,
            "decoded_sha256": self.decoded_sha256,
            "decoder_contract": self.decoder_contract,
        }


def decode_strict_json_bytes_with_receipt(
    raw: bytes,
    *,
    expected_sha256: str,
    maximum_bytes: int,
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    """Strictly decode bytes and retain evidence needed by a sealable record."""

    decoded = decode_strict_json_bytes(
        raw,
        expected_sha256=expected_sha256,
        maximum_bytes=maximum_bytes,
    )
    receipt = StrictJSONReceiptV1(
        raw_sha256=expected_sha256,
        raw_size_bytes=len(raw),
        decoded_sha256=_sha256(decoded),
        decoder_contract="strict-json-no-duplicates-v1",
    )
    return decoded, receipt.as_dict()


def decode_pinned_index_bytes(
    raw: bytes,
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    """Strictly decode pinned index bytes and return their binding receipt."""

    return decode_strict_json_bytes_with_receipt(
        raw,
        expected_sha256=HF_INDEX_SHA256,
        maximum_bytes=MAX_RAW_INDEX_BYTES,
    )


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise CheckpointInventoryError("inventory is not canonical JSON") from error


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CheckpointInventoryError(f"{name} must be a mapping")
    for key in value:
        if not isinstance(key, str):
            raise CheckpointInventoryError(f"{name} contains a non-string key")
    return value


def _exact_fields(
    value: Any,
    name: str,
    fields: frozenset[str],
) -> Mapping[str, Any]:
    result = _mapping(value, name)
    actual = set(result)
    if actual != fields:
        missing = sorted(fields - actual)
        unknown = sorted(actual - fields)
        details: list[str] = []
        if missing:
            details.append(f"missing fields={missing}")
        if unknown:
            details.append(f"unknown fields={unknown}")
        raise CheckpointInventoryError(f"invalid {name}: {'; '.join(details)}")
    return result


def _string(value: Any, name: str, *, maximum_length: int) -> str:
    if not isinstance(value, str) or not value:
        raise CheckpointInventoryError(f"{name} must be a non-empty string")
    if len(value) > maximum_length:
        raise CheckpointInventoryError(f"{name} exceeds {maximum_length} characters")
    return value


def _integer(
    value: Any,
    name: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if type(value) is not int:
        raise CheckpointInventoryError(
            f"{name} must be an integer (booleans are forbidden)"
        )
    if value < minimum:
        raise CheckpointInventoryError(f"{name} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise CheckpointInventoryError(f"{name} must be <= {maximum}")
    return value


def _shape(value: Any, name: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise CheckpointInventoryError(f"{name} must be an array")
    if not value or len(value) > MAX_RANK:
        raise CheckpointInventoryError(f"{name} rank must be in [1, {MAX_RANK}]")
    return tuple(
        _integer(item, f"{name}[]", minimum=1, maximum=MAX_DIMENSION) for item in value
    )


def _numel(shape: tuple[int, ...]) -> int:
    return math.prod(shape)


def _scale_shape(shape: tuple[int, ...], block_size: int) -> tuple[int, int]:
    if len(shape) != 2:
        raise CheckpointInventoryError(
            "FP8 source tensors must be rank two for blockwise dequantization"
        )
    rows, columns = shape
    return (
        (rows + block_size - 1) // block_size,
        (columns + block_size - 1) // block_size,
    )


@dataclass(frozen=True, slots=True)
class DeepSeekR1CheckpointGeometryV1:
    """Source tensor geometry used to derive, never guess, the inventory."""

    vocab_size: int = 129_280
    hidden_dim: int = 7_168
    main_layers: int = 61
    first_dense_layers: int = 3
    target_layer_zero_indexed: int = 30
    attention_heads: int = 128
    q_lora_rank: int = 1_536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    value_head_dim: int = 128
    dense_mlp_dim: int = 18_432
    routed_mlp_dim: int = 2_048
    routed_experts: int = 256
    shared_experts: int = 1
    fp8_block_size: int = 128

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                if name == "target_layer_zero_indexed" and value == 0:
                    continue
                raise CheckpointInventoryError(f"{name} must be a positive integer")
        if self.first_dense_layers >= self.main_layers:
            raise CheckpointInventoryError(
                "first_dense_layers must be below main_layers"
            )
        if not (
            self.first_dense_layers <= self.target_layer_zero_indexed < self.main_layers
        ):
            raise CheckpointInventoryError(
                "target layer must be a main MoE layer after the dense prefix"
            )
        if self.shared_experts != 1:
            raise CheckpointInventoryError("R1-0528 requires exactly one shared expert")

    @property
    def target_layer_one_indexed(self) -> int:
        return self.target_layer_zero_indexed + 1

    @property
    def mtp_layer_zero_indexed(self) -> int:
        return self.main_layers

    @property
    def q_projection_dim(self) -> int:
        return self.attention_heads * (self.qk_nope_head_dim + self.qk_rope_head_dim)

    @property
    def kv_a_projection_dim(self) -> int:
        return self.kv_lora_rank + self.qk_rope_head_dim

    @property
    def kv_b_projection_dim(self) -> int:
        return self.attention_heads * (self.qk_nope_head_dim + self.value_head_dim)

    @property
    def attention_output_dim(self) -> int:
        return self.attention_heads * self.value_head_dim

    def as_dict(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @property
    def digest(self) -> str:
        return _sha256(
            {"schema": "deepseek-r1-checkpoint-geometry-v1", **self.as_dict()}
        )


PINNED_GEOMETRY = DeepSeekR1CheckpointGeometryV1()


@dataclass(frozen=True, slots=True)
class ExpectedTensorSpecV1:
    """Expected logical source tensor before any FP8 scale companion."""

    key: str
    dtype: str
    shape: tuple[int, ...]
    family: str
    layer_zero_indexed: int | None = None
    expert_index: int | None = None

    @property
    def numel(self) -> int:
        return _numel(self.shape)


def _attention_specs(
    layer: int,
    geometry: DeepSeekR1CheckpointGeometryV1,
) -> list[ExpectedTensorSpecV1]:
    g = geometry
    prefix = f"model.layers.{layer}."
    definitions = (
        ("input_layernorm.weight", BF16_DTYPE, (g.hidden_dim,), "input_norm"),
        (
            "post_attention_layernorm.weight",
            BF16_DTYPE,
            (g.hidden_dim,),
            "post_attention_norm",
        ),
        (
            "self_attn.q_a_proj.weight",
            FP8_DTYPE,
            (g.q_lora_rank, g.hidden_dim),
            "q_a_projection",
        ),
        (
            "self_attn.q_a_layernorm.weight",
            BF16_DTYPE,
            (g.q_lora_rank,),
            "q_a_norm",
        ),
        (
            "self_attn.q_b_proj.weight",
            FP8_DTYPE,
            (g.q_projection_dim, g.q_lora_rank),
            "q_b_projection",
        ),
        (
            "self_attn.kv_a_proj_with_mqa.weight",
            FP8_DTYPE,
            (g.kv_a_projection_dim, g.hidden_dim),
            "kv_a_projection",
        ),
        (
            "self_attn.kv_a_layernorm.weight",
            BF16_DTYPE,
            (g.kv_lora_rank,),
            "kv_a_norm",
        ),
        (
            "self_attn.kv_b_proj.weight",
            FP8_DTYPE,
            (g.kv_b_projection_dim, g.kv_lora_rank),
            "kv_b_projection",
        ),
        (
            "self_attn.o_proj.weight",
            FP8_DTYPE,
            (g.hidden_dim, g.attention_output_dim),
            "attention_output_projection",
        ),
    )
    return [
        ExpectedTensorSpecV1(prefix + suffix, dtype, shape, family, layer)
        for suffix, dtype, shape, family in definitions
    ]


def _dense_specs(
    layer: int,
    geometry: DeepSeekR1CheckpointGeometryV1,
) -> list[ExpectedTensorSpecV1]:
    g = geometry
    prefix = f"model.layers.{layer}.mlp."
    return [
        ExpectedTensorSpecV1(
            prefix + "gate_proj.weight",
            FP8_DTYPE,
            (g.dense_mlp_dim, g.hidden_dim),
            "dense_gate_projection",
            layer,
        ),
        ExpectedTensorSpecV1(
            prefix + "up_proj.weight",
            FP8_DTYPE,
            (g.dense_mlp_dim, g.hidden_dim),
            "dense_up_projection",
            layer,
        ),
        ExpectedTensorSpecV1(
            prefix + "down_proj.weight",
            FP8_DTYPE,
            (g.hidden_dim, g.dense_mlp_dim),
            "dense_down_projection",
            layer,
        ),
    ]


def _moe_specs(
    layer: int,
    geometry: DeepSeekR1CheckpointGeometryV1,
) -> list[ExpectedTensorSpecV1]:
    g = geometry
    prefix = f"model.layers.{layer}.mlp."
    specs = [
        ExpectedTensorSpecV1(
            prefix + "gate.weight",
            BF16_DTYPE,
            (g.routed_experts, g.hidden_dim),
            "router_gate",
            layer,
        ),
        ExpectedTensorSpecV1(
            prefix + "gate.e_score_correction_bias",
            F32_DTYPE,
            (g.routed_experts,),
            "router_correction_bias",
            layer,
        ),
    ]
    for expert in range(g.routed_experts):
        expert_prefix = prefix + f"experts.{expert}."
        specs.extend(
            (
                ExpectedTensorSpecV1(
                    expert_prefix + "gate_proj.weight",
                    FP8_DTYPE,
                    (g.routed_mlp_dim, g.hidden_dim),
                    "routed_expert_gate_projection",
                    layer,
                    expert,
                ),
                ExpectedTensorSpecV1(
                    expert_prefix + "up_proj.weight",
                    FP8_DTYPE,
                    (g.routed_mlp_dim, g.hidden_dim),
                    "routed_expert_up_projection",
                    layer,
                    expert,
                ),
                ExpectedTensorSpecV1(
                    expert_prefix + "down_proj.weight",
                    FP8_DTYPE,
                    (g.hidden_dim, g.routed_mlp_dim),
                    "routed_expert_down_projection",
                    layer,
                    expert,
                ),
            )
        )
    shared_width = g.shared_experts * g.routed_mlp_dim
    specs.extend(
        (
            ExpectedTensorSpecV1(
                prefix + "shared_experts.gate_proj.weight",
                FP8_DTYPE,
                (shared_width, g.hidden_dim),
                "shared_expert_gate_projection",
                layer,
            ),
            ExpectedTensorSpecV1(
                prefix + "shared_experts.up_proj.weight",
                FP8_DTYPE,
                (shared_width, g.hidden_dim),
                "shared_expert_up_projection",
                layer,
            ),
            ExpectedTensorSpecV1(
                prefix + "shared_experts.down_proj.weight",
                FP8_DTYPE,
                (g.hidden_dim, shared_width),
                "shared_expert_down_projection",
                layer,
            ),
        )
    )
    return specs


def expected_full_logical_specs(
    geometry: DeepSeekR1CheckpointGeometryV1 = PINNED_GEOMETRY,
) -> tuple[ExpectedTensorSpecV1, ...]:
    """Return the exact full released logical key set for a geometry."""

    g = geometry
    specs = [
        ExpectedTensorSpecV1(
            "model.embed_tokens.weight",
            BF16_DTYPE,
            (g.vocab_size, g.hidden_dim),
            "token_embedding",
        )
    ]
    for layer in range(g.main_layers):
        specs.extend(_attention_specs(layer, g))
        if layer < g.first_dense_layers:
            specs.extend(_dense_specs(layer, g))
        else:
            specs.extend(_moe_specs(layer, g))

    # The released checkpoint carries one complete MoE MTP block at the next
    # layer index plus its six MTP-only tensors.
    mtp = g.mtp_layer_zero_indexed
    specs.extend(_attention_specs(mtp, g))
    specs.extend(_moe_specs(mtp, g))
    mtp_prefix = f"model.layers.{mtp}."
    specs.extend(
        (
            ExpectedTensorSpecV1(
                mtp_prefix + "embed_tokens.weight",
                BF16_DTYPE,
                (g.vocab_size, g.hidden_dim),
                "mtp_embedding",
                mtp,
            ),
            ExpectedTensorSpecV1(
                mtp_prefix + "enorm.weight",
                BF16_DTYPE,
                (g.hidden_dim,),
                "mtp_embedding_norm",
                mtp,
            ),
            ExpectedTensorSpecV1(
                mtp_prefix + "hnorm.weight",
                BF16_DTYPE,
                (g.hidden_dim,),
                "mtp_hidden_norm",
                mtp,
            ),
            ExpectedTensorSpecV1(
                mtp_prefix + "eh_proj.weight",
                BF16_DTYPE,
                (g.hidden_dim, 2 * g.hidden_dim),
                "mtp_eh_projection",
                mtp,
            ),
            ExpectedTensorSpecV1(
                mtp_prefix + "shared_head.norm.weight",
                BF16_DTYPE,
                (g.hidden_dim,),
                "mtp_shared_head_norm",
                mtp,
            ),
            ExpectedTensorSpecV1(
                mtp_prefix + "shared_head.head.weight",
                BF16_DTYPE,
                (g.vocab_size, g.hidden_dim),
                "mtp_shared_head",
                mtp,
            ),
            ExpectedTensorSpecV1(
                "model.norm.weight",
                BF16_DTYPE,
                (g.hidden_dim,),
                "final_norm",
            ),
            ExpectedTensorSpecV1(
                "lm_head.weight",
                BF16_DTYPE,
                (g.vocab_size, g.hidden_dim),
                "lm_head",
            ),
        )
    )
    ordered = tuple(sorted(specs, key=lambda item: item.key))
    keys = tuple(item.key for item in ordered)
    if len(keys) != len(set(keys)):
        raise CheckpointInventoryError("generated full logical inventory is not unique")
    return ordered


_TARGET_ALLOWED_FAMILIES = frozenset(
    {
        "input_norm",
        "post_attention_norm",
        "q_a_projection",
        "q_a_norm",
        "q_b_projection",
        "kv_a_projection",
        "kv_a_norm",
        "kv_b_projection",
        "attention_output_projection",
        "router_gate",
        "router_correction_bias",
        "routed_expert_gate_projection",
        "routed_expert_up_projection",
        "routed_expert_down_projection",
    }
)


def expected_prefix_logical_specs(
    geometry: DeepSeekR1CheckpointGeometryV1 = PINNED_GEOMETRY,
) -> tuple[ExpectedTensorSpecV1, ...]:
    """Return embedding, complete preceding blocks, and routed-only target."""

    target = geometry.target_layer_zero_indexed
    result: list[ExpectedTensorSpecV1] = []
    for spec in expected_full_logical_specs(geometry):
        if spec.family == "token_embedding":
            result.append(spec)
        elif spec.layer_zero_indexed is not None and spec.layer_zero_indexed < target:
            result.append(spec)
        elif (
            spec.layer_zero_indexed == target
            and spec.family in _TARGET_ALLOWED_FAMILIES
        ):
            result.append(spec)
    return tuple(result)


def expected_full_source_specs(
    geometry: DeepSeekR1CheckpointGeometryV1 = PINNED_GEOMETRY,
) -> tuple[ExpectedTensorSpecV1, ...]:
    """Return logical tensors plus the mandatory FP8 inverse-scale tensors."""

    result: list[ExpectedTensorSpecV1] = []
    for logical in expected_full_logical_specs(geometry):
        result.append(logical)
        if logical.dtype == FP8_DTYPE:
            result.append(
                ExpectedTensorSpecV1(
                    logical.key + _SCALE_SUFFIX,
                    F32_DTYPE,
                    _scale_shape(logical.shape, geometry.fp8_block_size),
                    "fp8_inverse_scale",
                    logical.layer_zero_indexed,
                    logical.expert_index,
                )
            )
    return tuple(sorted(result, key=lambda item: item.key))


@dataclass(frozen=True, slots=True)
class SourceTensorEntryV1:
    key: str
    shard: str
    dtype: str
    shape: tuple[int, ...]
    data_offsets: tuple[int, int]

    def __post_init__(self) -> None:
        _string(self.key, "source tensor key", maximum_length=MAX_KEY_LENGTH)
        _string(self.shard, "source shard", maximum_length=MAX_SHARD_NAME_LENGTH)
        if _SHARD_NAME.fullmatch(self.shard) is None:
            raise CheckpointInventoryError(
                f"invalid safetensor shard name: {self.shard}"
            )
        if self.dtype not in _DTYPE_BYTES:
            raise CheckpointInventoryError(
                f"unsupported source dtype for {self.key}: {self.dtype}"
            )
        parsed_shape = _shape(self.shape, f"{self.key}.shape")
        if parsed_shape != self.shape:
            raise CheckpointInventoryError(f"{self.key}.shape must be a tuple")
        if not isinstance(self.data_offsets, tuple) or len(self.data_offsets) != 2:
            raise CheckpointInventoryError(
                f"{self.key}.data_offsets must be a two-integer tuple"
            )
        start = _integer(self.data_offsets[0], f"{self.key}.data_offsets[0]", minimum=0)
        end = _integer(self.data_offsets[1], f"{self.key}.data_offsets[1]", minimum=0)
        if end <= start:
            raise CheckpointInventoryError(
                f"{self.key}.data_offsets must be increasing"
            )
        expected_bytes = _numel(self.shape) * _DTYPE_BYTES[self.dtype]
        if end - start != expected_bytes:
            raise CheckpointInventoryError(
                f"payload byte mismatch for {self.key}: "
                f"expected {expected_bytes}, got {end - start}"
            )

    @property
    def numel(self) -> int:
        return _numel(self.shape)

    @property
    def payload_bytes(self) -> int:
        return self.data_offsets[1] - self.data_offsets[0]

    @property
    def is_scale(self) -> bool:
        return self.key.endswith(_SCALE_SUFFIX)

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "shard": self.shard,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "data_offsets": list(self.data_offsets),
        }


@dataclass(frozen=True, slots=True)
class InventoryTotalsV1:
    source_entry_count: int
    logical_tensor_count: int
    scale_tensor_count: int
    logical_numel: int
    scale_numel: int
    source_numel: int
    payload_bytes: int
    dtype_tensor_counts: tuple[tuple[str, int], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_entry_count": self.source_entry_count,
            "logical_tensor_count": self.logical_tensor_count,
            "scale_tensor_count": self.scale_tensor_count,
            "logical_numel": self.logical_numel,
            "scale_numel": self.scale_numel,
            "source_numel": self.source_numel,
            "payload_bytes": self.payload_bytes,
            "dtype_tensor_counts": {
                dtype: count for dtype, count in self.dtype_tensor_counts
            },
        }


def _totals(entries: Iterable[SourceTensorEntryV1]) -> InventoryTotalsV1:
    materialized = tuple(entries)
    logical = tuple(entry for entry in materialized if not entry.is_scale)
    scales = tuple(entry for entry in materialized if entry.is_scale)
    dtype_counts = tuple(
        (dtype, sum(entry.dtype == dtype for entry in materialized))
        for dtype in sorted(_DTYPE_BYTES)
    )
    return InventoryTotalsV1(
        source_entry_count=len(materialized),
        logical_tensor_count=len(logical),
        scale_tensor_count=len(scales),
        logical_numel=sum(entry.numel for entry in logical),
        scale_numel=sum(entry.numel for entry in scales),
        source_numel=sum(entry.numel for entry in materialized),
        payload_bytes=sum(entry.payload_bytes for entry in materialized),
        dtype_tensor_counts=dtype_counts,
    )


PINNED_FULL_TOTALS = InventoryTotalsV1(
    source_entry_count=91_991,
    logical_tensor_count=46_183,
    scale_tensor_count=45_808,
    logical_numel=684_489_845_504,
    scale_numel=41_540_496,
    source_numel=684_531_386_000,
    payload_bytes=688_574_839_360,
    dtype_tensor_counts=((BF16_DTYPE, 316), (F32_DTYPE, 45_867), (FP8_DTYPE, 45_808)),
)

PINNED_PREFIX_TOTALS = InventoryTotalsV1(
    source_entry_count=43_679,
    logical_tensor_count=21_930,
    scale_tensor_count=21_749,
    logical_numel=324_837_104_640,
    scale_numel=19_767_624,
    source_numel=324_856_872_264,
    payload_bytes=325_894_763_808,
    dtype_tensor_counts=((BF16_DTYPE, 153), (F32_DTYPE, 21_777), (FP8_DTYPE, 21_749)),
)

PINNED_FULL_SHARDS = tuple(
    f"model-{index:05d}-of-000163.safetensors" for index in range(1, 164)
)
PINNED_PREFIX_SHARDS = PINNED_FULL_SHARDS[:78]


@dataclass(frozen=True, slots=True)
class ShardSourceDescriptorV1:
    """Authenticated-source receipt for one safetensor shard header.

    ``expected_lfs_sha256`` and ``expected_blob_size_bytes`` describe the
    complete immutable blob. ``raw_header_sha256`` covers exactly the
    ``header_length_bytes`` following the eight-byte safetensor length prefix.
    ``decoded_header_sha256`` binds that raw decode to the mapping consumed by
    this module. Payload bytes are not fetched or verified here.
    """

    shard: str
    expected_lfs_sha256: str
    expected_blob_size_bytes: int
    raw_header_sha256: str
    header_length_bytes: int
    decoded_header_sha256: str
    decoder_contract: str

    def __post_init__(self) -> None:
        _string(self.shard, "descriptor shard", maximum_length=MAX_SHARD_NAME_LENGTH)
        if _SHARD_NAME.fullmatch(self.shard) is None:
            raise CheckpointInventoryError(
                f"invalid descriptor shard name: {self.shard}"
            )
        for name in (
            "expected_lfs_sha256",
            "raw_header_sha256",
            "decoded_header_sha256",
        ):
            if _SHA256.fullmatch(getattr(self, name)) is None:
                raise CheckpointInventoryError(
                    f"descriptor {name} must be lowercase SHA-256"
                )
        _integer(
            self.expected_blob_size_bytes,
            "descriptor expected_blob_size_bytes",
            minimum=1,
            maximum=MAX_TOTAL_PAYLOAD_BYTES,
        )
        _integer(
            self.header_length_bytes,
            "descriptor header_length_bytes",
            minimum=1,
            maximum=MAX_RAW_HEADER_BYTES,
        )
        if self.decoder_contract != "strict-json-no-duplicates-v1":
            raise CheckpointInventoryError("unsupported strict header decoder contract")

    def as_dict(self) -> dict[str, Any]:
        return {
            "shard": self.shard,
            "expected_lfs_sha256": self.expected_lfs_sha256,
            "expected_blob_size_bytes": self.expected_blob_size_bytes,
            "raw_header_sha256": self.raw_header_sha256,
            "header_length_bytes": self.header_length_bytes,
            "decoded_header_sha256": self.decoded_header_sha256,
            "decoder_contract": self.decoder_contract,
        }

    def lfs_manifest_dict(self) -> dict[str, Any]:
        return {
            "path": self.shard,
            "sha256": self.expected_lfs_sha256,
            "size": self.expected_blob_size_bytes,
        }


def decode_safetensor_header_bytes(
    raw_header: bytes,
    *,
    shard: str,
    expected_lfs_sha256: str,
    expected_blob_size_bytes: int,
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    """Strictly decode raw header bytes and emit their source receipt mapping."""

    if type(raw_header) is not bytes:
        raise CheckpointInventoryError("raw safetensor header must be bytes")
    if not raw_header or len(raw_header) % 8 != 0:
        raise CheckpointInventoryError(
            "raw safetensor header length must be positive and eight-byte aligned"
        )
    raw_sha256 = hashlib.sha256(raw_header).hexdigest()
    decoded = decode_strict_json_bytes(
        raw_header,
        expected_sha256=raw_sha256,
        maximum_bytes=MAX_RAW_HEADER_BYTES,
    )
    descriptor = ShardSourceDescriptorV1(
        shard=shard,
        expected_lfs_sha256=expected_lfs_sha256,
        expected_blob_size_bytes=expected_blob_size_bytes,
        raw_header_sha256=raw_sha256,
        header_length_bytes=len(raw_header),
        decoded_header_sha256=_sha256(decoded),
        decoder_contract="strict-json-no-duplicates-v1",
    )
    return decoded, descriptor.as_dict()


@dataclass(frozen=True, slots=True)
class TensorInventoryV1:
    """Canonical, content-addressed inventory built from enumerated entries."""

    SCHEMA: ClassVar[str] = "deepseek-r1-source-tensor-inventory-v1"

    source_kind: str
    scope: str
    repository: str
    revision: str
    config_sha256: str
    modeling_sha256: str
    verified_index_sha256: str
    index_decode_receipt: StrictJSONReceiptV1 | None
    index_declared_total_size: int
    geometry: DeepSeekR1CheckpointGeometryV1
    entries: tuple[SourceTensorEntryV1, ...]
    header_metadata: tuple[tuple[str, tuple[tuple[str, str], ...]], ...]
    shard_descriptors: tuple[ShardSourceDescriptorV1, ...]
    authenticated_full_lfs_manifest_sha256: str | None

    def __post_init__(self) -> None:
        if self.source_kind not in {"pinned-r1-0528", "synthetic-test-fixture"}:
            raise CheckpointInventoryError("unknown inventory source kind")
        allowed_scope = {
            "pinned-r1-0528": {
                "full-released-snapshot",
                "block-31-routed-prefix",
            },
            "synthetic-test-fixture": {
                "full-synthetic-snapshot",
                "synthetic-routed-prefix",
            },
        }[self.source_kind]
        if self.scope not in allowed_scope:
            raise CheckpointInventoryError("unknown inventory scope")
        if type(self.geometry) is not DeepSeekR1CheckpointGeometryV1:
            raise CheckpointInventoryError(
                "geometry must be an exact immutable DeepSeekR1CheckpointGeometryV1"
            )
        if type(self.entries) is not tuple or not all(
            type(entry) is SourceTensorEntryV1 for entry in self.entries
        ):
            raise CheckpointInventoryError(
                "enumerated inventory entries must be SourceTensorEntryV1 tuples"
            )
        if type(self.header_metadata) is not tuple:
            raise CheckpointInventoryError("header_metadata must be a tuple")
        if type(self.shard_descriptors) is not tuple or not all(
            type(item) is ShardSourceDescriptorV1 for item in self.shard_descriptors
        ):
            raise CheckpointInventoryError(
                "shard_descriptors must be a ShardSourceDescriptorV1 tuple"
            )
        if _SHA256.fullmatch(self.verified_index_sha256) is None:
            raise CheckpointInventoryError(
                "verified_index_sha256 must be lowercase SHA-256"
            )
        if (
            self.index_decode_receipt is not None
            and type(self.index_decode_receipt) is not StrictJSONReceiptV1
        ):
            raise CheckpointInventoryError(
                "index_decode_receipt must be StrictJSONReceiptV1 or None"
            )
        for name in ("config_sha256", "modeling_sha256"):
            if _SHA256.fullmatch(getattr(self, name)) is None:
                raise CheckpointInventoryError(f"{name} must be lowercase SHA-256")
        _string(self.repository, "repository", maximum_length=MAX_KEY_LENGTH)
        _string(self.revision, "revision", maximum_length=MAX_KEY_LENGTH)
        _integer(
            self.index_declared_total_size,
            "index_declared_total_size",
            minimum=1,
        )
        if not self.entries:
            raise CheckpointInventoryError("enumerated inventory must not be empty")
        keys = tuple(entry.key for entry in self.entries)
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise CheckpointInventoryError(
                "enumerated inventory entries must be unique and key-sorted"
            )
        if len(self.entries) > MAX_SOURCE_ENTRIES:
            raise CheckpointInventoryError("enumerated inventory exceeds entry bound")
        metadata_shards = tuple(shard for shard, _ in self.header_metadata)
        if metadata_shards != tuple(sorted(metadata_shards)) or len(
            metadata_shards
        ) != len(set(metadata_shards)):
            raise CheckpointInventoryError(
                "header metadata must have unique sorted shard names"
            )
        for item in self.header_metadata:
            if type(item) is not tuple or len(item) != 2:
                raise CheckpointInventoryError(
                    "each header_metadata entry must be a two-item tuple"
                )
            shard, metadata = item
            if type(metadata) is not tuple or not all(
                type(pair) is tuple
                and len(pair) == 2
                and isinstance(pair[0], str)
                and isinstance(pair[1], str)
                for pair in metadata
            ):
                raise CheckpointInventoryError(
                    f"header metadata for {shard} must be nested immutable tuples"
                )
            if _SHARD_NAME.fullmatch(shard) is None:
                raise CheckpointInventoryError(
                    f"invalid header metadata shard name: {shard}"
                )
            if metadata != tuple(sorted(metadata)) or len(metadata) != len(
                {key for key, _ in metadata}
            ):
                raise CheckpointInventoryError(
                    f"header metadata for {shard} must have unique sorted fields"
                )
            for key, value in metadata:
                _string(key, "header metadata key", maximum_length=MAX_KEY_LENGTH)
                _string(value, "header metadata value", maximum_length=MAX_KEY_LENGTH)
        if set(metadata_shards) != set(self.required_shards):
            raise CheckpointInventoryError(
                "header metadata shard set must equal the inventory shard set"
            )

        if self.index_decode_receipt is not None:
            if self.index_decode_receipt.raw_sha256 != self.verified_index_sha256:
                raise CheckpointInventoryError(
                    "index decode receipt raw SHA-256 does not match inventory"
                )
            if self.scope in {
                "full-released-snapshot",
                "full-synthetic-snapshot",
            }:
                reconstructed_index = {
                    "metadata": {"total_size": self.index_declared_total_size},
                    "weight_map": {entry.key: entry.shard for entry in self.entries},
                }
                if self.index_decode_receipt.decoded_sha256 != _sha256(
                    reconstructed_index
                ):
                    raise CheckpointInventoryError(
                        "index decode receipt is not bound to inventory entries"
                    )

        descriptor_shards = tuple(item.shard for item in self.shard_descriptors)
        if self.shard_descriptors:
            if (
                descriptor_shards != tuple(sorted(descriptor_shards))
                or len(descriptor_shards) != len(set(descriptor_shards))
                or set(descriptor_shards) != set(self.required_shards)
            ):
                raise CheckpointInventoryError(
                    "authenticated shard descriptors must cover every required shard exactly"
                )
            if self.authenticated_full_lfs_manifest_sha256 is None:
                raise CheckpointInventoryError(
                    "complete shard descriptors require an authenticated full LFS manifest"
                )
            if self.scope in {
                "full-released-snapshot",
                "full-synthetic-snapshot",
            }:
                entries_by_shard: dict[str, list[SourceTensorEntryV1]] = {
                    shard: [] for shard in descriptor_shards
                }
                for entry in self.entries:
                    entries_by_shard[entry.shard].append(entry)
                metadata_lookup = dict(self.header_metadata)
                for descriptor in self.shard_descriptors:
                    reconstructed_header: dict[str, Any] = {}
                    metadata = metadata_lookup[descriptor.shard]
                    if metadata:
                        reconstructed_header["__metadata__"] = dict(metadata)
                    for entry in entries_by_shard[descriptor.shard]:
                        reconstructed_header[entry.key] = {
                            "dtype": entry.dtype,
                            "shape": list(entry.shape),
                            "data_offsets": list(entry.data_offsets),
                        }
                    if descriptor.decoded_header_sha256 != _sha256(
                        reconstructed_header
                    ):
                        raise CheckpointInventoryError(
                            "descriptor decoded header is not bound to "
                            f"{descriptor.shard} entries"
                        )
                    payload_end = max(
                        entry.data_offsets[1]
                        for entry in entries_by_shard[descriptor.shard]
                    )
                    exact_blob_size = 8 + descriptor.header_length_bytes + payload_end
                    if descriptor.expected_blob_size_bytes != exact_blob_size:
                        raise CheckpointInventoryError(
                            f"descriptor blob size is inconsistent for {descriptor.shard}"
                        )
                computed_manifest = _sha256(
                    [item.lfs_manifest_dict() for item in self.shard_descriptors]
                )
                if computed_manifest != self.authenticated_full_lfs_manifest_sha256:
                    raise CheckpointInventoryError(
                        "authenticated full LFS manifest digest does not match descriptors"
                    )
        elif self.authenticated_full_lfs_manifest_sha256 is not None:
            raise CheckpointInventoryError(
                "an authenticated LFS manifest cannot exist without shard descriptors"
            )
        if (
            self.authenticated_full_lfs_manifest_sha256 is not None
            and _SHA256.fullmatch(self.authenticated_full_lfs_manifest_sha256) is None
        ):
            raise CheckpointInventoryError(
                "authenticated_full_lfs_manifest_sha256 must be lowercase SHA-256"
            )

        _validate_fp8_scale_bijection(self.entries, self.geometry)
        if self.scope in {"full-released-snapshot", "full-synthetic-snapshot"}:
            _compare_expected_specs(self.entries, self.geometry)
        else:
            logical_specs = expected_prefix_logical_specs(self.geometry)
            expected: dict[str, ExpectedTensorSpecV1] = {
                spec.key: spec for spec in logical_specs
            }
            expected.update(
                {
                    spec.key + _SCALE_SUFFIX: ExpectedTensorSpecV1(
                        spec.key + _SCALE_SUFFIX,
                        F32_DTYPE,
                        _scale_shape(spec.shape, self.geometry.fp8_block_size),
                        "fp8_inverse_scale",
                        spec.layer_zero_indexed,
                        spec.expert_index,
                    )
                    for spec in logical_specs
                    if spec.dtype == FP8_DTYPE
                }
            )
            actual = {entry.key: entry for entry in self.entries}
            if set(actual) != set(expected):
                raise CheckpointInventoryError(
                    "selected inventory does not contain the exact routed prefix key set"
                )
            for key, spec in expected.items():
                entry = actual[key]
                if entry.dtype != spec.dtype or entry.shape != spec.shape:
                    raise CheckpointInventoryError(
                        f"selected source metadata mismatch for {key}"
                    )
            validate_prefix_source_keys(actual, self.geometry)

        if self.source_kind == "pinned-r1-0528":
            if self.geometry != PINNED_GEOMETRY:
                raise CheckpointInventoryError(
                    "pinned R1 source kind requires the exact block-31 geometry"
                )
            if self.repository != HF_REPOSITORY or self.revision != HF_REVISION:
                raise CheckpointInventoryError(
                    "pinned geometry requires the pinned HF identity"
                )
            if self.config_sha256 != HF_CONFIG_SHA256:
                raise CheckpointInventoryError("pinned config SHA-256 mismatch")
            if self.modeling_sha256 != HF_MODELING_SHA256:
                raise CheckpointInventoryError("pinned modeling SHA-256 mismatch")
            if self.verified_index_sha256 != HF_INDEX_SHA256:
                raise CheckpointInventoryError(
                    "pinned geometry requires the pinned index SHA-256"
                )
            if self.index_declared_total_size != HF_INDEX_DECLARED_TOTAL_SIZE:
                raise CheckpointInventoryError(
                    "pinned index metadata.total_size does not match R1-0528"
                )
            if self.scope == "full-released-snapshot":
                _assert_exact_totals(self.totals, PINNED_FULL_TOTALS, "full")
                expected_shards = PINNED_FULL_SHARDS
            else:
                _assert_exact_totals(self.totals, PINNED_PREFIX_TOTALS, "prefix")
                expected_shards = PINNED_PREFIX_SHARDS
            if self.required_shards != expected_shards:
                raise CheckpointInventoryError(
                    f"{self.scope} required shard set does not match the pinned source"
                )
            if self.shard_descriptors:
                if (
                    self.authenticated_full_lfs_manifest_sha256
                    != PINNED_LFS_MANIFEST_SHA256
                ):
                    raise CheckpointInventoryError(
                        "authenticated shard descriptors do not match the pinned LFS manifest"
                    )

    @property
    def metadata_authenticated(self) -> bool:
        return self.index_decode_receipt is not None and bool(self.shard_descriptors)

    @property
    def inventory_sealable(self) -> bool:
        return (
            self.source_kind == "pinned-r1-0528"
            and self.scope == "full-released-snapshot"
            and self.metadata_authenticated
        )

    @property
    def conversion_sealable(self) -> bool:
        # Tensor payload hashes, conversion outputs, restore, and parity live in
        # later contracts. Metadata qualification can never imply those gates.
        return False

    @property
    def totals(self) -> InventoryTotalsV1:
        return _totals(self.entries)

    @property
    def required_shards(self) -> tuple[str, ...]:
        return tuple(sorted({entry.shard for entry in self.entries}))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "source_kind": self.source_kind,
            "scope": self.scope,
            "repository": self.repository,
            "revision": self.revision,
            "config_sha256": self.config_sha256,
            "modeling_sha256": self.modeling_sha256,
            "verified_index_sha256": self.verified_index_sha256,
            "index_decode_receipt": (
                self.index_decode_receipt.as_dict()
                if self.index_decode_receipt is not None
                else None
            ),
            "index_declared_total_size": self.index_declared_total_size,
            "geometry": self.geometry.as_dict(),
            "geometry_sha256": self.geometry.digest,
            "totals": self.totals.as_dict(),
            "required_shards": list(self.required_shards),
            "header_metadata": {
                shard: dict(items) for shard, items in self.header_metadata
            },
            "shard_descriptors": [
                descriptor.as_dict() for descriptor in self.shard_descriptors
            ],
            "authenticated_full_lfs_manifest_sha256": (
                self.authenticated_full_lfs_manifest_sha256
            ),
            "metadata_authenticated": self.metadata_authenticated,
            "inventory_sealable": self.inventory_sealable,
            "conversion_sealable": self.conversion_sealable,
            "entries": [entry.as_dict() for entry in self.entries],
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.as_dict())

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class MaxTextMappingDescriptorV1:
    """One deterministic source derivation into an unscanned MaxText leaf."""

    source_key: str
    source_dtype: str
    source_shape: tuple[int, ...]
    scale_key: str | None
    destination_key: str
    destination_dtype: str
    destination_shape: tuple[int, ...]
    destination_slice_axis: int | None
    destination_slice_index: int | None
    transform: str
    prefix_semantics: str

    def __post_init__(self) -> None:
        if (
            type(self.source_shape) is not tuple
            or type(self.destination_shape) is not tuple
        ):
            raise CheckpointInventoryError(
                "mapping source and destination shapes must be immutable tuples"
            )
        _string(self.source_key, "mapping source_key", maximum_length=MAX_KEY_LENGTH)
        _string(
            self.destination_key,
            "mapping destination_key",
            maximum_length=MAX_KEY_LENGTH,
        )
        if self.source_dtype not in _DTYPE_BYTES:
            raise CheckpointInventoryError("mapping source_dtype is unsupported")
        _shape(self.source_shape, "mapping source_shape")
        _shape(self.destination_shape, "mapping destination_shape")
        if self.destination_dtype not in {"bfloat16", "float32"}:
            raise CheckpointInventoryError("mapping destination_dtype is unsupported")
        if (self.destination_slice_axis is None) != (
            self.destination_slice_index is None
        ):
            raise CheckpointInventoryError(
                "mapping destination slice axis and index must be set together"
            )
        if self.destination_slice_axis is not None:
            axis = _integer(
                self.destination_slice_axis,
                "mapping destination_slice_axis",
                minimum=0,
            )
            index = _integer(
                self.destination_slice_index,
                "mapping destination_slice_index",
                minimum=0,
            )
            if axis >= len(self.destination_shape):
                raise CheckpointInventoryError(
                    "mapping destination slice axis is outside destination rank"
                )
            if index >= self.destination_shape[axis]:
                raise CheckpointInventoryError(
                    "mapping destination slice index is outside destination shape"
                )
            destination_numel = (
                _numel(self.destination_shape) // self.destination_shape[axis]
            )
        else:
            destination_numel = _numel(self.destination_shape)
        if destination_numel != _numel(self.source_shape):
            raise CheckpointInventoryError(
                "mapping source and destination slice element counts differ"
            )
        if self.source_dtype == FP8_DTYPE:
            if self.scale_key != self.source_key + _SCALE_SUFFIX:
                raise CheckpointInventoryError(
                    "FP8 mapping must name its exact inverse-scale source"
                )
        elif self.scale_key is not None:
            raise CheckpointInventoryError("non-FP8 mapping cannot consume a scale")
        if self.transform not in {
            "identity-v1",
            "transpose-reshape-v1",
            "blockwise-fp8-f32-dequantize-then-transpose-reshape-bf16-v1",
        }:
            raise CheckpointInventoryError("unknown mapping transform")
        if self.prefix_semantics not in {
            "prefix-input",
            "complete-prefix-layer",
            "target-routed-only-before-shared-expert",
        }:
            raise CheckpointInventoryError("unknown mapping prefix semantics")

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_key": self.source_key,
            "source_dtype": self.source_dtype,
            "source_shape": list(self.source_shape),
            "scale_key": self.scale_key,
            "destination_key": self.destination_key,
            "destination_dtype": self.destination_dtype,
            "destination_shape": list(self.destination_shape),
            "destination_slice_axis": self.destination_slice_axis,
            "destination_slice_index": self.destination_slice_index,
            "transform": self.transform,
            "prefix_semantics": self.prefix_semantics,
        }


@dataclass(frozen=True, slots=True)
class CheckpointInventorySelectionV1:
    """Full inventory, exact prefix subset, and its conversion descriptors."""

    SCHEMA: ClassVar[str] = "deepseek-r1-checkpoint-selection-v1"

    full: TensorInventoryV1
    selected: TensorInventoryV1
    mappings: tuple[MaxTextMappingDescriptorV1, ...]

    def __post_init__(self) -> None:
        if (
            type(self.full) is not TensorInventoryV1
            or type(self.selected) is not TensorInventoryV1
        ):
            raise CheckpointInventoryError(
                "selection inventories must be exact TensorInventoryV1 records"
            )
        if type(self.mappings) is not tuple or not all(
            type(item) is MaxTextMappingDescriptorV1 for item in self.mappings
        ):
            raise CheckpointInventoryError(
                "selection mappings must be a MaxTextMappingDescriptorV1 tuple"
            )
        expected_scopes = {
            "pinned-r1-0528": (
                "full-released-snapshot",
                "block-31-routed-prefix",
            ),
            "synthetic-test-fixture": (
                "full-synthetic-snapshot",
                "synthetic-routed-prefix",
            ),
        }.get(self.full.source_kind)
        if expected_scopes is None or self.full.scope != expected_scopes[0]:
            raise CheckpointInventoryError("full inventory has the wrong scope")
        if self.selected.scope != expected_scopes[1]:
            raise CheckpointInventoryError("selected inventory has the wrong scope")
        identity_fields = (
            "source_kind",
            "repository",
            "revision",
            "config_sha256",
            "modeling_sha256",
            "verified_index_sha256",
            "index_decode_receipt",
            "index_declared_total_size",
            "geometry",
            "authenticated_full_lfs_manifest_sha256",
        )
        if any(
            getattr(self.full, field) != getattr(self.selected, field)
            for field in identity_fields
        ):
            raise CheckpointInventoryError(
                "full and selected inventories do not share one source identity"
            )
        full_entries = {entry.key: entry for entry in self.full.entries}
        if any(full_entries.get(entry.key) != entry for entry in self.selected.entries):
            raise CheckpointInventoryError(
                "selected inventory is not an exact subset of the full inventory"
            )
        selected_shards = set(self.selected.required_shards)
        expected_metadata = tuple(
            item for item in self.full.header_metadata if item[0] in selected_shards
        )
        expected_descriptors = tuple(
            item
            for item in self.full.shard_descriptors
            if item.shard in selected_shards
        )
        if self.selected.header_metadata != expected_metadata:
            raise CheckpointInventoryError(
                "selected header metadata is not the exact full-inventory subset"
            )
        if self.selected.shard_descriptors != expected_descriptors:
            raise CheckpointInventoryError(
                "selected shard descriptors are not the exact authenticated subset"
            )
        source_keys = tuple(item.source_key for item in self.mappings)
        if source_keys != tuple(sorted(source_keys)) or len(source_keys) != len(
            set(source_keys)
        ):
            raise CheckpointInventoryError(
                "mapping descriptors must have unique sorted source keys"
            )
        selected_logical = tuple(
            entry.key for entry in self.selected.entries if not entry.is_scale
        )
        if source_keys != selected_logical:
            raise CheckpointInventoryError(
                "mapping descriptors do not cover selected logical tensors exactly"
            )
        destinations = tuple(
            (
                item.destination_key,
                item.destination_slice_axis,
                item.destination_slice_index,
            )
            for item in self.mappings
        )
        if len(destinations) != len(set(destinations)):
            raise CheckpointInventoryError("mapping descriptors collide at destination")
        canonical_mappings = tuple(
            sorted(
                (
                    _mapping_descriptor(spec, self.full.geometry)
                    for spec in expected_prefix_logical_specs(self.full.geometry)
                ),
                key=lambda item: item.source_key,
            )
        )
        if self.mappings != canonical_mappings:
            raise CheckpointInventoryError(
                "mapping descriptors differ from the canonical HF-to-MaxText map"
            )
        if (
            self.full.source_kind == "pinned-r1-0528"
            and self.converted_payload_bytes != 649_674_223_616
        ):
            raise CheckpointInventoryError(
                "converted prefix byte total does not match retained-FP32 contract"
            )

    @property
    def metadata_authenticated(self) -> bool:
        return self.full.metadata_authenticated and self.selected.metadata_authenticated

    @property
    def inventory_sealable(self) -> bool:
        return self.full.inventory_sealable and self.selected.metadata_authenticated

    @property
    def conversion_sealable(self) -> bool:
        return False

    @property
    def converted_payload_bytes(self) -> int:
        total = 0
        for mapping in self.mappings:
            elements = _numel(mapping.destination_shape)
            if mapping.destination_slice_axis is not None:
                elements //= mapping.destination_shape[mapping.destination_slice_axis]
            total += elements * (4 if mapping.destination_dtype == "float32" else 2)
        return total

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "maxtext_revision": MAXTEXT_REVISION,
            "mapping_contract": "deepseek-r1-hf-to-unscanned-maxtext-routed-prefix-v1",
            "full_inventory_sha256": self.full.digest,
            "selected_inventory_sha256": self.selected.digest,
            "selected_totals": self.selected.totals.as_dict(),
            "selected_required_shards": list(self.selected.required_shards),
            "converted_payload_bytes": self.converted_payload_bytes,
            "metadata_authenticated": self.metadata_authenticated,
            "inventory_sealable": self.inventory_sealable,
            "conversion_sealable": self.conversion_sealable,
            "target_layer_one_indexed": self.full.geometry.target_layer_one_indexed,
            "target_semantics": "attention-router-and-routed-experts-only-before-shared-expert",
            "mappings": [mapping.as_dict() for mapping in self.mappings],
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.as_dict())

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


def _parse_index_decode_receipt(
    raw_receipt: Any,
    *,
    index: Mapping[str, Any],
    verified_index_sha256: str,
) -> StrictJSONReceiptV1 | None:
    if raw_receipt is None:
        return None
    raw = _exact_fields(
        raw_receipt,
        "index decode receipt",
        frozenset(
            {
                "raw_sha256",
                "raw_size_bytes",
                "decoded_sha256",
                "decoder_contract",
            }
        ),
    )
    receipt = StrictJSONReceiptV1(
        raw_sha256=_string(
            raw["raw_sha256"], "index receipt raw_sha256", maximum_length=64
        ),
        raw_size_bytes=_integer(
            raw["raw_size_bytes"],
            "index receipt raw_size_bytes",
            minimum=1,
            maximum=MAX_RAW_INDEX_BYTES,
        ),
        decoded_sha256=_string(
            raw["decoded_sha256"],
            "index receipt decoded_sha256",
            maximum_length=64,
        ),
        decoder_contract=_string(
            raw["decoder_contract"],
            "index receipt decoder_contract",
            maximum_length=128,
        ),
    )
    if receipt.raw_sha256 != verified_index_sha256:
        raise CheckpointInventoryError(
            "index decode receipt does not match verified raw index SHA-256"
        )
    if receipt.decoded_sha256 != _sha256(index):
        raise CheckpointInventoryError(
            "index decode receipt does not bind the supplied decoded mapping"
        )
    return receipt


def _parse_index(index: Any) -> tuple[int, dict[str, str]]:
    root = _exact_fields(
        index, "safetensor index", frozenset({"metadata", "weight_map"})
    )
    metadata = _exact_fields(
        root["metadata"], "safetensor index metadata", frozenset({"total_size"})
    )
    total_size = _integer(metadata["total_size"], "metadata.total_size", minimum=1)
    weight_map_raw = _mapping(root["weight_map"], "weight_map")
    if not weight_map_raw or len(weight_map_raw) > MAX_SOURCE_ENTRIES:
        raise CheckpointInventoryError(
            f"weight_map entry count must be in [1, {MAX_SOURCE_ENTRIES}]"
        )
    weight_map: dict[str, str] = {}
    for raw_key, raw_shard in weight_map_raw.items():
        key = _string(raw_key, "weight_map key", maximum_length=MAX_KEY_LENGTH)
        shard = _string(
            raw_shard, f"weight_map[{key}]", maximum_length=MAX_SHARD_NAME_LENGTH
        )
        if _SHARD_NAME.fullmatch(shard) is None:
            raise CheckpointInventoryError(f"invalid safetensor shard name: {shard}")
        weight_map[key] = shard
    if len(set(weight_map.values())) > MAX_SHARDS:
        raise CheckpointInventoryError("weight_map exceeds shard bound")
    return total_size, weight_map


def _parse_header_metadata(value: Any, shard: str) -> tuple[tuple[str, str], ...]:
    if value is None:
        return ()
    metadata = _mapping(value, f"{shard}.__metadata__")
    if len(metadata) > MAX_HEADER_METADATA_FIELDS:
        raise CheckpointInventoryError(f"{shard} header metadata exceeds field bound")
    result: list[tuple[str, str]] = []
    for key, item in metadata.items():
        clean_key = _string(key, f"{shard} metadata key", maximum_length=MAX_KEY_LENGTH)
        clean_value = _string(
            item, f"{shard} metadata[{clean_key}]", maximum_length=MAX_KEY_LENGTH
        )
        result.append((clean_key, clean_value))
    return tuple(sorted(result))


def _parse_tensor_entry(
    *,
    key: str,
    shard: str,
    raw: Any,
) -> SourceTensorEntryV1:
    metadata = _exact_fields(
        raw,
        f"header tensor {key}",
        frozenset({"dtype", "shape", "data_offsets"}),
    )
    dtype = _string(metadata["dtype"], f"{key}.dtype", maximum_length=32)
    if dtype not in _DTYPE_BYTES:
        raise CheckpointInventoryError(f"unsupported source dtype for {key}: {dtype}")
    shape = _shape(metadata["shape"], f"{key}.shape")
    offsets_raw = metadata["data_offsets"]
    if not isinstance(offsets_raw, (list, tuple)) or len(offsets_raw) != 2:
        raise CheckpointInventoryError(f"{key}.data_offsets must contain two integers")
    start = _integer(offsets_raw[0], f"{key}.data_offsets[0]", minimum=0)
    end = _integer(offsets_raw[1], f"{key}.data_offsets[1]", minimum=0)
    if end <= start:
        raise CheckpointInventoryError(f"{key}.data_offsets must be increasing")
    expected_bytes = _numel(shape) * _DTYPE_BYTES[dtype]
    if end - start != expected_bytes:
        raise CheckpointInventoryError(
            f"payload byte mismatch for {key}: expected {expected_bytes}, got {end - start}"
        )
    return SourceTensorEntryV1(key, shard, dtype, shape, (start, end))


def _parse_headers(
    headers_by_shard: Any,
    weight_map: Mapping[str, str],
) -> tuple[
    tuple[SourceTensorEntryV1, ...],
    tuple[tuple[str, tuple[tuple[str, str], ...]], ...],
]:
    headers = _mapping(headers_by_shard, "headers_by_shard")
    declared_shards = set(weight_map.values())
    if len(declared_shards) > MAX_SHARDS:
        raise CheckpointInventoryError("declared shard set exceeds bound")
    if set(headers) != declared_shards:
        missing = sorted(declared_shards - set(headers))
        extra = sorted(set(headers) - declared_shards)
        raise CheckpointInventoryError(
            f"header shard set mismatch: missing={missing}; extra={extra}"
        )

    entries_by_key: dict[str, SourceTensorEntryV1] = {}
    metadata_by_shard: list[tuple[str, tuple[tuple[str, str], ...]]] = []
    for shard in sorted(headers):
        header = _mapping(headers[shard], f"header {shard}")
        metadata_by_shard.append(
            (shard, _parse_header_metadata(header.get("__metadata__"), shard))
        )
        shard_entries: list[SourceTensorEntryV1] = []
        for raw_key, raw_entry in header.items():
            if raw_key == "__metadata__":
                continue
            key = _string(raw_key, "header tensor key", maximum_length=MAX_KEY_LENGTH)
            if len(entries_by_key) >= MAX_SOURCE_ENTRIES:
                raise CheckpointInventoryError("header entries exceed bound")
            if key in entries_by_key:
                raise CheckpointInventoryError(
                    f"duplicate source tensor key across headers: {key}"
                )
            if weight_map.get(key) != shard:
                raise CheckpointInventoryError(
                    f"header/index shard disagreement for {key}"
                )
            parsed = _parse_tensor_entry(key=key, shard=shard, raw=raw_entry)
            entries_by_key[key] = parsed
            shard_entries.append(parsed)

        ordered_offsets = sorted(
            (entry.data_offsets[0], entry.data_offsets[1], entry.key)
            for entry in shard_entries
        )
        cursor = 0
        for start, end, key in ordered_offsets:
            if start != cursor:
                raise CheckpointInventoryError(
                    f"non-contiguous or overlapping payload offsets in {shard} at {key}"
                )
            cursor = end

    if set(entries_by_key) != set(weight_map):
        missing = sorted(set(weight_map) - set(entries_by_key))
        extra = sorted(set(entries_by_key) - set(weight_map))
        raise CheckpointInventoryError(
            f"index/header tensor key mismatch: missing={missing[:5]}; extra={extra[:5]}"
        )
    entries = tuple(sorted(entries_by_key.values(), key=lambda entry: entry.key))
    if sum(entry.payload_bytes for entry in entries) > MAX_TOTAL_PAYLOAD_BYTES:
        raise CheckpointInventoryError("source payload exceeds inventory bound")
    return entries, tuple(metadata_by_shard)


def _parse_shard_descriptors(
    raw_descriptors: Any,
    *,
    headers_by_shard: Mapping[str, Mapping[str, Any]],
    entries: tuple[SourceTensorEntryV1, ...],
    geometry: DeepSeekR1CheckpointGeometryV1,
) -> tuple[tuple[ShardSourceDescriptorV1, ...], str | None]:
    """Validate complete receipts, or explicitly return an unauthenticated set."""

    if raw_descriptors is None:
        return (), None
    descriptors = _mapping(raw_descriptors, "shard_descriptors")
    required_shards = {entry.shard for entry in entries}
    if set(descriptors) != required_shards:
        missing = sorted(required_shards - set(descriptors))
        extra = sorted(set(descriptors) - required_shards)
        raise CheckpointInventoryError(
            "shard descriptor set must cover parsed shards exactly: "
            f"missing={missing}; extra={extra}"
        )

    entries_by_shard: dict[str, list[SourceTensorEntryV1]] = {
        shard: [] for shard in required_shards
    }
    for entry in entries:
        entries_by_shard[entry.shard].append(entry)

    parsed: list[ShardSourceDescriptorV1] = []
    for shard in sorted(descriptors):
        raw = _exact_fields(
            descriptors[shard],
            f"shard descriptor {shard}",
            frozenset(
                {
                    "shard",
                    "expected_lfs_sha256",
                    "expected_blob_size_bytes",
                    "raw_header_sha256",
                    "header_length_bytes",
                    "decoded_header_sha256",
                    "decoder_contract",
                }
            ),
        )
        descriptor = ShardSourceDescriptorV1(
            shard=_string(
                raw["shard"],
                f"{shard}.shard",
                maximum_length=MAX_SHARD_NAME_LENGTH,
            ),
            expected_lfs_sha256=_string(
                raw["expected_lfs_sha256"],
                f"{shard}.expected_lfs_sha256",
                maximum_length=64,
            ),
            expected_blob_size_bytes=_integer(
                raw["expected_blob_size_bytes"],
                f"{shard}.expected_blob_size_bytes",
                minimum=1,
                maximum=MAX_TOTAL_PAYLOAD_BYTES,
            ),
            raw_header_sha256=_string(
                raw["raw_header_sha256"],
                f"{shard}.raw_header_sha256",
                maximum_length=64,
            ),
            header_length_bytes=_integer(
                raw["header_length_bytes"],
                f"{shard}.header_length_bytes",
                minimum=1,
                maximum=MAX_RAW_HEADER_BYTES,
            ),
            decoded_header_sha256=_string(
                raw["decoded_header_sha256"],
                f"{shard}.decoded_header_sha256",
                maximum_length=64,
            ),
            decoder_contract=_string(
                raw["decoder_contract"],
                f"{shard}.decoder_contract",
                maximum_length=128,
            ),
        )
        if descriptor.shard != shard:
            raise CheckpointInventoryError(f"descriptor key/name mismatch for {shard}")
        if descriptor.header_length_bytes % 8 != 0:
            raise CheckpointInventoryError(
                f"safetensor raw header length is not eight-byte aligned for {shard}"
            )
        decoded_digest = _sha256(headers_by_shard[shard])
        if descriptor.decoded_header_sha256 != decoded_digest:
            raise CheckpointInventoryError(
                f"decoded header digest mismatch for {shard}"
            )
        payload_end = max(entry.data_offsets[1] for entry in entries_by_shard[shard])
        exact_blob_size = 8 + descriptor.header_length_bytes + payload_end
        if descriptor.expected_blob_size_bytes != exact_blob_size:
            raise CheckpointInventoryError(
                f"blob/header/payload size mismatch for {shard}: "
                f"expected {exact_blob_size}, got {descriptor.expected_blob_size_bytes}"
            )
        parsed.append(descriptor)

    result = tuple(parsed)
    lfs_manifest = [descriptor.lfs_manifest_dict() for descriptor in result]
    manifest_digest = _sha256(lfs_manifest)
    if geometry == PINNED_GEOMETRY:
        if len(result) != PINNED_LFS_MANIFEST_ENTRY_COUNT:
            raise CheckpointInventoryError(
                "pinned LFS descriptor manifest must contain exactly 163 shards"
            )
        if sum(item.expected_blob_size_bytes for item in result) != (
            PINNED_LFS_TOTAL_BLOB_BYTES
        ):
            raise CheckpointInventoryError(
                "pinned LFS descriptor blob-size total mismatch"
            )
        if manifest_digest != PINNED_LFS_MANIFEST_SHA256:
            raise CheckpointInventoryError(
                "pinned LFS descriptor manifest SHA-256 mismatch"
            )
    return result, manifest_digest


def _validate_fp8_scale_bijection(
    entries: tuple[SourceTensorEntryV1, ...],
    geometry: DeepSeekR1CheckpointGeometryV1,
) -> None:
    by_key = {entry.key: entry for entry in entries}
    fp8_keys = {entry.key for entry in entries if entry.dtype == FP8_DTYPE}
    scale_keys = {entry.key for entry in entries if entry.is_scale}
    expected_scale_keys = {key + _SCALE_SUFFIX for key in fp8_keys}
    missing = sorted(expected_scale_keys - scale_keys)
    orphaned = sorted(scale_keys - expected_scale_keys)
    if missing:
        raise CheckpointInventoryError(f"missing FP8 inverse scale: {missing[0]}")
    if orphaned:
        raise CheckpointInventoryError(f"orphan FP8 inverse scale: {orphaned[0]}")
    for fp8_key in sorted(fp8_keys):
        weight = by_key[fp8_key]
        scale = by_key[fp8_key + _SCALE_SUFFIX]
        if scale.dtype != F32_DTYPE:
            raise CheckpointInventoryError(
                f"inverse scale for {fp8_key} must have dtype {F32_DTYPE}"
            )
        expected_shape = _scale_shape(weight.shape, geometry.fp8_block_size)
        if scale.shape != expected_shape:
            raise CheckpointInventoryError(
                f"inverse scale shape mismatch for {fp8_key}: "
                f"expected {expected_shape}, got {scale.shape}"
            )


def _compare_expected_specs(
    entries: tuple[SourceTensorEntryV1, ...],
    geometry: DeepSeekR1CheckpointGeometryV1,
) -> None:
    expected = {spec.key: spec for spec in expected_full_source_specs(geometry)}
    actual = {entry.key: entry for entry in entries}
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise CheckpointInventoryError(
            "full source key set mismatch: "
            f"missing={missing[:5]}; unexpected={extra[:5]}"
        )
    for key in sorted(expected):
        spec = expected[key]
        entry = actual[key]
        if entry.shape != spec.shape:
            raise CheckpointInventoryError(
                f"source shape mismatch for {key}: expected {spec.shape}, got {entry.shape}"
            )
        if entry.dtype != spec.dtype:
            raise CheckpointInventoryError(
                f"source dtype mismatch for {key}: expected {spec.dtype}, got {entry.dtype}"
            )


def _assert_exact_totals(
    actual: InventoryTotalsV1,
    expected: InventoryTotalsV1,
    name: str,
) -> None:
    if actual != expected:
        raise CheckpointInventoryError(
            f"{name} totals do not match pinned R1-0528 inventory: "
            f"expected={expected.as_dict()}, actual={actual.as_dict()}"
        )


def _parse_full_hf_inventory(
    index: Mapping[str, Any],
    headers_by_shard: Mapping[str, Mapping[str, Any]],
    *,
    verified_index_sha256: str,
    index_decode_receipt: Mapping[str, Any] | None,
    shard_descriptors: Mapping[str, Mapping[str, Any]] | None,
    geometry: DeepSeekR1CheckpointGeometryV1,
    source_kind: str,
    repository: str,
    revision: str,
    config_sha256: str,
    modeling_sha256: str,
) -> TensorInventoryV1:
    if _SHA256.fullmatch(verified_index_sha256) is None:
        raise CheckpointInventoryError(
            "verified_index_sha256 must be a lowercase SHA-256"
        )
    parsed_index_receipt = _parse_index_decode_receipt(
        index_decode_receipt,
        index=index,
        verified_index_sha256=verified_index_sha256,
    )
    total_size, weight_map = _parse_index(index)
    entries, header_metadata = _parse_headers(headers_by_shard, weight_map)
    _validate_fp8_scale_bijection(entries, geometry)
    _compare_expected_specs(entries, geometry)
    parsed_descriptors, lfs_manifest_sha256 = _parse_shard_descriptors(
        shard_descriptors,
        headers_by_shard=headers_by_shard,
        entries=entries,
        geometry=geometry,
    )

    inventory = TensorInventoryV1(
        source_kind=source_kind,
        scope=(
            "full-released-snapshot"
            if source_kind == "pinned-r1-0528"
            else "full-synthetic-snapshot"
        ),
        repository=repository,
        revision=revision,
        config_sha256=config_sha256,
        modeling_sha256=modeling_sha256,
        verified_index_sha256=verified_index_sha256,
        index_decode_receipt=parsed_index_receipt,
        index_declared_total_size=total_size,
        geometry=geometry,
        entries=entries,
        header_metadata=header_metadata,
        shard_descriptors=parsed_descriptors,
        authenticated_full_lfs_manifest_sha256=lfs_manifest_sha256,
    )
    return inventory


def parse_full_hf_inventory(
    index: Mapping[str, Any],
    headers_by_shard: Mapping[str, Mapping[str, Any]],
    *,
    verified_index_sha256: str,
    index_decode_receipt: Mapping[str, Any] | None = None,
    shard_descriptors: Mapping[str, Mapping[str, Any]] | None = None,
) -> TensorInventoryV1:
    """Parse the exact pinned R1-0528 inventory.

    ``verified_index_sha256`` is the independently measured digest of the raw
    index bytes. Decoded mappings are useful for inventory derivation but are
    not authenticated headers: ``inventory_sealable`` remains false unless an
    exact complete ``shard_descriptors`` manifest is supplied and validated.
    Even then, ``conversion_sealable`` remains false until payload conversion,
    restore, and parity gates have passed.
    """

    return _parse_full_hf_inventory(
        index,
        headers_by_shard,
        verified_index_sha256=verified_index_sha256,
        index_decode_receipt=index_decode_receipt,
        shard_descriptors=shard_descriptors,
        geometry=PINNED_GEOMETRY,
        source_kind="pinned-r1-0528",
        repository=HF_REPOSITORY,
        revision=HF_REVISION,
        config_sha256=HF_CONFIG_SHA256,
        modeling_sha256=HF_MODELING_SHA256,
    )


def parse_synthetic_hf_inventory(
    index: Mapping[str, Any],
    headers_by_shard: Mapping[str, Mapping[str, Any]],
    *,
    verified_index_sha256: str,
    index_decode_receipt: Mapping[str, Any] | None,
    shard_descriptors: Mapping[str, Mapping[str, Any]] | None,
    geometry: DeepSeekR1CheckpointGeometryV1,
) -> TensorInventoryV1:
    """Explicit test-only parser; its result can never authorize production."""

    if geometry == PINNED_GEOMETRY:
        raise CheckpointInventoryError(
            "synthetic parser cannot be used with the pinned production geometry"
        )
    return _parse_full_hf_inventory(
        index,
        headers_by_shard,
        verified_index_sha256=verified_index_sha256,
        index_decode_receipt=index_decode_receipt,
        shard_descriptors=shard_descriptors,
        geometry=geometry,
        source_kind="synthetic-test-fixture",
        repository="synthetic/deepseek-r1-fixture",
        revision="synthetic-v1",
        config_sha256="0" * 64,
        modeling_sha256="0" * 64,
    )


def _strip_scale_suffix(key: str) -> str:
    return key[: -len(_SCALE_SUFFIX)] if key.endswith(_SCALE_SUFFIX) else key


def validate_prefix_source_keys(
    keys: Iterable[str],
    geometry: DeepSeekR1CheckpointGeometryV1 = PINNED_GEOMETRY,
) -> None:
    """Reject every key category forbidden from the routed prefix."""

    logical_specs = expected_prefix_logical_specs(geometry)
    expected = {spec.key for spec in logical_specs}
    expected.update(
        spec.key + _SCALE_SUFFIX for spec in logical_specs if spec.dtype == FP8_DTYPE
    )
    target = geometry.target_layer_zero_indexed
    for raw_key in keys:
        if raw_key in expected:
            continue
        key = _strip_scale_suffix(raw_key)
        if key.startswith(f"model.layers.{target}.mlp.shared_experts."):
            reason = "target shared expert"
        elif key in {"model.norm.weight", "lm_head.weight"}:
            reason = "final normalization or LM head"
        else:
            match = _LAYER_KEY.fullmatch(key)
            if (
                match is not None
                and int(match.group(1)) == geometry.mtp_layer_zero_indexed
            ):
                reason = "MTP block"
            elif match is not None and int(match.group(1)) > target:
                reason = "suffix layer"
            elif match is not None and int(match.group(1)) == target:
                reason = "unsupported target component"
            else:
                reason = "unknown or non-prefix tensor"
        raise CheckpointInventoryError(f"prefix leakage ({reason}): {raw_key}")


_DESTINATION_SUFFIX = {
    "input_norm": "pre_self_attention_layer_norm-scale",
    "post_attention_norm": "post_self_attention_layer_norm-scale",
    "q_a_projection": "self_attention-wq_a-kernel",
    "q_a_norm": "self_attention-q_norm-scale",
    "q_b_projection": "self_attention-wq_b-kernel",
    "kv_a_projection": "self_attention-wkv_a-kernel",
    "kv_a_norm": "self_attention-kv_norm-scale",
    "kv_b_projection": "self_attention-wkv_b-kernel",
    "attention_output_projection": "self_attention-out-kernel",
    "dense_gate_projection": "mlp-wi_0-kernel",
    "dense_up_projection": "mlp-wi_1-kernel",
    "dense_down_projection": "mlp-wo-kernel",
    "router_gate": "DeepSeekMoeBlock_0-MoeBlock_0-gate-kernel",
    "router_correction_bias": "DeepSeekMoeBlock_0-MoeBlock_0-gate-bias",
    "routed_expert_gate_projection": "DeepSeekMoeBlock_0-MoeBlock_0-wi_0",
    "routed_expert_up_projection": "DeepSeekMoeBlock_0-MoeBlock_0-wi_1",
    "routed_expert_down_projection": "DeepSeekMoeBlock_0-MoeBlock_0-wo",
    "shared_expert_gate_projection": "DeepSeekMoeBlock_0-shared_experts-wi_0-kernel",
    "shared_expert_up_projection": "DeepSeekMoeBlock_0-shared_experts-wi_1-kernel",
    "shared_expert_down_projection": "DeepSeekMoeBlock_0-shared_experts-wo-kernel",
}


def _destination_shape(
    spec: ExpectedTensorSpecV1,
    geometry: DeepSeekR1CheckpointGeometryV1,
) -> tuple[int, ...]:
    g = geometry
    shapes = {
        "token_embedding": (g.vocab_size, g.hidden_dim),
        "input_norm": (g.hidden_dim,),
        "post_attention_norm": (g.hidden_dim,),
        "q_a_projection": (g.hidden_dim, g.q_lora_rank),
        "q_a_norm": (g.q_lora_rank,),
        "q_b_projection": (
            g.q_lora_rank,
            g.attention_heads,
            g.qk_nope_head_dim + g.qk_rope_head_dim,
        ),
        "kv_a_projection": (g.hidden_dim, g.kv_a_projection_dim),
        "kv_a_norm": (g.kv_lora_rank,),
        "kv_b_projection": (
            g.kv_lora_rank,
            g.attention_heads,
            g.qk_nope_head_dim + g.value_head_dim,
        ),
        "attention_output_projection": (
            g.attention_heads,
            g.value_head_dim,
            g.hidden_dim,
        ),
        "dense_gate_projection": (g.hidden_dim, g.dense_mlp_dim),
        "dense_up_projection": (g.hidden_dim, g.dense_mlp_dim),
        "dense_down_projection": (g.dense_mlp_dim, g.hidden_dim),
        "router_gate": (g.hidden_dim, g.routed_experts),
        "router_correction_bias": (g.routed_experts,),
        "routed_expert_gate_projection": (
            g.routed_experts,
            g.hidden_dim,
            g.routed_mlp_dim,
        ),
        "routed_expert_up_projection": (
            g.routed_experts,
            g.hidden_dim,
            g.routed_mlp_dim,
        ),
        "routed_expert_down_projection": (
            g.routed_experts,
            g.routed_mlp_dim,
            g.hidden_dim,
        ),
        "shared_expert_gate_projection": (g.hidden_dim, g.routed_mlp_dim),
        "shared_expert_up_projection": (g.hidden_dim, g.routed_mlp_dim),
        "shared_expert_down_projection": (g.routed_mlp_dim, g.hidden_dim),
    }
    try:
        return shapes[spec.family]
    except KeyError as error:
        raise CheckpointInventoryError(
            f"unsupported destination tensor family: {spec.family}"
        ) from error


def _mapping_descriptor(
    spec: ExpectedTensorSpecV1,
    geometry: DeepSeekR1CheckpointGeometryV1,
) -> MaxTextMappingDescriptorV1:
    if spec.family == "token_embedding":
        destination_key = "params-token_embedder-embedding"
    else:
        if spec.layer_zero_indexed is None:
            raise CheckpointInventoryError(f"layer missing for {spec.key}")
        layer = spec.layer_zero_indexed
        if layer < geometry.first_dense_layers:
            layer_prefix = f"params-decoder-dense_layers_{layer}"
        else:
            layer_prefix = (
                f"params-decoder-moe_layers_{layer - geometry.first_dense_layers}"
            )
        try:
            destination_key = layer_prefix + "-" + _DESTINATION_SUFFIX[spec.family]
        except KeyError as error:
            raise CheckpointInventoryError(
                f"unsupported source mapping family: {spec.family}"
            ) from error

    is_expert = spec.family.startswith("routed_expert_")
    if is_expert and spec.expert_index is None:
        raise CheckpointInventoryError(f"expert index missing for {spec.key}")
    if not is_expert and spec.expert_index is not None:
        raise CheckpointInventoryError(f"unexpected expert index for {spec.key}")

    if spec.dtype == FP8_DTYPE:
        transform = "blockwise-fp8-f32-dequantize-then-transpose-reshape-bf16-v1"
        destination_dtype = "bfloat16"
        scale_key = spec.key + _SCALE_SUFFIX
    elif len(spec.shape) == 2 and spec.family != "token_embedding":
        transform = "transpose-reshape-v1"
        destination_dtype = "float32" if spec.dtype == F32_DTYPE else "bfloat16"
        scale_key = None
    else:
        transform = "identity-v1"
        destination_dtype = "float32" if spec.dtype == F32_DTYPE else "bfloat16"
        scale_key = None

    if spec.family == "token_embedding":
        semantics = "prefix-input"
    elif spec.layer_zero_indexed == geometry.target_layer_zero_indexed:
        semantics = "target-routed-only-before-shared-expert"
    else:
        semantics = "complete-prefix-layer"
    return MaxTextMappingDescriptorV1(
        source_key=spec.key,
        source_dtype=spec.dtype,
        source_shape=spec.shape,
        scale_key=scale_key,
        destination_key=destination_key,
        destination_dtype=destination_dtype,
        destination_shape=_destination_shape(spec, geometry),
        destination_slice_axis=0 if is_expert else None,
        destination_slice_index=spec.expert_index if is_expert else None,
        transform=transform,
        prefix_semantics=semantics,
    )


def _select_routed_prefix(full: TensorInventoryV1) -> CheckpointInventorySelectionV1:
    if full.scope not in {"full-released-snapshot", "full-synthetic-snapshot"}:
        raise CheckpointInventoryError("prefix selection requires a full inventory")
    geometry = full.geometry
    logical_specs = expected_prefix_logical_specs(geometry)
    selected_keys = {spec.key for spec in logical_specs}
    selected_keys.update(
        spec.key + _SCALE_SUFFIX for spec in logical_specs if spec.dtype == FP8_DTYPE
    )
    selected_entries = tuple(
        entry for entry in full.entries if entry.key in selected_keys
    )
    if {entry.key for entry in selected_entries} != selected_keys:
        raise CheckpointInventoryError(
            "validated full inventory lost required prefix keys"
        )
    validate_prefix_source_keys((entry.key for entry in selected_entries), geometry)
    selected_shards = {entry.shard for entry in selected_entries}
    selected_metadata = tuple(
        item for item in full.header_metadata if item[0] in selected_shards
    )
    selected_descriptors = tuple(
        item for item in full.shard_descriptors if item.shard in selected_shards
    )
    selected = TensorInventoryV1(
        source_kind=full.source_kind,
        scope=(
            "block-31-routed-prefix"
            if full.source_kind == "pinned-r1-0528"
            else "synthetic-routed-prefix"
        ),
        repository=full.repository,
        revision=full.revision,
        config_sha256=full.config_sha256,
        modeling_sha256=full.modeling_sha256,
        verified_index_sha256=full.verified_index_sha256,
        index_decode_receipt=full.index_decode_receipt,
        index_declared_total_size=full.index_declared_total_size,
        geometry=geometry,
        entries=selected_entries,
        header_metadata=selected_metadata,
        shard_descriptors=selected_descriptors,
        authenticated_full_lfs_manifest_sha256=(
            full.authenticated_full_lfs_manifest_sha256
        ),
    )
    mappings = tuple(
        sorted(
            (_mapping_descriptor(spec, geometry) for spec in logical_specs),
            key=lambda item: item.source_key,
        )
    )
    return CheckpointInventorySelectionV1(full, selected, mappings)


def select_block31_prefix(full: TensorInventoryV1) -> CheckpointInventorySelectionV1:
    """Derive the exact one-indexed block-31 prefix from pinned R1-0528."""

    if full.source_kind != "pinned-r1-0528" or full.geometry != PINNED_GEOMETRY:
        raise CheckpointInventoryError(
            "select_block31_prefix accepts only the exact pinned R1-0528 inventory"
        )
    return _select_routed_prefix(full)


def select_synthetic_routed_prefix(
    full: TensorInventoryV1,
) -> CheckpointInventorySelectionV1:
    """Explicit test-fixture selector whose result is never inventory-sealable."""

    if full.source_kind != "synthetic-test-fixture":
        raise CheckpointInventoryError(
            "synthetic selector accepts only an explicit synthetic inventory"
        )
    return _select_routed_prefix(full)


def build_checkpoint_inventory_selection(
    index: Mapping[str, Any],
    headers_by_shard: Mapping[str, Mapping[str, Any]],
    *,
    verified_index_sha256: str,
    index_decode_receipt: Mapping[str, Any] | None = None,
    shard_descriptors: Mapping[str, Mapping[str, Any]] | None = None,
) -> CheckpointInventorySelectionV1:
    """Parse the full source and derive its exact block-31 prefix in one call."""

    full = parse_full_hf_inventory(
        index,
        headers_by_shard,
        verified_index_sha256=verified_index_sha256,
        index_decode_receipt=index_decode_receipt,
        shard_descriptors=shard_descriptors,
    )
    return select_block31_prefix(full)


def build_synthetic_checkpoint_inventory_selection(
    index: Mapping[str, Any],
    headers_by_shard: Mapping[str, Mapping[str, Any]],
    *,
    verified_index_sha256: str,
    index_decode_receipt: Mapping[str, Any] | None,
    shard_descriptors: Mapping[str, Mapping[str, Any]] | None,
    geometry: DeepSeekR1CheckpointGeometryV1,
) -> CheckpointInventorySelectionV1:
    """Build a non-production synthetic selection for bounded contract tests."""

    full = parse_synthetic_hf_inventory(
        index,
        headers_by_shard,
        verified_index_sha256=verified_index_sha256,
        index_decode_receipt=index_decode_receipt,
        shard_descriptors=shard_descriptors,
        geometry=geometry,
    )
    return select_synthetic_routed_prefix(full)


@dataclass(frozen=True, slots=True)
class AuditedAggregateTermV1:
    """One prose-audit component; never a substitute for tensor entries."""

    label: str
    logical_tensor_count: int
    fp8_tensor_count: int
    bf16_tensor_count: int
    fp32_tensor_count: int
    fp8_numel: int
    bf16_numel: int
    fp32_numel: int
    scale_numel: int

    def __post_init__(self) -> None:
        _string(self.label, "aggregate label", maximum_length=128)
        for name in self.__dataclass_fields__:
            if name == "label":
                continue
            _integer(getattr(self, name), name, minimum=0)
        if (
            self.fp8_tensor_count + self.bf16_tensor_count + self.fp32_tensor_count
            != self.logical_tensor_count
        ):
            raise CheckpointInventoryError(
                f"aggregate dtype tensor counts do not sum for {self.label}"
            )


@dataclass(frozen=True, slots=True)
class AuditedSummaryCrossCheckV1:
    """Non-sealable arithmetic result with deliberately no digest method."""

    logical_tensor_count: int
    scale_tensor_count: int
    logical_numel: int
    scale_numel: int
    payload_bytes: int

    @property
    def metadata_authenticated(self) -> bool:
        return False

    @property
    def inventory_sealable(self) -> bool:
        return False

    @property
    def conversion_sealable(self) -> bool:
        return False


def audited_summary_cross_check(
    terms: Iterable[AuditedAggregateTermV1],
) -> AuditedSummaryCrossCheckV1:
    """Reduce compact audit terms without creating a source inventory."""

    materialized = tuple(terms)
    if not materialized:
        raise CheckpointInventoryError("at least one aggregate audit term is required")
    labels = tuple(term.label for term in materialized)
    if len(labels) != len(set(labels)):
        raise CheckpointInventoryError("aggregate audit labels must be unique")
    fp8_numel = sum(term.fp8_numel for term in materialized)
    bf16_numel = sum(term.bf16_numel for term in materialized)
    fp32_numel = sum(term.fp32_numel for term in materialized)
    scale_numel = sum(term.scale_numel for term in materialized)
    return AuditedSummaryCrossCheckV1(
        logical_tensor_count=sum(term.logical_tensor_count for term in materialized),
        scale_tensor_count=sum(term.fp8_tensor_count for term in materialized),
        logical_numel=fp8_numel + bf16_numel + fp32_numel,
        scale_numel=scale_numel,
        payload_bytes=(fp8_numel + 2 * bf16_numel + 4 * fp32_numel + 4 * scale_numel),
    )


__all__ = [
    "AuditedAggregateTermV1",
    "AuditedSummaryCrossCheckV1",
    "BF16_DTYPE",
    "CheckpointInventoryError",
    "CheckpointInventorySelectionV1",
    "DeepSeekR1CheckpointGeometryV1",
    "ExpectedTensorSpecV1",
    "F32_DTYPE",
    "FP8_DTYPE",
    "HF_CONFIG_SHA256",
    "HF_INDEX_DECLARED_TOTAL_SIZE",
    "HF_INDEX_SHA256",
    "HF_MODELING_SHA256",
    "HF_REPOSITORY",
    "HF_REVISION",
    "MAXTEXT_REVISION",
    "MaxTextMappingDescriptorV1",
    "PINNED_FULL_SHARDS",
    "PINNED_FULL_TOTALS",
    "PINNED_GEOMETRY",
    "PINNED_LFS_MANIFEST_ENTRY_COUNT",
    "PINNED_LFS_MANIFEST_SHA256",
    "PINNED_LFS_TOTAL_BLOB_BYTES",
    "PINNED_PREFIX_SHARDS",
    "PINNED_PREFIX_TOTALS",
    "SourceTensorEntryV1",
    "ShardSourceDescriptorV1",
    "TensorInventoryV1",
    "audited_summary_cross_check",
    "build_checkpoint_inventory_selection",
    "build_synthetic_checkpoint_inventory_selection",
    "decode_pinned_index_bytes",
    "decode_safetensor_header_bytes",
    "decode_strict_json_bytes",
    "expected_full_logical_specs",
    "expected_full_source_specs",
    "expected_prefix_logical_specs",
    "parse_full_hf_inventory",
    "parse_synthetic_hf_inventory",
    "select_block31_prefix",
    "select_synthetic_routed_prefix",
    "validate_prefix_source_keys",
]
