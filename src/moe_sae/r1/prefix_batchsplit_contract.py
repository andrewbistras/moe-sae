"""Immutable geometry authority for the forward-only R1 batch-split prefix.

This module is deliberately repository-side only.  It imports neither JAX nor
MaxText and implements no kernel, restore path, writer, controller, or cloud
action.  It freezes the first performance candidate and derives the proposed
scanned parameter tree from the canonical checkpoint mapping descriptors.

The unbound performance plan is an exact schema expectation, not evidence that
a production checkpoint was observed or restored.  Even a plan bound to an
authenticated inventory remains non-sealable and cannot authorize admission.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import re
from typing import Any, ClassVar

from .checkpoint_inventory import (
    MAXTEXT_REVISION,
    PINNED_GEOMETRY,
    MaxTextMappingDescriptorV1,
    expected_prefix_mapping_descriptors,
    validate_prefix_source_keys,
)
from .protocol import canonical_json_bytes, canonical_sha256


PINNED_MAXTEXT_REVISION = "c03f8bd307d18049fa814fc8c104bd9ec8c292aa"
PINNED_PREFIX_MAPPING_DESCRIPTOR_SHA256 = (
    "9a586b5fd2b3dd698e1aa3dea3fe942d67ab4742a0880c0a20385bf7e08c5ab6"
)
PINNED_UNBOUND_PREFIX_PERFORMANCE_TREE_PLAN_SHA256 = (
    "6de49e49d81384b764b0958462e2cce58b4b39eb8a7037053cfb881f414332c2"
)

PINNED_SEQUENCE_LENGTH = 1_024
PINNED_GLOBAL_BATCH_SIZE = 128
PINNED_BATCH_SPLIT_FACTOR = 2
PINNED_HOST_COUNT = 16
PINNED_CHIPS_PER_HOST = 4
PINNED_DEVICE_COUNT = 64
PINNED_PARAMETER_SCAN_AXIS = 1

PINNED_DENSE_SCAN_COUNT = 3
PINNED_COMPLETED_MOE_SCAN_COUNT = 27
PINNED_PARTIAL_TARGET_COUNT = 1
PINNED_TARGET_LAYER_ZERO_INDEXED = 30

PINNED_SOURCE_ENTRY_COUNT = 43_679
PINNED_SOURCE_MAPPING_COUNT = 21_930
PINNED_SCALE_MAPPING_COUNT = 21_749
PINNED_SOURCE_DESTINATION_COUNT = 510
PINNED_PERFORMANCE_LEAF_COUNT = 44

_ACTIVATION_BATCH_AXES = ("data", "fsdp", "expert")
_TARGET_EXCLUSIONS = (
    "target-shared-expert",
    "suffix-layers-31-through-60-zero-indexed",
    "final-normalization",
    "logits",
    "mtp-layer-61-zero-indexed",
)
_SECTION_LEAF_COUNTS = {
    "embedding": 1,
    "scanned-dense": 12,
    "scanned-completed-moe": 17,
    "target-routed-only": 14,
}
_SECTION_DESTINATION_COUNTS = {
    "embedding": 1,
    "scanned-dense": 36,
    "scanned-completed-moe": 459,
    "target-routed-only": 14,
}
_DTYPE_BYTES = {
    "bfloat16": 2,
    "float32": 4,
    "int32": 4,
    "int64": 8,
    "uint16": 2,
}
_DENSE_DESTINATION = re.compile(r"params-decoder-dense_layers_(\d+)-(.+)")
_MOE_DESTINATION = re.compile(r"params-decoder-moe_layers_(\d+)-(.+)")
_SOURCE_LAYER = re.compile(r"model\.layers\.(\d+)\.")


class PrefixBatchSplitContractError(ValueError):
    """The proposed batch-split geometry or consumption plan is not exact."""


def _integer(value: Any, name: str, *, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise PrefixBatchSplitContractError(
            f"{name} must be an integer >= {minimum} (booleans are forbidden)"
        )
    return value


def _shape(value: Any, name: str) -> tuple[int, ...]:
    if type(value) is not tuple or not value:
        raise PrefixBatchSplitContractError(f"{name} must be a non-empty tuple")
    for dimension in value:
        _integer(dimension, f"{name}[]")
    return value


def _mapping_descriptor_sha256(
    mappings: Iterable[MaxTextMappingDescriptorV1],
) -> str:
    return canonical_sha256(
        {
            "schema": "deepseek-r1-prefix-mapping-descriptors-v1",
            "mappings": [mapping.as_dict() for mapping in mappings],
        }
    )


@lru_cache(maxsize=1)
def _canonical_mappings() -> tuple[MaxTextMappingDescriptorV1, ...]:
    if MAXTEXT_REVISION != PINNED_MAXTEXT_REVISION:
        raise PrefixBatchSplitContractError(
            "checkpoint inventory MaxText revision differs from batch-split authority"
        )
    mappings = expected_prefix_mapping_descriptors(PINNED_GEOMETRY)
    if len(mappings) != PINNED_SOURCE_MAPPING_COUNT:
        raise PrefixBatchSplitContractError(
            "canonical checkpoint mapping count is not 21,930"
        )
    actual_sha256 = _mapping_descriptor_sha256(mappings)
    if actual_sha256 != PINNED_PREFIX_MAPPING_DESCRIPTOR_SHA256:
        raise PrefixBatchSplitContractError(
            "canonical checkpoint mapping descriptor digest changed"
        )
    return mappings


@dataclass(frozen=True, slots=True)
class BatchSplitMeshLayoutV1:
    """One 64-device layout with no replica-only context or tensor axis."""

    name: str
    data_parallelism: int
    fsdp_parallelism: int
    expert_parallelism: int
    context_parallelism: int = 1
    tensor_parallelism: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise PrefixBatchSplitContractError("layout name must be non-empty")
        for field in (
            "data_parallelism",
            "fsdp_parallelism",
            "expert_parallelism",
            "context_parallelism",
            "tensor_parallelism",
        ):
            _integer(getattr(self, field), field)
        if self.context_parallelism != 1 or self.tensor_parallelism != 1:
            raise PrefixBatchSplitContractError(
                "replica-only context/TP axes are forbidden; both must equal one"
            )
        if self.activation_shard_product != PINNED_DEVICE_COUNT:
            raise PrefixBatchSplitContractError(
                "data×fsdp×expert must shard activation batch over exactly 64 devices"
            )
        if self.device_count != PINNED_DEVICE_COUNT:
            raise PrefixBatchSplitContractError(
                "batch-split layout must use exactly 64 devices"
            )
        expected_name = f"EP{self.expert_parallelism}/FSDP{self.fsdp_parallelism}"
        if self.data_parallelism != 1 or self.name != expected_name:
            raise PrefixBatchSplitContractError(
                "pinned layout names require data=1 and exact EP/FSDP dimensions"
            )

    @property
    def activation_shard_product(self) -> int:
        return self.data_parallelism * self.fsdp_parallelism * self.expert_parallelism

    @property
    def device_count(self) -> int:
        return (
            self.activation_shard_product
            * self.context_parallelism
            * self.tensor_parallelism
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "data_parallelism": self.data_parallelism,
            "fsdp_parallelism": self.fsdp_parallelism,
            "expert_parallelism": self.expert_parallelism,
            "context_parallelism": self.context_parallelism,
            "tensor_parallelism": self.tensor_parallelism,
            "activation_shard_product": self.activation_shard_product,
            "device_count": self.device_count,
        }


PINNED_LAYOUT_LADDER = (
    BatchSplitMeshLayoutV1("EP16/FSDP4", 1, 4, 16),
    BatchSplitMeshLayoutV1("EP32/FSDP2", 1, 2, 32),
    BatchSplitMeshLayoutV1("EP8/FSDP8", 1, 8, 8),
)


@dataclass(frozen=True, slots=True)
class CaptureTensorSchemaV1:
    """One logical capture output and its device/durable dtype boundary."""

    name: str
    shape: tuple[int, ...]
    device_dtype: str
    durable_dtype: str
    semantics: str

    def __post_init__(self) -> None:
        _shape(self.shape, f"{self.name}.shape")
        expected = {
            "routed_activation": (
                (
                    PINNED_GLOBAL_BATCH_SIZE,
                    PINNED_SEQUENCE_LENGTH,
                    PINNED_GEOMETRY.hidden_dim,
                ),
                "bfloat16",
                "bfloat16",
                "target-routed-aggregate-before-shared-expert-and-residual",
            ),
            "global_expert_ids": (
                (
                    PINNED_GLOBAL_BATCH_SIZE,
                    PINNED_SEQUENCE_LENGTH,
                    8,
                ),
                "int32",
                "uint16",
                "global-ids-before-shard-local-remap",
            ),
            "scaled_router_weights": (
                (
                    PINNED_GLOBAL_BATCH_SIZE,
                    PINNED_SEQUENCE_LENGTH,
                    8,
                ),
                "float32",
                "bfloat16",
                "normalized-unbiased-selected-scores-times-2.5",
            ),
        }.get(self.name)
        if expected is None:
            raise PrefixBatchSplitContractError("unknown capture output")
        if (
            self.shape,
            self.device_dtype,
            self.durable_dtype,
            self.semantics,
        ) != expected:
            raise PrefixBatchSplitContractError(
                f"{self.name} differs from the pinned output ABI"
            )

    @property
    def elements(self) -> int:
        result = 1
        for dimension in self.shape:
            result *= dimension
        return result

    @property
    def device_bytes(self) -> int:
        return self.elements * _DTYPE_BYTES[self.device_dtype]

    @property
    def durable_bytes(self) -> int:
        return self.elements * _DTYPE_BYTES[self.durable_dtype]

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "device_dtype": self.device_dtype,
            "durable_dtype": self.durable_dtype,
            "semantics": self.semantics,
            "elements": self.elements,
            "device_bytes_per_step": self.device_bytes,
            "durable_bytes_per_step": self.durable_bytes,
        }


PINNED_CAPTURE_OUTPUTS = (
    CaptureTensorSchemaV1(
        "routed_activation",
        (128, 1_024, 7_168),
        "bfloat16",
        "bfloat16",
        "target-routed-aggregate-before-shared-expert-and-residual",
    ),
    CaptureTensorSchemaV1(
        "global_expert_ids",
        (128, 1_024, 8),
        "int32",
        "uint16",
        "global-ids-before-shard-local-remap",
    ),
    CaptureTensorSchemaV1(
        "scaled_router_weights",
        (128, 1_024, 8),
        "float32",
        "bfloat16",
        "normalized-unbiased-selected-scores-times-2.5",
    ),
)


@dataclass(frozen=True, slots=True)
class DurableMetadataFieldSchemaV1:
    """One host-created durable field, never a device model output."""

    name: str
    shape: tuple[int, ...]
    durable_dtype: str
    semantics: str

    def __post_init__(self) -> None:
        _shape(self.shape, f"{self.name}.shape")
        expected = {
            "sequence_id": (
                (PINNED_GLOBAL_BATCH_SIZE, PINNED_SEQUENCE_LENGTH),
                "int64",
                "stable-sequence-id-repeated-for-each-token",
            ),
            "position": (
                (PINNED_GLOBAL_BATCH_SIZE, PINNED_SEQUENCE_LENGTH),
                "uint16",
                "zero-based-position-within-sequence",
            ),
        }.get(self.name)
        if expected is None:
            raise PrefixBatchSplitContractError("unknown durable metadata field")
        if (self.shape, self.durable_dtype, self.semantics) != expected:
            raise PrefixBatchSplitContractError(
                f"{self.name} differs from the pinned durable metadata ABI"
            )

    @property
    def elements(self) -> int:
        result = 1
        for dimension in self.shape:
            result *= dimension
        return result

    @property
    def durable_bytes(self) -> int:
        return self.elements * _DTYPE_BYTES[self.durable_dtype]

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "durable_dtype": self.durable_dtype,
            "semantics": self.semantics,
            "elements": self.elements,
            "durable_bytes_per_step": self.durable_bytes,
        }


PINNED_DURABLE_METADATA = (
    DurableMetadataFieldSchemaV1(
        "sequence_id",
        (128, 1_024),
        "int64",
        "stable-sequence-id-repeated-for-each-token",
    ),
    DurableMetadataFieldSchemaV1(
        "position",
        (128, 1_024),
        "uint16",
        "zero-based-position-within-sequence",
    ),
)


@dataclass(frozen=True, slots=True)
class PrefixBatchSplitContractV1:
    """Frozen first candidate; construction rejects every geometry drift."""

    SCHEMA: ClassVar[str] = "moe-sae.r1.prefix-batchsplit-contract.v1"

    maxtext_revision: str = PINNED_MAXTEXT_REVISION
    checkpoint_geometry_sha256: str = PINNED_GEOMETRY.digest
    sequence_length: int = PINNED_SEQUENCE_LENGTH
    global_batch_size: int = PINNED_GLOBAL_BATCH_SIZE
    batch_split_factor: int = PINNED_BATCH_SPLIT_FACTOR
    host_count: int = PINNED_HOST_COUNT
    chips_per_host: int = PINNED_CHIPS_PER_HOST
    device_count: int = PINNED_DEVICE_COUNT
    activation_batch_axes: tuple[str, ...] = _ACTIVATION_BATCH_AXES
    layout_ladder: tuple[BatchSplitMeshLayoutV1, ...] = PINNED_LAYOUT_LADDER
    parameter_scan_axis: int = PINNED_PARAMETER_SCAN_AXIS
    dense_scan_count: int = PINNED_DENSE_SCAN_COUNT
    completed_moe_scan_count: int = PINNED_COMPLETED_MOE_SCAN_COUNT
    partial_target_count: int = PINNED_PARTIAL_TARGET_COUNT
    target_layer_zero_indexed: int = PINNED_TARGET_LAYER_ZERO_INDEXED
    source_entry_count: int = PINNED_SOURCE_ENTRY_COUNT
    source_mapping_count: int = PINNED_SOURCE_MAPPING_COUNT
    source_destination_count: int = PINNED_SOURCE_DESTINATION_COUNT
    performance_leaf_count: int = PINNED_PERFORMANCE_LEAF_COUNT
    mapping_descriptor_sha256: str = PINNED_PREFIX_MAPPING_DESCRIPTOR_SHA256
    router_bias_storage_dtype: str = "float32"
    target_exclusions: tuple[str, ...] = _TARGET_EXCLUSIONS
    capture_outputs: tuple[CaptureTensorSchemaV1, ...] = PINNED_CAPTURE_OUTPUTS
    durable_metadata: tuple[DurableMetadataFieldSchemaV1, ...] = PINNED_DURABLE_METADATA
    production_sealable: bool = False
    production_admission_enabled: bool = False

    def __post_init__(self) -> None:
        if self.maxtext_revision != PINNED_MAXTEXT_REVISION:
            raise PrefixBatchSplitContractError("MaxText revision is not pinned")
        if self.checkpoint_geometry_sha256 != PINNED_GEOMETRY.digest:
            raise PrefixBatchSplitContractError(
                "checkpoint geometry digest differs from inventory authority"
            )
        for name, expected in (
            ("sequence_length", PINNED_SEQUENCE_LENGTH),
            ("batch_split_factor", PINNED_BATCH_SPLIT_FACTOR),
            ("host_count", PINNED_HOST_COUNT),
            ("chips_per_host", PINNED_CHIPS_PER_HOST),
            ("device_count", PINNED_DEVICE_COUNT),
            ("parameter_scan_axis", PINNED_PARAMETER_SCAN_AXIS),
            ("dense_scan_count", PINNED_DENSE_SCAN_COUNT),
            ("completed_moe_scan_count", PINNED_COMPLETED_MOE_SCAN_COUNT),
            ("partial_target_count", PINNED_PARTIAL_TARGET_COUNT),
            ("target_layer_zero_indexed", PINNED_TARGET_LAYER_ZERO_INDEXED),
            ("source_entry_count", PINNED_SOURCE_ENTRY_COUNT),
            ("source_mapping_count", PINNED_SOURCE_MAPPING_COUNT),
            ("source_destination_count", PINNED_SOURCE_DESTINATION_COUNT),
            ("performance_leaf_count", PINNED_PERFORMANCE_LEAF_COUNT),
        ):
            value = getattr(self, name)
            _integer(
                value, name, minimum=0 if name == "target_layer_zero_indexed" else 1
            )
            if value != expected:
                raise PrefixBatchSplitContractError(
                    f"{name} must be exactly {expected}"
                )
        _integer(self.global_batch_size, "global_batch_size")
        if self.global_batch_size % self.device_count:
            raise PrefixBatchSplitContractError(
                "global batch must divide across all 64 activation shards"
            )
        sequences_per_chip = self.global_batch_size // self.device_count
        if sequences_per_chip % self.batch_split_factor:
            raise PrefixBatchSplitContractError(
                "B64 is forbidden under product-64 activation sharding: each chip "
                "would receive one sequence, which cannot split by factor two"
            )
        if self.global_batch_size != PINNED_GLOBAL_BATCH_SIZE:
            raise PrefixBatchSplitContractError(
                "first batch-split candidate must be exactly B128"
            )
        if self.host_count * self.chips_per_host != self.device_count:
            raise PrefixBatchSplitContractError("host/chip geometry must total 64")
        if (
            type(self.activation_batch_axes) is not tuple
            or self.activation_batch_axes != _ACTIVATION_BATCH_AXES
        ):
            raise PrefixBatchSplitContractError(
                "activation batch axes must be exactly data×fsdp×expert"
            )
        if (
            type(self.layout_ladder) is not tuple
            or self.layout_ladder != PINNED_LAYOUT_LADDER
        ):
            raise PrefixBatchSplitContractError(
                "layout ladder must be EP16/FSDP4, EP32/FSDP2, EP8/FSDP8"
            )
        if any(
            type(layout) is not BatchSplitMeshLayoutV1 for layout in self.layout_ladder
        ):
            raise PrefixBatchSplitContractError("layout ladder is not immutable")
        if (
            self.dense_scan_count
            + self.completed_moe_scan_count
            + self.partial_target_count
            != self.target_layer_zero_indexed + 1
        ):
            raise PrefixBatchSplitContractError(
                "three dense, 27 complete MoE, and one partial target must end at layer 30"
            )
        if self.mapping_descriptor_sha256 != PINNED_PREFIX_MAPPING_DESCRIPTOR_SHA256:
            raise PrefixBatchSplitContractError(
                "checkpoint mapping descriptor digest is not pinned"
            )
        if self.router_bias_storage_dtype != "float32":
            raise PrefixBatchSplitContractError(
                "router correction bias must remain FP32"
            )
        if (
            type(self.target_exclusions) is not tuple
            or self.target_exclusions != _TARGET_EXCLUSIONS
        ):
            raise PrefixBatchSplitContractError(
                "target exclusions must cover shared, suffix, norm/logits, and MTP"
            )
        if (
            type(self.capture_outputs) is not tuple
            or self.capture_outputs != PINNED_CAPTURE_OUTPUTS
            or any(
                type(output) is not CaptureTensorSchemaV1
                for output in self.capture_outputs
            )
        ):
            raise PrefixBatchSplitContractError("capture output ABI changed")
        if (
            type(self.durable_metadata) is not tuple
            or self.durable_metadata != PINNED_DURABLE_METADATA
            or any(
                type(field) is not DurableMetadataFieldSchemaV1
                for field in self.durable_metadata
            )
        ):
            raise PrefixBatchSplitContractError("durable metadata ABI changed")
        if (
            self.production_sealable is not False
            or self.production_admission_enabled is not False
        ):
            raise PrefixBatchSplitContractError(
                "this first-tranche contract cannot seal or authorize production"
            )

    @property
    def sequences_per_host(self) -> int:
        return self.global_batch_size // self.host_count

    @property
    def sequences_per_chip(self) -> int:
        return self.global_batch_size // self.device_count

    @property
    def sequences_per_chip_split_microbatch(self) -> int:
        return self.sequences_per_chip // self.batch_split_factor

    def capture_byte_arithmetic(self) -> dict[str, Any]:
        """Return separate device-output and complete durable byte totals."""

        token_count = self.global_batch_size * self.sequence_length

        def geometry(per_step: int) -> dict[str, int]:
            if (
                per_step % token_count
                or per_step % self.host_count
                or per_step % self.device_count
                or per_step % self.batch_split_factor
            ):
                raise PrefixBatchSplitContractError(
                    "capture bytes do not divide across pinned geometry"
                )
            return {
                "per_token": per_step // token_count,
                "per_step": per_step,
                "per_host": per_step // self.host_count,
                "per_chip": per_step // self.device_count,
                "per_split_microbatch": per_step // self.batch_split_factor,
                "per_chip_split_microbatch": (
                    per_step // self.device_count // self.batch_split_factor
                ),
            }

        device_outputs = {
            output.name: output.device_bytes for output in self.capture_outputs
        }
        durable_outputs = {
            output.name: output.durable_bytes for output in self.capture_outputs
        }
        durable_metadata = {
            field.name: field.durable_bytes for field in self.durable_metadata
        }
        device_total = sum(device_outputs.values())
        durable_total = sum(durable_outputs.values()) + sum(durable_metadata.values())
        return {
            "device": {
                "scope": "three-device-model-outputs-only",
                "per_model_output_per_step": device_outputs,
                **geometry(device_total),
            },
            "durable": {
                "scope": "durable-model-outputs-plus-host-metadata",
                "per_model_output_per_step": durable_outputs,
                "per_metadata_field_per_step": durable_metadata,
                **geometry(durable_total),
            },
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "maxtext_revision": self.maxtext_revision,
            "checkpoint_geometry_sha256": self.checkpoint_geometry_sha256,
            "sequence_length": self.sequence_length,
            "global_batch_size": self.global_batch_size,
            "batch_split_factor": self.batch_split_factor,
            "host_count": self.host_count,
            "chips_per_host": self.chips_per_host,
            "device_count": self.device_count,
            "activation_batch_axes": list(self.activation_batch_axes),
            "layout_ladder": [layout.as_dict() for layout in self.layout_ladder],
            "parameter_scan_axis": self.parameter_scan_axis,
            "dense_scan_count": self.dense_scan_count,
            "completed_moe_scan_count": self.completed_moe_scan_count,
            "partial_target_count": self.partial_target_count,
            "target_layer_zero_indexed": self.target_layer_zero_indexed,
            "source_entry_count": self.source_entry_count,
            "source_mapping_count": self.source_mapping_count,
            "source_destination_count": self.source_destination_count,
            "performance_leaf_count": self.performance_leaf_count,
            "mapping_descriptor_sha256": self.mapping_descriptor_sha256,
            "router_bias_storage_dtype": self.router_bias_storage_dtype,
            "target_exclusions": list(self.target_exclusions),
            "capture_outputs": [output.as_dict() for output in self.capture_outputs],
            "durable_metadata": [field.as_dict() for field in self.durable_metadata],
            "capture_byte_arithmetic": self.capture_byte_arithmetic(),
            "production_sealable": self.production_sealable,
            "production_admission_enabled": self.production_admission_enabled,
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.as_dict())

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


PINNED_PREFIX_BATCHSPLIT_CONTRACT = PrefixBatchSplitContractV1()
PINNED_PREFIX_BATCHSPLIT_CONTRACT_SHA256 = (
    "3e9c8bf4aae62ea8853cab95a868eba67e122ee19679e354f94314e6739d05b3"
)
if PINNED_PREFIX_BATCHSPLIT_CONTRACT.digest != PINNED_PREFIX_BATCHSPLIT_CONTRACT_SHA256:
    raise PrefixBatchSplitContractError(
        "pinned batch-split contract digest changed: "
        f"{PINNED_PREFIX_BATCHSPLIT_CONTRACT.digest}"
    )


@dataclass(frozen=True, slots=True)
class DestinationConsumptionV1:
    """Every canonical source derivation consumed by one unscanned destination."""

    destination_key: str
    performance_section: str
    performance_leaf_key: str
    scan_index: int | None
    mappings: tuple[MaxTextMappingDescriptorV1, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.destination_key, str) or not self.destination_key:
            raise PrefixBatchSplitContractError(
                "destination consumption key must be non-empty"
            )
        if self.performance_section not in _SECTION_LEAF_COUNTS:
            raise PrefixBatchSplitContractError(
                "destination consumption has an unknown performance section"
            )
        if (
            not isinstance(self.performance_leaf_key, str)
            or not self.performance_leaf_key
        ):
            raise PrefixBatchSplitContractError(
                "destination consumption leaf key must be non-empty"
            )
        if type(self.mappings) is not tuple or not self.mappings:
            raise PrefixBatchSplitContractError(
                "destination consumption mappings must be a non-empty tuple"
            )
        if any(
            type(mapping) is not MaxTextMappingDescriptorV1 for mapping in self.mappings
        ):
            raise PrefixBatchSplitContractError(
                "destination consumption contains a non-canonical mapping type"
            )
        if any(
            mapping.destination_key != self.destination_key for mapping in self.mappings
        ):
            raise PrefixBatchSplitContractError(
                "destination consumption crosses destination keys"
            )
        first = self.mappings[0]
        if any(
            (
                mapping.destination_dtype,
                mapping.destination_shape,
                mapping.destination_slice_axis,
            )
            != (
                first.destination_dtype,
                first.destination_shape,
                first.destination_slice_axis,
            )
            for mapping in self.mappings
        ):
            raise PrefixBatchSplitContractError(
                "destination consumption has inconsistent shape/dtype/slice axes"
            )
        if first.destination_slice_axis is None:
            if len(self.mappings) != 1 or first.destination_slice_index is not None:
                raise PrefixBatchSplitContractError(
                    "unsliced destination must consume exactly one mapping"
                )
        else:
            indices = tuple(
                mapping.destination_slice_index for mapping in self.mappings
            )
            if (
                first.destination_slice_axis != 0
                or indices != tuple(range(PINNED_GEOMETRY.routed_experts))
                or len(self.mappings) != PINNED_GEOMETRY.routed_experts
            ):
                raise PrefixBatchSplitContractError(
                    "expert destination must consume slices 0..255 exactly once"
                )
        if self.performance_section in {
            "embedding",
            "target-routed-only",
        }:
            if self.scan_index is not None:
                raise PrefixBatchSplitContractError(
                    "embedding/target destinations cannot have a scan index"
                )
        elif type(self.scan_index) is not int or self.scan_index < 0:
            raise PrefixBatchSplitContractError(
                "scanned destination requires a nonnegative scan index"
            )

    @property
    def destination_dtype(self) -> str:
        return self.mappings[0].destination_dtype

    @property
    def destination_shape(self) -> tuple[int, ...]:
        return self.mappings[0].destination_shape

    @property
    def source_keys(self) -> tuple[str, ...]:
        return tuple(mapping.source_key for mapping in self.mappings)

    @property
    def scale_keys(self) -> tuple[str, ...]:
        return tuple(
            mapping.scale_key
            for mapping in self.mappings
            if mapping.scale_key is not None
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "destination_key": self.destination_key,
            "performance_section": self.performance_section,
            "performance_leaf_key": self.performance_leaf_key,
            "scan_index": self.scan_index,
            "destination_dtype": self.destination_dtype,
            "destination_shape": list(self.destination_shape),
            "mappings": [mapping.as_dict() for mapping in self.mappings],
        }


@dataclass(frozen=True, slots=True)
class PerformanceLeafV1:
    """One leaf in the proposed 44-leaf scanned/partial parameter tree."""

    leaf_key: str
    section: str
    family: str
    dtype: str
    logical_shape: tuple[int, ...]
    scan_axis: int | None
    scan_count: int
    source_destination_keys: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.leaf_key, str) or not self.leaf_key:
            raise PrefixBatchSplitContractError("performance leaf key is empty")
        if self.section not in _SECTION_LEAF_COUNTS:
            raise PrefixBatchSplitContractError("performance leaf section is unknown")
        if not isinstance(self.family, str) or not self.family:
            raise PrefixBatchSplitContractError("performance leaf family is empty")
        if self.dtype not in {"bfloat16", "float32"}:
            raise PrefixBatchSplitContractError("performance leaf dtype is unsupported")
        _shape(self.logical_shape, "performance leaf logical_shape")
        _integer(self.scan_count, "performance leaf scan_count")
        if self.section in {
            "scanned-dense",
            "scanned-completed-moe",
        }:
            if self.scan_axis != PINNED_PARAMETER_SCAN_AXIS:
                raise PrefixBatchSplitContractError(
                    "scanned performance leaves must insert parameter axis one"
                )
        elif self.scan_axis is not None or self.scan_count != 1:
            raise PrefixBatchSplitContractError(
                "embedding/target leaves must be unscanned"
            )
        if (
            type(self.source_destination_keys) is not tuple
            or not self.source_destination_keys
            or len(set(self.source_destination_keys))
            != len(self.source_destination_keys)
        ):
            raise PrefixBatchSplitContractError(
                "performance leaf destinations must be unique immutable keys"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "leaf_key": self.leaf_key,
            "section": self.section,
            "family": self.family,
            "dtype": self.dtype,
            "logical_shape": list(self.logical_shape),
            "scan_axis": self.scan_axis,
            "scan_count": self.scan_count,
            "source_destination_keys": list(self.source_destination_keys),
        }


@dataclass(frozen=True, slots=True)
class PrefixPerformanceTreePlanV1:
    """Exact 21,930-source / 510-destination / 44-leaf consumption plan."""

    SCHEMA: ClassVar[str] = "moe-sae.r1.prefix-performance-tree-plan.v1"

    maxtext_revision: str
    mapping_descriptor_sha256: str
    inventory_selection_sha256: str | None
    inventory_bound: bool
    destinations: tuple[DestinationConsumptionV1, ...]
    leaves: tuple[PerformanceLeafV1, ...]
    production_sealable: bool = False
    production_admission_enabled: bool = False

    def __post_init__(self) -> None:
        if self.maxtext_revision != PINNED_MAXTEXT_REVISION:
            raise PrefixBatchSplitContractError(
                "performance plan MaxText revision is not pinned"
            )
        if self.mapping_descriptor_sha256 != PINNED_PREFIX_MAPPING_DESCRIPTOR_SHA256:
            raise PrefixBatchSplitContractError(
                "performance plan mapping descriptor digest is not pinned"
            )
        if (
            self.inventory_bound is not False
            or self.inventory_selection_sha256 is not None
        ):
            raise PrefixBatchSplitContractError(
                "first-tranche performance plan must remain unbound"
            )
        if (
            type(self.destinations) is not tuple
            or len(self.destinations) != PINNED_SOURCE_DESTINATION_COUNT
            or any(
                type(item) is not DestinationConsumptionV1 for item in self.destinations
            )
        ):
            raise PrefixBatchSplitContractError(
                "performance plan must contain exactly 510 immutable destinations"
            )
        if (
            type(self.leaves) is not tuple
            or len(self.leaves) != PINNED_PERFORMANCE_LEAF_COUNT
            or any(type(item) is not PerformanceLeafV1 for item in self.leaves)
        ):
            raise PrefixBatchSplitContractError(
                "performance plan must contain exactly 44 immutable leaves"
            )
        destination_keys = tuple(item.destination_key for item in self.destinations)
        leaf_keys = tuple(item.leaf_key for item in self.leaves)
        if destination_keys != tuple(sorted(destination_keys)) or len(
            set(destination_keys)
        ) != len(destination_keys):
            raise PrefixBatchSplitContractError(
                "destination plan keys must be unique and sorted"
            )
        if leaf_keys != tuple(sorted(leaf_keys)) or len(set(leaf_keys)) != len(
            leaf_keys
        ):
            raise PrefixBatchSplitContractError(
                "performance leaf keys must be unique and sorted"
            )

        canonical_family_by_leaf: dict[str, str] = {}
        for destination in self.destinations:
            (
                expected_section,
                expected_leaf_key,
                expected_scan_index,
                expected_family,
            ) = _destination_route(destination.destination_key)
            if (
                destination.performance_section,
                destination.performance_leaf_key,
                destination.scan_index,
            ) != (
                expected_section,
                expected_leaf_key,
                expected_scan_index,
            ):
                raise PrefixBatchSplitContractError(
                    "destination route differs from the canonical route"
                )
            prior_family = canonical_family_by_leaf.setdefault(
                expected_leaf_key, expected_family
            )
            if prior_family != expected_family:
                raise PrefixBatchSplitContractError(
                    "canonical destination routes disagree on leaf family"
                )
        for leaf in self.leaves:
            if leaf.family != canonical_family_by_leaf.get(leaf.leaf_key):
                raise PrefixBatchSplitContractError(
                    "performance leaf family differs from its canonical route"
                )

        flattened = tuple(
            sorted(
                (
                    mapping
                    for destination in self.destinations
                    for mapping in destination.mappings
                ),
                key=lambda mapping: mapping.source_key,
            )
        )
        if flattened != _canonical_mappings():
            raise PrefixBatchSplitContractError(
                "performance plan does not consume canonical mappings exactly once"
            )
        if sum(len(item.mappings) for item in self.destinations) != (
            PINNED_SOURCE_MAPPING_COUNT
        ):
            raise PrefixBatchSplitContractError(
                "performance plan source mapping total is not 21,930"
            )
        scale_keys = tuple(
            scale
            for destination in self.destinations
            for scale in destination.scale_keys
        )
        if len(scale_keys) != PINNED_SCALE_MAPPING_COUNT or len(set(scale_keys)) != len(
            scale_keys
        ):
            raise PrefixBatchSplitContractError(
                "performance plan must consume 21,749 unique FP8 scales"
            )
        if len(flattened) + len(scale_keys) != PINNED_SOURCE_ENTRY_COUNT:
            raise PrefixBatchSplitContractError(
                "logical and scale consumption must cover 43,679 source entries"
            )
        validate_prefix_source_keys(
            (
                key
                for destination in self.destinations
                for key in (*destination.source_keys, *destination.scale_keys)
            ),
            PINNED_GEOMETRY,
        )
        canonical_destinations, canonical_leaves = _canonical_plan_components()
        if self.destinations != canonical_destinations:
            raise PrefixBatchSplitContractError(
                "performance destinations differ from the canonical derivation"
            )

        destinations_by_leaf: dict[str, list[DestinationConsumptionV1]] = defaultdict(
            list
        )
        for destination in self.destinations:
            destinations_by_leaf[destination.performance_leaf_key].append(destination)
        if set(destinations_by_leaf) != set(leaf_keys):
            raise PrefixBatchSplitContractError(
                "destination plan and performance leaves are not bijective"
            )
        consumed_destination_keys: list[str] = []
        for leaf in self.leaves:
            consumed = destinations_by_leaf[leaf.leaf_key]
            consumed.sort(
                key=lambda item: (
                    -1 if item.scan_index is None else item.scan_index,
                    item.destination_key,
                )
            )
            actual_keys = tuple(item.destination_key for item in consumed)
            if leaf.source_destination_keys != actual_keys:
                raise PrefixBatchSplitContractError(
                    "performance leaf destination order is not canonical"
                )
            if any(item.performance_section != leaf.section for item in consumed):
                raise PrefixBatchSplitContractError(
                    "performance leaf crosses tree sections"
                )
            if any(
                item.destination_dtype != leaf.dtype
                or item.destination_shape != consumed[0].destination_shape
                for item in consumed
            ):
                raise PrefixBatchSplitContractError(
                    "scanned leaf source shapes/dtypes are inconsistent"
                )
            expected_scan_count = {
                "embedding": 1,
                "scanned-dense": PINNED_DENSE_SCAN_COUNT,
                "scanned-completed-moe": PINNED_COMPLETED_MOE_SCAN_COUNT,
                "target-routed-only": 1,
            }[leaf.section]
            if leaf.scan_count != expected_scan_count or len(consumed) != (
                expected_scan_count
            ):
                raise PrefixBatchSplitContractError(
                    "performance leaf scan count differs from its source destinations"
                )
            if leaf.scan_axis is None:
                expected_shape = consumed[0].destination_shape
            else:
                dimensions = list(consumed[0].destination_shape)
                dimensions.insert(leaf.scan_axis, leaf.scan_count)
                expected_shape = tuple(dimensions)
            if leaf.logical_shape != expected_shape:
                raise PrefixBatchSplitContractError(
                    "performance leaf scan-axis shape is incorrect"
                )
            consumed_destination_keys.extend(actual_keys)
        if self.leaves != canonical_leaves:
            raise PrefixBatchSplitContractError(
                "performance leaves differ from the canonical derivation"
            )
        if len(consumed_destination_keys) != len(set(consumed_destination_keys)) or set(
            consumed_destination_keys
        ) != set(destination_keys):
            raise PrefixBatchSplitContractError(
                "510 source destinations are not consumed exactly once"
            )

        leaf_sections = Counter(leaf.section for leaf in self.leaves)
        destination_sections = Counter(
            destination.performance_section for destination in self.destinations
        )
        if leaf_sections != Counter(_SECTION_LEAF_COUNTS):
            raise PrefixBatchSplitContractError(
                "performance tree must contain 1/12/17/14 leaves"
            )
        if destination_sections != Counter(_SECTION_DESTINATION_COUNTS):
            raise PrefixBatchSplitContractError(
                "performance tree must consume 1/36/459/14 destinations"
            )
        fan_in = Counter(len(destination.mappings) for destination in self.destinations)
        if fan_in != Counter({1: 426, PINNED_GEOMETRY.routed_experts: 84}):
            raise PrefixBatchSplitContractError(
                "destination fan-in must be 426 singleton and 84 expert-packed"
            )
        if (
            self.source_entry_count,
            self.source_mapping_count,
            len(self.destinations),
            len(self.leaves),
        ) != (
            PINNED_PREFIX_BATCHSPLIT_CONTRACT.source_entry_count,
            PINNED_PREFIX_BATCHSPLIT_CONTRACT.source_mapping_count,
            PINNED_PREFIX_BATCHSPLIT_CONTRACT.source_destination_count,
            PINNED_PREFIX_BATCHSPLIT_CONTRACT.performance_leaf_count,
        ):
            raise PrefixBatchSplitContractError(
                "derived plan counts differ from the pinned batch-split contract"
            )

        dense_families = {
            leaf.family for leaf in self.leaves if leaf.section == "scanned-dense"
        }
        completed_families = {
            leaf.family
            for leaf in self.leaves
            if leaf.section == "scanned-completed-moe"
        }
        target_families = {
            leaf.family for leaf in self.leaves if leaf.section == "target-routed-only"
        }
        if (
            len(dense_families) != 12
            or len(completed_families) != 17
            or len(target_families) != 14
            or not target_families < completed_families
        ):
            raise PrefixBatchSplitContractError(
                "performance family inventory is not 12 dense / 17 MoE / 14 target"
            )
        omitted_target_families = completed_families - target_families
        if len(omitted_target_families) != 3 or any(
            "shared_experts" not in family for family in omitted_target_families
        ):
            raise PrefixBatchSplitContractError(
                "target must omit exactly the three shared-expert families"
            )

        bias_destinations = tuple(
            destination
            for destination in self.destinations
            if any(
                mapping.source_key.endswith(".mlp.gate.e_score_correction_bias")
                for mapping in destination.mappings
            )
        )
        float32_destinations = tuple(
            destination
            for destination in self.destinations
            if destination.destination_dtype == "float32"
        )
        if len(bias_destinations) != 28 or set(bias_destinations) != set(
            float32_destinations
        ):
            raise PrefixBatchSplitContractError(
                "all and only 28 router-bias destinations must remain FP32"
            )
        self._validate_layer_consumption()
        if (
            self.production_sealable is not False
            or self.production_admission_enabled is not False
        ):
            raise PrefixBatchSplitContractError(
                "performance plan cannot seal or authorize production"
            )
        actual_digest = self.digest
        if actual_digest != PINNED_UNBOUND_PREFIX_PERFORMANCE_TREE_PLAN_SHA256:
            raise PrefixBatchSplitContractError(
                f"unbound performance plan digest differs from its pin: {actual_digest}"
            )

    def _validate_layer_consumption(self) -> None:
        for destination in self.destinations:
            if destination.performance_section == "embedding":
                expected_layer = None
            elif destination.performance_section == "scanned-dense":
                expected_layer = destination.scan_index
            elif destination.performance_section == "scanned-completed-moe":
                expected_layer = PINNED_DENSE_SCAN_COUNT + destination.scan_index
            else:
                expected_layer = PINNED_TARGET_LAYER_ZERO_INDEXED
            for source_key in destination.source_keys:
                match = _SOURCE_LAYER.match(source_key)
                actual_layer = int(match.group(1)) if match is not None else None
                if actual_layer != expected_layer:
                    raise PrefixBatchSplitContractError(
                        "destination consumes a source from the wrong logical layer"
                    )
                if (
                    destination.performance_section == "target-routed-only"
                    and ".shared_experts." in source_key
                ):
                    raise PrefixBatchSplitContractError(
                        "target destination consumed a shared expert"
                    )

    @property
    def source_mapping_count(self) -> int:
        return sum(len(item.mappings) for item in self.destinations)

    @property
    def scale_mapping_count(self) -> int:
        return sum(len(item.scale_keys) for item in self.destinations)

    @property
    def source_entry_count(self) -> int:
        return self.source_mapping_count + self.scale_mapping_count

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "maxtext_revision": self.maxtext_revision,
            "mapping_descriptor_sha256": self.mapping_descriptor_sha256,
            "inventory_selection_sha256": self.inventory_selection_sha256,
            "inventory_bound": self.inventory_bound,
            "source_entry_count": self.source_entry_count,
            "source_mapping_count": self.source_mapping_count,
            "scale_mapping_count": self.scale_mapping_count,
            "source_destination_count": len(self.destinations),
            "performance_leaf_count": len(self.leaves),
            "destinations": [
                destination.as_dict() for destination in self.destinations
            ],
            "leaves": [leaf.as_dict() for leaf in self.leaves],
            "production_sealable": self.production_sealable,
            "production_admission_enabled": self.production_admission_enabled,
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.as_dict())

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


def _destination_route(
    destination_key: str,
) -> tuple[str, str, int | None, str]:
    if destination_key == "params-token_embedder-embedding":
        return (
            "embedding",
            destination_key,
            None,
            "token_embedder-embedding",
        )
    dense = _DENSE_DESTINATION.fullmatch(destination_key)
    if dense is not None:
        index = int(dense.group(1))
        if index >= PINNED_DENSE_SCAN_COUNT:
            raise PrefixBatchSplitContractError(
                "dense destination index is outside the three-layer scan"
            )
        family = dense.group(2)
        return (
            "scanned-dense",
            f"params-decoder-dense_layers-{family}",
            index,
            family,
        )
    moe = _MOE_DESTINATION.fullmatch(destination_key)
    if moe is None:
        raise PrefixBatchSplitContractError(
            f"unknown canonical destination key: {destination_key}"
        )
    index = int(moe.group(1))
    family = moe.group(2)
    if index < PINNED_COMPLETED_MOE_SCAN_COUNT:
        return (
            "scanned-completed-moe",
            f"params-decoder-moe_layers-{family}",
            index,
            family,
        )
    if index == PINNED_COMPLETED_MOE_SCAN_COUNT:
        return (
            "target-routed-only",
            f"target-routed-only-{family}",
            None,
            family,
        )
    raise PrefixBatchSplitContractError(
        "MoE destination index crosses the partial target"
    )


def _ordered_destination_mappings(
    mappings: list[MaxTextMappingDescriptorV1],
) -> tuple[MaxTextMappingDescriptorV1, ...]:
    if len(mappings) == 1:
        return (mappings[0],)
    return tuple(
        sorted(
            mappings,
            key=lambda mapping: (
                -1
                if mapping.destination_slice_index is None
                else mapping.destination_slice_index
            ),
        )
    )


def _derive_destinations(
    mappings: tuple[MaxTextMappingDescriptorV1, ...],
) -> tuple[DestinationConsumptionV1, ...]:
    by_destination: dict[str, list[MaxTextMappingDescriptorV1]] = defaultdict(list)
    for mapping in mappings:
        by_destination[mapping.destination_key].append(mapping)
    result = []
    for destination_key in sorted(by_destination):
        section, leaf_key, scan_index, _ = _destination_route(destination_key)
        result.append(
            DestinationConsumptionV1(
                destination_key=destination_key,
                performance_section=section,
                performance_leaf_key=leaf_key,
                scan_index=scan_index,
                mappings=_ordered_destination_mappings(by_destination[destination_key]),
            )
        )
    return tuple(result)


def _derive_leaves(
    destinations: tuple[DestinationConsumptionV1, ...],
) -> tuple[PerformanceLeafV1, ...]:
    by_leaf: dict[str, list[DestinationConsumptionV1]] = defaultdict(list)
    for destination in destinations:
        by_leaf[destination.performance_leaf_key].append(destination)
    result = []
    for leaf_key in sorted(by_leaf):
        consumed = sorted(
            by_leaf[leaf_key],
            key=lambda item: (
                -1 if item.scan_index is None else item.scan_index,
                item.destination_key,
            ),
        )
        section = consumed[0].performance_section
        _, _, _, family = _destination_route(consumed[0].destination_key)
        scan_count = {
            "embedding": 1,
            "scanned-dense": PINNED_DENSE_SCAN_COUNT,
            "scanned-completed-moe": PINNED_COMPLETED_MOE_SCAN_COUNT,
            "target-routed-only": 1,
        }[section]
        scan_axis = (
            PINNED_PARAMETER_SCAN_AXIS
            if section in {"scanned-dense", "scanned-completed-moe"}
            else None
        )
        dimensions = list(consumed[0].destination_shape)
        if scan_axis is not None:
            dimensions.insert(scan_axis, scan_count)
        result.append(
            PerformanceLeafV1(
                leaf_key=leaf_key,
                section=section,
                family=family,
                dtype=consumed[0].destination_dtype,
                logical_shape=tuple(dimensions),
                scan_axis=scan_axis,
                scan_count=scan_count,
                source_destination_keys=tuple(
                    item.destination_key for item in consumed
                ),
            )
        )
    return tuple(result)


@lru_cache(maxsize=1)
def _canonical_plan_components() -> tuple[
    tuple[DestinationConsumptionV1, ...],
    tuple[PerformanceLeafV1, ...],
]:
    destinations = _derive_destinations(_canonical_mappings())
    return destinations, _derive_leaves(destinations)


def build_pinned_prefix_performance_tree_plan() -> PrefixPerformanceTreePlanV1:
    """Derive the exact unbound 510-to-44 first-tranche schema plan.

    Authenticated checkpoint binding belongs to restore authority.  This
    repository-side planning contract accepts no inventory selection and can
    neither claim binding nor authorize production.
    """

    destinations, leaves = _canonical_plan_components()
    return PrefixPerformanceTreePlanV1(
        maxtext_revision=PINNED_MAXTEXT_REVISION,
        mapping_descriptor_sha256=PINNED_PREFIX_MAPPING_DESCRIPTOR_SHA256,
        inventory_selection_sha256=None,
        inventory_bound=False,
        destinations=destinations,
        leaves=leaves,
    )


__all__ = [
    "BatchSplitMeshLayoutV1",
    "CaptureTensorSchemaV1",
    "DestinationConsumptionV1",
    "DurableMetadataFieldSchemaV1",
    "PINNED_BATCH_SPLIT_FACTOR",
    "PINNED_CAPTURE_OUTPUTS",
    "PINNED_COMPLETED_MOE_SCAN_COUNT",
    "PINNED_DENSE_SCAN_COUNT",
    "PINNED_DEVICE_COUNT",
    "PINNED_DURABLE_METADATA",
    "PINNED_GLOBAL_BATCH_SIZE",
    "PINNED_LAYOUT_LADDER",
    "PINNED_MAXTEXT_REVISION",
    "PINNED_PARAMETER_SCAN_AXIS",
    "PINNED_PARTIAL_TARGET_COUNT",
    "PINNED_PERFORMANCE_LEAF_COUNT",
    "PINNED_PREFIX_BATCHSPLIT_CONTRACT",
    "PINNED_PREFIX_BATCHSPLIT_CONTRACT_SHA256",
    "PINNED_PREFIX_MAPPING_DESCRIPTOR_SHA256",
    "PINNED_SEQUENCE_LENGTH",
    "PINNED_SOURCE_DESTINATION_COUNT",
    "PINNED_SOURCE_ENTRY_COUNT",
    "PINNED_SOURCE_MAPPING_COUNT",
    "PINNED_TARGET_LAYER_ZERO_INDEXED",
    "PINNED_UNBOUND_PREFIX_PERFORMANCE_TREE_PLAN_SHA256",
    "PerformanceLeafV1",
    "PrefixBatchSplitContractError",
    "PrefixBatchSplitContractV1",
    "PrefixPerformanceTreePlanV1",
    "build_pinned_prefix_performance_tree_plan",
]
