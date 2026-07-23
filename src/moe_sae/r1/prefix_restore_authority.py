"""Fail-closed, lazy restore authority for the R1 forward-only prefix.

This module joins three already-frozen inputs without importing JAX or
MaxText:

* an exact, generation-pinned V2 regional checkpoint;
* an explicitly expected inventory identity; and
* the canonical 510-destination to 44-leaf performance-tree plan.

Opening verifies only the bounded control plane.  Payload objects remain lazy
and can be touched only through leaf-scoped views.  The resulting binding is
inventory-bound, but it is deliberately not evidence of a completed payload
restore, framework interoperability, qualification, admission, or production
readiness.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
import hashlib
import math
from types import MappingProxyType
from typing import Any, ClassVar, Protocol, Self

import numpy as np

from .checkpoint_conversion import (
    CheckpointConversionError,
    conversion_engine_sha256,
    numeric_codec_identity_sha256,
)
from .checkpoint_inventory import PINNED_GEOMETRY
from .gcs_immutable_byte_store import GCSPhysicalObjectReceiptV1
from .prefix_batchsplit_contract import (
    PINNED_PERFORMANCE_LEAF_COUNT,
    PINNED_MAXTEXT_REVISION,
    PINNED_PREFIX_MAPPING_DESCRIPTOR_SHA256,
    PINNED_SOURCE_DESTINATION_COUNT,
    PINNED_UNBOUND_PREFIX_PERFORMANCE_TREE_PLAN_SHA256,
    DestinationConsumptionV1,
    PrefixPerformanceTreePlanV1,
    build_pinned_prefix_performance_tree_plan,
)
from .prefix_checkpoint_v2 import (
    MAX_PAYLOAD_READ_BYTES,
    GenerationPinnedPrefixObjectReader,
    LazyRegionalPrefixCheckpointReaderV1,
    LogicalStoredDestinationV2,
    PrefixCheckpointV2Error,
)
from .protocol import (
    CanonicalRecord,
    IDENTIFIER_PATTERN,
    SHA256_PATTERN,
    ProtocolError,
    canonical_json_bytes,
    require_integer,
    require_string,
)


DIRECT_PREFIX_RESTORE_CONTRACT = (
    "generation-pinned-v2-to-derived-44-leaf-lazy-restore-authority-v1"
)
DEFAULT_LEAF_REGION_READ_BYTES = 8 * 1024**2
MAX_LEAF_REGION_READ_BYTES = MAX_PAYLOAD_READ_BYTES

_DTYPE_BYTES = {"bfloat16": 2, "float32": 4}
_MAX_GCS_GENERATION = (1 << 63) - 1
_MAPPING_PROXY_TYPE = type(MappingProxyType({}))
_AUTHORITY_OPEN_CAPABILITY = object()
_LEAF_VIEW_CAPABILITY = object()


class PrefixRestoreAuthorityError(PrefixCheckpointV2Error):
    """A direct-restore identity, schema, or bounded-read invariant failed."""


def _sha256(value: Any, name: str) -> str:
    if type(value) is not str:
        raise PrefixRestoreAuthorityError(f"{name} must be an exact immutable string")
    try:
        return require_string(value, name, pattern=SHA256_PATTERN)
    except ProtocolError as error:
        raise PrefixRestoreAuthorityError(f"{name} is not an exact SHA-256") from error


def _identifier(value: Any, name: str) -> str:
    if type(value) is not str:
        raise PrefixRestoreAuthorityError(f"{name} must be an exact immutable string")
    try:
        return require_string(value, name, pattern=IDENTIFIER_PATTERN)
    except ProtocolError as error:
        raise PrefixRestoreAuthorityError(f"{name} is not an identifier") from error


def _strict_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise PrefixRestoreAuthorityError(f"{name} must be an exact boolean")
    return value


def _positive_bound(value: Any, name: str, *, maximum: int) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise PrefixRestoreAuthorityError(
            f"{name} must be an integer in [1, {maximum}]"
        )
    return value


@dataclass(frozen=True, slots=True)
class PrefixRestoreIdentityExpectationV1(CanonicalRecord):
    """Operator-provided exact identity required before an inventory binding."""

    SCHEMA: ClassVar[str] = "moe-sae.r1.prefix-restore-identity-expectation.v1"

    logical_manifest_sha256: str
    selection_sha256: str
    full_inventory_sha256: str
    selected_inventory_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "logical_manifest_sha256",
            "selection_sha256",
            "full_inventory_sha256",
            "selected_inventory_sha256",
        ):
            _sha256(getattr(self, name), name)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "logical_manifest_sha256": self.logical_manifest_sha256,
            "selection_sha256": self.selection_sha256,
            "full_inventory_sha256": self.full_inventory_sha256,
            "selected_inventory_sha256": self.selected_inventory_sha256,
        }


@dataclass(frozen=True, slots=True)
class PrefixRestoreBindingV1(CanonicalRecord):
    """Auditable control-plane binding, never a payload-restore claim."""

    SCHEMA: ClassVar[str] = "moe-sae.r1.prefix-restore-binding.v1"

    logical_manifest_sha256: str
    selection_sha256: str
    full_inventory_sha256: str
    selected_inventory_sha256: str
    regional_envelope_sha256: str
    regional_envelope_receipt_sha256: str
    regional_envelope_generation: str
    store_id: str
    region: str
    performance_plan_sha256: str
    mapping_descriptor_sha256: str
    geometry_sha256: str
    maxtext_revision: str
    source_destination_count: int
    performance_leaf_count: int
    inventory_bound: bool = True
    control_plane_verified: bool = True
    released_checkpoint_observed: bool = False
    payload_restore_completed: bool = False
    qualification_complete: bool = False
    production_sealable: bool = False
    production_admission_enabled: bool = False
    restore_contract: str = DIRECT_PREFIX_RESTORE_CONTRACT

    def __post_init__(self) -> None:
        for name in (
            "logical_manifest_sha256",
            "selection_sha256",
            "full_inventory_sha256",
            "selected_inventory_sha256",
            "regional_envelope_sha256",
            "regional_envelope_receipt_sha256",
            "performance_plan_sha256",
            "mapping_descriptor_sha256",
            "geometry_sha256",
        ):
            _sha256(getattr(self, name), name)
        _identifier(self.store_id, "store_id")
        _identifier(self.region, "region")
        _identifier(self.maxtext_revision, "maxtext_revision")
        _identifier(self.restore_contract, "restore_contract")
        if (
            type(self.regional_envelope_generation) is not str
            or not self.regional_envelope_generation.isascii()
            or not self.regional_envelope_generation.isdecimal()
            or self.regional_envelope_generation.startswith("0")
            or len(self.regional_envelope_generation) > 19
            or int(self.regional_envelope_generation) > _MAX_GCS_GENERATION
        ):
            raise PrefixRestoreAuthorityError(
                "regional_envelope_generation must be a positive canonical decimal"
            )
        try:
            require_integer(
                self.source_destination_count,
                "source_destination_count",
                minimum=1,
                maximum=PINNED_SOURCE_DESTINATION_COUNT,
            )
            require_integer(
                self.performance_leaf_count,
                "performance_leaf_count",
                minimum=1,
                maximum=PINNED_PERFORMANCE_LEAF_COUNT,
            )
        except ProtocolError as error:
            raise PrefixRestoreAuthorityError(
                "restore binding counts are invalid"
            ) from error
        if (
            self.source_destination_count != PINNED_SOURCE_DESTINATION_COUNT
            or self.performance_leaf_count != PINNED_PERFORMANCE_LEAF_COUNT
        ):
            raise PrefixRestoreAuthorityError(
                "restore binding is not the exact 510-to-44 schema"
            )
        for name in (
            "inventory_bound",
            "control_plane_verified",
            "released_checkpoint_observed",
            "payload_restore_completed",
            "qualification_complete",
            "production_sealable",
            "production_admission_enabled",
        ):
            _strict_bool(getattr(self, name), name)
        if self.inventory_bound is not True or self.control_plane_verified is not True:
            raise PrefixRestoreAuthorityError(
                "restore authority rejects unbound or unverified control-plane claims"
            )
        if (
            self.released_checkpoint_observed is not False
            or self.payload_restore_completed is not False
            or self.qualification_complete is not False
            or self.production_sealable is not False
            or self.production_admission_enabled is not False
        ):
            raise PrefixRestoreAuthorityError(
                "control-plane binding cannot claim source observation, restore, "
                "qualification, or production"
            )
        if (
            self.performance_plan_sha256
            != PINNED_UNBOUND_PREFIX_PERFORMANCE_TREE_PLAN_SHA256
            or self.mapping_descriptor_sha256 != PINNED_PREFIX_MAPPING_DESCRIPTOR_SHA256
            or self.geometry_sha256 != PINNED_GEOMETRY.digest
            or self.maxtext_revision != PINNED_MAXTEXT_REVISION
            or self.restore_contract != DIRECT_PREFIX_RESTORE_CONTRACT
        ):
            raise PrefixRestoreAuthorityError(
                "restore binding differs from the pinned authority contract"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "logical_manifest_sha256": self.logical_manifest_sha256,
            "selection_sha256": self.selection_sha256,
            "full_inventory_sha256": self.full_inventory_sha256,
            "selected_inventory_sha256": self.selected_inventory_sha256,
            "regional_envelope_sha256": self.regional_envelope_sha256,
            "regional_envelope_receipt_sha256": (self.regional_envelope_receipt_sha256),
            "regional_envelope_generation": self.regional_envelope_generation,
            "store_id": self.store_id,
            "region": self.region,
            "performance_plan_sha256": self.performance_plan_sha256,
            "mapping_descriptor_sha256": self.mapping_descriptor_sha256,
            "geometry_sha256": self.geometry_sha256,
            "maxtext_revision": self.maxtext_revision,
            "source_destination_count": self.source_destination_count,
            "performance_leaf_count": self.performance_leaf_count,
            "inventory_bound": self.inventory_bound,
            "control_plane_verified": self.control_plane_verified,
            "released_checkpoint_observed": self.released_checkpoint_observed,
            "payload_restore_completed": self.payload_restore_completed,
            "qualification_complete": self.qualification_complete,
            "production_sealable": self.production_sealable,
            "production_admission_enabled": self.production_admission_enabled,
            "restore_contract": self.restore_contract,
        }


@dataclass(frozen=True, slots=True)
class DestinationRestoreSpecV1:
    """One exact V2 destination in a canonically derived leaf arrangement."""

    destination_key: str
    dtype: str
    shape: tuple[int, ...]
    performance_section: str
    performance_leaf_key: str
    scan_index: int | None

    def __post_init__(self) -> None:
        _identifier(self.destination_key, "destination_key")
        _identifier(self.performance_section, "performance_section")
        _identifier(self.performance_leaf_key, "performance_leaf_key")
        dtype = _identifier(self.dtype, "dtype")
        if dtype not in _DTYPE_BYTES:
            raise PrefixRestoreAuthorityError(
                "destination restore dtype is unsupported"
            )
        if (
            type(self.shape) is not tuple
            or not self.shape
            or any(type(item) is not int or item <= 0 for item in self.shape)
        ):
            raise PrefixRestoreAuthorityError("destination restore shape is invalid")
        if self.scan_index is not None and (
            type(self.scan_index) is not int or self.scan_index < 0
        ):
            raise PrefixRestoreAuthorityError(
                "destination restore scan index is invalid"
            )

    @property
    def element_count(self) -> int:
        return math.prod(self.shape)


@dataclass(frozen=True, slots=True)
class PerformanceLeafRestoreSpecV1:
    """Framework-neutral source order and scan geometry for one output leaf."""

    leaf_key: str
    section: str
    family: str
    dtype: str
    logical_shape: tuple[int, ...]
    scan_axis: int | None
    sources: tuple[DestinationRestoreSpecV1, ...]

    def __post_init__(self) -> None:
        _identifier(self.leaf_key, "leaf_key")
        _identifier(self.section, "section")
        _identifier(self.family, "family")
        dtype = _identifier(self.dtype, "dtype")
        if dtype not in _DTYPE_BYTES:
            raise PrefixRestoreAuthorityError("performance leaf dtype is unsupported")
        if (
            type(self.logical_shape) is not tuple
            or not self.logical_shape
            or any(
                type(dimension) is not int or dimension <= 0
                for dimension in self.logical_shape
            )
        ):
            raise PrefixRestoreAuthorityError(
                "performance leaf logical shape is invalid"
            )
        if (
            type(self.sources) is not tuple
            or not self.sources
            or any(type(item) is not DestinationRestoreSpecV1 for item in self.sources)
        ):
            raise PrefixRestoreAuthorityError("performance leaf sources are invalid")
        if any(
            item.performance_leaf_key != self.leaf_key
            or item.performance_section != self.section
            or item.dtype != self.dtype
            for item in self.sources
        ):
            raise PrefixRestoreAuthorityError(
                "performance leaf sources cross a canonical route"
            )
        source_shapes = {item.shape for item in self.sources}
        if len(source_shapes) != 1:
            raise PrefixRestoreAuthorityError(
                "performance leaf source shapes are inconsistent"
            )
        source_shape = self.sources[0].shape
        if self.scan_axis is None:
            expected_shape = source_shape
            if len(self.sources) != 1 or self.sources[0].scan_index is not None:
                raise PrefixRestoreAuthorityError(
                    "unscanned leaf must contain one unscanned source"
                )
        else:
            if (
                type(self.scan_axis) is not int
                or not 0 <= self.scan_axis <= len(source_shape)
                or tuple(item.scan_index for item in self.sources)
                != tuple(range(len(self.sources)))
            ):
                raise PrefixRestoreAuthorityError(
                    "scanned leaf source order is not canonical"
                )
            dimensions = list(source_shape)
            dimensions.insert(self.scan_axis, len(self.sources))
            expected_shape = tuple(dimensions)
        if self.logical_shape != expected_shape:
            raise PrefixRestoreAuthorityError(
                "performance leaf shape differs from its source arrangement"
            )


class PerformanceLeafAdapter(Protocol):
    """Future framework adapter; returning an object creates no evidence claim."""

    def arrange_leaf(self, leaf: LazyPerformanceLeafV1) -> object: ...


@dataclass(frozen=True, slots=True, init=False)
class LazyPerformanceLeafV1:
    """Leaf-scoped, bounded access to exact generation-pinned payload regions."""

    spec: PerformanceLeafRestoreSpecV1
    maximum_read_bytes: int
    _reader: LazyRegionalPrefixCheckpointReaderV1
    _sources: Mapping[str, DestinationRestoreSpecV1]

    def __init__(
        self,
        spec: PerformanceLeafRestoreSpecV1,
        reader: LazyRegionalPrefixCheckpointReaderV1,
        *,
        maximum_read_bytes: int,
        _capability: object | None = None,
    ) -> None:
        if _capability is not _LEAF_VIEW_CAPABILITY:
            raise PrefixRestoreAuthorityError(
                "lazy leaf construction is available only through restore authority"
            )
        if (
            type(spec) is not PerformanceLeafRestoreSpecV1
            or type(reader) is not LazyRegionalPrefixCheckpointReaderV1
        ):
            raise PrefixRestoreAuthorityError("lazy leaf state has invalid types")
        bound = _positive_bound(
            maximum_read_bytes,
            "maximum_read_bytes",
            maximum=MAX_LEAF_REGION_READ_BYTES,
        )
        if bound != reader.maximum_read_bytes:
            raise PrefixRestoreAuthorityError(
                "leaf read bound differs from its exact regional reader"
            )
        sources = {item.destination_key: item for item in spec.sources}
        if len(sources) != len(spec.sources):
            raise PrefixRestoreAuthorityError("leaf source keys are not unique")
        object.__setattr__(self, "spec", spec)
        object.__setattr__(self, "maximum_read_bytes", bound)
        object.__setattr__(self, "_reader", reader)
        object.__setattr__(self, "_sources", MappingProxyType(sources))

    def source(self, source_index: int) -> DestinationRestoreSpecV1:
        if type(source_index) is not int or not 0 <= source_index < len(
            self.spec.sources
        ):
            raise PrefixRestoreAuthorityError("leaf source index is out of bounds")
        return self.spec.sources[source_index]

    def read_source_flat(
        self,
        source_index: int,
        flat_start: int,
        flat_stop: int,
    ) -> np.ndarray:
        source = self.source(source_index)
        return self.read_destination_flat(
            source.destination_key,
            flat_start,
            flat_stop,
        )

    def read_destination_flat(
        self,
        destination_key: str,
        flat_start: int,
        flat_stop: int,
    ) -> np.ndarray:
        _identifier(destination_key, "destination_key")
        source = self._sources.get(destination_key)
        if source is None:
            raise PrefixRestoreAuthorityError(
                "destination is outside this performance leaf"
            )
        if type(flat_start) is not int or type(flat_stop) is not int:
            raise PrefixRestoreAuthorityError("flat read bounds must be integers")
        if (
            flat_start < 0
            or flat_stop <= flat_start
            or flat_stop > source.element_count
        ):
            raise PrefixRestoreAuthorityError("flat read interval is invalid")
        requested_bytes = (flat_stop - flat_start) * _DTYPE_BYTES[source.dtype]
        if requested_bytes > self.maximum_read_bytes:
            raise PrefixRestoreAuthorityError(
                "leaf payload request exceeds the configured byte bound"
            )
        try:
            return self._reader.read_flat_region(
                destination_key,
                flat_start,
                flat_stop,
            )
        except (
            PrefixCheckpointV2Error,
            CheckpointConversionError,
            ProtocolError,
        ) as error:
            raise PrefixRestoreAuthorityError(
                "generation-pinned leaf payload read failed"
            ) from error

    def iter_source_regions(
        self,
        source_index: int,
        *,
        maximum_region_bytes: int | None = None,
    ) -> Iterator[tuple[int, int, np.ndarray]]:
        """Yield bounded flat intervals without assembling a destination."""

        source = self.source(source_index)
        bound = (
            self.maximum_read_bytes
            if maximum_region_bytes is None
            else _positive_bound(
                maximum_region_bytes,
                "maximum_region_bytes",
                maximum=self.maximum_read_bytes,
            )
        )
        elements_per_region = bound // _DTYPE_BYTES[source.dtype]
        if elements_per_region <= 0:
            raise PrefixRestoreAuthorityError(
                "region bound cannot hold one destination element"
            )
        start = 0
        while start < source.element_count:
            stop = min(source.element_count, start + elements_per_region)
            yield start, stop, self.read_source_flat(source_index, start, stop)
            start = stop


@dataclass(frozen=True, slots=True, init=False)
class PrefixDirectRestoreAuthorityV1:
    """Opened, inventory-bound authority over 44 lazy performance leaves."""

    binding: PrefixRestoreBindingV1
    plan: PrefixPerformanceTreePlanV1
    _reader: LazyRegionalPrefixCheckpointReaderV1
    _envelope_receipt: GCSPhysicalObjectReceiptV1
    leaves: tuple[LazyPerformanceLeafV1, ...]
    _leaves_by_key: Mapping[str, LazyPerformanceLeafV1]

    def __init__(
        self,
        binding: PrefixRestoreBindingV1,
        plan: PrefixPerformanceTreePlanV1,
        reader: LazyRegionalPrefixCheckpointReaderV1,
        leaves: tuple[LazyPerformanceLeafV1, ...],
        *,
        envelope_receipt: GCSPhysicalObjectReceiptV1 | None = None,
        _capability: object | None = None,
    ) -> None:
        if _capability is not _AUTHORITY_OPEN_CAPABILITY:
            raise PrefixRestoreAuthorityError(
                "direct restore authority is available only through open()"
            )
        if (
            type(binding) is not PrefixRestoreBindingV1
            or type(plan) is not PrefixPerformanceTreePlanV1
            or type(reader) is not LazyRegionalPrefixCheckpointReaderV1
            or type(envelope_receipt) is not GCSPhysicalObjectReceiptV1
            or type(leaves) is not tuple
            or len(leaves) != PINNED_PERFORMANCE_LEAF_COUNT
            or any(type(item) is not LazyPerformanceLeafV1 for item in leaves)
        ):
            raise PrefixRestoreAuthorityError(
                "direct restore authority state is invalid"
            )
        keys = tuple(item.spec.leaf_key for item in leaves)
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise PrefixRestoreAuthorityError(
                "direct restore leaf index is not exact and sorted"
            )
        expected_binding_identity = (
            reader.manifest.digest,
            reader.manifest.selection_sha256,
            reader.manifest.full_inventory_sha256,
            reader.manifest.selected_inventory_sha256,
            reader.envelope.digest,
            envelope_receipt.digest,
            envelope_receipt.generation,
            reader.envelope.store_id,
            reader.envelope.region,
            plan.digest,
            plan.mapping_descriptor_sha256,
            reader.manifest.geometry_sha256,
            reader.manifest.maxtext_revision,
            len(plan.destinations),
            len(plan.leaves),
        )
        actual_binding_identity = (
            binding.logical_manifest_sha256,
            binding.selection_sha256,
            binding.full_inventory_sha256,
            binding.selected_inventory_sha256,
            binding.regional_envelope_sha256,
            binding.regional_envelope_receipt_sha256,
            binding.regional_envelope_generation,
            binding.store_id,
            binding.region,
            binding.performance_plan_sha256,
            binding.mapping_descriptor_sha256,
            binding.geometry_sha256,
            binding.maxtext_revision,
            binding.source_destination_count,
            binding.performance_leaf_count,
        )
        expected_leaf_specs = self._derive_leaf_specs(reader, plan)
        if (
            actual_binding_identity != expected_binding_identity
            or envelope_receipt != reader._opening_envelope_receipt
            or envelope_receipt.sha256 != reader.envelope.digest
            or envelope_receipt.store_id != reader.envelope.store_id
            or envelope_receipt.region != reader.envelope.region
            or tuple(item.spec for item in leaves) != expected_leaf_specs
            or any(
                item._reader is not reader
                or item.maximum_read_bytes != reader.maximum_read_bytes
                or type(item._sources) is not _MAPPING_PROXY_TYPE
                or tuple(item._sources.items())
                != tuple(
                    (source.destination_key, source) for source in item.spec.sources
                )
                for item in leaves
            )
        ):
            raise PrefixRestoreAuthorityError(
                "direct restore authority components are not cross-bound"
            )
        object.__setattr__(self, "binding", binding)
        object.__setattr__(self, "plan", plan)
        object.__setattr__(self, "_reader", reader)
        object.__setattr__(self, "_envelope_receipt", envelope_receipt)
        object.__setattr__(self, "leaves", leaves)
        object.__setattr__(
            self,
            "_leaves_by_key",
            MappingProxyType({item.spec.leaf_key: item for item in leaves}),
        )

    @classmethod
    def open(
        cls,
        store: GenerationPinnedPrefixObjectReader,
        envelope_receipt: GCSPhysicalObjectReceiptV1,
        *,
        expected_identity: PrefixRestoreIdentityExpectationV1,
        maximum_control_bytes: int,
        maximum_read_bytes: int = DEFAULT_LEAF_REGION_READ_BYTES,
    ) -> Self:
        """Open two exact control records; do not touch any payload object."""

        if type(expected_identity) is not PrefixRestoreIdentityExpectationV1:
            raise PrefixRestoreAuthorityError(
                "restore authority requires an exact bound identity expectation"
            )
        bound = _positive_bound(
            maximum_read_bytes,
            "maximum_read_bytes",
            maximum=MAX_LEAF_REGION_READ_BYTES,
        )
        try:
            reader = LazyRegionalPrefixCheckpointReaderV1.open(
                store,
                envelope_receipt,
                expected_logical_manifest_sha256=(
                    expected_identity.logical_manifest_sha256
                ),
                maximum_control_bytes=maximum_control_bytes,
                maximum_read_bytes=bound,
            )
        except (
            PrefixCheckpointV2Error,
            CheckpointConversionError,
            ProtocolError,
        ) as error:
            raise PrefixRestoreAuthorityError(
                "generation-pinned V2 checkpoint open failed"
            ) from error

        plan = build_pinned_prefix_performance_tree_plan()
        cls._validate_opened_control_plane(
            reader,
            envelope_receipt,
            expected_identity,
            plan,
        )
        binding = PrefixRestoreBindingV1(
            logical_manifest_sha256=reader.manifest.digest,
            selection_sha256=reader.manifest.selection_sha256,
            full_inventory_sha256=reader.manifest.full_inventory_sha256,
            selected_inventory_sha256=reader.manifest.selected_inventory_sha256,
            regional_envelope_sha256=reader.envelope.digest,
            regional_envelope_receipt_sha256=envelope_receipt.digest,
            regional_envelope_generation=envelope_receipt.generation,
            store_id=reader.envelope.store_id,
            region=reader.envelope.region,
            performance_plan_sha256=plan.digest,
            mapping_descriptor_sha256=plan.mapping_descriptor_sha256,
            geometry_sha256=reader.manifest.geometry_sha256,
            maxtext_revision=reader.manifest.maxtext_revision,
            source_destination_count=len(plan.destinations),
            performance_leaf_count=len(plan.leaves),
        )
        leaf_specs = cls._derive_leaf_specs(reader, plan)
        leaves = tuple(
            LazyPerformanceLeafV1(
                spec,
                reader,
                maximum_read_bytes=bound,
                _capability=_LEAF_VIEW_CAPABILITY,
            )
            for spec in leaf_specs
        )
        return cls(
            binding,
            plan,
            reader,
            leaves,
            envelope_receipt=envelope_receipt,
            _capability=_AUTHORITY_OPEN_CAPABILITY,
        )

    @staticmethod
    def _validate_opened_control_plane(
        reader: LazyRegionalPrefixCheckpointReaderV1,
        envelope_receipt: GCSPhysicalObjectReceiptV1,
        expected: PrefixRestoreIdentityExpectationV1,
        plan: PrefixPerformanceTreePlanV1,
    ) -> None:
        manifest = reader.manifest
        envelope = reader.envelope
        if (
            type(plan) is not PrefixPerformanceTreePlanV1
            or plan.digest != PINNED_UNBOUND_PREFIX_PERFORMANCE_TREE_PLAN_SHA256
            or plan.inventory_bound is not False
            or plan.inventory_selection_sha256 is not None
            or plan.production_sealable is not False
            or plan.production_admission_enabled is not False
        ):
            raise PrefixRestoreAuthorityError(
                "canonical input plan is not the exact unbound planning contract"
            )
        try:
            manifest.require_pinned_production_invariants()
        except (ProtocolError, CheckpointConversionError) as error:
            raise PrefixRestoreAuthorityError(
                "logical manifest is not the pinned R1 prefix profile"
            ) from error
        if (
            manifest.digest != expected.logical_manifest_sha256
            or manifest.selection_sha256 != expected.selection_sha256
            or manifest.full_inventory_sha256 != expected.full_inventory_sha256
            or manifest.selected_inventory_sha256 != expected.selected_inventory_sha256
        ):
            raise PrefixRestoreAuthorityError(
                "opened manifest differs from the expected inventory binding"
            )
        if (
            manifest.geometry_sha256 != PINNED_GEOMETRY.digest
            or manifest.maxtext_revision != plan.maxtext_revision
            or manifest.engine_sha256 != conversion_engine_sha256()
            or manifest.numeric_codec_sha256 != numeric_codec_identity_sha256()
            or manifest.storage_complete is not True
            or envelope.storage_complete is not True
            or manifest.production_sealable is not False
            or envelope.production_sealable is not False
        ):
            raise PrefixRestoreAuthorityError(
                "opened V2 control plane differs from the pinned restore contract"
            )
        if (
            envelope_receipt.sha256 != envelope.digest
            or envelope_receipt.generation == ""
            or envelope.logical_manifest_sha256 != manifest.digest
        ):
            raise PrefixRestoreAuthorityError(
                "regional envelope is not bound to the exact opening receipt"
            )
        manifest_destinations = {
            item.destination_key: item for item in manifest.destinations
        }
        expected_keys = tuple(item.destination_key for item in plan.destinations)
        if tuple(manifest_destinations) != expected_keys:
            raise PrefixRestoreAuthorityError(
                "V2 manifest destinations differ from the canonical 510-plan"
            )
        for destination in plan.destinations:
            stored = manifest_destinations[destination.destination_key]
            if (
                stored.dtype != destination.destination_dtype
                or stored.shape != destination.destination_shape
            ):
                raise PrefixRestoreAuthorityError(
                    "V2 destination shape or dtype differs from its canonical mapping"
                )

    @staticmethod
    def _derive_leaf_specs(
        reader: LazyRegionalPrefixCheckpointReaderV1,
        plan: PrefixPerformanceTreePlanV1,
    ) -> tuple[PerformanceLeafRestoreSpecV1, ...]:
        stored_by_key = {
            item.destination_key: item for item in reader.manifest.destinations
        }
        planned_by_key = {item.destination_key: item for item in plan.destinations}
        result: list[PerformanceLeafRestoreSpecV1] = []
        for leaf in plan.leaves:
            sources = tuple(
                PrefixDirectRestoreAuthorityV1._derive_source_spec(
                    planned_by_key[key],
                    stored_by_key[key],
                )
                for key in leaf.source_destination_keys
            )
            result.append(
                PerformanceLeafRestoreSpecV1(
                    leaf_key=leaf.leaf_key,
                    section=leaf.section,
                    family=leaf.family,
                    dtype=leaf.dtype,
                    logical_shape=leaf.logical_shape,
                    scan_axis=leaf.scan_axis,
                    sources=sources,
                )
            )
        return tuple(result)

    @staticmethod
    def _derive_source_spec(
        planned: DestinationConsumptionV1,
        stored: LogicalStoredDestinationV2,
    ) -> DestinationRestoreSpecV1:
        if (
            planned.destination_key != stored.destination_key
            or planned.destination_dtype != stored.dtype
            or planned.destination_shape != stored.shape
        ):
            raise PrefixRestoreAuthorityError(
                "cannot derive a restore source from inconsistent inputs"
            )
        return DestinationRestoreSpecV1(
            destination_key=planned.destination_key,
            dtype=stored.dtype,
            shape=stored.shape,
            performance_section=planned.performance_section,
            performance_leaf_key=planned.performance_leaf_key,
            scan_index=planned.scan_index,
        )

    def leaf(self, leaf_key: str) -> LazyPerformanceLeafV1:
        _identifier(leaf_key, "leaf_key")
        result = self._leaves_by_key.get(leaf_key)
        if result is None:
            raise PrefixRestoreAuthorityError("unknown performance leaf")
        return result

    def iter_leaves(self) -> Iterator[LazyPerformanceLeafV1]:
        return iter(self.leaves)

    def arrange_leaf(
        self,
        leaf_key: str,
        adapter: PerformanceLeafAdapter,
    ) -> object:
        """Delegate one leaf without changing any restore/qualification flag."""

        arrange = getattr(adapter, "arrange_leaf", None)
        if not callable(arrange):
            raise PrefixRestoreAuthorityError(
                "performance adapter must implement arrange_leaf"
            )
        return arrange(self.leaf(leaf_key))

    @property
    def digest(self) -> str:
        """Digest of the binding only; lazy views and adapter results are excluded."""

        return hashlib.sha256(canonical_json_bytes(self.binding)).hexdigest()


__all__ = [
    "DEFAULT_LEAF_REGION_READ_BYTES",
    "DIRECT_PREFIX_RESTORE_CONTRACT",
    "DestinationRestoreSpecV1",
    "LazyPerformanceLeafV1",
    "MAX_LEAF_REGION_READ_BYTES",
    "PerformanceLeafAdapter",
    "PerformanceLeafRestoreSpecV1",
    "PrefixDirectRestoreAuthorityV1",
    "PrefixRestoreAuthorityError",
    "PrefixRestoreBindingV1",
    "PrefixRestoreIdentityExpectationV1",
]
