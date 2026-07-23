"""Bounded, fail-closed execution core for the R1 prefix conversion.

This module deliberately stops before Orbax/TensorStore publication.  It turns
the already-audited :class:`CheckpointInventorySelectionV1` into immutable
chunk work, executes those chunks against two pluggable trust boundaries, and
returns a diagnostic attempt record.  It cannot seal a production checkpoint.

The source is responsible for authenticating a complete safetensor blob before
exposing any tensor region.  The sink is responsible for create-only durable
writes and for re-hashing an immutable object before reporting a resumed
commit.  Those responsibilities are explicit protocols rather than fake local
implementations masquerading as production storage.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
from importlib import import_module
from importlib.metadata import PackageNotFoundError, distribution
import json
import math
from pathlib import Path
import platform
import re
import sys
import tomllib
from typing import Any, ClassVar, Protocol

import ml_dtypes
import numpy as np

from .checkpoint_inventory import (
    BF16_DTYPE,
    F32_DTYPE,
    FP8_DTYPE,
    HF_REVISION,
    MAXTEXT_REVISION,
    CheckpointInventorySelectionV1,
    MaxTextMappingDescriptorV1,
    ShardSourceDescriptorV1,
    SourceTensorEntryV1,
)


CONVERSION_ENGINE_CONTRACT = "deepseek-r1-bounded-numpy-conversion-core-v1"
CHUNKING_CONTRACT = "source-axis-aligned-contiguous-destination-chunks-v1"
DEQUANTIZATION_CONTRACT = (
    "fp8-e4m3fn-to-f32-times-f32-inverse-scale-blockwise-then-bf16-v1"
)
SINK_VERIFICATION_CONTRACT = "sink-rehashed-create-only-immutable-bytes-v1"
SOURCE_VERIFICATION_CONTRACT = "full-blob-sha256-and-size-before-read-v1"

_BF16 = np.dtype(ml_dtypes.bfloat16)
_FP8 = np.dtype(ml_dtypes.float8_e4m3fn)
_F32 = np.dtype("<f4")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,511}")
_DESTINATION_DTYPE_BYTES = {"bfloat16": 2, "float32": 4}
_SOURCE_NUMPY_DTYPES = {
    FP8_DTYPE: _FP8,
    BF16_DTYPE: _BF16,
    F32_DTYPE: _F32,
}


class CheckpointConversionError(RuntimeError):
    """The conversion plan, source, transform, or sink failed closed."""


class ProductionSealUnavailable(CheckpointConversionError):
    """A diagnostic conversion attempt cannot become a production seal."""


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
        raise CheckpointConversionError(
            "conversion record is not canonical JSON"
        ) from error


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _require_sha256(value: str, name: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise CheckpointConversionError(f"{name} must be a lowercase SHA-256")


def _require_identifier(value: str, name: str) -> None:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise CheckpointConversionError(f"{name} is not a bounded identifier")


def _require_int(
    value: int,
    name: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> None:
    if type(value) is not int or value < minimum:
        raise CheckpointConversionError(f"{name} must be an integer >= {minimum}")
    if maximum is not None and value > maximum:
        raise CheckpointConversionError(f"{name} must be <= {maximum}")


def _numel(shape: Sequence[int]) -> int:
    return math.prod(shape)


def conversion_engine_sha256() -> str:
    """Hash the exact installed execution module used to derive chunk IDs."""

    try:
        raw = Path(__file__).read_bytes()
    except OSError as error:  # pragma: no cover - packaging failure, not a fixture
        raise CheckpointConversionError(
            "cannot hash the installed conversion implementation"
        ) from error
    return hashlib.sha256(raw).hexdigest()


def _hash_installed_file(path: str | None, name: str) -> str:
    if path is None:
        raise CheckpointConversionError(f"installed {name} has no filesystem path")
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError as error:
        raise CheckpointConversionError(
            f"cannot hash installed numerical component {name}"
        ) from error


@lru_cache(maxsize=1)
def _numeric_codec_identity_items() -> tuple[tuple[str, str], ...]:
    """Measure the installed codecs once, including their native extensions."""

    try:
        numpy_distribution = distribution("numpy")
        ml_dtypes_distribution = distribution("ml-dtypes")
    except PackageNotFoundError as error:  # pragma: no cover - dependency corruption
        raise CheckpointConversionError(
            "cannot identify installed numerical codec distributions"
        ) from error
    numpy_record = numpy_distribution.read_text("RECORD")
    ml_dtypes_record = ml_dtypes_distribution.read_text("RECORD")
    if numpy_record is None or ml_dtypes_record is None:
        raise CheckpointConversionError(
            "numerical codec distribution RECORD is unavailable"
        )
    numpy_extension = import_module("numpy._core._multiarray_umath")
    ml_dtypes_extension = import_module("ml_dtypes._ml_dtypes_ext")
    values = {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform_machine": platform.machine(),
        "platform_system": platform.system(),
        "byteorder": sys.byteorder,
        "numpy_version": np.__version__,
        "numpy_record_sha256": hashlib.sha256(numpy_record.encode("utf-8")).hexdigest(),
        "numpy_core_extension_sha256": _hash_installed_file(
            getattr(numpy_extension, "__file__", None), "NumPy core extension"
        ),
        "ml_dtypes_version": ml_dtypes.__version__,
        "ml_dtypes_record_sha256": hashlib.sha256(
            ml_dtypes_record.encode("utf-8")
        ).hexdigest(),
        "ml_dtypes_extension_sha256": _hash_installed_file(
            getattr(ml_dtypes_extension, "__file__", None),
            "ml_dtypes extension",
        ),
        "fp8_dtype": str(_FP8),
        "bf16_dtype": str(_BF16),
        "float32_dtype": _F32.str,
    }
    return tuple(sorted(values.items()))


def numeric_codec_identity() -> dict[str, str]:
    """Return the measured numerical runtime bound into conversion identity."""

    return dict(_numeric_codec_identity_items())


def numeric_codec_identity_sha256() -> str:
    return _sha256(numeric_codec_identity())


@dataclass(frozen=True, slots=True)
class SourceAdapterIdentityV1:
    """Externally attested implementation identity for a verified source."""

    adapter_id: str
    implementation_sha256: str

    def __post_init__(self) -> None:
        _require_identifier(self.adapter_id, "source adapter_id")
        _require_sha256(
            self.implementation_sha256, "source adapter implementation_sha256"
        )

    def as_dict(self) -> dict[str, str]:
        return {
            "adapter_id": self.adapter_id,
            "implementation_sha256": self.implementation_sha256,
        }


@dataclass(frozen=True, slots=True)
class R1RuntimeLockIdentityV1:
    """Measured identity of an isolated project lock pinned to R1 MaxText."""

    maxtext_revision: str
    pyproject_sha256: str
    uv_lock_sha256: str

    def __post_init__(self) -> None:
        if self.maxtext_revision != MAXTEXT_REVISION:
            raise CheckpointConversionError("runtime lock has the wrong MaxText pin")
        _require_sha256(self.pyproject_sha256, "runtime pyproject_sha256")
        _require_sha256(self.uv_lock_sha256, "runtime uv_lock_sha256")

    def as_dict(self) -> dict[str, str]:
        return {
            "maxtext_revision": self.maxtext_revision,
            "pyproject_sha256": self.pyproject_sha256,
            "uv_lock_sha256": self.uv_lock_sha256,
        }


def attest_r1_runtime_lock(project_root: str | Path) -> R1RuntimeLockIdentityV1:
    """Measure a local lock and reject the shared stale Gemma MaxText pin."""

    root = Path(project_root).resolve()
    pyproject_path = root / "pyproject.toml"
    uv_lock_path = root / "uv.lock"
    try:
        pyproject_raw = pyproject_path.read_bytes()
        uv_lock_raw = uv_lock_path.read_bytes()
    except OSError as error:
        raise CheckpointConversionError(
            "R1 runtime attestation requires readable pyproject.toml and uv.lock"
        ) from error
    try:
        pyproject = tomllib.loads(pyproject_raw.decode("utf-8"))
        lock = tomllib.loads(uv_lock_raw.decode("utf-8"))
        source = pyproject["tool"]["uv"]["sources"]["maxtext"]
    except (KeyError, TypeError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise CheckpointConversionError(
            "R1 runtime lock metadata is malformed"
        ) from error
    if (
        not isinstance(source, dict)
        or source.get("git") != "https://github.com/AI-Hypercomputer/maxtext.git"
        or source.get("rev") != MAXTEXT_REVISION
    ):
        raise CheckpointConversionError(
            "active project MaxText source is not pinned to the audited R1 revision"
        )
    packages = lock.get("package")
    if not isinstance(packages, list):
        raise CheckpointConversionError("uv.lock has no package inventory")
    locked_maxtext = [
        package
        for package in packages
        if isinstance(package, dict) and package.get("name") == "maxtext"
    ]
    expected_fragment = f"?rev={MAXTEXT_REVISION}#{MAXTEXT_REVISION}"
    if len(locked_maxtext) != 1:
        raise CheckpointConversionError(
            "uv.lock must contain exactly one MaxText package"
        )
    locked_source = locked_maxtext[0].get("source")
    if (
        not isinstance(locked_source, dict)
        or not isinstance(locked_source.get("git"), str)
        or expected_fragment not in locked_source["git"]
    ):
        raise CheckpointConversionError(
            "uv.lock MaxText package is not resolved to the audited R1 revision"
        )
    return R1RuntimeLockIdentityV1(
        maxtext_revision=MAXTEXT_REVISION,
        pyproject_sha256=hashlib.sha256(pyproject_raw).hexdigest(),
        uv_lock_sha256=hashlib.sha256(uv_lock_raw).hexdigest(),
    )


@dataclass(frozen=True, slots=True)
class ConversionMemoryLimitsV1:
    """Hard bounds applied before source I/O.

    ``max_source_blob_bytes`` bounds the source adapter's one-shard local
    scratch.  The remaining fields bound resident arrays owned explicitly by
    the core and sink.  A sink implementation may retain at most
    ``sink_staging_buffer_bytes`` additional application-owned bytes while the
    supplied memoryview is live.  This is not a claim about allocator overhead
    or opaque third-party client RSS; those require a measured qualification.
    """

    max_source_blob_bytes: int = 6 * 1024**3
    max_resident_bytes: int = 512 * 1024**2
    max_output_chunk_bytes: int = 64 * 1024**2
    # The production GCS sink snapshots a complete maximum-sized conversion
    # chunk into immutable bytes before it can occupy a canonical SHA path and
    # may concurrently own one bounded 8-MiB upload/verification slice.
    sink_staging_buffer_bytes: int = 72 * 1024**2

    def __post_init__(self) -> None:
        _require_int(
            self.max_source_blob_bytes,
            "max_source_blob_bytes",
            minimum=1,
            maximum=64 * 1024**3,
        )
        _require_int(
            self.max_resident_bytes,
            "max_resident_bytes",
            minimum=1,
            maximum=8 * 1024**3,
        )
        _require_int(
            self.max_output_chunk_bytes,
            "max_output_chunk_bytes",
            minimum=1,
            maximum=self.max_resident_bytes,
        )
        _require_int(
            self.sink_staging_buffer_bytes,
            "sink_staging_buffer_bytes",
            minimum=0,
            maximum=self.max_resident_bytes,
        )
        if self.sink_staging_buffer_bytes >= self.max_resident_bytes:
            raise CheckpointConversionError(
                "sink staging buffer leaves no resident transform budget"
            )

    def as_dict(self) -> dict[str, int]:
        return {
            "max_source_blob_bytes": self.max_source_blob_bytes,
            "max_resident_bytes": self.max_resident_bytes,
            "max_output_chunk_bytes": self.max_output_chunk_bytes,
            "sink_staging_buffer_bytes": self.sink_staging_buffer_bytes,
        }


@dataclass(frozen=True, slots=True)
class TensorReadRegionV1:
    """One bounded half-open hyperrectangle requested from a verified shard."""

    entry: SourceTensorEntryV1
    starts: tuple[int, ...]
    stops: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.entry) is not SourceTensorEntryV1:
            raise CheckpointConversionError(
                "tensor read entry must be SourceTensorEntryV1"
            )
        if type(self.starts) is not tuple or type(self.stops) is not tuple:
            raise CheckpointConversionError("tensor read bounds must be tuples")
        if len(self.starts) != len(self.entry.shape) or len(self.stops) != len(
            self.entry.shape
        ):
            raise CheckpointConversionError("tensor read rank mismatch")
        for axis, (start, stop, dimension) in enumerate(
            zip(self.starts, self.stops, self.entry.shape, strict=True)
        ):
            _require_int(start, f"tensor read start[{axis}]", minimum=0)
            _require_int(stop, f"tensor read stop[{axis}]", minimum=1)
            if start >= stop or stop > dimension:
                raise CheckpointConversionError(
                    f"tensor read bounds are outside axis {axis}"
                )

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(
            stop - start for start, stop in zip(self.starts, self.stops, strict=True)
        )

    @property
    def payload_bytes(self) -> int:
        return _numel(self.shape) * self.entry.payload_bytes // self.entry.numel


@dataclass(frozen=True, slots=True)
class ConversionChunkIdentityV1:
    """Immutable source/transform/destination identity for one output chunk."""

    selection_sha256: str
    engine_sha256: str
    numeric_codec_sha256: str
    source_adapter_identity: SourceAdapterIdentityV1
    runtime_lock_identity: R1RuntimeLockIdentityV1 | None
    source_revision: str
    source_descriptor: ShardSourceDescriptorV1
    source_entry: SourceTensorEntryV1
    scale_descriptor: ShardSourceDescriptorV1 | None
    scale_entry: SourceTensorEntryV1 | None
    mapping: MaxTextMappingDescriptorV1
    source_stream_axis: int
    source_start: int
    source_stop: int
    destination_flat_start: int
    destination_flat_stop: int
    fp8_block_size: int
    estimated_peak_resident_bytes: int
    chunk_id: str = field(init=False)

    def __post_init__(self) -> None:
        _require_sha256(self.selection_sha256, "selection_sha256")
        _require_sha256(self.engine_sha256, "engine_sha256")
        _require_sha256(self.numeric_codec_sha256, "numeric_codec_sha256")
        if type(self.source_adapter_identity) is not SourceAdapterIdentityV1:
            raise CheckpointConversionError(
                "chunk source adapter identity has the wrong type"
            )
        if (
            self.runtime_lock_identity is not None
            and type(self.runtime_lock_identity) is not R1RuntimeLockIdentityV1
        ):
            raise CheckpointConversionError(
                "chunk runtime lock identity has the wrong type"
            )
        _require_identifier(self.source_revision, "source_revision")
        if type(self.source_descriptor) is not ShardSourceDescriptorV1:
            raise CheckpointConversionError(
                "source descriptor must be ShardSourceDescriptorV1"
            )
        if type(self.source_entry) is not SourceTensorEntryV1:
            raise CheckpointConversionError("source entry must be SourceTensorEntryV1")
        if type(self.mapping) is not MaxTextMappingDescriptorV1:
            raise CheckpointConversionError(
                "mapping must be MaxTextMappingDescriptorV1"
            )
        if self.source_entry.shard != self.source_descriptor.shard:
            raise CheckpointConversionError("source descriptor/entry shard mismatch")
        if self.mapping.source_key != self.source_entry.key:
            raise CheckpointConversionError("mapping/source key mismatch")
        if (
            self.mapping.source_dtype != self.source_entry.dtype
            or self.mapping.source_shape != self.source_entry.shape
        ):
            raise CheckpointConversionError("mapping/source metadata mismatch")
        if (self.scale_entry is None) != (self.scale_descriptor is None):
            raise CheckpointConversionError(
                "scale entry and descriptor must be present together"
            )
        _require_int(self.fp8_block_size, "fp8_block_size", minimum=1)
        if self.mapping.scale_key is None:
            if self.scale_entry is not None:
                raise CheckpointConversionError("non-FP8 chunk consumed a scale")
            if self.source_entry.dtype == FP8_DTYPE:
                raise CheckpointConversionError("FP8 chunk is missing a scale mapping")
        else:
            if self.scale_entry is None or self.scale_descriptor is None:
                raise CheckpointConversionError("FP8 chunk is missing its scale")
            if self.scale_entry.key != self.mapping.scale_key:
                raise CheckpointConversionError("FP8 scale key mismatch")
            if self.scale_entry.dtype != F32_DTYPE:
                raise CheckpointConversionError("FP8 scale must be F32")
            if self.scale_entry.shard != self.scale_descriptor.shard:
                raise CheckpointConversionError("scale descriptor/entry shard mismatch")
            expected_scale_shape = tuple(
                (dimension + self.fp8_block_size - 1) // self.fp8_block_size
                for dimension in self.source_entry.shape
            )
            if self.scale_entry.shape != expected_scale_shape:
                raise CheckpointConversionError("FP8 scale geometry mismatch")
        _require_int(self.source_stream_axis, "source_stream_axis", minimum=0)
        if self.source_stream_axis >= len(self.source_entry.shape):
            raise CheckpointConversionError("source stream axis is outside source rank")
        _require_int(self.source_start, "source_start", minimum=0)
        _require_int(self.source_stop, "source_stop", minimum=1)
        if (
            self.source_start >= self.source_stop
            or self.source_stop > self.source_entry.shape[self.source_stream_axis]
        ):
            raise CheckpointConversionError("source stream interval is invalid")
        _require_int(self.destination_flat_start, "destination_flat_start", minimum=0)
        _require_int(self.destination_flat_stop, "destination_flat_stop", minimum=1)
        destination_numel = _numel(self.mapping.destination_shape)
        if (
            self.destination_flat_start >= self.destination_flat_stop
            or self.destination_flat_stop > destination_numel
        ):
            raise CheckpointConversionError("destination flat interval is invalid")
        _require_int(
            self.estimated_peak_resident_bytes,
            "estimated_peak_resident_bytes",
            minimum=1,
        )
        expected_elements = self._expected_output_elements()
        if (
            self.destination_flat_stop - self.destination_flat_start
            != expected_elements
        ):
            raise CheckpointConversionError(
                "source chunk and destination interval element counts differ"
            )
        if self.mapping.destination_slice_axis is None:
            destination_base = 0
        else:
            if self.mapping.destination_slice_axis != 0:
                raise CheckpointConversionError(
                    "streaming conversion supports only leading-axis destination slices"
                )
            slice_numel = (
                destination_numel
                // self.mapping.destination_shape[self.mapping.destination_slice_axis]
            )
            destination_base = self.mapping.destination_slice_index * slice_numel
        source_trailing = (
            _numel(self.source_entry.shape[1:])
            if self.mapping.transform == "identity-v1"
            else self.source_entry.shape[0]
        )
        if (
            self.destination_flat_start
            != destination_base + self.source_start * source_trailing
            or self.destination_flat_stop
            != destination_base + self.source_stop * source_trailing
        ):
            raise CheckpointConversionError(
                "destination interval does not match the canonical source stream slice"
            )
        object.__setattr__(self, "chunk_id", _sha256(self.identity_dict()))

    def _expected_output_elements(self) -> int:
        shape = self.source_entry.shape
        width = self.source_stop - self.source_start
        if self.mapping.transform == "identity-v1":
            if self.source_stream_axis != 0:
                raise CheckpointConversionError(
                    "identity chunks must stream source axis zero"
                )
            return width * _numel(shape[self.source_stream_axis + 1 :])
        if self.source_stream_axis != 1 or len(shape) != 2:
            raise CheckpointConversionError(
                "transpose/dequant chunks must stream rank-two source columns"
            )
        return shape[0] * width

    @property
    def output_elements(self) -> int:
        return self.destination_flat_stop - self.destination_flat_start

    @property
    def output_bytes(self) -> int:
        return (
            self.output_elements
            * _DESTINATION_DTYPE_BYTES[self.mapping.destination_dtype]
        )

    def source_region(self) -> TensorReadRegionV1:
        starts = [0] * len(self.source_entry.shape)
        stops = list(self.source_entry.shape)
        starts[self.source_stream_axis] = self.source_start
        stops[self.source_stream_axis] = self.source_stop
        return TensorReadRegionV1(self.source_entry, tuple(starts), tuple(stops))

    def scale_region(self) -> TensorReadRegionV1 | None:
        if self.scale_entry is None:
            return None
        if self.source_stream_axis != 1:
            raise CheckpointConversionError("FP8 chunks must stream source columns")
        scale_start = self.source_start // self.fp8_block_size
        scale_stop = (self.source_stop + self.fp8_block_size - 1) // self.fp8_block_size
        return TensorReadRegionV1(
            self.scale_entry,
            (0, scale_start),
            (self.scale_entry.shape[0], scale_stop),
        )

    def identity_dict(self) -> dict[str, Any]:
        return {
            "schema": "deepseek-r1-conversion-chunk-identity-v1",
            "selection_sha256": self.selection_sha256,
            "engine_sha256": self.engine_sha256,
            "numeric_codec_sha256": self.numeric_codec_sha256,
            "source_adapter_identity": self.source_adapter_identity.as_dict(),
            "runtime_lock_identity": (
                self.runtime_lock_identity.as_dict()
                if self.runtime_lock_identity is not None
                else None
            ),
            "engine_contract": CONVERSION_ENGINE_CONTRACT,
            "chunking_contract": CHUNKING_CONTRACT,
            "dequantization_contract": DEQUANTIZATION_CONTRACT,
            "source_revision": self.source_revision,
            "source_shard": self.source_descriptor.lfs_manifest_dict(),
            "source_entry": self.source_entry.as_dict(),
            "scale_shard": (
                self.scale_descriptor.lfs_manifest_dict()
                if self.scale_descriptor is not None
                else None
            ),
            "scale_entry": (
                self.scale_entry.as_dict() if self.scale_entry is not None else None
            ),
            "mapping": self.mapping.as_dict(),
            "source_stream_axis": self.source_stream_axis,
            "source_start": self.source_start,
            "source_stop": self.source_stop,
            "destination_flat_start": self.destination_flat_start,
            "destination_flat_stop": self.destination_flat_stop,
            "fp8_block_size": self.fp8_block_size,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.identity_dict(),
            "chunk_id": self.chunk_id,
            "output_elements": self.output_elements,
            "output_bytes": self.output_bytes,
            "estimated_peak_resident_bytes": self.estimated_peak_resident_bytes,
        }


@dataclass(frozen=True, slots=True)
class ConversionShardWorkV1:
    source_descriptor: ShardSourceDescriptorV1
    chunks: tuple[ConversionChunkIdentityV1, ...]

    def __post_init__(self) -> None:
        if type(self.source_descriptor) is not ShardSourceDescriptorV1:
            raise CheckpointConversionError("invalid shard-work descriptor")
        if type(self.chunks) is not tuple or not self.chunks:
            raise CheckpointConversionError("shard work must contain chunk tuples")
        if any(type(chunk) is not ConversionChunkIdentityV1 for chunk in self.chunks):
            raise CheckpointConversionError("invalid shard-work chunk type")
        if any(
            chunk.source_descriptor != self.source_descriptor for chunk in self.chunks
        ):
            raise CheckpointConversionError("shard work contains another shard")
        ordering = tuple(
            (chunk.source_entry.key, chunk.source_start) for chunk in self.chunks
        )
        if ordering != tuple(sorted(ordering)):
            raise CheckpointConversionError("shard chunks are not deterministic")

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_descriptor": self.source_descriptor.as_dict(),
            "cross_shard_scale_cache_bytes": self.cross_shard_scale_cache_bytes,
            "estimated_peak_resident_bytes": self.estimated_peak_resident_bytes,
            "chunk_ids": [chunk.chunk_id for chunk in self.chunks],
        }

    @property
    def cross_shard_scale_cache_bytes(self) -> int:
        return sum(
            region.payload_bytes
            for chunk in self.chunks
            if chunk.scale_descriptor is not None
            and chunk.scale_descriptor != chunk.source_descriptor
            for region in (chunk.scale_region(),)
            if region is not None
        )

    @property
    def estimated_peak_resident_bytes(self) -> int:
        return self.cross_shard_scale_cache_bytes + max(
            chunk.estimated_peak_resident_bytes for chunk in self.chunks
        )


@dataclass(frozen=True, slots=True)
class DestinationTensorPlanV1:
    destination_key: str
    dtype: str
    shape: tuple[int, ...]
    chunks: tuple[tuple[int, int, str], ...]

    def __post_init__(self) -> None:
        _require_identifier(self.destination_key, "destination_key")
        if self.dtype not in _DESTINATION_DTYPE_BYTES:
            raise CheckpointConversionError("unsupported destination dtype")
        if type(self.shape) is not tuple or not self.shape:
            raise CheckpointConversionError("destination shape must be a tuple")
        if any(type(item) is not int or item <= 0 for item in self.shape):
            raise CheckpointConversionError("destination dimensions must be positive")
        if type(self.chunks) is not tuple or not self.chunks:
            raise CheckpointConversionError("destination must contain chunk tuples")
        cursor = 0
        for item in self.chunks:
            if type(item) is not tuple or len(item) != 3:
                raise CheckpointConversionError("invalid destination chunk interval")
            start, stop, chunk_id = item
            _require_sha256(chunk_id, "destination chunk_id")
            if start != cursor or stop <= start:
                raise CheckpointConversionError(
                    "destination chunks have a gap, overlap, or unstable order"
                )
            cursor = stop
        if cursor != _numel(self.shape):
            raise CheckpointConversionError(
                "destination chunks do not cover the complete logical tensor"
            )

    @property
    def payload_bytes(self) -> int:
        return _numel(self.shape) * _DESTINATION_DTYPE_BYTES[self.dtype]

    def as_dict(self) -> dict[str, Any]:
        return {
            "destination_key": self.destination_key,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "payload_bytes": self.payload_bytes,
            "chunks": [
                {"flat_start": start, "flat_stop": stop, "chunk_id": chunk_id}
                for start, stop, chunk_id in self.chunks
            ],
        }


@dataclass(frozen=True, slots=True)
class StreamingConversionPlanV1:
    """Immutable shard-ordered conversion work derived from one selection."""

    SCHEMA: ClassVar[str] = "deepseek-r1-streaming-conversion-plan-v1"

    selection: CheckpointInventorySelectionV1 = field(repr=False)
    mode: str
    selection_sha256: str
    full_inventory_sha256: str
    selected_inventory_sha256: str
    source_revision: str
    maxtext_revision: str
    geometry_sha256: str
    engine_sha256: str
    numeric_codec_sha256: str
    source_adapter_identity: SourceAdapterIdentityV1
    runtime_lock_identity: R1RuntimeLockIdentityV1 | None
    limits: ConversionMemoryLimitsV1
    shard_work: tuple[ConversionShardWorkV1, ...]
    destinations: tuple[DestinationTensorPlanV1, ...]
    converted_payload_bytes: int
    plan_id: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.selection) is not CheckpointInventorySelectionV1:
            raise CheckpointConversionError(
                "plan selection must be an exact CheckpointInventorySelectionV1"
            )
        if self.mode not in {"pinned-r1-prefix", "synthetic-test-prefix"}:
            raise CheckpointConversionError("unknown conversion plan mode")
        for name in (
            "selection_sha256",
            "full_inventory_sha256",
            "selected_inventory_sha256",
            "geometry_sha256",
            "engine_sha256",
            "numeric_codec_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        _require_identifier(self.source_revision, "source_revision")
        _require_identifier(self.maxtext_revision, "maxtext_revision")
        if type(self.source_adapter_identity) is not SourceAdapterIdentityV1:
            raise CheckpointConversionError(
                "plan source adapter identity has the wrong type"
            )
        if (
            self.runtime_lock_identity is not None
            and type(self.runtime_lock_identity) is not R1RuntimeLockIdentityV1
        ):
            raise CheckpointConversionError(
                "plan runtime lock identity has the wrong type"
            )
        if (
            self.selection.digest != self.selection_sha256
            or self.selection.full.digest != self.full_inventory_sha256
            or self.selection.selected.digest != self.selected_inventory_sha256
            or self.selection.full.revision != self.source_revision
            or self.selection.full.geometry.digest != self.geometry_sha256
            or self.selection.converted_payload_bytes != self.converted_payload_bytes
        ):
            raise CheckpointConversionError(
                "plan identity does not match its authoritative inventory selection"
            )
        if not self.selection.metadata_authenticated:
            raise CheckpointConversionError(
                "plan selection lacks authenticated source metadata"
            )
        if type(self.limits) is not ConversionMemoryLimitsV1:
            raise CheckpointConversionError("plan limits have the wrong type")
        if type(self.shard_work) is not tuple or not self.shard_work:
            raise CheckpointConversionError("plan shard work must be a tuple")
        if any(type(item) is not ConversionShardWorkV1 for item in self.shard_work):
            raise CheckpointConversionError("plan shard work has an invalid type")
        if type(self.destinations) is not tuple or not self.destinations:
            raise CheckpointConversionError("plan destinations must be a tuple")
        if any(type(item) is not DestinationTensorPlanV1 for item in self.destinations):
            raise CheckpointConversionError("plan destination has an invalid type")
        shard_names = tuple(item.source_descriptor.shard for item in self.shard_work)
        if shard_names != tuple(sorted(shard_names)) or len(shard_names) != len(
            set(shard_names)
        ):
            raise CheckpointConversionError("plan shard work is not unique and sorted")
        destination_keys = tuple(item.destination_key for item in self.destinations)
        if destination_keys != tuple(sorted(destination_keys)) or len(
            destination_keys
        ) != len(set(destination_keys)):
            raise CheckpointConversionError(
                "plan destinations are not unique and sorted"
            )
        chunks = self.chunks
        chunk_ids = tuple(chunk.chunk_id for chunk in chunks)
        if len(chunk_ids) != len(set(chunk_ids)):
            raise CheckpointConversionError("conversion chunk IDs collide")
        planned_relations = tuple(
            sorted(
                (destination.destination_key, start, stop, chunk_id)
                for destination in self.destinations
                for start, stop, chunk_id in destination.chunks
            )
        )
        chunk_relations = tuple(
            sorted(
                (
                    chunk.mapping.destination_key,
                    chunk.destination_flat_start,
                    chunk.destination_flat_stop,
                    chunk.chunk_id,
                )
                for chunk in chunks
            )
        )
        if planned_relations != chunk_relations:
            raise CheckpointConversionError(
                "destination plan does not exactly match conversion chunk intervals"
            )
        planned_ids = [
            chunk_id
            for destination in self.destinations
            for _, _, chunk_id in destination.chunks
        ]
        if len(planned_ids) != len(set(planned_ids)):
            raise CheckpointConversionError(
                "destination plan contains a duplicate conversion chunk ID"
            )
        selected_entries = {
            entry.key: entry for entry in self.selection.selected.entries
        }
        selected_descriptors = {
            descriptor.shard: descriptor
            for descriptor in self.selection.selected.shard_descriptors
        }
        selected_mappings = {
            mapping.source_key: mapping for mapping in self.selection.mappings
        }
        chunks_by_source: dict[str, list[ConversionChunkIdentityV1]] = defaultdict(list)
        for chunk in chunks:
            chunks_by_source[chunk.source_entry.key].append(chunk)
            if (
                chunk.selection_sha256 != self.selection_sha256
                or chunk.engine_sha256 != self.engine_sha256
                or chunk.numeric_codec_sha256 != self.numeric_codec_sha256
                or chunk.source_adapter_identity != self.source_adapter_identity
                or chunk.runtime_lock_identity != self.runtime_lock_identity
                or chunk.source_revision != self.source_revision
                or chunk.fp8_block_size != self.selection.full.geometry.fp8_block_size
                or selected_entries.get(chunk.source_entry.key) != chunk.source_entry
                or selected_descriptors.get(chunk.source_entry.shard)
                != chunk.source_descriptor
                or selected_mappings.get(chunk.source_entry.key) != chunk.mapping
            ):
                raise CheckpointConversionError(
                    "conversion chunk is not derived from the authoritative selection"
                )
            if chunk.scale_entry is not None:
                if (
                    selected_entries.get(chunk.scale_entry.key) != chunk.scale_entry
                    or selected_descriptors.get(chunk.scale_entry.shard)
                    != chunk.scale_descriptor
                ):
                    raise CheckpointConversionError(
                        "conversion scale is not derived from the authoritative selection"
                    )
            expected_peak = _estimate_peak_resident_bytes(
                chunk.mapping,
                chunk.source_entry,
                chunk.scale_entry,
                stream_width=chunk.source_stop - chunk.source_start,
                block_size=chunk.fp8_block_size,
                sink_staging_bytes=self.limits.sink_staging_buffer_bytes,
            )
            if chunk.estimated_peak_resident_bytes != expected_peak:
                raise CheckpointConversionError(
                    "conversion chunk resident estimate is not canonical"
                )
        if set(chunks_by_source) != set(selected_mappings):
            raise CheckpointConversionError(
                "conversion chunks do not cover every authoritative mapping"
            )
        for source_key, source_chunks in chunks_by_source.items():
            ordered = sorted(source_chunks, key=lambda item: item.source_start)
            cursor = 0
            for index, chunk in enumerate(ordered):
                if chunk.source_start != cursor:
                    raise CheckpointConversionError(
                        f"source stream has a gap or overlap for {source_key}"
                    )
                cursor = chunk.source_stop
                if (
                    chunk.mapping.transform != "identity-v1"
                    and index < len(ordered) - 1
                    and cursor % chunk.fp8_block_size != 0
                ):
                    raise CheckpointConversionError(
                        f"source stream has an unaligned interior boundary for {source_key}"
                    )
            stream_axis = ordered[0].source_stream_axis
            if cursor != ordered[0].source_entry.shape[stream_axis]:
                raise CheckpointConversionError(
                    f"source stream is incomplete for {source_key}"
                )
        _require_int(
            self.converted_payload_bytes,
            "converted_payload_bytes",
            minimum=1,
        )
        if sum(item.payload_bytes for item in self.destinations) != (
            self.converted_payload_bytes
        ):
            raise CheckpointConversionError("converted payload total is inconsistent")
        all_descriptors = {
            descriptor
            for chunk in chunks
            for descriptor in (chunk.source_descriptor, chunk.scale_descriptor)
            if descriptor is not None
        }
        for descriptor in all_descriptors:
            if descriptor.expected_blob_size_bytes > self.limits.max_source_blob_bytes:
                raise CheckpointConversionError(
                    f"source shard exceeds local scratch bound: {descriptor.shard}"
                )
        if any(
            chunk.output_bytes > self.limits.max_output_chunk_bytes
            or chunk.estimated_peak_resident_bytes > self.limits.max_resident_bytes
            for chunk in chunks
        ):
            raise CheckpointConversionError("plan contains a chunk above memory bounds")
        if any(
            shard.estimated_peak_resident_bytes > self.limits.max_resident_bytes
            for shard in self.shard_work
        ):
            raise CheckpointConversionError(
                "cross-shard scale cache exceeds the declared resident bound"
            )
        if self.mode == "pinned-r1-prefix":
            if (
                self.selection.full.source_kind != "pinned-r1-0528"
                or not self.selection.inventory_sealable
            ):
                raise CheckpointConversionError(
                    "pinned plan does not carry the sealed R1 inventory selection"
                )
            if self.runtime_lock_identity is None:
                raise CheckpointConversionError(
                    "pinned plan has no isolated R1 runtime lock identity"
                )
            if self.source_revision != HF_REVISION:
                raise CheckpointConversionError("pinned plan revision mismatch")
            if self.maxtext_revision != MAXTEXT_REVISION:
                raise CheckpointConversionError("pinned MaxText revision mismatch")
            if self.converted_payload_bytes != 649_674_223_616:
                raise CheckpointConversionError("pinned converted byte total mismatch")
        else:
            if self.selection.full.source_kind != "synthetic-test-fixture":
                raise CheckpointConversionError(
                    "synthetic plan does not carry an explicit synthetic selection"
                )
            if self.runtime_lock_identity is not None:
                raise CheckpointConversionError(
                    "synthetic plan must not carry a production runtime attestation"
                )
        object.__setattr__(self, "plan_id", _sha256(self.as_dict()))

    @property
    def chunks(self) -> tuple[ConversionChunkIdentityV1, ...]:
        return tuple(chunk for shard in self.shard_work for chunk in shard.chunks)

    @property
    def production_input_qualified(self) -> bool:
        return self.mode == "pinned-r1-prefix"

    @property
    def conversion_sealable(self) -> bool:
        return False

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "mode": self.mode,
            "selection_sha256": self.selection_sha256,
            "full_inventory_sha256": self.full_inventory_sha256,
            "selected_inventory_sha256": self.selected_inventory_sha256,
            "source_revision": self.source_revision,
            "maxtext_revision": self.maxtext_revision,
            "geometry_sha256": self.geometry_sha256,
            "engine_sha256": self.engine_sha256,
            "numeric_codec_sha256": self.numeric_codec_sha256,
            "source_adapter_identity": self.source_adapter_identity.as_dict(),
            "runtime_lock_identity": (
                self.runtime_lock_identity.as_dict()
                if self.runtime_lock_identity is not None
                else None
            ),
            "engine_contract": CONVERSION_ENGINE_CONTRACT,
            "chunking_contract": CHUNKING_CONTRACT,
            "dequantization_contract": DEQUANTIZATION_CONTRACT,
            "limits": self.limits.as_dict(),
            "converted_payload_bytes": self.converted_payload_bytes,
            "production_input_qualified": self.production_input_qualified,
            "conversion_sealable": self.conversion_sealable,
            "shard_work": [item.as_dict() for item in self.shard_work],
            "destinations": [item.as_dict() for item in self.destinations],
        }


def _estimate_peak_resident_bytes(
    mapping: MaxTextMappingDescriptorV1,
    source_entry: SourceTensorEntryV1,
    scale_entry: SourceTensorEntryV1 | None,
    *,
    stream_width: int,
    block_size: int,
    sink_staging_bytes: int,
) -> int:
    source_shape = source_entry.shape
    source_itemsize = source_entry.payload_bytes // source_entry.numel
    destination_itemsize = _DESTINATION_DTYPE_BYTES[mapping.destination_dtype]
    if mapping.transform == "identity-v1":
        trailing = _numel(source_shape[1:])
        source_bytes = stream_width * trailing * source_itemsize
        return source_bytes + sink_staging_bytes

    rows = source_shape[0]
    source_bytes = rows * stream_width * source_itemsize
    destination_bytes = rows * stream_width * destination_itemsize
    if mapping.transform == "transpose-reshape-v1":
        return source_bytes + destination_bytes + sink_staging_bytes
    if mapping.transform != (
        "blockwise-fp8-f32-dequantize-then-transpose-reshape-bf16-v1"
    ):
        raise CheckpointConversionError("unsupported conversion transform")
    if scale_entry is None:
        raise CheckpointConversionError("FP8 peak estimate is missing a scale")
    scale_rows = (rows + block_size - 1) // block_size
    scale_columns = (stream_width + block_size - 1) // block_size
    scale_bytes = scale_rows * scale_columns * 4
    fp32_work_bytes = rows * stream_width * 4
    return (
        source_bytes
        + scale_bytes
        + fp32_work_bytes
        + destination_bytes
        + sink_staging_bytes
    )


def _chunk_width(
    mapping: MaxTextMappingDescriptorV1,
    source_entry: SourceTensorEntryV1,
    scale_entry: SourceTensorEntryV1 | None,
    limits: ConversionMemoryLimitsV1,
    block_size: int,
) -> int:
    axis = 0 if mapping.transform == "identity-v1" else 1
    extent = source_entry.shape[axis]
    trailing = (
        _numel(source_entry.shape[1:])
        if mapping.transform == "identity-v1"
        else source_entry.shape[0]
    )
    output_itemsize = _DESTINATION_DTYPE_BYTES[mapping.destination_dtype]
    output_bound = limits.max_output_chunk_bytes // (trailing * output_itemsize)
    if output_bound < 1:
        raise CheckpointConversionError(
            f"output chunk bound cannot hold one stream unit for {source_entry.key}"
        )
    candidate = min(extent, output_bound)
    if mapping.transform != "identity-v1" and candidate < extent:
        candidate = candidate // block_size * block_size
    while candidate > 0:
        peak = _estimate_peak_resident_bytes(
            mapping,
            source_entry,
            scale_entry,
            stream_width=candidate,
            block_size=block_size,
            sink_staging_bytes=limits.sink_staging_buffer_bytes,
        )
        if peak <= limits.max_resident_bytes:
            return candidate
        if mapping.transform == "identity-v1":
            candidate //= 2
        elif candidate > block_size:
            candidate = (candidate - block_size) // block_size * block_size
        else:
            candidate = 0
    raise CheckpointConversionError(
        f"resident memory bound cannot hold the minimum aligned chunk for "
        f"{source_entry.key}"
    )


def _build_plan(
    selection: CheckpointInventorySelectionV1,
    limits: ConversionMemoryLimitsV1,
    source_adapter_identity: SourceAdapterIdentityV1,
    runtime_lock_identity: R1RuntimeLockIdentityV1 | None,
    *,
    mode: str,
) -> StreamingConversionPlanV1:
    if type(selection) is not CheckpointInventorySelectionV1:
        raise CheckpointConversionError(
            "conversion requires an exact CheckpointInventorySelectionV1"
        )
    if type(limits) is not ConversionMemoryLimitsV1:
        raise CheckpointConversionError("conversion limits have the wrong type")
    if type(source_adapter_identity) is not SourceAdapterIdentityV1:
        raise CheckpointConversionError(
            "conversion source adapter identity has the wrong type"
        )
    if not selection.metadata_authenticated:
        raise CheckpointConversionError(
            "conversion planning requires authenticated index and shard metadata"
        )
    if mode == "pinned-r1-prefix":
        if selection.full.source_kind != "pinned-r1-0528":
            raise CheckpointConversionError(
                "production planning accepts only the pinned R1-0528 selection"
            )
        if not selection.inventory_sealable:
            raise CheckpointConversionError(
                "pinned R1 inventory has not passed its metadata seal gate"
            )
    elif mode == "synthetic-test-prefix":
        if selection.full.source_kind != "synthetic-test-fixture":
            raise CheckpointConversionError(
                "synthetic planning accepts only an explicit synthetic fixture"
            )
    else:  # pragma: no cover - private caller contract
        raise CheckpointConversionError("unknown conversion planning mode")

    selected_entries = {entry.key: entry for entry in selection.selected.entries}
    descriptors = {
        descriptor.shard: descriptor
        for descriptor in selection.selected.shard_descriptors
    }
    engine_sha256 = conversion_engine_sha256()
    numeric_codec_sha256 = numeric_codec_identity_sha256()
    selection_sha256 = selection.digest
    full_inventory_sha256 = selection.full.digest
    selected_inventory_sha256 = selection.selected.digest
    geometry_sha256 = selection.full.geometry.digest
    chunks: list[ConversionChunkIdentityV1] = []
    block_size = selection.full.geometry.fp8_block_size

    for mapping in selection.mappings:
        try:
            source_entry = selected_entries[mapping.source_key]
            source_descriptor = descriptors[source_entry.shard]
        except KeyError as error:  # inventory should make this unreachable
            raise CheckpointConversionError(
                f"mapping lost authenticated source metadata: {mapping.source_key}"
            ) from error
        if mapping.scale_key is None:
            scale_entry = None
            scale_descriptor = None
        else:
            try:
                scale_entry = selected_entries[mapping.scale_key]
                scale_descriptor = descriptors[scale_entry.shard]
            except KeyError as error:
                raise CheckpointConversionError(
                    f"mapping lost authenticated scale metadata: {mapping.scale_key}"
                ) from error
        stream_axis = 0 if mapping.transform == "identity-v1" else 1
        width = _chunk_width(
            mapping,
            source_entry,
            scale_entry,
            limits,
            block_size,
        )
        extent = source_entry.shape[stream_axis]
        if mapping.destination_slice_axis is None:
            destination_base = 0
        else:
            slice_numel = (
                _numel(mapping.destination_shape)
                // mapping.destination_shape[mapping.destination_slice_axis]
            )
            destination_base = mapping.destination_slice_index * slice_numel

        start = 0
        while start < extent:
            stop = min(start + width, extent)
            if mapping.transform != "identity-v1" and stop < extent:
                if stop % block_size != 0:
                    raise CheckpointConversionError(
                        "interior transformed chunk is not FP8-block aligned"
                    )
            trailing = (
                _numel(source_entry.shape[1:])
                if mapping.transform == "identity-v1"
                else source_entry.shape[0]
            )
            destination_start = destination_base + start * trailing
            destination_stop = destination_base + stop * trailing
            peak = _estimate_peak_resident_bytes(
                mapping,
                source_entry,
                scale_entry,
                stream_width=stop - start,
                block_size=block_size,
                sink_staging_bytes=limits.sink_staging_buffer_bytes,
            )
            chunks.append(
                ConversionChunkIdentityV1(
                    selection_sha256=selection_sha256,
                    engine_sha256=engine_sha256,
                    numeric_codec_sha256=numeric_codec_sha256,
                    source_adapter_identity=source_adapter_identity,
                    runtime_lock_identity=runtime_lock_identity,
                    source_revision=selection.full.revision,
                    source_descriptor=source_descriptor,
                    source_entry=source_entry,
                    scale_descriptor=scale_descriptor,
                    scale_entry=scale_entry,
                    mapping=mapping,
                    source_stream_axis=stream_axis,
                    source_start=start,
                    source_stop=stop,
                    destination_flat_start=destination_start,
                    destination_flat_stop=destination_stop,
                    fp8_block_size=block_size,
                    estimated_peak_resident_bytes=peak,
                )
            )
            start = stop

    chunks.sort(
        key=lambda chunk: (
            chunk.source_descriptor.shard,
            chunk.source_entry.key,
            chunk.source_start,
        )
    )
    by_shard: dict[str, list[ConversionChunkIdentityV1]] = defaultdict(list)
    for chunk in chunks:
        by_shard[chunk.source_descriptor.shard].append(chunk)
    shard_work = tuple(
        ConversionShardWorkV1(descriptors[shard], tuple(by_shard[shard]))
        for shard in sorted(by_shard)
    )

    by_destination: dict[str, list[ConversionChunkIdentityV1]] = defaultdict(list)
    for chunk in chunks:
        by_destination[chunk.mapping.destination_key].append(chunk)
    destinations: list[DestinationTensorPlanV1] = []
    for key in sorted(by_destination):
        destination_chunks = by_destination[key]
        first = destination_chunks[0].mapping
        if any(
            item.mapping.destination_dtype != first.destination_dtype
            or item.mapping.destination_shape != first.destination_shape
            for item in destination_chunks
        ):
            raise CheckpointConversionError(
                f"destination metadata disagreement for {key}"
            )
        intervals = tuple(
            sorted(
                (
                    item.destination_flat_start,
                    item.destination_flat_stop,
                    item.chunk_id,
                )
                for item in destination_chunks
            )
        )
        destinations.append(
            DestinationTensorPlanV1(
                destination_key=key,
                dtype=first.destination_dtype,
                shape=first.destination_shape,
                chunks=intervals,
            )
        )

    return StreamingConversionPlanV1(
        selection=selection,
        mode=mode,
        selection_sha256=selection_sha256,
        full_inventory_sha256=full_inventory_sha256,
        selected_inventory_sha256=selected_inventory_sha256,
        source_revision=selection.full.revision,
        maxtext_revision=MAXTEXT_REVISION,
        geometry_sha256=geometry_sha256,
        engine_sha256=engine_sha256,
        numeric_codec_sha256=numeric_codec_sha256,
        source_adapter_identity=source_adapter_identity,
        runtime_lock_identity=runtime_lock_identity,
        limits=limits,
        shard_work=shard_work,
        destinations=tuple(destinations),
        converted_payload_bytes=selection.converted_payload_bytes,
    )


def build_pinned_prefix_conversion_plan(
    selection: CheckpointInventorySelectionV1,
    limits: ConversionMemoryLimitsV1 = ConversionMemoryLimitsV1(),
    *,
    source_adapter_identity: SourceAdapterIdentityV1,
    runtime_project_root: str | Path,
) -> StreamingConversionPlanV1:
    """Build work only from the authenticated, exact pinned R1 selection."""

    if (
        type(selection) is not CheckpointInventorySelectionV1
        or selection.full.source_kind != "pinned-r1-0528"
    ):
        raise CheckpointConversionError(
            "production planning accepts only the pinned R1-0528 selection"
        )
    runtime_lock_identity = attest_r1_runtime_lock(runtime_project_root)
    return _build_plan(
        selection,
        limits,
        source_adapter_identity,
        runtime_lock_identity,
        mode="pinned-r1-prefix",
    )


def build_synthetic_prefix_conversion_plan(
    selection: CheckpointInventorySelectionV1,
    limits: ConversionMemoryLimitsV1,
    *,
    source_adapter_identity: SourceAdapterIdentityV1,
) -> StreamingConversionPlanV1:
    """Build explicitly non-production work for bounded synthetic tests."""

    return _build_plan(
        selection,
        limits,
        source_adapter_identity,
        None,
        mode="synthetic-test-prefix",
    )


@dataclass(frozen=True, slots=True)
class VerifiedSourceShardReceiptV1:
    shard: str
    observed_sha256: str
    observed_size_bytes: int
    source_adapter_identity: SourceAdapterIdentityV1
    verification_contract: str = SOURCE_VERIFICATION_CONTRACT

    def __post_init__(self) -> None:
        _require_identifier(self.shard, "verified source shard")
        _require_sha256(self.observed_sha256, "observed source SHA-256")
        _require_int(self.observed_size_bytes, "observed source size", minimum=1)
        if type(self.source_adapter_identity) is not SourceAdapterIdentityV1:
            raise CheckpointConversionError(
                "source receipt adapter identity has the wrong type"
            )
        if self.verification_contract != SOURCE_VERIFICATION_CONTRACT:
            raise CheckpointConversionError("unsupported source verification contract")

    def validate_for(
        self,
        descriptor: ShardSourceDescriptorV1,
        source_adapter_identity: SourceAdapterIdentityV1,
    ) -> None:
        if (
            self.shard != descriptor.shard
            or self.observed_sha256 != descriptor.expected_lfs_sha256
            or self.observed_size_bytes != descriptor.expected_blob_size_bytes
            or self.source_adapter_identity != source_adapter_identity
        ):
            raise CheckpointConversionError(
                f"source blob receipt mismatch for {descriptor.shard}"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "shard": self.shard,
            "observed_sha256": self.observed_sha256,
            "observed_size_bytes": self.observed_size_bytes,
            "source_adapter_identity": self.source_adapter_identity.as_dict(),
            "verification_contract": self.verification_contract,
        }


class VerifiedShardReader(Protocol):
    """Reader exposed only after its complete blob receipt has been measured."""

    @property
    def receipt(self) -> VerifiedSourceShardReceiptV1: ...

    def read_region(self, region: TensorReadRegionV1) -> np.ndarray:
        """Return one owned, C-contiguous array with the exact logical dtype."""


class VerifiedTensorSource(Protocol):
    """Trust boundary for one-at-a-time authenticated shard access."""

    @property
    def adapter_identity(self) -> SourceAdapterIdentityV1: ...

    def open_verified_shard(
        self, descriptor: ShardSourceDescriptorV1
    ) -> AbstractContextManager[VerifiedShardReader]:
        """Verify the full blob before yielding; never nest local shard scratch."""


@dataclass(frozen=True, slots=True)
class OutputChunkReceiptV1:
    plan_id: str
    chunk_id: str
    sink_id: str
    destination_key: str
    destination_flat_start: int
    destination_flat_stop: int
    dtype: str
    output_sha256: str
    output_size_bytes: int
    immutable_object_id: str
    verification_contract: str = SINK_VERIFICATION_CONTRACT

    def __post_init__(self) -> None:
        _require_sha256(self.plan_id, "receipt plan_id")
        _require_sha256(self.chunk_id, "receipt chunk_id")
        _require_identifier(self.sink_id, "receipt sink_id")
        _require_identifier(self.destination_key, "receipt destination_key")
        _require_int(
            self.destination_flat_start,
            "receipt destination_flat_start",
            minimum=0,
        )
        _require_int(
            self.destination_flat_stop,
            "receipt destination_flat_stop",
            minimum=1,
        )
        if self.destination_flat_stop <= self.destination_flat_start:
            raise CheckpointConversionError("receipt destination interval is empty")
        if self.dtype not in _DESTINATION_DTYPE_BYTES:
            raise CheckpointConversionError("receipt dtype is unsupported")
        _require_sha256(self.output_sha256, "receipt output_sha256")
        _require_int(self.output_size_bytes, "receipt output_size_bytes", minimum=1)
        _require_identifier(self.immutable_object_id, "immutable_object_id")
        if self.verification_contract != SINK_VERIFICATION_CONTRACT:
            raise CheckpointConversionError("unsupported sink verification contract")

    def validate_for(
        self,
        plan: StreamingConversionPlanV1,
        chunk: ConversionChunkIdentityV1,
        *,
        sink_id: str,
        expected_output_sha256: str | None = None,
    ) -> None:
        if (
            self.plan_id != plan.plan_id
            or self.chunk_id != chunk.chunk_id
            or self.sink_id != sink_id
            or self.destination_key != chunk.mapping.destination_key
            or self.destination_flat_start != chunk.destination_flat_start
            or self.destination_flat_stop != chunk.destination_flat_stop
            or self.dtype != chunk.mapping.destination_dtype
            or self.output_size_bytes != chunk.output_bytes
        ):
            raise CheckpointConversionError(
                f"sink receipt conflicts with chunk identity {chunk.chunk_id}"
            )
        if (
            expected_output_sha256 is not None
            and self.output_sha256 != expected_output_sha256
        ):
            raise CheckpointConversionError(
                f"sink output hash mismatch for chunk {chunk.chunk_id}"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "chunk_id": self.chunk_id,
            "sink_id": self.sink_id,
            "destination_key": self.destination_key,
            "destination_flat_start": self.destination_flat_start,
            "destination_flat_stop": self.destination_flat_stop,
            "dtype": self.dtype,
            "output_sha256": self.output_sha256,
            "output_size_bytes": self.output_size_bytes,
            "immutable_object_id": self.immutable_object_id,
            "verification_contract": self.verification_contract,
        }


class ConversionOutputSink(Protocol):
    """Create-only output boundary; no implementation here is an Orbax sink."""

    @property
    def sink_id(self) -> str: ...

    def probe_committed(
        self,
        plan: StreamingConversionPlanV1,
        chunk: ConversionChunkIdentityV1,
    ) -> OutputChunkReceiptV1 | None:
        """Re-hash immutable durable bytes before returning a receipt."""

    def commit_chunk(
        self,
        plan: StreamingConversionPlanV1,
        chunk: ConversionChunkIdentityV1,
        payload: memoryview,
        *,
        output_sha256: str,
    ) -> OutputChunkReceiptV1:
        """Create and verify once; never retain ``payload`` after returning."""


def _validate_source_array(
    value: np.ndarray,
    region: TensorReadRegionV1,
) -> np.ndarray:
    if type(value) is not np.ndarray:
        raise CheckpointConversionError(
            f"source did not return an exact ndarray for {region.entry.key}"
        )
    if value.shape != region.shape:
        raise CheckpointConversionError(
            f"source region shape mismatch for {region.entry.key}: "
            f"expected {region.shape}, got {value.shape}"
        )
    expected_dtype = _SOURCE_NUMPY_DTYPES[region.entry.dtype]
    if value.dtype != expected_dtype:
        raise CheckpointConversionError(
            f"source region dtype mismatch for {region.entry.key}: "
            f"expected {expected_dtype}, got {value.dtype}"
        )
    if not value.flags.c_contiguous or not value.flags.owndata:
        raise CheckpointConversionError(
            f"source region for {region.entry.key} must be owned and C-contiguous"
        )
    if value.nbytes != region.payload_bytes:
        raise CheckpointConversionError(
            f"source region byte mismatch for {region.entry.key}"
        )
    return value


def _reader_for(
    source: VerifiedTensorSource,
    descriptor: ShardSourceDescriptorV1,
) -> AbstractContextManager[VerifiedShardReader]:
    context = source.open_verified_shard(descriptor)
    if not isinstance(context, AbstractContextManager):
        raise CheckpointConversionError(
            "source open_verified_shard did not return a context manager"
        )
    return context


def _validated_reader_receipt(
    reader: VerifiedShardReader,
    descriptor: ShardSourceDescriptorV1,
    source_adapter_identity: SourceAdapterIdentityV1,
) -> VerifiedSourceShardReceiptV1:
    receipt = reader.receipt
    if type(receipt) is not VerifiedSourceShardReceiptV1:
        raise CheckpointConversionError("source returned an invalid receipt type")
    receipt.validate_for(descriptor, source_adapter_identity)
    return receipt


def _read_cross_shard_scales(
    source: VerifiedTensorSource,
    chunks: Sequence[ConversionChunkIdentityV1],
    source_adapter_identity: SourceAdapterIdentityV1,
) -> tuple[dict[str, np.ndarray], tuple[VerifiedSourceShardReceiptV1, ...]]:
    by_scale_shard: dict[ShardSourceDescriptorV1, list[ConversionChunkIdentityV1]] = (
        defaultdict(list)
    )
    for chunk in chunks:
        if (
            chunk.scale_descriptor is not None
            and chunk.scale_descriptor != chunk.source_descriptor
        ):
            by_scale_shard[chunk.scale_descriptor].append(chunk)
    result: dict[str, np.ndarray] = {}
    receipts: list[VerifiedSourceShardReceiptV1] = []
    for descriptor in sorted(by_scale_shard, key=lambda item: item.shard):
        with _reader_for(source, descriptor) as reader:
            receipts.append(
                _validated_reader_receipt(reader, descriptor, source_adapter_identity)
            )
            for chunk in by_scale_shard[descriptor]:
                region = chunk.scale_region()
                if region is None:  # pragma: no cover - grouping invariant
                    raise CheckpointConversionError("cross-shard scale vanished")
                result[chunk.chunk_id] = _validate_source_array(
                    reader.read_region(region), region
                )
    return result, tuple(receipts)


def _transform_chunk(
    chunk: ConversionChunkIdentityV1,
    source_value: np.ndarray,
    scale_value: np.ndarray | None,
) -> np.ndarray:
    transform = chunk.mapping.transform
    if transform == "identity-v1":
        if scale_value is not None:
            raise CheckpointConversionError("identity transform received a scale")
        output = source_value.reshape(-1)
    elif transform == "transpose-reshape-v1":
        if scale_value is not None:
            raise CheckpointConversionError("transpose transform received a scale")
        output = np.ascontiguousarray(source_value.T).reshape(-1)
    elif transform == ("blockwise-fp8-f32-dequantize-then-transpose-reshape-bf16-v1"):
        if scale_value is None:
            raise CheckpointConversionError("FP8 transform is missing its scale")
        if not np.isfinite(scale_value).all():
            # Positivity is plausible but remains unverified until the real
            # 21,749-scale payload scan.  The released transform semantics are
            # multiplication, so this core must not invent a stricter gate.
            raise CheckpointConversionError("FP8 inverse scales must be finite")
        work = source_value.astype(np.float32)
        if not np.isfinite(work).all():
            raise CheckpointConversionError("FP8 source contains non-finite values")
        block_size = chunk.fp8_block_size
        source_global_column = chunk.source_start
        for row_start in range(0, work.shape[0], block_size):
            row_stop = min(row_start + block_size, work.shape[0])
            scale_row = row_start // block_size
            for local_start in range(0, work.shape[1], block_size):
                local_stop = min(local_start + block_size, work.shape[1])
                global_column = source_global_column + local_start
                scale_column = global_column // block_size - (
                    chunk.source_start // block_size
                )
                scale = np.float32(scale_value[scale_row, scale_column])
                np.multiply(
                    work[row_start:row_stop, local_start:local_stop],
                    scale,
                    out=work[row_start:row_stop, local_start:local_stop],
                )
        if not np.isfinite(work).all():
            raise CheckpointConversionError(
                "FP8 dequantization produced non-finite values"
            )
        output = np.ascontiguousarray(work.T, dtype=_BF16).reshape(-1)
    else:  # pragma: no cover - inventory mapping rejects this first
        raise CheckpointConversionError(f"unknown transform: {transform}")

    expected_dtype = _BF16 if chunk.mapping.destination_dtype == "bfloat16" else _F32
    if output.dtype != expected_dtype:
        raise CheckpointConversionError(
            f"transform dtype mismatch for {chunk.source_entry.key}"
        )
    if output.size != chunk.output_elements:
        raise CheckpointConversionError(
            f"transform element mismatch for {chunk.source_entry.key}"
        )
    if not output.flags.c_contiguous:
        raise CheckpointConversionError("transform output is not C-contiguous")
    if output.nbytes != chunk.output_bytes:
        raise CheckpointConversionError("transform output byte count is inconsistent")
    return output


def _canonical_payload_view(output: np.ndarray) -> memoryview:
    if sys.byteorder != "little":  # pragma: no cover - unsupported production host
        raise CheckpointConversionError(
            "conversion payload hashing requires a little-endian host"
        )
    if not output.flags.c_contiguous:
        raise CheckpointConversionError("canonical payload must be C-contiguous")
    try:
        # Python's buffer protocol does not recognize ml_dtypes' BF16 format.
        # An unsigned integer view is zero-copy and exposes the exact stored
        # little-endian bits without a temporary ``bytes`` allocation.
        if output.dtype == _BF16:
            storage = output.view(np.uint16)
        elif output.dtype == _F32:
            storage = output.view(np.uint32)
        else:  # pragma: no cover - transform validation runs immediately before
            raise CheckpointConversionError("canonical output dtype is unsupported")
        return memoryview(storage).cast("B")
    except (TypeError, ValueError) as error:
        raise CheckpointConversionError(
            "cannot expose canonical output bytes"
        ) from error


@dataclass(frozen=True, slots=True)
class ConversionAttemptSummaryV1:
    SCHEMA: ClassVar[str] = "deepseek-r1-conversion-attempt-summary-v1"

    plan_id: str
    sink_id: str
    total_chunks: int
    reused_chunks: int
    written_chunks: int
    converted_payload_bytes: int
    max_estimated_resident_bytes: int
    source_receipts: tuple[VerifiedSourceShardReceiptV1, ...]
    receipts: tuple[OutputChunkReceiptV1, ...]

    def __post_init__(self) -> None:
        _require_sha256(self.plan_id, "summary plan_id")
        _require_identifier(self.sink_id, "summary sink_id")
        for name in (
            "total_chunks",
            "reused_chunks",
            "written_chunks",
            "converted_payload_bytes",
            "max_estimated_resident_bytes",
        ):
            _require_int(getattr(self, name), f"summary {name}", minimum=0)
        if self.total_chunks != self.reused_chunks + self.written_chunks:
            raise CheckpointConversionError("summary chunk accounting is inconsistent")
        if type(self.receipts) is not tuple or len(self.receipts) != self.total_chunks:
            raise CheckpointConversionError("summary receipts are incomplete")
        if any(type(item) is not OutputChunkReceiptV1 for item in self.receipts):
            raise CheckpointConversionError("summary receipt type is invalid")
        if type(self.source_receipts) is not tuple or any(
            type(item) is not VerifiedSourceShardReceiptV1
            for item in self.source_receipts
        ):
            raise CheckpointConversionError("summary source receipt type is invalid")
        source_shards = tuple(item.shard for item in self.source_receipts)
        if source_shards != tuple(sorted(source_shards)) or len(source_shards) != len(
            set(source_shards)
        ):
            raise CheckpointConversionError(
                "summary source receipts must have unique sorted shards"
            )

    @property
    def conversion_sealable(self) -> bool:
        return False

    @property
    def unmet_production_gates(self) -> tuple[str, ...]:
        return (
            "isolated-r1-maxtext-runtime-and-image-attestation",
            "real-full-blob-validating-safetensor-source-adapter",
            "real-tensorstore-or-orbax-sink-and-fresh-process-restore",
            "independent-r1-fp8-and-bf16-conversion-oracle",
            "full-model-r1-logit-parity",
            "prefix-versus-full-routed-aggregate-parity",
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "plan_id": self.plan_id,
            "sink_id": self.sink_id,
            "total_chunks": self.total_chunks,
            "reused_chunks": self.reused_chunks,
            "written_chunks": self.written_chunks,
            "converted_payload_bytes": self.converted_payload_bytes,
            "max_estimated_resident_bytes": self.max_estimated_resident_bytes,
            "conversion_sealable": self.conversion_sealable,
            "unmet_production_gates": list(self.unmet_production_gates),
            "source_receipts": [receipt.as_dict() for receipt in self.source_receipts],
            "receipts": [receipt.as_dict() for receipt in self.receipts],
        }


def execute_streaming_conversion(
    plan: StreamingConversionPlanV1,
    source: VerifiedTensorSource,
    sink: ConversionOutputSink,
) -> ConversionAttemptSummaryV1:
    """Execute shard-ordered chunks and resume only exact verified commits."""

    if type(plan) is not StreamingConversionPlanV1:
        raise CheckpointConversionError("executor requires StreamingConversionPlanV1")
    if plan.engine_sha256 != conversion_engine_sha256():
        raise CheckpointConversionError(
            "conversion implementation changed after the plan was derived"
        )
    if plan.numeric_codec_sha256 != numeric_codec_identity_sha256():
        raise CheckpointConversionError(
            "numerical codec runtime changed after the plan was derived"
        )
    source_adapter_identity = source.adapter_identity
    if (
        type(source_adapter_identity) is not SourceAdapterIdentityV1
        or source_adapter_identity != plan.source_adapter_identity
    ):
        raise CheckpointConversionError(
            "verified source adapter does not match the conversion plan"
        )
    sink_id = sink.sink_id
    _require_identifier(sink_id, "sink_id")
    receipts: dict[str, OutputChunkReceiptV1] = {}
    source_receipts: dict[str, VerifiedSourceShardReceiptV1] = {}
    reused = 0
    written = 0

    for shard_work in plan.shard_work:
        pending: list[ConversionChunkIdentityV1] = []
        for chunk in shard_work.chunks:
            receipt = sink.probe_committed(plan, chunk)
            if receipt is None:
                pending.append(chunk)
                continue
            if type(receipt) is not OutputChunkReceiptV1:
                raise CheckpointConversionError("sink returned an invalid receipt type")
            receipt.validate_for(plan, chunk, sink_id=sink_id)
            receipts[chunk.chunk_id] = receipt
            reused += 1
        if not pending:
            continue

        cross_scale_bytes = sum(
            region.payload_bytes
            for chunk in pending
            if chunk.scale_descriptor is not None
            and chunk.scale_descriptor != chunk.source_descriptor
            for region in (chunk.scale_region(),)
            if region is not None
        )
        if cross_scale_bytes and any(
            chunk.estimated_peak_resident_bytes + cross_scale_bytes
            > plan.limits.max_resident_bytes
            for chunk in pending
        ):
            raise CheckpointConversionError(
                "cross-shard scale cache exceeds the declared resident bound"
            )
        cross_scales, cross_receipts = _read_cross_shard_scales(
            source, pending, source_adapter_identity
        )
        for source_receipt in cross_receipts:
            existing = source_receipts.get(source_receipt.shard)
            if existing is not None and existing != source_receipt:
                raise CheckpointConversionError(
                    f"conflicting source verifier receipts for {source_receipt.shard}"
                )
            source_receipts[source_receipt.shard] = source_receipt
        if sum(value.nbytes for value in cross_scales.values()) != cross_scale_bytes:
            raise CheckpointConversionError(
                "cross-shard scale cache byte accounting changed after source I/O"
            )

        with _reader_for(source, shard_work.source_descriptor) as reader:
            source_receipt = _validated_reader_receipt(
                reader, shard_work.source_descriptor, source_adapter_identity
            )
            existing = source_receipts.get(source_receipt.shard)
            if existing is not None and existing != source_receipt:
                raise CheckpointConversionError(
                    f"conflicting source verifier receipts for {source_receipt.shard}"
                )
            source_receipts[source_receipt.shard] = source_receipt
            for chunk in pending:
                scale_region = chunk.scale_region()
                if scale_region is None:
                    scale_value = None
                elif chunk.scale_descriptor == chunk.source_descriptor:
                    scale_value = _validate_source_array(
                        reader.read_region(scale_region), scale_region
                    )
                else:
                    try:
                        scale_value = cross_scales[chunk.chunk_id]
                    except KeyError as error:  # pragma: no cover - preload invariant
                        raise CheckpointConversionError(
                            "cross-shard scale cache is incomplete"
                        ) from error
                source_region = chunk.source_region()
                source_value = _validate_source_array(
                    reader.read_region(source_region), source_region
                )
                output = _transform_chunk(chunk, source_value, scale_value)
                payload = _canonical_payload_view(output)
                output_sha256 = hashlib.sha256(payload).hexdigest()
                receipt = sink.commit_chunk(
                    plan,
                    chunk,
                    payload,
                    output_sha256=output_sha256,
                )
                if type(receipt) is not OutputChunkReceiptV1:
                    raise CheckpointConversionError(
                        "sink commit returned an invalid receipt type"
                    )
                receipt.validate_for(
                    plan,
                    chunk,
                    sink_id=sink_id,
                    expected_output_sha256=output_sha256,
                )
                receipts[chunk.chunk_id] = receipt
                written += 1
                # The sink contract is synchronous and forbids retaining this
                # view.  Release every numerical reference before the next
                # source read so two chunks cannot overlap the admitted array
                # working-set bound.
                payload.release()
                del payload, output, source_value, scale_value

    ordered_receipts = tuple(receipts[chunk.chunk_id] for chunk in plan.chunks)
    if sum(receipt.output_size_bytes for receipt in ordered_receipts) != (
        plan.converted_payload_bytes
    ):
        raise CheckpointConversionError("completed output byte total is inconsistent")
    return ConversionAttemptSummaryV1(
        plan_id=plan.plan_id,
        sink_id=sink_id,
        total_chunks=len(plan.chunks),
        reused_chunks=reused,
        written_chunks=written,
        converted_payload_bytes=plan.converted_payload_bytes,
        max_estimated_resident_bytes=max(
            shard.estimated_peak_resident_bytes for shard in plan.shard_work
        ),
        source_receipts=tuple(
            source_receipts[shard] for shard in sorted(source_receipts)
        ),
        receipts=ordered_receipts,
    )


def seal_production_conversion(_: ConversionAttemptSummaryV1) -> None:
    """Unconditionally refuse a production seal in this implementation tranche."""

    raise ProductionSealUnavailable(
        "production sealing is disabled until an isolated R1 runtime, real "
        "full-blob-validating source, Orbax/TensorStore sink, fresh-process "
        "restore, independent R1 conversion oracle, full-model logit parity, "
        "and prefix/full routed-aggregate parity all exist"
    )
