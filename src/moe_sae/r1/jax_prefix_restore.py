"""Runtime-only JAX restore for the canonical 44-leaf DeepSeek-R1 prefix.

This module intentionally imports no JAX at module import time.  The default
factory loads JAX only when a TPU worker constructs arrays.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import importlib
import itertools
import math
from types import MappingProxyType
from typing import Any, Protocol

import ml_dtypes
import numpy as np

from .prefix_batchsplit_contract import PINNED_PERFORMANCE_LEAF_COUNT
from .prefix_restore_authority import (
    LazyPerformanceLeafV1,
    PerformanceLeafRestoreSpecV1,
    PrefixDirectRestoreAuthorityV1,
    PrefixRestoreAuthorityError,
)


_DTYPES = {
    "bfloat16": np.dtype(ml_dtypes.bfloat16),
    "float32": np.dtype(np.float32),
}
_SECTIONS = {
    "scanned-dense": "dense_params",
    "scanned-completed-moe": "completed_moe_params",
    "target-routed-only": "target_params",
}


class JAXPrefixRestoreError(PrefixRestoreAuthorityError):
    """The exact JAX restore contract failed."""


class ArrayFromCallback(Protocol):
    def __call__(
        self,
        shape: tuple[int, ...],
        sharding: object,
        callback: Callable[[tuple[slice, ...]], np.ndarray],
    ) -> object: ...


class LeafReader(Protocol):
    spec: PerformanceLeafRestoreSpecV1
    maximum_read_bytes: int

    def read_source_flat(
        self, source_index: int, flat_start: int, flat_stop: int
    ) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class LeafSharding:
    leaf_key: str
    sharding: object

    def __post_init__(self) -> None:
        if type(self.leaf_key) is not str or not self.leaf_key:
            raise JAXPrefixRestoreError("leaf_key must be an exact nonempty string")
        if self.sharding is None:
            raise JAXPrefixRestoreError("leaf sharding cannot be None")


@dataclass(frozen=True, slots=True)
class RestoredPrefixParameters:
    """Concrete 30-model/14-target parameter bundle for the MaxText adapter."""

    embedding: object
    dense_params: dict[str, Any]
    completed_moe_params: dict[str, Any]
    target_params: dict[str, Any]
    model_by_path: Mapping[str, object]
    leaves_by_key: Mapping[str, object]
    authority_digest: str

    def __post_init__(self) -> None:
        if (
            len(self.leaves_by_key) != PINNED_PERFORMANCE_LEAF_COUNT
            or len(self.model_by_path) != 30
            or _leaf_count(self.dense_params) != 12
            or _leaf_count(self.completed_moe_params) != 17
            or _leaf_count(self.target_params) != 14
            or type(self.authority_digest) is not str
            or len(self.authority_digest) != 64
        ):
            raise JAXPrefixRestoreError(
                "restored bundle is not the exact 1+12+17 model / 14 target tree"
            )


def _leaf_count(tree: Mapping[str, Any]) -> int:
    return sum(
        _leaf_count(value) if isinstance(value, Mapping) else 1
        for value in tree.values()
    )


def _insert(root: dict[str, Any], path: tuple[str, ...], value: object) -> None:
    current = root
    for part in path[:-1]:
        child = current.setdefault(part, {})
        if type(child) is not dict:
            raise JAXPrefixRestoreError("restored parameter paths collide")
        current = child
    if not path or path[-1] in current:
        raise JAXPrefixRestoreError("restored parameter path is duplicated")
    current[path[-1]] = value


def _bounds(
    index: Any, shape: tuple[int, ...]
) -> tuple[tuple[int, int], ...]:
    if type(index) is not tuple or len(index) != len(shape):
        raise JAXPrefixRestoreError("callback index rank differs from leaf rank")
    result = []
    for item, dimension in zip(index, shape, strict=True):
        if type(item) is not slice or item.step not in (None, 1):
            raise JAXPrefixRestoreError("callback requires contiguous exact slices")
        start = 0 if item.start is None else item.start
        stop = dimension if item.stop is None else item.stop
        if (
            type(start) is not int
            or type(stop) is not int
            or not 0 <= start < stop <= dimension
        ):
            raise JAXPrefixRestoreError("callback slice is outside the logical leaf")
        result.append((start, stop))
    return tuple(result)


def _read_interval(
    leaf: LeafReader,
    source_index: int,
    start: int,
    stop: int,
    dtype: np.dtype,
) -> np.ndarray:
    result = np.empty((stop - start,), dtype=dtype)
    per_read = leaf.maximum_read_bytes // dtype.itemsize
    if per_read < 1:
        raise JAXPrefixRestoreError("leaf read bound cannot hold one element")
    offset = 0
    while start < stop:
        end = min(stop, start + per_read)
        values = leaf.read_source_flat(source_index, start, end)
        if (
            type(values) is not np.ndarray
            or values.shape != (end - start,)
            or np.dtype(values.dtype) != dtype
        ):
            raise JAXPrefixRestoreError("leaf reader returned a wrong shape or dtype")
        result[offset : offset + values.size] = values
        offset += values.size
        start = end
    return result


def _read_source_slice(
    leaf: LeafReader,
    source_index: int,
    source_shape: tuple[int, ...],
    bounds: tuple[tuple[int, int], ...],
    dtype: np.dtype,
) -> np.ndarray:
    """Read a rectangle as maximal covered C-order flat intervals."""

    if len(source_shape) != len(bounds):
        raise JAXPrefixRestoreError("source slice rank mismatch")
    output = np.empty(tuple(stop - start for start, stop in bounds), dtype=dtype)
    pivot = len(source_shape) - 1
    for candidate in range(len(source_shape)):
        if all(
            bounds[axis] == (0, source_shape[axis])
            for axis in range(candidate + 1, len(source_shape))
        ):
            pivot = candidate
            break
    prefix_ranges = [
        range(bounds[axis][0], bounds[axis][1]) for axis in range(pivot)
    ]
    prefixes = itertools.product(*prefix_ranges) if prefix_ranges else ((),)
    suffix_size = math.prod(source_shape[pivot + 1 :])
    pivot_start, pivot_stop = bounds[pivot]
    block_size = (pivot_stop - pivot_start) * suffix_size
    block_shape = (pivot_stop - pivot_start, *source_shape[pivot + 1 :])
    for prefix in prefixes:
        flat_start = 0
        for coordinate, dimension in zip(
            (*prefix, pivot_start), source_shape[: pivot + 1], strict=True
        ):
            flat_start = flat_start * dimension + coordinate
        flat_start *= suffix_size
        block = _read_interval(
            leaf, source_index, flat_start, flat_start + block_size, dtype
        ).reshape(block_shape)
        relative = tuple(
            coordinate - bounds[axis][0]
            for axis, coordinate in enumerate(prefix)
        )
        output[
            (*relative, *([slice(None)] * (len(source_shape) - pivot)))
        ] = block
    return output


class LeafSliceCallback:
    """Global-slice callback for one lazy authority leaf."""

    __slots__ = ("_leaf", "dtype", "leaf_key", "shape")

    def __init__(self, leaf: LeafReader) -> None:
        spec = getattr(leaf, "spec", None)
        if (
            type(spec) is not PerformanceLeafRestoreSpecV1
            or type(getattr(leaf, "maximum_read_bytes", None)) is not int
            or leaf.maximum_read_bytes <= 0
            or not callable(getattr(leaf, "read_source_flat", None))
        ):
            raise JAXPrefixRestoreError("invalid lazy performance leaf")
        self._leaf = leaf
        self.dtype = _DTYPES[spec.dtype]
        self.leaf_key = spec.leaf_key
        self.shape = spec.logical_shape

    def __call__(self, index: tuple[slice, ...]) -> np.ndarray:
        spec = self._leaf.spec
        bounds = _bounds(index, spec.logical_shape)
        if spec.scan_axis is None:
            return _read_source_slice(
                self._leaf, 0, spec.sources[0].shape, bounds, self.dtype
            )
        scan_start, scan_stop = bounds[spec.scan_axis]
        source_bounds = (*bounds[: spec.scan_axis], *bounds[spec.scan_axis + 1 :])
        result = np.empty(
            tuple(stop - start for start, stop in bounds), dtype=self.dtype
        )
        output_index = [slice(None)] * len(bounds)
        for local_index, source_index in enumerate(range(scan_start, scan_stop)):
            source = spec.sources[source_index]
            if source.scan_index != source_index:
                raise JAXPrefixRestoreError("noncanonical scan source order")
            output_index[spec.scan_axis] = local_index
            result[tuple(output_index)] = _read_source_slice(
                self._leaf,
                source_index,
                source.shape,
                source_bounds,
                self.dtype,
            )
        return result


def _jax_factory(
    shape: tuple[int, ...],
    sharding: object,
    callback: Callable[[tuple[slice, ...]], np.ndarray],
) -> object:
    try:
        jax = importlib.import_module("jax")
    except ImportError as error:  # pragma: no cover - TPU runtime dependency
        raise JAXPrefixRestoreError("JAX is unavailable") from error
    return jax.make_array_from_callback(shape, sharding, callback)


def _check_array(
    value: object, leaf: LazyPerformanceLeafV1, sharding: object
) -> None:
    if (
        tuple(getattr(value, "shape", ())) != leaf.spec.logical_shape
        or np.dtype(getattr(value, "dtype", None)) != _DTYPES[leaf.spec.dtype]
    ):
        raise JAXPrefixRestoreError(
            f"array factory returned a wrong ABI for {leaf.spec.leaf_key}"
        )
    actual = getattr(value, "sharding", None)
    if actual is not None and actual != sharding:
        raise JAXPrefixRestoreError("array factory changed the requested sharding")


def restore_prefix_parameters(
    authority: PrefixDirectRestoreAuthorityV1,
    shardings: tuple[LeafSharding, ...],
    *,
    array_factory: ArrayFromCallback | None = None,
) -> RestoredPrefixParameters:
    """Construct the exact 44 callback-backed arrays and adapter-ready trees."""

    if type(authority) is not PrefixDirectRestoreAuthorityV1:
        raise JAXPrefixRestoreError("restore requires an exact opened authority")
    if type(shardings) is not tuple or any(
        type(item) is not LeafSharding for item in shardings
    ):
        raise JAXPrefixRestoreError("shardings must be an exact immutable tuple")
    expected = tuple(leaf.spec.leaf_key for leaf in authority.leaves)
    supplied = tuple(item.leaf_key for item in shardings)
    duplicates = sorted({key for key in supplied if supplied.count(key) > 1})
    if duplicates:
        raise JAXPrefixRestoreError(f"duplicate restore leaves: {duplicates!r}")
    missing = sorted(set(expected).difference(supplied))
    unused = sorted(set(supplied).difference(expected))
    if missing:
        raise JAXPrefixRestoreError(f"missing restore leaves: {missing!r}")
    if unused:
        raise JAXPrefixRestoreError(f"unused restore leaves: {unused!r}")
    if (
        len(expected) != PINNED_PERFORMANCE_LEAF_COUNT
        or len(shardings) != PINNED_PERFORMANCE_LEAF_COUNT
    ):
        raise JAXPrefixRestoreError("restore must consume exactly 44 leaves")

    factory = _jax_factory if array_factory is None else array_factory
    if not callable(factory):
        raise JAXPrefixRestoreError("array_factory must be callable")
    by_sharding = {item.leaf_key: item.sharding for item in shardings}
    leaves: dict[str, object] = {}
    embedding: object | None = None
    dense: dict[str, Any] = {}
    completed: dict[str, Any] = {}
    target: dict[str, Any] = {}
    model_by_path: dict[str, object] = {}
    trees = {
        "scanned-dense": dense,
        "scanned-completed-moe": completed,
        "target-routed-only": target,
    }
    for leaf in authority.leaves:
        sharding = by_sharding[leaf.spec.leaf_key]
        callback = LeafSliceCallback(leaf)
        try:
            value = factory(leaf.spec.logical_shape, sharding, callback)
        except Exception as error:
            if isinstance(error, JAXPrefixRestoreError):
                raise
            raise JAXPrefixRestoreError(
                f"array construction failed for {leaf.spec.leaf_key}"
            ) from error
        _check_array(value, leaf, sharding)
        leaves[leaf.spec.leaf_key] = value
        if leaf.spec.section == "embedding":
            if embedding is not None:
                raise JAXPrefixRestoreError("embedding leaf is duplicated")
            embedding = value
        else:
            _insert(trees[leaf.spec.section], tuple(leaf.spec.family.split("-")), value)
        if leaf.spec.section != "target-routed-only":
            path = leaf.spec.leaf_key.removeprefix("params-").replace("-", "/")
            if path in model_by_path:
                raise JAXPrefixRestoreError("model binding path is duplicated")
            model_by_path[path] = value
    if tuple(leaves) != expected or embedding is None:
        raise JAXPrefixRestoreError("restore did not consume each leaf exactly once")
    return RestoredPrefixParameters(
        embedding=embedding,
        dense_params=dense,
        completed_moe_params=completed,
        target_params=target,
        model_by_path=MappingProxyType(model_by_path),
        leaves_by_key=MappingProxyType(leaves),
        authority_digest=authority.digest,
    )


__all__ = [
    "ArrayFromCallback",
    "JAXPrefixRestoreError",
    "LeafSharding",
    "LeafSliceCallback",
    "RestoredPrefixParameters",
    "restore_prefix_parameters",
]
