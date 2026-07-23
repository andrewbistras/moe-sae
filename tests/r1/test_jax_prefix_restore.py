"""Tests for slice-addressed restoration of the 44-leaf prefix tree."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ml_dtypes
import numpy as np
import pytest

from moe_sae.r1.jax_prefix_restore import (
    JAXPrefixRestoreError,
    LeafSharding,
    LeafSliceCallback,
    restore_prefix_parameters,
)
from moe_sae.r1.prefix_restore_authority import (
    DestinationRestoreSpecV1,
    PerformanceLeafRestoreSpecV1,
)
from tests.r1.test_prefix_restore_authority import (
    _PinnedControlFixture,
    pinned_control,
)


@dataclass
class _FakeLeaf:
    spec: PerformanceLeafRestoreSpecV1
    arrays: tuple[np.ndarray, ...]
    maximum_read_bytes: int

    def __post_init__(self) -> None:
        self.reads: list[tuple[int, int, int]] = []

    def read_source_flat(
        self, source_index: int, start: int, stop: int
    ) -> np.ndarray:
        self.reads.append((source_index, start, stop))
        return self.arrays[source_index].reshape(-1)[start:stop].copy()


@dataclass(frozen=True)
class _FakeArray:
    shape: tuple[int, ...]
    dtype: np.dtype
    sharding: object


def _source(
    key: str,
    dtype: str,
    shape: tuple[int, ...],
    *,
    leaf_key: str,
    section: str,
    scan_index: int | None,
) -> DestinationRestoreSpecV1:
    return DestinationRestoreSpecV1(
        destination_key=key,
        dtype=dtype,
        shape=shape,
        performance_section=section,
        performance_leaf_key=leaf_key,
        scan_index=scan_index,
    )


def test_import_does_not_initialize_jax() -> None:
    source_root = Path(__file__).parents[2] / "src"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                f"import sys; sys.path.insert(0, {str(source_root)!r}); "
                "import moe_sae.r1.jax_prefix_restore; "
                "assert 'jax' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_unscanned_callback_reads_only_requested_flat_regions() -> None:
    key = "test-unscanned"
    source = _source(
        "source-zero",
        "float32",
        (4, 5),
        leaf_key=key,
        section="embedding",
        scan_index=None,
    )
    backing = np.arange(20, dtype=np.float32).reshape(4, 5)
    leaf = _FakeLeaf(
        PerformanceLeafRestoreSpecV1(
            leaf_key=key,
            section="embedding",
            family="test",
            dtype="float32",
            logical_shape=(4, 5),
            scan_axis=None,
            sources=(source,),
        ),
        (backing,),
        maximum_read_bytes=8,
    )
    actual = LeafSliceCallback(leaf)((slice(1, 3), slice(2, 5)))
    np.testing.assert_array_equal(actual, backing[1:3, 2:5])
    assert actual.dtype == np.float32
    assert leaf.reads == [(0, 7, 9), (0, 9, 10), (0, 12, 14), (0, 14, 15)]


def test_scanned_callback_places_axis_one_and_preserves_bfloat16() -> None:
    key = "test-scanned"
    arrays = tuple(
        (np.arange(8).reshape(2, 4) + 10 * index).astype(ml_dtypes.bfloat16)
        for index in range(3)
    )
    sources = tuple(
        _source(
            f"source-{index}",
            "bfloat16",
            (2, 4),
            leaf_key=key,
            section="scanned-dense",
            scan_index=index,
        )
        for index in range(3)
    )
    leaf = _FakeLeaf(
        PerformanceLeafRestoreSpecV1(
            leaf_key=key,
            section="scanned-dense",
            family="test",
            dtype="bfloat16",
            logical_shape=(2, 3, 4),
            scan_axis=1,
            sources=sources,
        ),
        arrays,
        maximum_read_bytes=4,
    )
    actual = LeafSliceCallback(leaf)(
        (slice(1, 2), slice(1, 3), slice(1, 4))
    )
    expected = np.stack(
        (arrays[1][1:2, 1:4], arrays[2][1:2, 1:4]), axis=1
    )
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.dtype(ml_dtypes.bfloat16)
    assert leaf.reads == [
        (1, 5, 7),
        (1, 7, 8),
        (2, 5, 7),
        (2, 7, 8),
    ]


def test_callback_rejects_strides_before_payload_reads() -> None:
    key = "test-invalid"
    source = _source(
        "source-zero",
        "float32",
        (2,),
        leaf_key=key,
        section="embedding",
        scan_index=None,
    )
    leaf = _FakeLeaf(
        PerformanceLeafRestoreSpecV1(
            leaf_key=key,
            section="embedding",
            family="test",
            dtype="float32",
            logical_shape=(2,),
            scan_axis=None,
            sources=(source,),
        ),
        (np.arange(2, dtype=np.float32),),
        maximum_read_bytes=8,
    )
    with pytest.raises(JAXPrefixRestoreError, match="contiguous"):
        LeafSliceCallback(leaf)((slice(None, None, 2),))
    assert leaf.reads == []


def test_exact_44_leaf_bundle_and_router_dtypes(
    pinned_control: _PinnedControlFixture,
) -> None:
    authority = pinned_control.authority
    marker = object()
    bindings = tuple(
        LeafSharding(leaf.spec.leaf_key, marker) for leaf in authority.leaves
    )
    callbacks: dict[str, LeafSliceCallback] = {}

    def factory(
        shape: tuple[int, ...], sharding: object, callback: LeafSliceCallback
    ) -> _FakeArray:
        assert sharding is marker
        callbacks[callback.leaf_key] = callback
        return _FakeArray(shape, callback.dtype, sharding)

    restored = restore_prefix_parameters(
        authority, bindings, array_factory=factory
    )
    assert len(restored.leaves_by_key) == 44
    assert len(restored.model_by_path) == 30
    assert restored.embedding is restored.leaves_by_key[
        "params-token_embedder-embedding"
    ]
    assert len(callbacks) == 44
    assert callbacks[
        "params-decoder-moe_layers-DeepSeekMoeBlock_0-MoeBlock_0-gate-bias"
    ].dtype == np.float32
    assert callbacks[
        "params-decoder-moe_layers-DeepSeekMoeBlock_0-MoeBlock_0-gate-kernel"
    ].dtype == np.dtype(ml_dtypes.bfloat16)
    assert restored.completed_moe_params["DeepSeekMoeBlock_0"]["MoeBlock_0"][
        "gate"
    ]["bias"].dtype == np.float32
    assert restored.target_params["DeepSeekMoeBlock_0"]["MoeBlock_0"]["gate"][
        "bias"
    ].dtype == np.float32
    assert restored.authority_digest == authority.digest


def test_missing_unused_and_duplicate_bindings_fail_before_factory(
    pinned_control: _PinnedControlFixture,
) -> None:
    authority = pinned_control.authority
    bindings = tuple(
        LeafSharding(leaf.spec.leaf_key, object()) for leaf in authority.leaves
    )
    calls = 0

    def factory(*_args: Any) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError("factory must not run")

    with pytest.raises(JAXPrefixRestoreError, match="missing"):
        restore_prefix_parameters(
            authority, bindings[:-1], array_factory=factory
        )
    with pytest.raises(JAXPrefixRestoreError, match="unused"):
        restore_prefix_parameters(
            authority,
            (*bindings, LeafSharding("unused-leaf", object())),
            array_factory=factory,
        )
    with pytest.raises(JAXPrefixRestoreError, match="duplicate"):
        restore_prefix_parameters(
            authority, (*bindings[:-1], bindings[0]), array_factory=factory
        )
    assert calls == 0


def test_default_factory_accepts_explicit_named_sharding() -> None:
    import jax
    from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

    devices = np.asarray(jax.devices()[:1])
    mesh = Mesh(devices, ("data",), axis_types=(AxisType.Explicit,))
    sharding = NamedSharding(mesh, P("data", None))
    key = "test-jax"
    source = _source(
        "source-zero",
        "float32",
        (2, 3),
        leaf_key=key,
        section="embedding",
        scan_index=None,
    )
    backing = np.arange(6, dtype=np.float32).reshape(2, 3)
    leaf = _FakeLeaf(
        PerformanceLeafRestoreSpecV1(
            leaf_key=key,
            section="embedding",
            family="test",
            dtype="float32",
            logical_shape=(2, 3),
            scan_axis=None,
            sources=(source,),
        ),
        (backing,),
        maximum_read_bytes=24,
    )
    module = __import__(
        "moe_sae.r1.jax_prefix_restore", fromlist=["_jax_factory"]
    )
    array = module._jax_factory((2, 3), sharding, LeafSliceCallback(leaf))
    np.testing.assert_array_equal(np.asarray(array), backing)
    assert array.sharding == sharding
