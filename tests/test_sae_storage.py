from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from moe_sae.object_store import LocalObjectStore
from moe_sae.sae_storage import (
    SAE_CACHE_MATERIALIZATION_WORKERS,
    SAEStorageQualificationError,
    _assigned_objects,
    _token_partition,
    qualify_sae_streaming_cache,
    validate_local_sae_cache_residency,
)


def _manifest(store: LocalObjectStore, root: Path, sizes: tuple[int, ...]):
    shards = []
    for index, size in enumerate(sizes):
        source = root / f"source-{index}-{size}.bin"
        payload = bytes([index + 1]) * size
        source.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        identity = store.put_if_absent(
            source,
            f"gs://bucket/capture/{index}-{digest}.safetensors",
            sha256=digest,
        )
        shards.append({"tokens": size, "object": identity.as_dict()})
    return {"total_tokens": sum(sizes), "shards": shards}


def test_partition_matches_distributed_floor_boundaries() -> None:
    assert [_token_partition(10, 3, index) for index in range(3)] == [
        (0, 3),
        (3, 6),
        (6, 10),
    ]


def test_assignment_uses_every_object_overlapping_exact_partition(tmp_path: Path) -> None:
    store = LocalObjectStore(tmp_path / "objects")
    manifest = _manifest(store, tmp_path, (2, 2, 2, 2, 2))
    assignments = [
        _assigned_objects((manifest,), index, 3) for index in range(3)
    ]
    assert [[item.size_bytes for item in values] for values in assignments] == [
        [2, 2],
        [2, 2],
        [2, 2],
    ]
    # A boundary-cut shard can be cached by both adjacent consumers, but each
    # host receives only its exact contiguous token-overlap inventory.
    assert assignments[0][-1] == assignments[1][0]


def test_local_qualification_cold_fills_all_assigned_objects(tmp_path: Path) -> None:
    store = LocalObjectStore(tmp_path / "objects")
    # Expose the production transfer tuning in the qualification evidence.
    store.parallel_download_workers = 16
    store.parallel_download_chunk_bytes = 16 * 1024**2
    manifest = _manifest(store, tmp_path, (64, 64, 64, 64))
    result = qualify_sae_streaming_cache(
        tmp_path / "cache",
        (manifest,),
        store,
        cache_max_bytes=256,
        minimum_free_after_bytes=1,
        minimum_aggregate_download_bytes_per_second=0.0,
    )
    assert result["passed"] is True
    assert result["cache"]["materialization_workers"] == SAE_CACHE_MATERIALIZATION_WORKERS
    assert result["cache"]["all_assigned_resident"] is True
    assert result["cache"]["per_process"][0]["assigned_objects"] == 4
    assert result["cache"]["per_process"][0]["retained_bytes"] == 256
    assert result["download"]["inner_object_workers"] == 16
    assert result["download"]["chunk_bytes"] == 16 * 1024**2


def test_runtime_residency_requires_exact_qualified_local_inventory(
    tmp_path: Path,
) -> None:
    store = LocalObjectStore(tmp_path / "objects")
    manifest = _manifest(store, tmp_path, (64, 64, 64, 64))
    cache_root = (tmp_path / "cache").resolve()
    qualification = qualify_sae_streaming_cache(
        cache_root,
        (manifest,),
        store,
        cache_max_bytes=256,
        minimum_free_after_bytes=1,
        minimum_aggregate_download_bytes_per_second=0.0,
    )
    payload = {
        "qualification": qualification,
        "layout": {"cache_directories": [str(cache_root)]},
    }

    residency = validate_local_sae_cache_residency(
        payload, (manifest,), process_index=0
    )
    assert residency["objects"] == 4
    assert residency["bytes"] == 256

    next(cache_root.glob("*.safetensors")).unlink()
    with pytest.raises(SAEStorageQualificationError, match="requalify a fresh cache"):
        validate_local_sae_cache_residency(
            payload, (manifest,), process_index=0
        )
