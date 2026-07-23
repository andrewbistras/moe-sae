from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import threading

import google_crc32c
import pytest

from moe_sae.r1.gcs_immutable_byte_store import GCSPhysicalObjectReceiptV1
from moe_sae.r1.gcs_read_cache import ExactReceiptReadCache, GCSReadCacheError


class Source:
    store_id = "gcs-prefix-byte-store-v2:" + "a" * 64
    region = "us-east1"
    distributed_identity_qualified = True

    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.calls: list[tuple[int, int, int]] = []

    def read_exact(
        self,
        _receipt: GCSPhysicalObjectReceiptV1,
        *,
        offset: int,
        length: int,
        maximum_bytes: int,
    ) -> bytes:
        self.calls.append((offset, length, maximum_bytes))
        return self.payload[offset : offset + length]


def _receipt(source: Source) -> GCSPhysicalObjectReceiptV1:
    checksum = google_crc32c.Checksum()
    checksum.update(source.payload)
    digest = hashlib.sha256(source.payload).hexdigest()
    return GCSPhysicalObjectReceiptV1(
        store_id=source.store_id,
        object_id=f"objects/sha256/{digest[:2]}/{digest}.bin",
        uri=f"gs://r1-us-east1/checkpoint/objects/sha256/{digest[:2]}/{digest}.bin",
        region=source.region,
        generation="1",
        metageneration="1",
        size_bytes=len(source.payload),
        crc32c=base64.b64encode(checksum.digest()).decode("ascii"),
        sha256=digest,
        content_type="application/octet-stream",
    )


def _receipt_for_payload(source: Source, payload: bytes) -> GCSPhysicalObjectReceiptV1:
    original = source.payload
    source.payload = payload
    try:
        return _receipt(source)
    finally:
        source.payload = original


def test_one_full_remote_fill_serves_many_local_ranges(tmp_path) -> None:
    source = Source(bytes(range(251)) * 100)
    receipt = _receipt(source)
    with ExactReceiptReadCache(source, tmp_path) as cache:
        assert cache.read_exact(receipt, offset=5, length=9, maximum_bytes=9) == (
            source.payload[5:14]
        )
        assert cache.read_exact(receipt, offset=50, length=17, maximum_bytes=32) == (
            source.payload[50:67]
        )
        assert source.calls == [(0, len(source.payload), len(source.payload))]
        assert cache.remote_objects == 1
        assert cache.remote_bytes == len(source.payload)
        assert cache.local_read_calls == 2
        assert cache.local_read_bytes == 26

    # A fresh process-equivalent wrapper re-hashes the local object once and
    # performs no source request.
    second = Source(source.payload)
    with ExactReceiptReadCache(second, tmp_path) as cache:
        assert cache.read_exact(receipt, offset=0, length=3, maximum_bytes=3) == (
            source.payload[:3]
        )
        assert second.calls == []


def test_corrupt_existing_cache_fails_instead_of_refetching(tmp_path) -> None:
    source = Source(b"correct bytes")
    receipt = _receipt(source)
    with ExactReceiptReadCache(source, tmp_path) as cache:
        cache.read_exact(receipt, offset=0, length=1, maximum_bytes=1)
    cached = next(path for path in tmp_path.rglob("*.bin"))
    cached.write_bytes(b"x" * len(source.payload))
    source.calls.clear()

    with ExactReceiptReadCache(source, tmp_path) as cache:
        with pytest.raises(GCSReadCacheError, match="SHA-256"):
            cache.read_exact(receipt, offset=0, length=1, maximum_bytes=1)
    assert source.calls == []


def test_wrong_receipt_and_range_fail_before_source_io(tmp_path) -> None:
    source = Source(b"some bytes")
    receipt = _receipt(source)
    wrong = Source(source.payload)
    wrong.store_id = "gcs-prefix-byte-store-v2:" + "b" * 64
    with ExactReceiptReadCache(wrong, tmp_path) as cache:
        with pytest.raises(GCSReadCacheError, match="another source"):
            cache.read_exact(receipt, offset=0, length=1, maximum_bytes=1)
    with ExactReceiptReadCache(source, tmp_path) as cache:
        with pytest.raises(GCSReadCacheError, match="range"):
            cache.read_exact(receipt, offset=0, length=10, maximum_bytes=1)
    assert source.calls == []


def test_same_object_converges_and_different_objects_fill_concurrently(tmp_path) -> None:
    source = Source(b"shared payload")
    receipt = _receipt(source)
    with ExactReceiptReadCache(source, tmp_path) as cache:
        with ThreadPoolExecutor(max_workers=8) as pool:
            values = tuple(
                pool.map(
                    lambda _index: cache.read_exact(
                        receipt, offset=0, length=6, maximum_bytes=6
                    ),
                    range(8),
                )
            )
        assert values == (b"shared",) * 8
        assert len(source.calls) == 1

    class ParallelSource(Source):
        def __init__(self, payloads: tuple[bytes, bytes]) -> None:
            super().__init__(b"")
            self.payloads = {
                hashlib.sha256(payload).hexdigest(): payload for payload in payloads
            }
            self.barrier = threading.Barrier(2)

        def read_exact(self, receipt, *, offset, length, maximum_bytes):
            self.barrier.wait(timeout=2)
            payload = self.payloads[receipt.sha256]
            self.calls.append((offset, length, maximum_bytes))
            return payload[offset : offset + length]

    first = b"first parallel object"
    second = b"second parallel object"
    parallel = ParallelSource((first, second))
    first_receipt = _receipt_for_payload(parallel, first)
    second_receipt = _receipt_for_payload(parallel, second)
    other_root = tmp_path / "parallel"
    other_root.mkdir()
    with ExactReceiptReadCache(parallel, other_root) as cache:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = tuple(
                pool.submit(
                    cache.read_exact,
                    candidate,
                    offset=0,
                    length=5,
                    maximum_bytes=5,
                )
                for candidate in (first_receipt, second_receipt)
            )
            assert tuple(future.result() for future in futures) == (
                first[:5],
                second[:5],
            )
