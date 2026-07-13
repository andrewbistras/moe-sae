"""A3 keyed-hash split (frozen v0.1.2 addendum): the frozen formula, the
vectorized wrapper, and the contamination regression test — a training
composer over shards spanning all buckets yields zero dev/eval rows under the
new membership function."""

from dataclasses import dataclass
from hashlib import sha256
from itertools import islice

import numpy as np
import pytest
import torch

from moe_sae.capture.writer import DOMAINS, ShardWriter
from moe_sae.data.batches import BatchComposer
from moe_sae.data.splits import (
    SPLIT_RANGES,
    split_bucket,
    split_buckets,
    split_mask,
)


@dataclass
class TrainBatchStub:
    c: torch.Tensor
    experts: torch.Tensor
    gates: torch.Tensor
    token_type: torch.Tensor
    example_id: torch.Tensor
    domain: torch.Tensor


D = 4
SHARD = 16


def test_frozen_formula_exact():
    """split_bucket must be exactly the A3 formula — no salt, no reordering."""
    for key, byte in (("coco", 0), ("textvqa", 1), ("chartqa", 2), ("ai2d", 3),
                      ("text_control", 255)):
        for eid in (0, 1, 79, 12345, 250_000_001, 2**31 - 1):
            want = int(sha256(f"{key}:{eid}".encode("utf-8")).hexdigest()[:8], 16) % 100
            assert split_bucket(byte, eid) == want, (key, eid)
            assert split_bucket(key, eid) == want  # name and byte agree


def test_buckets_differ_across_domains():
    # the keyed hash decorrelates datasets sharing raw ids (A3 reason)
    ids = np.arange(500)
    b_coco = split_buckets("coco", ids)
    b_textvqa = split_buckets("textvqa", ids)
    assert (b_coco != b_textvqa).any()
    # and buckets are not a function of id % 100
    same_residue = [3, 103, 203, 303, 403]
    buckets = {split_bucket("coco", i) for i in same_residue}
    assert len(buckets) > 1


def test_vectorized_matches_scalar_and_types():
    ids = np.array([0, 7, 79, 80, 99, 100_000_000, 250_000_085], dtype=np.int64)
    doms = np.array([0, 1, 2, 3, 255, 1, 2], dtype=np.uint8)
    vec = split_buckets(doms, ids)
    assert isinstance(vec, np.ndarray) and vec.dtype == np.int64
    for d, e, b in zip(doms, ids, vec):
        assert split_bucket(int(d), int(e)) == int(b)
    # torch in -> torch out; scalar domain broadcast
    t = split_buckets(torch.from_numpy(doms), torch.from_numpy(ids))
    assert isinstance(t, torch.Tensor)
    assert np.array_equal(t.numpy(), vec)
    t2 = split_buckets(2, torch.from_numpy(ids))
    for e, b in zip(ids, t2.tolist()):
        assert split_bucket(2, int(e)) == b


def test_split_mask_ranges():
    ids = np.arange(1000)
    doms = ids % 4
    buckets = split_buckets(doms, ids)
    for split, (lo, hi) in SPLIT_RANGES.items():
        mask = split_mask(doms, ids, split)
        np.testing.assert_array_equal(mask, (buckets >= lo) & (buckets < hi))
    # the three splits partition the id space
    total = sum(split_mask(doms, ids, s).sum() for s in SPLIT_RANGES)
    assert total == ids.size


def test_unknown_domain_raises():
    with pytest.raises(KeyError):
        split_bucket(7, 0)
    with pytest.raises(KeyError):
        split_bucket("imagenet", 0)


# --------------------------------------------------------------------------- #
# A3 contamination regression test (consequence freeze): under the NEW         #
# membership function, the training composer must yield zero dev/eval rows.    #
# --------------------------------------------------------------------------- #


def build_pool_spanning_buckets(root, pool, domain_byte, n_shards=6):
    """Sequential ids with a fixed domain: the hash scatters them over all
    100 buckets, so every shard mixes train/dev/eval rows."""
    torch.manual_seed(domain_byte)
    w = ShardWriter(root, pool, shard_size=SHARD)
    eid = 0
    for _ in range(n_shards):
        ids = torch.arange(eid, eid + SHARD, dtype=torch.int32)
        eid += SHARD
        w.add(
            c=ids.to(torch.float32).view(-1, 1, 1).expand(SHARD, 6, D).contiguous(),
            experts=torch.randint(0, 64, (SHARD, 6)),
            gates=torch.rand(SHARD, 6),
            example_id=ids,
            domain=torch.full((SHARD,), domain_byte, dtype=torch.uint8),
        )
    assert w.close() == n_shards * SHARD


def test_composer_zero_train_dev_eval_contamination(tmp_path):
    """A composer over synthetic shards spanning all buckets yields zero rows
    with split_bucket in 80-99 (A3 consequence freeze)."""
    build_pool_spanning_buckets(tmp_path, "visual", domain_byte=0)
    build_pool_spanning_buckets(tmp_path, "text", domain_byte=1)

    # the raw shards really span all three splits for both domains
    for dom in (0, 1):
        buckets = split_buckets(dom, np.arange(6 * SHARD))
        assert (buckets < 80).any() and ((buckets >= 80) & (buckets < 90)).any()
        assert (buckets >= 90).any()

    composer = BatchComposer(
        tmp_path, batch_tokens=32, seed=7, buffer_shards=2,
        batch_cls=TrainBatchStub, split="train",
    )
    batches = list(islice(iter(composer), 16))  # multiple epochs of both pools
    assert batches
    for b in batches:
        buckets = split_buckets(b.domain, b.example_id)
        assert (buckets < 80).all(), "dev/eval rows leaked into the training stream"
        # row integrity: c still matches the example id after filtering
        expected = b.example_id.to(torch.bfloat16).view(-1, 1, 1).expand_as(b.c)
        assert torch.equal(b.c, expected)
    assert composer.excluded_counts["visual"] > 0
    assert composer.excluded_counts["text"] > 0


def test_dataset_keys_match_capture_domains():
    # A3 pins the four training dataset keys as exact lowercase strings.
    assert DOMAINS == {
        "coco": 0, "textvqa": 1, "chartqa": 2, "ai2d": 3, "text_control": 255,
    }
