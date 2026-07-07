"""BatchComposer: exact 50/50 halves, seed determinism, independent pool
cycling, shuffle-buffer locality (spec §9) — on synthetic ShardWriter shards
(tiny d; the writer only constrains ndim)."""

from dataclasses import dataclass
from itertools import islice

import pytest
import torch

from moe_sae.capture.writer import ShardWriter
from moe_sae.data.batches import BatchComposer

try:  # parallel module; duck-typed fallback per docs/INTERFACES.md contract
    from moe_sae.sae import TrainBatch
except Exception:

    @dataclass
    class TrainBatch:
        c: torch.Tensor
        experts: torch.Tensor
        gates: torch.Tensor
        token_type: torch.Tensor
        example_id: torch.Tensor
        domain: torch.Tensor


SHARD = 16  # tokens per synthetic shard
D = 4


def build_pool(root, pool, n_shards, id_base):
    """One example_id per shard so shard membership is recoverable from rows;
    c is filled with the example_id so row integrity is checkable."""
    torch.manual_seed(id_base)
    w = ShardWriter(root, pool, shard_size=SHARD)
    for s in range(n_shards):
        eid = id_base + s
        w.add(
            c=torch.full((SHARD, 6, D), float(eid)),
            experts=torch.randint(0, 64, (SHARD, 6)),
            gates=torch.rand(SHARD, 6),
            example_id=torch.full((SHARD,), eid, dtype=torch.int32),
            domain=torch.full((SHARD,), s % 4, dtype=torch.uint8),
        )
    assert w.close() == n_shards * SHARD


@pytest.fixture()
def capture_root(tmp_path):
    build_pool(tmp_path, "visual", n_shards=8, id_base=1000)
    build_pool(tmp_path, "text", n_shards=2, id_base=2000)
    return tmp_path


def make_composer(root, batch_tokens=32, seed=7, buffer_shards=2):
    return BatchComposer(
        root, batch_tokens=batch_tokens, seed=seed, buffer_shards=buffer_shards,
        batch_cls=TrainBatch,
    )


def test_exact_half_composition_and_dtypes(capture_root):
    batch = next(iter(make_composer(capture_root)))
    assert batch.c.shape == (32, 6, D) and batch.c.dtype == torch.bfloat16
    assert batch.experts.shape == (32, 6) and batch.experts.dtype == torch.int64
    assert batch.gates.dtype == torch.float32
    assert batch.token_type.dtype == torch.uint8
    assert batch.example_id.dtype == torch.int32
    assert batch.domain.dtype == torch.uint8
    assert int((batch.token_type == 1).sum()) == 16  # visual half
    assert int((batch.token_type == 0).sum()) == 16  # text half
    # pool membership: visual rows carry visual-pool ids, text rows text-pool ids
    vis_ids = batch.example_id[batch.token_type == 1]
    txt_ids = batch.example_id[batch.token_type == 0]
    assert ((vis_ids >= 1000) & (vis_ids < 2000)).all()
    assert (txt_ids >= 2000).all()


def test_row_integrity_c_matches_example_id(capture_root):
    for batch in islice(iter(make_composer(capture_root)), 3):
        # c was written as full(example_id); every row must still match its id
        expected = batch.example_id.to(torch.bfloat16).view(-1, 1, 1).expand_as(batch.c)
        assert torch.equal(batch.c, expected)


def test_deterministic_across_reinstantiation(capture_root):
    a = list(islice(iter(make_composer(capture_root, seed=7)), 6))
    b = list(islice(iter(make_composer(capture_root, seed=7)), 6))
    for x, y in zip(a, b):
        for f in ("c", "experts", "gates", "token_type", "example_id", "domain"):
            assert torch.equal(getattr(x, f), getattr(y, f)), f
    c = list(islice(iter(make_composer(capture_root, seed=8)), 6))
    assert any(
        not torch.equal(x.example_id, y.example_id) for x, y in zip(a, c)
    ), "different seed must reorder the stream"


def test_visual_epoch_is_a_permutation(capture_root):
    # 8 visual shards x 16 tokens = 128 = 8 batches x 16 visual rows: one epoch
    batches = list(islice(iter(make_composer(capture_root)), 8))
    vis_ids = torch.cat([b.example_id[b.token_type == 1] for b in batches])
    counts = {int(e): int((vis_ids == e).sum()) for e in vis_ids.unique()}
    assert counts == {1000 + s: SHARD for s in range(8)}, (
        "each visual token must appear exactly once per epoch"
    )


def test_shuffle_buffer_locality(capture_root):
    # buffer_shards=2: the first 2*SHARD visual rows must be a joint permutation
    # of exactly 2 shards (groups partition the epoch's shuffled shard order).
    batches = list(islice(iter(make_composer(capture_root, buffer_shards=2)), 8))
    vis_ids = torch.cat([b.example_id[b.token_type == 1] for b in batches])
    for g in range(4):  # 4 groups of 2 shards per epoch
        group = vis_ids[g * 2 * SHARD : (g + 1) * 2 * SHARD]
        uniq, cnt = group.unique(return_counts=True)
        assert len(uniq) == 2 and (cnt == SHARD).all(), (
            f"group {g} not a permutation of exactly 2 shards: {uniq.tolist()}"
        )


def test_pools_cycle_independently(capture_root):
    composer = make_composer(capture_root)
    batches = list(islice(iter(composer), 9))
    # text pool (2 shards = 32 tokens) epochs every 2 batches; visual every 8
    assert composer.epoch_counts["text"] > composer.epoch_counts["visual"]
    assert composer.epoch_counts["text"] >= 4
    # text rows repeat across epochs; every batch still has exact halves
    for b in batches:
        assert int((b.token_type == 1).sum()) == 16
        assert int((b.token_type == 0).sum()) == 16


def test_unaligned_half_size(capture_root):
    # half=12 does not divide the 32-token buffer groups: leftovers must carry
    batches = list(islice(iter(make_composer(capture_root, batch_tokens=24)), 11))
    for b in batches:
        assert b.c.shape[0] == 24
        assert int((b.token_type == 1).sum()) == 12
        assert int((b.token_type == 0).sum()) == 12
    vis_ids = torch.cat([b.example_id[b.token_type == 1] for b in batches])
    # first visual epoch (128 rows) is still an exact permutation of the pool
    counts = {int(e): int((vis_ids[:128] == e).sum()) for e in vis_ids[:128].unique()}
    assert counts == {1000 + s: SHARD for s in range(8)}


def build_mixed_pool(root, pool, n_shards, ids):
    """Rows cycle through `ids` (int32 domain-offset style), so every shard
    mixes train/dev/eval buckets; c is filled with the example_id per row."""
    torch.manual_seed(ids[0])
    w = ShardWriter(root, pool, shard_size=SHARD)
    for s in range(n_shards):
        eid = torch.tensor(
            [ids[(s * SHARD + r) % len(ids)] for r in range(SHARD)], dtype=torch.int32
        )
        w.add(
            c=eid.to(torch.float32).view(-1, 1, 1).expand(SHARD, 6, D).contiguous(),
            experts=torch.randint(0, 64, (SHARD, 6)),
            gates=torch.rand(SHARD, 6),
            example_id=eid,
            domain=torch.full((SHARD,), s % 4, dtype=torch.uint8),
        )
    assert w.close() == n_shards * SHARD


@pytest.fixture()
def mixed_root(tmp_path):
    # raw int32 domain-offset ids; %100 gives buckets 0, 1, 85 (dev), 92 (eval)
    vis_ids = [250000000, 250000001, 250000085, 250000092]
    txt_ids = [300000000, 300000001, 300000185, 300000292]
    build_mixed_pool(tmp_path, "visual", n_shards=8, ids=vis_ids)
    build_mixed_pool(tmp_path, "text", n_shards=2, ids=txt_ids)
    return tmp_path


def test_split_filter_excludes_dev_and_eval_buckets(mixed_root):
    # Regression (blocker): training stream must never see example_id%100 >= 80.
    composer = make_composer(mixed_root)
    batches = list(islice(iter(composer), 12))  # > 1 epoch of each pool
    for b in batches:
        assert (b.example_id.to(torch.int64) % 100 < 80).all(), (
            "dev/eval buckets leaked into the training stream"
        )
    # half of every shard's rows are dev/eval: exclusions must be tracked
    assert composer.excluded_counts["visual"] > 0
    assert composer.excluded_counts["text"] > 0


def test_split_filter_keeps_halves_and_determinism(mixed_root):
    a = list(islice(iter(make_composer(mixed_root, seed=7)), 8))
    b = list(islice(iter(make_composer(mixed_root, seed=7)), 8))
    for x, y in zip(a, b):
        for f in ("c", "experts", "gates", "token_type", "example_id", "domain"):
            assert torch.equal(getattr(x, f), getattr(y, f)), f
    for batch in a:
        assert int((batch.token_type == 1).sum()) == 16
        assert int((batch.token_type == 0).sum()) == 16
        # row integrity survives filtering: c still matches example_id
        expected = (
            batch.example_id.to(torch.bfloat16).view(-1, 1, 1).expand_as(batch.c)
        )
        assert torch.equal(batch.c, expected)


def test_split_filter_epoch_is_permutation_of_train_rows(mixed_root):
    # 8 visual shards x 8 kept rows = 64 = 4 batches x 16 visual rows: one epoch
    batches = list(islice(iter(make_composer(mixed_root)), 4))
    vis_ids = torch.cat([b.example_id[b.token_type == 1] for b in batches])
    counts = {int(e): int((vis_ids == e).sum()) for e in vis_ids.unique()}
    assert counts == {250000000: 32, 250000001: 32}, (
        "each kept visual token must appear exactly once per epoch"
    )


def test_split_none_disables_filter(mixed_root):
    composer = BatchComposer(
        mixed_root, batch_tokens=32, seed=7, buffer_shards=2,
        batch_cls=TrainBatch, split=None,
    )
    batches = list(islice(iter(composer), 8))  # full visual epoch (old behavior)
    ids = torch.cat([b.example_id for b in batches]).to(torch.int64) % 100
    assert (ids >= 80).any(), "split=None must keep dev/eval rows"
    assert composer.excluded_counts == {"visual": 0, "text": 0}


def test_missing_pool_raises(tmp_path):
    build_pool(tmp_path, "visual", n_shards=1, id_base=0)
    with pytest.raises(FileNotFoundError):
        make_composer(tmp_path)
