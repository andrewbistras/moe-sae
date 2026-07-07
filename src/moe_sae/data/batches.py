"""BatchComposer: 50/50 visual/text TrainBatch stream from a capture root.

Frozen contract (spec §9, docs/INTERFACES.md): every batch holds exactly
``batch_tokens/2`` visual + ``batch_tokens/2`` text positions; each half is
shuffled independently; the pools cycle independently (the smaller text pool
epochs faster — epoch counts are tracked and logged), and sampling repeats
across epochs. Deterministic given ``seed``.

Shuffling is the practical two-level implementation of §9's "examples and
positions shuffled independently within each half":

1. per epoch, the pool's shard order is reshuffled;
2. consecutive groups of ``buffer_shards`` shards (≈65k tokens at the default
   16,384-token shard size) are loaded, concatenated, and jointly permuted —
   a shuffle buffer of ``buffer_shards`` shards.

Memory: at most ``buffer_shards`` shards are resident per pool at any time
(one permuted group, sliced until consumed, plus nothing else).

``TrainBatch`` is imported from ``moe_sae.sae`` lazily (only when iteration
starts) so this module stays import-safe while that package is developed in
parallel; tests may inject a duck-typed substitute via ``batch_cls``.

Decisions:
- Split filtering (docs/INTERFACES.md §Splits): training consumes ONLY the
  train buckets (``example_id % 100`` in 0–79) by default. The constructor
  takes ``split='train'`` (any key of ``SPLIT_RANGES``); ``split=None``
  disables filtering (debugging only). Rows are filtered at shard-load time,
  before the shuffle buffer, so the 50/50 halves and seed determinism are
  unchanged; excluded row counts are tracked in ``excluded_counts`` per pool
  and logged at the end of each epoch. The modulo applies to the raw int32
  domain-offset example_id.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Iterator

import torch
from safetensors.torch import load_file

from moe_sae.capture.writer import POOLS, shard_paths

logger = logging.getLogger(__name__)

_FIELDS = ("c", "experts", "gates", "example_id", "domain")

# Frozen split convention (docs/INTERFACES.md §Splits): example_id % 100.
SPLIT_RANGES = {"train": (0, 80), "dev": (80, 90), "eval": (90, 100)}


def split_mask(example_id: torch.Tensor, split: str) -> torch.Tensor:
    """Frozen example-level split membership (never position-level; spec §16.4)."""
    lo, hi = SPLIT_RANGES[split]
    m = example_id.to(torch.int64) % 100
    return (m >= lo) & (m < hi)


def _default_batch_cls():
    from moe_sae.sae import TrainBatch  # lazy: parallel module (docs/INTERFACES.md)

    return TrainBatch


class BatchComposer:
    """Yields TrainBatch forever; take a fixed count with itertools.islice."""

    def __init__(
        self,
        capture_root: str | Path,
        batch_tokens: int,
        seed: int,
        buffer_shards: int = 4,
        batch_cls=None,
        split: str | None = "train",
    ):
        assert batch_tokens % 2 == 0, "batch must split into equal halves (spec §9)"
        assert buffer_shards >= 1
        assert split is None or split in SPLIT_RANGES, split
        self.capture_root = Path(capture_root)
        self.batch_tokens = batch_tokens
        self.seed = seed
        self.buffer_shards = buffer_shards
        self._batch_cls = batch_cls
        self.split = split
        # Shard lists are frozen at construction so the epoch permutation is
        # well-defined even if the directory changes mid-run.
        self._paths = {pool: shard_paths(self.capture_root, pool) for pool in POOLS}
        for pool in POOLS:
            if not self._paths[pool]:
                raise FileNotFoundError(
                    f"no shards in {self.capture_root / pool} — capture first"
                )
        self.epoch_counts = {pool: 0 for pool in POOLS}
        # Rows dropped by the split filter, cumulative per pool (contract:
        # exclusions are tracked and logged).
        self.excluded_counts = {pool: 0 for pool in POOLS}

    # ------------------------------------------------------------------ #

    def _pool_chunks(self, pool: str) -> Iterator[dict[str, torch.Tensor]]:
        """Infinite stream of jointly-permuted shard groups for one pool.

        String-seeded RNG (`"{seed}:{pool}"`) is stable across processes and
        platforms; torch permutations draw their seeds from it so the whole
        stream is reproducible from (seed, pool, shard list).
        """
        rng = random.Random(f"{self.seed}:{pool}")
        gen = torch.Generator()
        paths = self._paths[pool]
        while True:
            order = rng.sample(paths, len(paths))  # level 1: shard order/epoch
            for i in range(0, len(order), self.buffer_shards):
                group = order[i : i + self.buffer_shards]
                shards = [self._load_filtered(pool, p) for p in group]
                cat = {k: torch.cat([s[k] for s in shards], dim=0) for k in _FIELDS}
                n = cat["domain"].shape[0]
                gen.manual_seed(rng.getrandbits(62))
                if n == 0:  # every row of the group was filtered out
                    continue
                perm = torch.randperm(n, generator=gen)  # level 2: buffer shuffle
                yield {k: v[perm] for k, v in cat.items()}
            self.epoch_counts[pool] += 1
            logger.info(
                "pool %r finished epoch %d (%d shards, %d rows excluded by "
                "split=%r so far); pools cycle independently",
                pool,
                self.epoch_counts[pool],
                len(paths),
                self.excluded_counts[pool],
                self.split,
            )

    def _load_filtered(self, pool: str, path: Path) -> dict[str, torch.Tensor]:
        """Load one shard, dropping non-`split` rows before the shuffle buffer."""
        shard = load_file(str(path))
        if self.split is None:
            return {k: shard[k] for k in _FIELDS}
        keep = split_mask(shard["example_id"], self.split)
        self.excluded_counts[pool] += int((~keep).sum())
        return {k: shard[k][keep] for k in _FIELDS}

    @staticmethod
    def _take(
        chunks: Iterator[dict[str, torch.Tensor]],
        pending: list[dict[str, torch.Tensor]],
        n: int,
    ) -> dict[str, torch.Tensor]:
        """Pop exactly n rows from the chunk stream, keeping leftovers."""
        have = sum(p["domain"].shape[0] for p in pending)
        while have < n:
            chunk = next(chunks)
            pending.append(chunk)
            have += chunk["domain"].shape[0]
        taken: list[dict[str, torch.Tensor]] = []
        need = n
        while need > 0:
            head = pending[0]
            size = head["domain"].shape[0]
            if size <= need:
                taken.append(pending.pop(0))
                need -= size
            else:
                taken.append({k: v[:need] for k, v in head.items()})
                pending[0] = {k: v[need:] for k, v in head.items()}
                need = 0
        return {k: torch.cat([t[k] for t in taken], dim=0) for k in _FIELDS}

    # ------------------------------------------------------------------ #

    def __iter__(self):
        batch_cls = self._batch_cls if self._batch_cls is not None else _default_batch_cls()
        half = self.batch_tokens // 2
        chunk_iters = {pool: self._pool_chunks(pool) for pool in POOLS}
        pending: dict[str, list] = {pool: [] for pool in POOLS}
        while True:
            vis = self._take(chunk_iters["visual"], pending["visual"], half)
            txt = self._take(chunk_iters["text"], pending["text"], half)
            token_type = torch.cat(
                [
                    torch.ones(half, dtype=torch.uint8),  # visual rows first
                    torch.zeros(half, dtype=torch.uint8),
                ]
            )
            yield batch_cls(
                c=torch.cat([vis["c"], txt["c"]], dim=0).to(torch.bfloat16),
                experts=torch.cat([vis["experts"], txt["experts"]], dim=0).to(torch.int64),
                gates=torch.cat([vis["gates"], txt["gates"]], dim=0).to(torch.float32),
                token_type=token_type,
                example_id=torch.cat([vis["example_id"], txt["example_id"]], dim=0).to(
                    torch.int32
                ),
                domain=torch.cat([vis["domain"], txt["domain"]], dim=0).to(torch.uint8),
            )
