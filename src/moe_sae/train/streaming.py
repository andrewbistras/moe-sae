"""Bucketless streaming discovery: in-memory capture → frozen batch stream →
rotating on-disk windows consumed by all conditions (spec §9/§10 semantics).

Components:
- CollectingWriter: ShardWriter-compatible sink that keeps rows in memory, so
  the unchanged capture driver logic (`_capture_one`) feeds the stream without
  touching disk.
- StreamComposer: composes the frozen 2048+2048 batches from a visual shuffle
  buffer (rows consumed without replacement) and a growing text pool (rows
  drawn uniformly with replacement from the pool-so-far — the streaming
  reading of the contract's "sampling with replacement is permitted"). One
  fixed STREAM RNG: all runs share the batch sequence; seeds vary
  initialization only (user decision 2026-07-08; prereg wording owned by the
  author).
- WindowStore: K composed batches materialized to disk so every condition can
  replay them in identical order; windows are deleted after all conditions
  consume them (peak disk = one window).
- StateSwapper: one condition resident on GPU at a time; idle params live on
  CPU and idle FP32 Adam moments in disk-backed memmaps (durable for free).

Correctness invariant (unit-tested): windowed replay with swapping produces
bit-identical parameters and metrics to a plain sequential `loop.train` fed
the same batch list.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from moe_sae.sae import TrainBatch

logger = logging.getLogger(__name__)

STREAM_SEED = 1001  # fixed: the stream is shared by every run (decision above)


class CollectingWriter:
    """ShardWriter-compatible in-memory sink for the capture driver."""

    def __init__(self):
        self.rows: list[dict[str, torch.Tensor]] = []
        self.total_written = 0
        self._buffered = 0  # driver logging reads this

    def add(self, c, experts, gates, example_id, domain) -> None:
        self.rows.append(
            {
                "c": c.to(torch.bfloat16),
                "experts": experts.to(torch.uint8),
                "gates": gates.to(torch.float32),
                "example_id": example_id.to(torch.int32),
                "domain": domain.to(torch.uint8),
            }
        )
        self.total_written += c.shape[0]

    def drain(self) -> list[dict[str, torch.Tensor]]:
        out, self.rows = self.rows, []
        return out


def _cat_pool(chunks: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = ("c", "experts", "gates", "example_id", "domain")
    return {k: torch.cat([ch[k] for ch in chunks], dim=0) for k in keys}


class TokenPool:
    """Column store of token rows: append chunks, index-select, consume.

    Single source of truth: `_chunks` (lazily coalesced to one chunk). All
    reads go through `_flat()`, so `remove` results can never be shadowed by
    a later `append` (the bug class this replaces)."""

    KEYS = ("c", "experts", "gates", "example_id", "domain")

    def __init__(self):
        self._chunks: list[dict[str, torch.Tensor]] = []
        self.n = 0

    def append(self, rows: dict[str, torch.Tensor]) -> None:
        self._chunks.append(rows)
        self.n += rows["c"].shape[0]

    def _flat(self) -> dict[str, torch.Tensor]:
        assert self._chunks, "empty pool"
        if len(self._chunks) > 1:
            self._chunks = [_cat_pool(self._chunks)]
        return self._chunks[0]

    def take(self, idx: torch.Tensor) -> dict[str, torch.Tensor]:
        flat = self._flat()
        return {k: flat[k][idx] for k in self.KEYS}

    def remove(self, idx: torch.Tensor) -> None:
        flat = self._flat()
        keep = torch.ones(self.n, dtype=torch.bool)
        keep[idx] = False
        self._chunks = [{k: flat[k][keep] for k in self.KEYS}]
        self.n = int(keep.sum())


class StreamComposer:
    """Frozen batch composition from live capture (spec §9: B/2 visual + B/2
    text, halves independently shuffled; the visual half consumes unique rows,
    the text half draws with replacement from the growing pool)."""

    def __init__(
        self,
        example_source,  # iterator of per-example {pool: rows} dicts
        batch_tokens: int,
        min_visual_buffer: int = 65536,
        stream_seed: int = STREAM_SEED,
    ):
        self.source = iter(example_source)
        self.half = batch_tokens // 2
        self.min_visual_buffer = min_visual_buffer
        self.g = torch.Generator().manual_seed(stream_seed)
        self.visual = TokenPool()
        self.text = TokenPool()
        self.exhausted = False
        self.examples_consumed = 0

    def _pull_examples(self, need_visual: int) -> None:
        while not self.exhausted and self.visual.n < need_visual:
            try:
                per_pool = next(self.source)
            except StopIteration:
                self.exhausted = True
                return
            self.examples_consumed += 1
            if "visual" in per_pool:
                self.visual.append(per_pool["visual"])
            if "text" in per_pool:
                self.text.append(per_pool["text"])

    def next_batch(self) -> TrainBatch:
        self._pull_examples(self.half + self.min_visual_buffer)
        if self.visual.n < self.half:
            raise StopIteration(
                f"stream exhausted: visual buffer {self.visual.n} < {self.half}"
            )
        assert self.text.n > 0, "no text tokens captured yet"

        v_idx = torch.randperm(self.visual.n, generator=self.g)[: self.half]
        vis = self.visual.take(v_idx)
        self.visual.remove(v_idx)  # visual rows are consumed (drawn once)

        t_idx = torch.randint(0, self.text.n, (self.half,), generator=self.g)
        txt = self.text.take(t_idx)  # with replacement, pool keeps growing

        token_type = torch.cat(
            [
                torch.ones(self.half, dtype=torch.uint8),
                torch.zeros(self.half, dtype=torch.uint8),
            ]
        )
        cat = {
            k: torch.cat([vis[k], txt[k]], dim=0)
            for k in ("c", "experts", "gates", "example_id", "domain")
        }
        return TrainBatch(
            c=cat["c"],
            experts=cat["experts"].to(torch.int64),
            gates=cat["gates"],
            token_type=token_type,
            example_id=cat["example_id"],
            domain=cat["domain"],
        )


# --------------------------------------------------------------------------- #
# Window store                                                                 #
# --------------------------------------------------------------------------- #

_BATCH_KEYS = ("c", "experts", "gates", "token_type", "example_id", "domain")


class WindowStore:
    """K composed batches on disk, replayable in order, deletable as a unit."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, window: int) -> Path:
        return self.root / f"window_{window:04d}.pt"

    def write(self, window: int, batches: list[TrainBatch]) -> None:
        payload = [
            {k: getattr(b, k) for k in _BATCH_KEYS} for b in batches
        ]
        tmp = self._path(window).with_suffix(".tmp")
        torch.save(payload, tmp)
        tmp.rename(self._path(window))

    def read(self, window: int) -> list[TrainBatch]:
        payload = torch.load(self._path(window), map_location="cpu", weights_only=True)
        return [TrainBatch(**d) for d in payload]

    def delete(self, window: int) -> None:
        self._path(window).unlink(missing_ok=True)

    def exists(self, window: int) -> bool:
        return self._path(window).exists()


# --------------------------------------------------------------------------- #
# State swapping (one resident condition; durable Adam memmaps)                #
# --------------------------------------------------------------------------- #


class StateSwapper:
    """Moves a condition + its SpecAdam between GPU (active) and CPU/disk.

    FP32 Adam moments are disk-backed numpy memmaps — idle states cost no RAM
    and survive crashes; the active condition's moments are ordinary GPU/CPU
    tensors, copied back to the memmap on swap-out."""

    def __init__(self, state_dir: str | Path):
        self.dir = Path(state_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._maps: dict[str, list[tuple[np.memmap, np.memmap]]] = {}

    def _memmaps(self, name: str, opt) -> list[tuple[np.memmap, np.memmap]]:
        if name not in self._maps:
            pairs = []
            for i, (p, _ax) in enumerate(opt._params()):
                shape = tuple(p.shape)
                pair = []
                for kind in ("m", "v"):
                    f = self.dir / f"{name}_{i}_{kind}.f32"
                    mode = "r+" if f.exists() else "w+"
                    pair.append(np.memmap(f, dtype=np.float32, mode=mode, shape=shape))
                pairs.append(tuple(pair))
            self._maps[name] = pairs
        return self._maps[name]

    def swap_in(self, name: str, condition, opt, device: str) -> None:
        condition.to(device)
        for (p, _ax), (mm, vm) in zip(opt._params(), self._memmaps(name, opt)):
            st = opt.state[id(p)]
            st["exp_avg"] = torch.from_numpy(np.asarray(mm)).clone().to(device)
            st["exp_avg_sq"] = torch.from_numpy(np.asarray(vm)).clone().to(device)

    def swap_out(self, name: str, condition, opt) -> None:
        for (p, _ax), (mm, vm) in zip(opt._params(), self._memmaps(name, opt)):
            st = opt.state[id(p)]
            mm[:] = st["exp_avg"].detach().cpu().numpy()
            vm[:] = st["exp_avg_sq"].detach().cpu().numpy()
            mm.flush(); vm.flush()
            st["exp_avg"] = torch.zeros(1)  # release
            st["exp_avg_sq"] = torch.zeros(1)
        condition.to("cpu")
