"""Capture manifest storage (spec §3): sharded safetensors, one pool per token type.

Layout:
    captures/<name>/
      meta.json
      visual/shard_00000.safetensors
      text/shard_00000.safetensors

Shard fields (N tokens): c [N,6,1280] bf16, experts [N,6] u8, gates [N,6] f32,
example_id [N] i32, domain [N] u8.  See docs/INTERFACES.md.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

import torch
from safetensors.torch import load_file, save_file

POOLS = ("visual", "text")
DOMAINS = {"coco": 0, "textvqa": 1, "chartqa": 2, "ai2d": 3, "text_control": 255}
DOMAIN_NAMES = {v: k for k, v in DOMAINS.items()}

_FIELD_SPECS = {
    "c": (torch.bfloat16, 3),
    "experts": (torch.uint8, 2),
    "gates": (torch.float32, 2),
    "example_id": (torch.int32, 1),
    "domain": (torch.uint8, 1),
}


@dataclass
class CaptureMeta:
    name: str
    model_id: str
    spec_version: str
    target_layer_zero_indexed: int
    tokens: dict = field(default_factory=lambda: {"visual": 0, "text": 0})
    examples: int = 0
    config: dict = field(default_factory=dict)

    def save(self, root: Path) -> None:
        (root / "meta.json").write_text(json.dumps(asdict(self), indent=2))

    @staticmethod
    def load(root: Path) -> "CaptureMeta":
        return CaptureMeta(**json.loads((root / "meta.json").read_text()))


class ShardWriter:
    """Buffers token records for one pool and writes fixed-size shards."""

    def __init__(self, root: Path, pool: str, shard_size: int = 16384):
        assert pool in POOLS
        self.dir = Path(root) / pool
        self.dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        self.pool = pool
        self._buf: dict[str, list[torch.Tensor]] = {k: [] for k in _FIELD_SPECS}
        self._buffered = 0
        self._shard_idx = 0
        self.total_written = 0

    def add(
        self,
        c: torch.Tensor,
        experts: torch.Tensor,
        gates: torch.Tensor,
        example_id: torch.Tensor,
        domain: torch.Tensor,
    ) -> None:
        n = c.shape[0]
        batch = {
            "c": c.to(torch.bfloat16).cpu(),
            "experts": experts.to(torch.uint8).cpu(),
            "gates": gates.to(torch.float32).cpu(),
            "example_id": example_id.to(torch.int32).cpu(),
            "domain": domain.to(torch.uint8).cpu(),
        }
        for k, (dtype, ndim) in _FIELD_SPECS.items():
            t = batch[k]
            assert t.shape[0] == n and t.ndim == ndim, (k, tuple(t.shape))
            self._buf[k].append(t)
        self._buffered += n
        while self._buffered >= self.shard_size:
            self._write_shard(self.shard_size)

    def _write_shard(self, size: int) -> None:
        cat = {k: torch.cat(v, dim=0) for k, v in self._buf.items()}
        shard = {k: t[:size].contiguous() for k, t in cat.items()}
        rest = {k: [t[size:]] for k, t in cat.items()}
        path = self.dir / f"shard_{self._shard_idx:05d}.safetensors"
        save_file(shard, str(path))
        self._shard_idx += 1
        self.total_written += size
        self._buf = rest
        self._buffered -= size

    def close(self, drop_remainder: bool = False) -> int:
        """Flush the final partial shard (unless drop_remainder). Returns total."""
        if self._buffered > 0 and not drop_remainder:
            self._write_shard(self._buffered)
        self._buf = {k: [] for k in _FIELD_SPECS}
        self._buffered = 0
        return self.total_written


def shard_paths(root: Path, pool: str) -> list[Path]:
    return sorted((Path(root) / pool).glob("shard_*.safetensors"))


def iter_shards(root: Path, pool: str) -> Iterator[dict[str, torch.Tensor]]:
    for p in shard_paths(root, pool):
        yield load_file(str(p))


def pool_token_count(root: Path, pool: str) -> int:
    return sum(load_file(str(p))["domain"].shape[0] for p in shard_paths(root, pool))
