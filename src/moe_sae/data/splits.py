"""Keyed-hash data splits (frozen spec v0.1.2 addendum, A3).

Split assignment is frozen as:

    split(x) = int(sha256(f"{dataset_key}:{example_id}".encode("utf-8"))
                   .hexdigest()[:8], 16) % 100

with buckets 0-79 train / 80-89 dev / 90-99 eval, example-level, never
position-level (spec §16.4). ``dataset_key`` is the capture domain's lowercase
name resolved through ``moe_sae.capture.writer.DOMAIN_NAMES``:
``coco``, ``textvqa``, ``chartqa``, ``ai2d`` for the four training domains,
and ``text_control`` for the eval-only text-control pool (byte 255; it never
enters training captures, but its split membership is defined the same way).
``example_id`` is the raw int32 domain-offset id carried in capture shards.

Language-native ``hash()`` is forbidden (process-salted, non-reproducible;
A3). The former raw ``example_id % 100`` convention is superseded: sequential
ids can encode shard/source/collection order into the split and correlate
buckets across datasets sharing ids (A3 reason, Reviewer S).

This module is the ONLY split-membership implementation; every consumer
(BatchComposer, stats pass, reconstruction eval, probes) resolves membership
through ``split_bucket`` / ``split_buckets`` / ``split_mask`` here.
"""

from __future__ import annotations

from functools import lru_cache
from hashlib import sha256

import numpy as np
import torch

from moe_sae.capture.writer import DOMAIN_NAMES, DOMAINS

# Frozen bucket ranges (A3): half-open [lo, hi) over the 0-99 bucket space.
SPLIT_RANGES = {"train": (0, 80), "dev": (80, 90), "eval": (90, 100)}

# domain byte -> frozen dataset_key (lowercase names, exactly as in A3).
DATASET_KEYS = dict(DOMAIN_NAMES)


def _dataset_key(domain: int | str) -> str:
    """Resolve a capture domain byte (or an already-resolved name) to the
    frozen dataset_key string."""
    if isinstance(domain, str):
        if domain not in DOMAINS:
            raise KeyError(f"unknown dataset_key {domain!r}; expected one of {sorted(DOMAINS)}")
        return domain
    key = DATASET_KEYS.get(int(domain))
    if key is None:
        raise KeyError(f"unknown capture domain byte {int(domain)}; expected one of {sorted(DATASET_KEYS)}")
    return key


@lru_cache(maxsize=1 << 22)
def _bucket_cached(dataset_key: str, example_id: int) -> int:
    """The frozen A3 formula for one (dataset_key, example_id) pair."""
    digest = sha256(f"{dataset_key}:{example_id}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 100


def split_bucket(domain: int | str, example_id: int) -> int:
    """Frozen A3 split bucket (0-99) of one example.

    ``domain``: capture domain byte (0=coco, 1=textvqa, 2=chartqa, 3=ai2d,
    255=text_control) or the dataset_key string itself.
    """
    return _bucket_cached(_dataset_key(domain), int(example_id))


def split_buckets(domain, example_id) -> np.ndarray | torch.Tensor:
    """Vectorized ``split_bucket`` with an lru cache over unique (domain, id)
    pairs (sha256 is computed once per unique pair, not per row).

    ``domain`` may be an array (per-row capture domain bytes), a scalar byte,
    or a dataset_key string (broadcast over ``example_id``). Returns a torch
    int64 tensor when ``example_id`` is a torch tensor, else a numpy int64
    array of the broadcast shape.
    """
    is_torch = isinstance(example_id, torch.Tensor)
    eid = (
        example_id.detach().cpu().numpy() if is_torch else np.asarray(example_id)
    ).astype(np.int64)
    if isinstance(domain, str):
        dom = np.full(eid.shape, int(DOMAINS[_dataset_key(domain)]), dtype=np.int64)
    else:
        if isinstance(domain, torch.Tensor):
            domain = domain.detach().cpu().numpy()
        dom = np.asarray(domain).astype(np.int64)
    dom, eid = np.broadcast_arrays(dom, eid)

    pairs = np.stack([dom.reshape(-1), eid.reshape(-1)], axis=1)
    uniq, inverse = np.unique(pairs, axis=0, return_inverse=True)
    uniq_buckets = np.array(
        [_bucket_cached(_dataset_key(int(d)), int(e)) for d, e in uniq], dtype=np.int64
    )
    out = uniq_buckets[inverse].reshape(dom.shape)
    return torch.from_numpy(out) if is_torch else out


def split_mask(domain, example_id, split: str):
    """Frozen example-level split membership under the A3 keyed hash.

    Returns a boolean mask of the broadcast (domain, example_id) shape —
    torch when ``example_id`` is torch, else numpy.
    """
    lo, hi = SPLIT_RANGES[split]
    b = split_buckets(domain, example_id)
    return (b >= lo) & (b < hi)
