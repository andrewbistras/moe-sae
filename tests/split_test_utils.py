"""Shared test helpers for the A3 keyed-hash split (frozen v0.1.2 addendum).

Test fixtures can no longer plant `example_id % 100` residues; bucket
membership is pseudorandom in (domain, example_id). These helpers search for
example ids landing in a wanted bucket range for a given capture domain.
"""

from __future__ import annotations

from moe_sae.data.splits import SPLIT_RANGES, split_bucket


def ids_in_bucket_range(domain, lo: int, hi: int, n: int, start: int = 0) -> list[int]:
    """First `n` example ids >= start whose A3 bucket for `domain` lies in
    [lo, hi). `domain` is a capture domain byte or dataset_key string."""
    out: list[int] = []
    i = start
    while len(out) < n:
        if lo <= split_bucket(domain, i) < hi:
            out.append(i)
        i += 1
    return out


def ids_in_split(domain, split: str, n: int, start: int = 0) -> list[int]:
    lo, hi = SPLIT_RANGES[split]
    return ids_in_bucket_range(domain, lo, hi, n, start)


def ids_in_split_domain_mod4(split: str, n: int, start: int = 0) -> list[int]:
    """First `n` ids >= start in `split` under the fixture convention
    domain = example_id % 4 (used by the synthetic probe dumps)."""
    lo, hi = SPLIT_RANGES[split]
    out: list[int] = []
    i = start
    while len(out) < n:
        if lo <= split_bucket(i % 4, i) < hi:
            out.append(i)
        i += 1
    return out
