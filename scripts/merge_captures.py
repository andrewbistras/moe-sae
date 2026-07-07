"""Merge parallel capture-worker directories into one capture root.

    uv run python scripts/merge_captures.py --out captures/discovery_50m \\
        captures/discovery_50m_w0 captures/discovery_50m_w1 ...

Workers captured disjoint slices of the deterministic stream (run_capture
--start/--examples), so example ids are globally unique across workers; a
merge is just shard renumbering (hardlinked when possible) plus summed
metadata. Every worker must be complete. The merged root gets meta.json and
a complete progress.json so downstream runners treat it as a finished capture.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from moe_sae.capture.driver import PROGRESS_FILE, CaptureProgress
from moe_sae.capture.writer import POOLS, CaptureMeta, shard_paths


def merge(worker_roots: list[Path], out_root: Path) -> CaptureMeta:
    workers = []
    for root in worker_roots:
        prog = CaptureProgress.load(root)
        assert prog is not None and prog.complete, f"{root} is not a complete capture"
        workers.append((root, prog, CaptureMeta.load(root)))

    out_root.mkdir(parents=True, exist_ok=True)
    merged_prog = CaptureProgress(config=dict(workers[0][1].config, merged=True))
    tokens = {p: 0 for p in POOLS}
    shard_idx = {p: 0 for p in POOLS}

    for root, prog, _meta in workers:
        for pool in POOLS:
            dst_dir = out_root / pool
            dst_dir.mkdir(exist_ok=True)
            for src in shard_paths(root, pool):
                dst = dst_dir / f"shard_{shard_idx[pool]:05d}.safetensors"
                assert not dst.exists(), dst
                try:
                    os.link(src, dst)  # hardlink: instant, no extra disk
                except OSError:
                    dst.write_bytes(src.read_bytes())
                shard_idx[pool] += 1
            tokens[pool] += prog.tokens.get(pool, 0)
        merged_prog.stream_consumed += prog.stream_consumed
        merged_prog.captured += prog.captured
        merged_prog.skipped_too_long += prog.skipped_too_long
        merged_prog.failed += prog.failed
        for k, v in prog.domain_counts.items():
            merged_prog.domain_counts[k] = merged_prog.domain_counts.get(k, 0) + v

    merged_prog.tokens = tokens
    merged_prog.shards = dict(shard_idx)
    merged_prog.complete = True
    merged_prog.save(out_root)

    base = workers[0][2]
    meta = CaptureMeta(
        name=out_root.name,
        model_id=base.model_id,
        spec_version=base.spec_version,
        target_layer_zero_indexed=base.target_layer_zero_indexed,
        tokens=tokens,
        examples=merged_prog.captured,
        config={
            **base.config,
            "merged_from": [str(r) for r, _, _ in workers],
            "skipped_too_long": merged_prog.skipped_too_long,
            "failed": merged_prog.failed,
            "domain_example_counts": merged_prog.domain_counts,
        },
    )
    meta.save(out_root)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("workers", nargs="+", help="complete worker capture roots, in stream order")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    meta = merge([Path(w) for w in args.workers], Path(args.out))
    print(json.dumps({"examples": meta.examples, "tokens": meta.tokens}, indent=2))


if __name__ == "__main__":
    main()
