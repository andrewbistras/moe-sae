#!/usr/bin/env python
"""Server-side merge of capture worker slices inside a GCS bucket.

The canonical merger (scripts/merge_captures.py) renumbers shards with
hardlinks on one filesystem; the Colab capture lane
(scripts/cloud/colab_capture.sh) never holds all slices locally, so this ops
variant performs the identical renumbering with in-bucket server-side copies
(no data transits any machine) and writes the merged progress.json/meta.json
from the workers' metadata. merge_captures.merge remains the semantics
source; the metadata arithmetic below mirrors it line for line.

Dry-run by default — prints the plan and exits. Requires gcloud auth.

    python scripts/cloud/merge_in_bucket.py --bucket gs://<bucket> \
        --src "captures/discovery_50m_w{w}" --workers 12 \
        --out captures/discovery_50m --execute
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from moe_sae.capture.driver import CaptureProgress
from moe_sae.capture.writer import POOLS, CaptureMeta


def _gcloud(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["gcloud", "storage", *args], capture_output=True, text=True)


def _ls_shards(url_dir: str) -> list[str]:
    r = _gcloud("ls", f"{url_dir}/shard_*.safetensors")
    if r.returncode != 0:  # no matches
        return []
    return sorted(line.strip() for line in r.stdout.splitlines() if line.strip())


def _fetch_worker(bucket: str, src: str, tmp: Path) -> tuple[CaptureProgress, CaptureMeta]:
    tmp.mkdir(parents=True, exist_ok=True)
    for name in ("progress.json", "meta.json"):
        r = _gcloud("cp", f"{bucket}/{src}/{name}", str(tmp / name))
        assert r.returncode == 0, f"missing {bucket}/{src}/{name}: {r.stderr.strip()}"
    prog = CaptureProgress.load(tmp)
    meta = CaptureMeta.load(tmp)
    assert prog is not None and prog.complete, f"{src} is not a complete capture"
    return prog, meta


def plan(bucket: str, srcs: list[str], out: str) -> tuple[list[tuple[str, str]], dict, dict]:
    """Copy list [(src_url, dst_url)] + per-pool totals, after verifying each
    worker's bucket listing exactly matches its progress.json shard counts."""
    with tempfile.TemporaryDirectory() as td:
        workers = [
            (src, *_fetch_worker(bucket, src, Path(td) / f"w{i}")) for i, src in enumerate(srcs)
        ]
        copies: list[tuple[str, str]] = []
        shard_idx = {p: 0 for p in POOLS}
        for src, prog, _meta in workers:
            for pool in POOLS:
                urls = _ls_shards(f"{bucket}/{src}/{pool}")
                n = prog.shards.get(pool, 0)
                want = [f"{bucket}/{src}/{pool}/shard_{i:05d}.safetensors" for i in range(n)]
                assert urls == want, (
                    f"{src}/{pool}: bucket listing does not match progress.json "
                    f"({len(urls)} objects vs {n} recorded) — stray/missing shards; "
                    "re-push the slice (colab_capture.sh final rsync uses "
                    "--delete-unmatched-destination-objects) before merging"
                )
                for i in range(n):
                    dst = f"{bucket}/{out}/{pool}/shard_{shard_idx[pool]:05d}.safetensors"
                    copies.append((want[i], dst))
                    shard_idx[pool] += 1

        # merged metadata — mirrors scripts/merge_captures.py::merge exactly
        merged_prog = CaptureProgress(config=dict(workers[0][1].config, merged=True))
        tokens = {p: 0 for p in POOLS}
        for _src, prog, _meta in workers:
            for pool in POOLS:
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

        base = workers[0][2]
        merged_meta = CaptureMeta(
            name=out.rstrip("/").split("/")[-1],
            model_id=base.model_id,
            spec_version=base.spec_version,
            target_layer_zero_indexed=base.target_layer_zero_indexed,
            tokens=tokens,
            examples=merged_prog.captured,
            config={
                **base.config,
                "merged_from": [f"{bucket}/{s}" for s in srcs],
                "skipped_too_long": merged_prog.skipped_too_long,
                "failed": merged_prog.failed,
                "domain_example_counts": merged_prog.domain_counts,
            },
        )
        return copies, merged_prog.__dict__, merged_meta.__dict__ | {"_obj": merged_meta, "_prog": merged_prog}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bucket", required=True, help="gs://<bucket>, no trailing slash")
    ap.add_argument("--src", default="captures/discovery_50m_w{w}", help="worker path template, {w} = index")
    ap.add_argument("--workers", type=int, required=True)
    ap.add_argument("--out", default="captures/discovery_50m")
    ap.add_argument("--execute", action="store_true", help="perform the copies (default: dry-run)")
    ap.add_argument("--jobs", type=int, default=32)
    args = ap.parse_args()

    bucket = args.bucket.rstrip("/")
    srcs = [args.src.format(w=w) for w in range(args.workers)]
    copies, prog_dict, meta_extra = plan(bucket, srcs, args.out)
    merged_prog, merged_meta = meta_extra.pop("_prog"), meta_extra.pop("_obj")

    per_pool = {p: sum(1 for _s, d in copies if f"/{p}/" in d) for p in POOLS}
    print(f"plan: {len(copies)} server-side copies -> {bucket}/{args.out}  {per_pool}")
    print(f"      tokens={prog_dict['tokens']}  examples={prog_dict['captured']}")
    if copies:
        print(f"      first: {copies[0][0]} -> {copies[0][1]}")
        print(f"      last : {copies[-1][0]} -> {copies[-1][1]}")
    if not args.execute:
        print("dry-run only — pass --execute to merge")
        return 0

    def cp(pair: tuple[str, str]) -> tuple[str, str, bool]:
        r = _gcloud("cp", pair[0], pair[1])
        return (*pair, r.returncode == 0)

    with ThreadPoolExecutor(args.jobs) as ex:
        failed = [(s, d) for s, d, ok in ex.map(cp, copies) if not ok]
    assert not failed, f"{len(failed)} copies failed (rerun is safe/idempotent): {failed[:3]}"

    for pool in POOLS:  # verify the merged root before blessing it with metadata
        got = len(_ls_shards(f"{bucket}/{args.out}/{pool}"))
        assert got == per_pool[pool], f"{pool}: merged bucket has {got} shards, expected {per_pool[pool]}"

    with tempfile.TemporaryDirectory() as td:
        merged_prog.save(Path(td))
        merged_meta.save(Path(td))
        for name in ("progress.json", "meta.json"):
            r = _gcloud("cp", str(Path(td) / name), f"{bucket}/{args.out}/{name}")
            assert r.returncode == 0, r.stderr
    print(json.dumps({"examples": merged_meta.examples, "tokens": merged_meta.tokens}, indent=2))
    print(f"merged capture ready at {bucket}/{args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
