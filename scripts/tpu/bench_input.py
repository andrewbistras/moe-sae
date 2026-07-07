"""Input-pipeline throughput benchmark (smoke ladder rung 3).

Reads every shard under --path (repeatable: give the /dev/shm cache and/or a
gcsfuse mount / freshly-rsynced directory), reports raw GB/s, then converts to
achievable steps/s and MFU for the discovery-shape training steps:

  A-step: ~1.55 TFLOPs / 63 MB batch      F-step: ~16.6 TFLOPs / 63 MB batch

MFU here = FLOPs implied by the stream rate ÷ peak (275 TFLOPS/chip bf16),
i.e. the ceiling the input pipeline imposes — compute kernels can only lower it.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from safetensors import safe_open

PEAK_TFLOPS = 275.0  # v4 chip, bf16
BYTES_PER_BATCH = 4096 * (6 * 1280 * 2 + 6 + 6 * 4 + 4 + 1)  # ~63 MB (contract)
STEP_TFLOPS = {"A": 1.55, "F": 16.6}


def bench_dir(root: Path) -> tuple[float, float]:
    paths = sorted(root.rglob("shard_*.safetensors"))
    assert paths, f"no shards under {root}"
    total = 0
    t0 = time.monotonic()
    for p in paths:
        with safe_open(str(p), framework="numpy") as f:
            for k in f.keys():
                arr = f.get_tensor(k)
                total += arr.nbytes
    dt = time.monotonic() - t0
    return total / 1e9, total / 1e9 / dt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--path", action="append", required=True)
    args = ap.parse_args()
    for path in args.path:
        gb, gbps = bench_dir(Path(path))
        print(f"\n{path}: {gb:.1f} GB read at {gbps:.2f} GB/s")
        for cond, tf in STEP_TFLOPS.items():
            steps_s = gbps * 1e9 / BYTES_PER_BATCH
            mfu_ceiling = min(1.0, steps_s * tf / PEAK_TFLOPS)
            print(
                f"  {cond}-shape: stream supports {steps_s:6.1f} steps/s "
                f"→ input-imposed MFU ceiling {mfu_ceiling * 100:5.1f}% "
                f"(vs 1 chip; broadcast to all runs divides the need, not this rate)"
            )


if __name__ == "__main__":
    main()
