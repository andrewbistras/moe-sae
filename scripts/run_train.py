#!/usr/bin/env python
"""Train one SAE condition from a resolved RunConfig YAML (spec §10, §12).

Usage:
    uv run python scripts/run_train.py --config configs/A_seed1.yaml [--device mps]

Thin by design: validate the frozen config, load the statistics-pass means,
build the batch composer and condition, write the run manifest, hand off to
moe_sae.train.loop.train. `--resume` continues an interrupted run bitwise
from its latest checkpoint (docs/OPS.md §4); without it, an out_dir that
already holds run state is a hard error (mirrors capture semantics).
"""

from __future__ import annotations

import argparse

import torch
from safetensors.torch import load_file

from moe_sae.model_loading import HIDDEN_SIZE, NUM_ROUTED_EXPERTS, pick_device
from moe_sae.sae.conditions import build_condition
from moe_sae.train.config import RunConfig
from moe_sae.train.loop import train
from moe_sae.util.manifest import write_run_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to RunConfig YAML")
    parser.add_argument(
        "--device", default=None, choices=["cpu", "mps", "cuda"], help="override cfg.device"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue an interrupted run at cfg.out_dir from its latest "
        "checkpoint (bitwise, docs/OPS.md §4; a completed run is a no-op)",
    )
    parser.add_argument(
        "--peak-tflops",
        type=float,
        default=None,
        help="hardware peak TFLOP/s for MFU in the ops_metrics.jsonl sidecar "
        "(ops-only; omitted → steps/s + goodput only)",
    )
    args = parser.parse_args()

    cfg = RunConfig.from_yaml(args.config)
    if args.device is not None:
        cfg.device = args.device
    cfg.validate()  # frozen-constant enforcement lives here, not in the loop

    device = pick_device(None if cfg.device == "auto" else cfg.device)

    stats = load_file(cfg.stats_path)
    mu_e = stats["mu_e"].to(torch.float32)
    mu_route = stats["mu_route"].to(torch.float32)
    assert mu_e.shape == (NUM_ROUTED_EXPERTS, HIDDEN_SIZE), mu_e.shape
    assert mu_route.shape == (HIDDEN_SIZE,), mu_route.shape

    # Parallel-module import (contract: BatchComposer(capture_root,
    # batch_tokens, seed) yields TrainBatch); imported inside main so the SAE
    # core stays importable while data/ is developed independently.
    from moe_sae.data.batches import BatchComposer

    condition = build_condition(cfg.condition, cfg, mu_e, mu_route, cfg.seed).to(device)
    # record_draws (A15): per-(shard, row) text draw counts, persisted by the
    # loop as final/text_draw_counts.npz (bookkeeping only; stream unchanged).
    composer = BatchComposer(
        cfg.capture_root, cfg.batch_tokens, cfg.data_seed, record_draws=True
    )

    write_run_manifest(
        cfg.out_dir,
        cfg,
        extra={
            "entrypoint": "scripts/run_train.py",
            "backend": "torch",
            "device": str(device),
            "resume": args.resume,
        },
    )
    train(
        cfg,
        condition,
        iter(composer),
        cfg.out_dir,
        resume=args.resume,
        peak_flops=None if args.peak_tflops is None else args.peak_tflops * 1e12,
        composer=composer,
    )


if __name__ == "__main__":
    main()
