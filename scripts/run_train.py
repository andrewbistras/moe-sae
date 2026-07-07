#!/usr/bin/env python
"""Train one SAE condition from a resolved RunConfig YAML (spec §10, §12).

Usage:
    uv run python scripts/run_train.py --config configs/A_seed1.yaml [--device mps]

Thin by design: validate the frozen config, load the statistics-pass means,
build the batch composer and condition, hand off to moe_sae.train.loop.train.
"""

from __future__ import annotations

import argparse

import torch
from safetensors.torch import load_file

from moe_sae.model_loading import HIDDEN_SIZE, NUM_ROUTED_EXPERTS, pick_device
from moe_sae.sae.conditions import build_condition
from moe_sae.train.config import RunConfig
from moe_sae.train.loop import train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to RunConfig YAML")
    parser.add_argument(
        "--device", default=None, choices=["cpu", "mps", "cuda"], help="override cfg.device"
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
    composer = BatchComposer(cfg.capture_root, cfg.batch_tokens, cfg.data_seed)

    train(cfg, condition, iter(composer), cfg.out_dir)


if __name__ == "__main__":
    main()
