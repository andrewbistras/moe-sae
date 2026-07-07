"""JAX/TPU training CLI — mirrors scripts/run_train.py.

    uv run python scripts/run_train_jax.py --config configs/pipeline_1m_A.yaml \\
        [--out-suffix _jax] [--capacity 768] [--device-index 0]

Writes runs/<run_name><out-suffix>/ with artifacts identical in format to the
torch trainer (checkpoints readable by load_condition_from_run).
"""

from __future__ import annotations

import argparse
import logging

import torch
from safetensors.torch import load_file

from moe_sae.model_loading import HIDDEN_SIZE, NUM_ROUTED_EXPERTS
from moe_sae.sae.conditions import build_condition
from moe_sae.train.config import RunConfig


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--out-suffix", default="_jax", help="appended to run_name")
    p.add_argument("--capacity", type=int, default=None, help="per-bank group capacity")
    p.add_argument("--device-index", type=int, default=0, help="jax.devices() index")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    cfg = RunConfig.from_yaml(args.config)
    cfg.validate()

    import jax

    from moe_sae.data.batches import BatchComposer
    from moe_sae.tpu.train_jax import train_jax

    stats = load_file(cfg.stats_path)
    mu_e = stats["mu_e"].to(torch.float32)
    mu_route = stats["mu_route"].to(torch.float32)
    assert mu_e.shape == (NUM_ROUTED_EXPERTS, HIDDEN_SIZE)

    cond = build_condition(cfg.condition, cfg, mu_e, mu_route, cfg.seed)
    composer = BatchComposer(cfg.capture_root, cfg.batch_tokens, cfg.data_seed)
    out_dir = cfg.out_dir.parent / (cfg.out_dir.name + args.out_suffix)

    train_jax(
        cfg,
        cond,
        iter(composer),
        out_dir,
        capacity=args.capacity,
        device=jax.devices()[args.device_index],
    )


if __name__ == "__main__":
    main()
