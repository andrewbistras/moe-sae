"""Bucketless streaming discovery on one GPU (spec §10 step 4).

    uv run python scripts/run_stream_discovery.py \\
        --configs configs/discovery_*.yaml --stats stats/discovery_prefix \\
        --examples 96000 --window-batches 800 --device cuda

Prereqs (see RUNBOOK 'Bucketless A100 lane'): the prefix stats pass and the
retained dev/eval capture. All runs share one live capture stream (seeds vary
initialization only — user decision 2026-07-08); artifacts land under runs/
in the standard layout.
"""

from __future__ import annotations

import argparse
import glob
import logging
import os

import torch


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--configs", nargs="+", required=True, help="run config YAMLs (globs ok)")
    p.add_argument("--stats", required=True, help="stats dir or stats.safetensors (prefix pass)")
    p.add_argument("--examples", type=int, default=96000, help="stream budget (train buckets)")
    p.add_argument("--window-batches", type=int, default=800)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=7, help="capture stream seed (frozen)")
    p.add_argument("--visual-cap", type=int, default=512)
    p.add_argument("--work-dir", default="stream_work")
    p.add_argument("--pipeline", action="store_true",
                   help="A100-80: overlap capture with replay (model resident)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    from safetensors.torch import load_file

    from moe_sae.capture.driver import StopForward, _capture_one, instrumented_capture
    from moe_sae.data.streams import domain_stream, interleave_domains
    from moe_sae.model_loading import TARGET_LAYER_ZERO_INDEXED, load_model, load_processor
    from moe_sae.train.config import RunConfig
    from moe_sae.train.stream_discovery import stream_discovery
    from moe_sae.train.streaming import CollectingWriter, _cat_pool

    cfg_paths = sorted({q for g in args.configs for q in glob.glob(g)})
    configs = [RunConfig.from_yaml(q) for q in cfg_paths]
    for c in configs:
        c.validate()
    logging.info("suite: %s", [c.run_name for c in configs])

    stats_path = args.stats if args.stats.endswith(".safetensors") else os.path.join(
        args.stats, "stats.safetensors"
    )
    stats = load_file(stats_path)
    mu_e = stats["mu_e"].to(torch.float32)
    mu_route = stats["mu_route"].to(torch.float32)

    model, info = load_model(device=args.device)
    processor = load_processor()
    logging.info("model loaded: %s", info)
    language = model.language.model
    target_layer = language.layers[TARGET_LAYER_ZERO_INDEXED]
    moe = target_layer.mlp

    def example_source():
        """Live per-example capture, train buckets only (A3)."""
        stream = interleave_domains(
            {d: domain_stream(d, split="train")
             for d in ("coco", "textvqa", "chartqa", "ai2d")},
            seed=args.seed,
        )

        def early_exit(_m, _a, _o):
            raise StopForward

        hook = target_layer.register_forward_hook(early_exit)
        consumed = failed = 0
        try:
            with instrumented_capture(moe, store_dtype=torch.bfloat16, device="cpu") as sink:
                with torch.no_grad():
                    for ex in stream:
                        if consumed >= args.examples:
                            return
                        if ex.example_id % 100 >= 80:
                            continue  # dev/eval buckets never enter training
                        consumed += 1
                        writers = {"visual": CollectingWriter(), "text": CollectingWriter()}
                        try:
                            _capture_one(
                                model, processor, ex, sink, language,
                                args.visual_cap, args.seed, 2048, False, writers,
                            )
                        except Exception:
                            sink.drain()
                            failed += 1
                            if failed > max(20, consumed // 20):
                                raise
                            logging.warning("example %d failed (%d so far)",
                                            ex.example_id, failed)
                            continue
                        out = {}
                        for pool, w in writers.items():
                            rows = w.drain()
                            if rows:
                                out[pool] = _cat_pool(rows)
                        yield out
        finally:
            hook.remove()

    stream_discovery(
        configs,
        example_source(),
        mu_e,
        mu_route,
        window_batches=args.window_batches,
        device=args.device,
        work_dir=args.work_dir,
        model=model,
        pipelined=args.pipeline,
    )


if __name__ == "__main__":
    main()
