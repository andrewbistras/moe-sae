#!/usr/bin/env python
"""Reconstruction + downstream-fidelity evaluation of a trained run (spec §13,
§20 step 6), producing the FeatureDump consumed by the analysis stage.

Usage:
    uv run python scripts/run_eval.py \
        --run runs/<run_name> --capture captures/<name> --split dev \
        --dump dumps/<run_name>/dev \
        [--fidelity matched,full --eval-examples 200] \
        [--out runs/<run_name>/eval_dev]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

FIDELITY_DOMAINS = ("coco", "textvqa", "chartqa", "ai2d", "text_control")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="run directory (runs/<run_name>)")
    ap.add_argument("--capture", required=True, help="capture root (captures/<name>)")
    ap.add_argument(
        "--split",
        required=True,
        choices=("train", "dev", "eval"),
        help="capture split; 'train' produces the dump the §16.4 probes FIT on "
        "(the reconstruction metrics are then in-sample for that split)",
    )
    ap.add_argument("--dump", required=True, help="FeatureDump output dir (dumps/<run>/<split>)")
    ap.add_argument("--out", default=None, help="metrics dir inside the run (default eval_<split>)")
    ap.add_argument("--fidelity", default=None, help="comma list of modes: matched,full")
    ap.add_argument("--eval-examples", type=int, default=200, help="examples per domain")
    ap.add_argument("--eval-seed", type=int, default=7)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    run_dir = Path(args.run)
    out_dir = Path(args.out) if args.out else run_dir / f"eval_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    from moe_sae.eval.fidelity import run_fidelity
    from moe_sae.eval.reconstruction import evaluate_reconstruction
    from moe_sae.model_loading import pick_device
    from moe_sae.sae import load_condition_from_run

    device = str(pick_device(args.device))
    condition, theta_star, _extras = load_condition_from_run(run_dir, device=device)

    metrics = evaluate_reconstruction(
        condition, theta_star, args.capture, args.split, args.dump, device=device
    )
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(
        f"[{args.split}] local micro NMSE={metrics['local_micro_nmse']}, "
        f"aggregate NMSE={metrics['aggregate_nmse']}, "
        f"mean L0={metrics['mean_l0']['overall']:.2f}, "
        f"dead={metrics['dead_feature_pct']:.2f}%"
    )

    if args.fidelity:
        from itertools import chain

        from moe_sae.data.evalsets import iter_eval_examples
        from moe_sae.model_loading import load_model, load_processor

        model, _info = load_model(device=device)
        processor = load_processor()
        modes = [m.strip() for m in args.fidelity.split(",") if m.strip()]
        for mode in modes:
            examples = chain.from_iterable(
                iter_eval_examples(dom, args.eval_examples, seed=args.eval_seed)
                for dom in FIDELITY_DOMAINS
            )
            result = run_fidelity(
                model, processor, condition, theta_star, examples, mode=mode, device=device
            )
            path = out_dir / f"fidelity_{mode}.json"
            path.write_text(json.dumps(result, indent=2))
            print(f"[fidelity {mode}] overall ce_delta={result['overall']['ce_delta']}")


if __name__ == "__main__":
    main()
