#!/usr/bin/env python
"""Feature-subtraction intervention (spec §17, §20 steps 9–10).

Defines the frozen ablation set J on the DEV-split FeatureDump, builds matched
control sets, then runs the subtraction intervention on untouched evaluation
examples — optionally repeating the identical pipeline on the condition-C
negative-control run (changelog #10).

Usage:
    uv run python scripts/run_intervention.py \
        --run runs/<name> --dev-dump dumps/<name>/dev --capture captures/<name> \
        --out reports/intervention_<name>/ \
        [--stats stats/<capture-name>] \
        [--negative-control-run runs/<C-name>]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

INTERVENTION_DOMAINS = ("coco", "textvqa", "chartqa", "ai2d", "text_control")


def load_theta_star(run_dir: Path) -> float:
    raw = json.loads((run_dir / "final" / "theta_star.json").read_text())
    return float(raw["theta_star"]) if isinstance(raw, dict) else float(raw)


def run_pipeline(run_dir: Path, dev_dump: Path, stats, out_dir: Path, args, device: str) -> None:
    from itertools import chain

    from moe_sae.data.evalsets import iter_eval_examples
    from moe_sae.eval.intervention import (
        matched_control_sets,
        run_intervention,
        save_ablation_set,
        select_enriched_latents,
    )
    from moe_sae.model_loading import load_model
    from moe_sae.sae import conditions

    out_dir.mkdir(parents=True, exist_ok=True)
    theta_star = load_theta_star(run_dir)
    condition = conditions.load_checkpoint(run_dir, device=device)

    # Step 9: enriched-set definition on the DEV split; frozen before eval data.
    J = select_enriched_latents(
        dev_dump,
        stats,
        f=condition.f,
        top_n_experts=args.top_n_experts,
        quantile=args.quantile,
        load_floor=args.load_floor,
    )
    save_ablation_set(J, out_dir / "ablation_set.json")
    controls, match_report = matched_control_sets(
        J, dev_dump, stats, f=condition.f, n_controls=args.n_controls, seed=args.seed
    )
    (out_dir / "matched_controls.json").write_text(
        json.dumps({"controls": [[[b, l] for b, l in c] for c in controls],
                    "report": match_report}, indent=2)
    )
    print(f"[{run_dir.name}] |J| = {len(J)}, {len(controls)} matched control sets")

    # Step 10: intervention on untouched evaluation examples.
    model, _info = load_model(device=device)

    def examples():
        return chain.from_iterable(
            iter_eval_examples(dom, args.eval_examples, seed=args.eval_seed)
            for dom in INTERVENTION_DOMAINS
        )

    result = run_intervention(model, condition, theta_star, J, examples(), device=device)
    (out_dir / "intervention.json").write_text(json.dumps(result, indent=2))
    print(f"[{run_dir.name}] enriched-set ce_delta={result['overall']['ce_delta']}")

    for i, control in enumerate(controls):
        res = run_intervention(model, condition, theta_star, control, examples(), device=device)
        (out_dir / f"control_{i:02d}.json").write_text(json.dumps(res, indent=2))
        print(f"[{run_dir.name}] control {i:02d} ce_delta={res['overall']['ce_delta']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="run directory (runs/<name>)")
    ap.add_argument("--dev-dump", required=True, help="DEV-split FeatureDump dir")
    ap.add_argument("--capture", required=True, help="capture root (locates default stats)")
    ap.add_argument("--out", required=True, help="report dir (reports/intervention_<name>/)")
    ap.add_argument("--stats", default=None, help="stats dir (default stats/<capture-name>)")
    ap.add_argument("--negative-control-run", default=None, help="condition-C run dir")
    ap.add_argument("--negative-control-dump", default=None,
                    help="C run's DEV dump (default dumps/<C-name>/dev)")
    ap.add_argument("--top-n-experts", type=int, default=6)
    ap.add_argument("--quantile", type=float, default=0.9)
    ap.add_argument("--load-floor", type=int, default=500_000)
    ap.add_argument("--n-controls", type=int, default=10)
    ap.add_argument("--eval-examples", type=int, default=200, help="examples per domain")
    ap.add_argument("--eval-seed", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0, help="matched-control RNG seed")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    from moe_sae.model_loading import pick_device
    from moe_sae.train.stats_pass import load_stats

    device = str(pick_device(args.device))
    stats_dir = Path(args.stats) if args.stats else Path("stats") / Path(args.capture).name
    stats = load_stats(stats_dir)

    out_root = Path(args.out)
    run_pipeline(Path(args.run), Path(args.dev_dump), stats, out_root, args, device)

    if args.negative_control_run:
        c_run = Path(args.negative_control_run)
        c_dump = (
            Path(args.negative_control_dump)
            if args.negative_control_dump
            else Path("dumps") / c_run.name / "dev"
        )
        run_pipeline(c_run, c_dump, stats, out_root / "negative_control_C", args, device)


if __name__ == "__main__":
    main()
