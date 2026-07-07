#!/usr/bin/env python
"""Run the spec §14–§16 analyses over FeatureDumps and write JSON + summary.

Usage:
    uv run python scripts/run_analysis.py \
        --dump dumps/<run>/dev --stats stats/<name> \
        --out reports/analysis_<run>/ [--load-floor 500000] \
        [--train-dump dumps/<run>/train] \
        [--compare-dump dumps/<C-run>/dev] \
        [--seed-dumps dumps/<A-seed2>/dev dumps/<A-seed3>/dev] \
        [--condition A] [--f 4096] [--seed 0] [--n-shuffles 1000] \
        [--n-boot 1000] [--probe-c 1.0] [--include-eval-split]

--dump is the dev/primary dump: §14–§16.3 run on it, and the §16.4 probes are
scored on its dev-split tokens. The probes are FIT on --train-dump (a
TRAIN-split dump from the same run); without --train-dump the probes are
skipped with a logged notice. --include-eval-split additionally scores probes
on the frozen eval split (example_id % 100 in 90–99) — POST-INTERVENTION ONLY:
spec §17/§20 require evaluation data untouched until the intervention runs.
--condition/--f override missing meta.json fields (applied to --dump,
--train-dump and --seed-dumps, which share the run's condition; --compare-dump
is condition C and must carry its own meta.json).

Outputs: mass.json, enrichment.json, stability.json, probes.json, summary.md.
For global conditions (B/F) the bank-structured analyses (§14–§16.3) are
undefined and recorded as null with a note; the §16.4 probes still run
(minus the bank-mass probes).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from moe_sae.analysis.enrichment import enrichment_analysis, load_stats
from moe_sae.analysis.mass import (
    load_dump,
    modality_masks,
    per_example_masses,
    q_by_domain,
    q_by_modality,
    token_bank_masses,
)
from moe_sae.analysis.probes import run_probes
from moe_sae.analysis.stability import observed_q, stability_analysis

GLOBAL_NOTE = "global condition: single dictionary, no bank structure (spec §14)"


def _q_json(q_dict: dict) -> dict:
    return {k: (v.tolist() if v is not None else None) for k, v in q_dict.items()}


def _fmt(x, digits: int = 4) -> str:
    if x is None:
        return "—"
    return f"{x:.{digits}f}"


def _write_summary(out: Path, args, dump, mass_report, enr, stab, prob) -> None:
    lines = [
        f"# Analysis summary — {args.dump}",
        "",
        f"- condition: **{dump.condition}**, tokens: {dump.n_tokens} "
        f"({mass_report['n_tokens_image']} image / {mass_report['n_tokens_text']} text), "
        f"examples: {mass_report['n_examples']}",
        f"- stats: `{args.stats}`, load floor: {args.load_floor}, seed: {args.seed}",
        "",
    ]
    if stab is not None:
        sep = stab["separation"]
        lines += [
            "## §16.1 image-vs-text separation (JSD, base 2)",
            "",
            "| statistic | value |",
            "|---|---|",
            f"| JSD(q_image, q_text) | {_fmt(sep['jsd_observed'])} "
            f"[{_fmt(sep['jsd_ci_low'])}, {_fmt(sep['jsd_ci_high'])}] |",
            f"| shuffled-label null (mean / p95) | {_fmt(sep['shuffled_null_mean'])} / "
            f"{_fmt(sep['shuffled_null_p95'])} |",
            f"| permutation p-value | {_fmt(sep['shuffled_null_p_value'])} |",
            f"| gate-mass JSD | {_fmt(sep['gate_mass_jsd'])} |",
        ]
        if sep["compare_dump"] is not None:
            c = sep["compare_dump"]
            lines.append(
                f"| condition-C dump JSD | {_fmt(c['jsd_observed'])} "
                f"[{_fmt(c['jsd_ci_low'])}, {_fmt(c['jsd_ci_high'])}] |"
            )
        lines += ["", "## §16.2 concentration", "", "| modality | H_norm | top-1 | top-3 | top-6 | eff. experts |", "|---|---|---|---|---|---|"]
        for mod in ("image", "text"):
            cc = stab["concentration"][mod]
            lines.append(
                f"| {mod} | {_fmt(cc['H_norm'])} | {_fmt(cc['top1_mass'])} | "
                f"{_fmt(cc['top3_mass'])} | {_fmt(cc['top6_mass'])} | "
                f"{_fmt(cc['effective_experts'], 2)} |"
            )
        lines += ["", "## §16.3 stability (top-6 image-enriched, above load floor)", "", "| set | banks |", "|---|---|"]
        for name, banks in stab["top_enriched"].items():
            lines.append(f"| {name} | {banks} |")
        if stab["jaccard"]:
            mean_j = sum(stab["jaccard"].values()) / len(stab["jaccard"])
            lines.append(f"\nMean pairwise Jaccard: {_fmt(mean_j)}")
    else:
        lines += ["## §14–§16.3", "", GLOBAL_NOTE, ""]
    if "skipped" in prob:
        lines += ["", "## §16.4 incremental modality information", "", f"Skipped: {prob['skipped']}"]
    else:
        lines += ["", "## §16.4 incremental modality information", "", "| probe | split | accuracy | ROC-AUC | nnz coef |", "|---|---|---|---|---|"]
        for name, res in prob["probes"].items():
            if res is None or "skipped" in res:
                lines.append(f"| {name} | — | skipped | — | — |")
                continue
            nnz = res.get("n_nonzero_coef", "—")
            for part in ("dev", "eval"):
                r = res[part]
                if r is None or "skipped" in r:
                    continue
                lines.append(
                    f"| {name} | {part} | {_fmt(r['accuracy'])} | {_fmt(r['auc'])} | {nnz} |"
                )
        if not prob["include_eval_split"]:
            lines.append("\nEval split not scored (pre-intervention; spec §17/§20).")
        tb = prob["splits"]["train"]
        db = prob["splits"]["dev"]
        lines.append(
            f"\nClass balance (frac visual): train {_fmt(tb['frac_visual'])} "
            f"({tb['n_tokens']} tokens), dev {_fmt(db['frac_visual'])} "
            f"({db['n_tokens']} tokens)."
        )
    (out / "summary.md").write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump", required=True, help="dev/primary dump: dumps/<run>/dev")
    ap.add_argument("--stats", required=True, help="stats artifact (dir or .safetensors)")
    ap.add_argument("--out", required=True, help="report output directory")
    ap.add_argument("--load-floor", type=int, default=500_000)
    ap.add_argument(
        "--train-dump", default=None,
        help="TRAIN-split dump the §16.4 probes are fit on; probes are skipped when absent",
    )
    ap.add_argument(
        "--include-eval-split", action="store_true",
        help="also score probes on the frozen eval split — POST-INTERVENTION ONLY (spec §17/§20)",
    )
    ap.add_argument("--compare-dump", default=None, help="condition-C dump for §16.1")
    ap.add_argument("--seed-dumps", nargs="*", default=[], help="replicate-seed dumps for §16.3")
    ap.add_argument("--condition", default=None, help="override dump meta condition (metaless dumps)")
    ap.add_argument("--f", type=int, default=None, help="override dump meta per-bank width f (metaless dumps)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-shuffles", type=int, default=1000)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--probe-c", type=float, default=1.0)
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dump = load_dump(args.dump, condition=args.condition, f=args.f)
    stats = load_stats(args.stats)
    masses = token_bank_masses(dump)
    mods = modality_masks(dump)

    # ---- §14 masses ----
    mass_report: dict = {
        "condition": dump.condition,
        "n_tokens": dump.n_tokens,
        "n_tokens_image": int(mods["image"].sum()),
        "n_tokens_text": int(mods["text"].sum()),
        "n_examples": int(np.unique(dump.example_id).size),
    }
    if masses is not None:
        kinds = ["act", "count"] + (["energy"] if masses.m_energy is not None else [])
        mass_report["q"] = {k: _q_json(q_by_modality(masses, dump, k)) for k in kinds}
        mass_report["q_by_domain"] = {
            dom: _q_json(per_mod) for dom, per_mod in q_by_domain(masses, dump, "act").items()
        }
        mass_report["has_energy"] = masses.m_energy is not None
    else:
        mass_report["q"] = None
        mass_report["note"] = GLOBAL_NOTE
    (out / "mass.json").write_text(json.dumps(mass_report, indent=2))

    # ---- §15 enrichment + §16.1–§16.3 stability ----
    enr = None
    stab = None
    if masses is not None:
        enr = enrichment_analysis(q_by_modality(masses, dump, "act"), stats, args.load_floor)

        compare_ex = None
        if args.compare_dump:
            cdump = load_dump(args.compare_dump)
            cmasses = token_bank_masses(cdump)
            if cmasses is not None:
                compare_ex = per_example_masses(cmasses.m_act, cdump)

        seed_q = {}
        for sd in args.seed_dumps:
            sdump = load_dump(sd, condition=args.condition, f=args.f)
            smasses = token_bank_masses(sdump)
            if smasses is None:
                continue
            sq, _ = observed_q(per_example_masses(smasses.m_act, sdump))
            if sq is not None:
                seed_q[Path(sd).parent.name or str(sd)] = sq

        stab = stability_analysis(
            dump,
            masses,
            stats,
            load_floor=args.load_floor,
            seed=args.seed,
            n_shuffles=args.n_shuffles,
            n_boot=args.n_boot,
            compare_example_masses=compare_ex,
            seed_q_image=seed_q or None,
        )
    (out / "enrichment.json").write_text(
        json.dumps(enr if enr is not None else {"condition": dump.condition, "note": GLOBAL_NOTE}, indent=2)
    )
    (out / "stability.json").write_text(
        json.dumps(stab if stab is not None else {"condition": dump.condition, "note": GLOBAL_NOTE}, indent=2)
    )

    # ---- §16.4 probes (all conditions; fit on --train-dump, scored on --dump) ----
    if args.train_dump:
        tdump = load_dump(args.train_dump, condition=args.condition, f=args.f)
        prob = run_probes(
            tdump,
            token_bank_masses(tdump),
            dump,
            masses,
            c=args.probe_c,
            seed=args.seed,
            include_eval_split=args.include_eval_split,
        )
    else:
        prob = {
            "skipped": "--train-dump not provided; §16.4 probes fit on a "
            "TRAIN-split FeatureDump (dumps/<run>/train)"
        }
        print(f"probes skipped: {prob['skipped']}")
    (out / "probes.json").write_text(json.dumps(prob, indent=2))

    _write_summary(out, args, dump, mass_report, enr, stab, prob)
    print(f"analysis written to {out}")


if __name__ == "__main__":
    main()
