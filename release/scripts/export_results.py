#!/usr/bin/env python3
"""Export the paper's aggregate results from the sealed closure evidence.

    uv run python release/scripts/export_results.py \\
        --evidence release/results/component-closure-score-grid.json \\
        --report release/results/component-closure-report.json \\
        --output-dir release/results --check-paper

The evidence file (the score grid) holds per-sequence errors for all 72
evaluated cells and the training cost of all 24 runs. This script validates
it, rebuilds the report from it with the repository's own sealing code
(seeded, so the bootstrap intervals are reproducible), confirms the rebuilt
report matches the sealed one, and writes CSV tables. With --check-paper it
also compares every number printed in the paper with the evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from moe_sae.component_closure_config import CHECKPOINT_TOKENS, CONDITIONS, PANELS, SEEDS  # noqa: E402
from moe_sae.component_closure_report import (  # noqa: E402
    EVIDENCE_SCHEMA_VERSION,
    REPORT_SCHEMA_VERSION,
    build_closure_report,
    load_score_grid,
)
from moe_sae.contribution_artifacts import canonical_sha256  # noqa: E402

# Identities printed in the paper's appendix.
PAPER_REPORT_SHA256 = "fc1d6d4112dbd2f5cea075e1c257424c7fc6e917d66b9e6c5d42814a0940f460"
PAPER_EVIDENCE_SHA256 = "4b8bf760befc6b8dbca2efe209e5de9c1ed47ce67f46e3a1cd9e0dbc9370329f"

# Numbers printed in the paper, as printed.
PAPER_TABLE2 = {  # 10M: EV, and A minus layout as (estimate, low, high)
    "sae": {"A": ("0.518", None), "F": ("0.481", ("0.038", "0.035", "0.040")),
            "B": ("0.458", ("0.061", "0.059", "0.062")), "R": ("0.387", ("0.131", "0.127", "0.135"))},
    "transcoder": {"A": ("0.432", None), "F": ("0.347", ("0.085", "0.082", "0.088")),
                   "B": ("0.371", ("0.060", "0.058", "0.062")), "R": ("0.118", ("0.313", "0.308", "0.320"))},
}
PAPER_FIGURE2 = {  # three-seed EV at 2M, 5M, 10M
    "sae": {"R": ("0.2624", "0.3779", "0.3873"), "B": ("0.4069", "0.4491", "0.4576"),
            "F": ("0.4176", "0.4700", "0.4809"), "A": ("0.4452", "0.5075", "0.5184")},
    "transcoder": {"R": ("0.0116", "0.1035", "0.1183"), "B": ("0.3168", "0.3648", "0.3715"),
                   "F": ("0.2757", "0.3356", "0.3468"), "A": ("0.3579", "0.4217", "0.4317")},
}
PAPER_TABLE_A1 = {  # per-seed EV, ascending, at 2M, 5M, 10M
    "sae": {
        "A": (("0.444", "0.445", "0.447"), ("0.507", "0.508", "0.508"), ("0.518", "0.518", "0.519")),
        "F": (("0.417", "0.417", "0.418"), ("0.468", "0.471", "0.472"), ("0.478", "0.481", "0.483")),
        "B": (("0.405", "0.407", "0.409"), ("0.448", "0.449", "0.450"), ("0.457", "0.458", "0.458")),
        "R": (("0.262", "0.262", "0.263"), ("0.378", "0.378", "0.378"), ("0.387", "0.387", "0.388")),
    },
    "transcoder": {
        "A": (("0.357", "0.358", "0.359"), ("0.421", "0.422", "0.422"), ("0.431", "0.432", "0.432")),
        "F": (("0.275", "0.276", "0.276"), ("0.335", "0.336", "0.336"), ("0.345", "0.347", "0.348")),
        "B": (("0.315", "0.317", "0.318"), ("0.364", "0.365", "0.366"), ("0.371", "0.371", "0.373")),
        "R": (("0.011", "0.011", "0.013"), ("0.101", "0.104", "0.105"), ("0.117", "0.118", "0.120")),
    },
}
PAPER_A_OVER_F = {"sae": ("1.10", "1.12"), "transcoder": ("1.10", "1.16")}  # min-max over seeds


def per_seed_rows(evidence: dict) -> list[dict]:
    cells = {(c["panel"], c["seed"], c["checkpoint_tokens"], c["condition"]): c for c in evidence["cells"]}
    rows = []
    for panel in PANELS:
        for checkpoint in CHECKPOINT_TOKENS:
            for seed in SEEDS:
                # The A cell's per-sequence SST is the common denominator.
                sst = float(sum(cells[(panel, seed, checkpoint, "A")]["component_sst"]))
                for condition in CONDITIONS:
                    cell = cells[(panel, seed, checkpoint, condition)]
                    sse = float(sum(cell["component_sse"]))
                    rows.append({
                        "panel": panel, "checkpoint_tokens": checkpoint, "seed": seed, "condition": condition,
                        "ev": 1.0 - sse / sst, "sse": sse, "sst": sst,
                        "packed_sequences": len(cell["sequence_id"]),
                        "evaluation_tokens": cell["evaluation_token_count"],
                    })
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: repr(v) if isinstance(v, float) else v for k, v in row.items()})


def check_paper(report: dict, seeds: list[dict], costs: list[dict]) -> list[str]:
    problems: list[str] = []
    checked = 0

    def same(label: str, printed: str, value: float) -> None:
        nonlocal checked
        checked += 1
        digits = len(printed.split(".")[1])
        actual = f"{value:.{digits}f}"
        if actual != printed:
            problems.append(f"{label}: paper prints {printed}, evidence gives {value:.6f} -> {actual}")

    for panel in PANELS:
        milestones = report["panels"][panel]["milestones"]
        for i, checkpoint in enumerate(CHECKPOINT_TOKENS):
            row = milestones[str(checkpoint)]
            for condition in CONDITIONS:
                same(f"Figure 2 {panel} {condition} {checkpoint:,}", PAPER_FIGURE2[panel][condition][i],
                     row["component_endpoint"][condition]["ev"])
                values = sorted(r["ev"] for r in seeds if r["panel"] == panel
                                and r["checkpoint_tokens"] == checkpoint and r["condition"] == condition)
                for k, (printed, value) in enumerate(zip(PAPER_TABLE_A1[panel][condition][i], values)):
                    same(f"Table A1 {panel} {condition} {checkpoint:,} seed rank {k + 1}", printed, value)
        final = milestones[str(CHECKPOINT_TOKENS[-1])]
        for condition, (ev, margin) in PAPER_TABLE2[panel].items():
            same(f"Table 2 {panel} {condition} EV", ev, final["component_endpoint"][condition]["ev"])
            if margin:
                effect = final["contrasts"][f"A-{condition}"]
                for name, printed, value in zip(("estimate", "low", "high"), margin,
                                                (effect["estimate"], effect["ci95_low"], effect["ci95_high"])):
                    same(f"Table 2 {panel} A-{condition} {name}", printed, value)
        by_key = {(c["seed"], c["condition"]): c for c in costs if c["panel"] == panel}
        ratios = [by_key[(s, "A")]["chip_seconds"] / by_key[(s, "F")]["chip_seconds"] for s in SEEDS]
        same(f"timing {panel} A/F lowest", PAPER_A_OVER_F[panel][0], min(ratios))
        same(f"timing {panel} A/F highest", PAPER_A_OVER_F[panel][1], max(ratios))
        comparable = report["panels"][panel]["cost_compatibility"]["a_over_f_all_seeds_comparable"]
        if not comparable:
            problems.append(f"timing {panel}: the report marks A/F as not comparable on hardware or timing scope")
    print(f"paper check: {checked} printed values compared, {len(problems)} differ")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--report", type=Path, help="sealed report JSON to compare against")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--check-paper", action="store_true")
    args = parser.parse_args(argv)

    evidence = load_score_grid(args.evidence)  # validates the evidence identity
    report = build_closure_report(evidence, bootstrap_replicates=args.bootstrap_replicates)
    status = {
        "evidence_sha256": evidence["evidence_sha256"],
        "evidence_matches_paper": evidence["evidence_sha256"] == PAPER_EVIDENCE_SHA256,
        "rebuilt_report_sha256": report["report_sha256"],
        "rebuilt_report_matches_paper": report["report_sha256"] == PAPER_REPORT_SHA256,
    }
    if args.report:
        sealed = json.loads(args.report.read_text(encoding="ascii"))
        body = {k: v for k, v in sealed.items() if k != "report_sha256"}
        status["sealed_report_sha256"] = sealed["report_sha256"]
        status["sealed_report_self_consistent"] = canonical_sha256(body) == sealed["report_sha256"]
        status["rebuilt_equals_sealed"] = canonical_sha256(sealed) == canonical_sha256(report)
    for key, value in status.items():
        print(f"{key}: {value}")

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    endpoints, contrasts, ratios = [], [], []
    for panel in PANELS:
        for checkpoint in CHECKPOINT_TOKENS:
            row = report["panels"][panel]["milestones"][str(checkpoint)]
            for condition in CONDITIONS:
                e = row["component_endpoint"][condition]
                endpoints.append({"panel": panel, "checkpoint_tokens": checkpoint, "condition": condition,
                                  "ev": e["ev"], "nmse": e["nmse"], "sse": e["sse"], "sst": e["sst"]})
            for name, e in row["contrasts"].items():
                contrasts.append({"panel": panel, "checkpoint_tokens": checkpoint, "contrast": name,
                                  "estimate": e["estimate"], "ci95_low": e["ci95_low"], "ci95_high": e["ci95_high"]})
        for c in report["panels"][panel]["cost_compatibility"]["comparisons"]:
            ratios.append({"panel": panel, "seed": c["seed"], "contrast": c["contrast"],
                           "comparable": c["compatible"], "chip_seconds_ratio": c["chip_seconds_ratio"]})
    seeds = per_seed_rows(evidence)
    costs = [{k: c[k] for k in ("panel", "seed", "condition", "accelerator_type", "chip_count",
                                "marginal_wall_seconds", "chip_seconds", "training_source_revision")}
             for c in evidence["costs"]]
    write_csv(out / "endpoints.csv", endpoints)
    write_csv(out / "contrasts.csv", contrasts)
    write_csv(out / "per_seed.csv", seeds)
    write_csv(out / "training_costs.csv", costs)
    write_csv(out / "cost_ratios.csv", ratios)
    first = evidence["cells"][0]
    identities = {
        "report_sha256": report["report_sha256"],
        "evidence_sha256": evidence["evidence_sha256"],
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
        "protocol_id": report["protocol_id"],
        "bootstrap": report["bootstrap"],
        "evaluation_tokens": first["evaluation_token_count"],
        "evaluation_packed_sequences": first["evaluation_packed_sequences"],
        "evaluation_target_manifest_sha256": first["evaluation_target_manifest_sha256"],
        "training_source_revisions": sorted({c["training_source_revision"] for c in evidence["costs"]}),
        "note": "Canonical identities: SHA-256 of canonical JSON, embedded in each file. Not raw-file digests.",
    }
    (out / "identities.json").write_text(json.dumps(identities, indent=2) + "\n", encoding="utf-8")
    print(f"wrote endpoints, contrasts, per_seed, training_costs, cost_ratios CSVs and identities.json to {out}")

    ok = all(v for k, v in status.items() if isinstance(v, bool))
    if args.check_paper:
        problems = check_paper(report, seeds, evidence["costs"])
        for line in problems:
            print("  " + line)
        ok = ok and not problems
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
