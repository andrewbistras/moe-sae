#!/usr/bin/env python
"""A18 confirmatory-trigger evaluation (v0.1.2 addendum; §20 step 8+).

Mechanically evaluates the frozen two-route trigger rule — on the DEVELOPMENT
split only, with the evaluation split untouched — from run/eval/probe
artifacts, and records inputs + verdict in reports/confirmatory_trigger.json.
The verdict is posted as a registration addendum BEFORE any confirmatory
launch (registration requirement 7).

Frozen rule (A18): the confirmatory run proceeds iff the §20.5 gate (per A10
semantics) has passed AND at least one route holds:

- ARCHITECTURE route: A beats C on local micro NMSE by ≥10% relative in ≥2 of
  3 seeds, AND A ≥ B on aggregate NMSE or on matched-position downstream
  fidelity in ≥2 of 3 seeds.
- REPRESENTATION route: A beats C on local micro NMSE by ≥10% relative in ≥2
  of 3 seeds, AND the frozen representation criterion holds: the held-out
  (dev, example-level) accuracy advantage of probe 2, 3, or 4 over probe 1
  (router-only) has a 95% example-level bootstrap CI excluding zero in ≥2 of
  3 seeds, AND that advantage exceeds the corresponding condition-C advantage
  (point estimate) in those seeds.

If neither route triggers: discovery is reported as final; the negative
interpretation remains UNCLAIMED under the power clause; confirmation is
deferred (disclosed asymmetry: confirmation is reachable only from
positive-trending discovery).

Input artifact expectations:
- --gate: reports/sanity_gate.json (moe_sae.eval.sanity_gate output).
- --a-metrics/--c-metrics/--b-metrics: per-seed DEV-split eval metrics.json
  (moe_sae.eval.reconstruction output; "split" is checked to be "dev" when
  present). Metric keys: local_micro_nmse (A vs C), aggregate_nmse (A vs B;
  lower is better).
- --a-fidelity/--b-fidelity (optional): per-seed matched-mode fidelity
  reports (moe_sae.eval.fidelity output); "A ≥ B on matched-position
  downstream fidelity" is read as A's overall teacher-forced ce_delta ≤ B's
  (smaller CE degradation = higher fidelity).
- --a-probes/--c-probes (optional; required for the representation route):
  per-seed probes.json each carrying `advantages_vs_router` —
  {probe_name: {"point": x, "ci_low": lo, "ci_high": hi}} for the dev-split
  example-level bootstrap of the probe's accuracy advantage over probe 1
  (router-only). Probe names considered: sae_bank_mass (probe 2),
  feature_identity (probe 3), gates_plus_bank_mass / gates_plus_identity
  (probe 4 combinations).

Usage:
    uv run python scripts/evaluate_trigger.py \
        --gate reports/sanity_gate.json \
        --a-metrics runs/A_s1/eval_dev/metrics.json ... (3 seeds) \
        --c-metrics ... --b-metrics ... \
        [--a-fidelity ... --b-fidelity ...] \
        [--a-probes ... --c-probes ...] \
        --out reports/confirmatory_trigger.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

GATE_RATIO = 0.9  # frozen (§20.5 / A18): ≥10% relative on local micro NMSE
SEEDS_REQUIRED = 2  # "in ≥2 of 3 seeds"
PROBE_ADVANTAGE_KEYS = (
    "sae_bank_mass",  # probe 2
    "feature_identity",  # probe 3
    "gates_plus_bank_mass",  # probe 4 (1)+(2)
    "gates_plus_identity",  # probe 4 (1)+(3)
)


def _load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def _check_dev_split(record: dict, path: str) -> None:
    split = record.get("split")
    assert split in (None, "dev"), (
        f"{path}: A18 is evaluated on the DEVELOPMENT split only "
        f"(got split={split!r}); the evaluation split stays untouched"
    )


def nmse_route_condition(a_metrics: list[dict], c_metrics: list[dict]) -> dict:
    """Shared premise of both routes: A ≤ 0.9·C on local micro NMSE per seed;
    holds in ≥2 of 3 seeds."""
    per_seed = []
    for a, c in zip(a_metrics, c_metrics):
        na, nc = a["local_micro_nmse"], c["local_micro_nmse"]
        per_seed.append(
            {"nmse_a": na, "nmse_c": nc, "passes": bool(na <= GATE_RATIO * nc)}
        )
    n = sum(s["passes"] for s in per_seed)
    return {"per_seed": per_seed, "n_seeds_passing": n, "holds": n >= SEEDS_REQUIRED}


def architecture_route(
    a_metrics: list[dict],
    b_metrics: list[dict] | None,
    a_fidelity: list[dict] | None,
    b_fidelity: list[dict] | None,
) -> dict:
    """Second clause of the architecture route: A ≥ B on aggregate NMSE or on
    matched-position downstream fidelity, in ≥2 of 3 seeds."""
    agg = None
    if b_metrics is not None:
        per_seed = []
        for a, b in zip(a_metrics, b_metrics):
            per_seed.append(
                {
                    "aggregate_nmse_a": a["aggregate_nmse"],
                    "aggregate_nmse_b": b["aggregate_nmse"],
                    # lower aggregate NMSE = better: "A ≥ B" in quality
                    "passes": bool(a["aggregate_nmse"] <= b["aggregate_nmse"]),
                }
            )
        n = sum(s["passes"] for s in per_seed)
        agg = {"per_seed": per_seed, "n_seeds_passing": n, "holds": n >= SEEDS_REQUIRED}

    fid = None
    if a_fidelity is not None and b_fidelity is not None:
        per_seed = []
        for a, b in zip(a_fidelity, b_fidelity):
            da, db = a["overall"]["ce_delta"], b["overall"]["ce_delta"]
            # smaller teacher-forced CE degradation = higher fidelity
            per_seed.append({"ce_delta_a": da, "ce_delta_b": db, "passes": bool(da <= db)})
        n = sum(s["passes"] for s in per_seed)
        fid = {"per_seed": per_seed, "n_seeds_passing": n, "holds": n >= SEEDS_REQUIRED}

    holds = bool((agg is not None and agg["holds"]) or (fid is not None and fid["holds"]))
    return {"aggregate_nmse": agg, "matched_fidelity": fid, "second_clause_holds": holds}


def representation_route(
    a_probes: list[dict] | None, c_probes: list[dict] | None
) -> dict:
    """Frozen representation criterion: for some probe in {2,3,4}, the A-run
    dev advantage over probe 1 has a 95% bootstrap CI excluding zero in ≥2/3
    seeds AND exceeds C's point-estimate advantage in those seeds."""
    if a_probes is None or c_probes is None:
        return {"evaluated": False, "second_clause_holds": False,
                "note": "probe artifacts not provided; representation route not evaluated"}
    per_probe = {}
    for key in PROBE_ADVANTAGE_KEYS:
        per_seed = []
        for a, c in zip(a_probes, c_probes):
            adv_a = a.get("advantages_vs_router", {}).get(key)
            adv_c = c.get("advantages_vs_router", {}).get(key)
            if adv_a is None or adv_c is None:
                per_seed.append({"available": False, "passes": False})
                continue
            ci_excludes_zero = bool(adv_a["ci_low"] > 0.0 or adv_a["ci_high"] < 0.0)
            positive = bool(adv_a["ci_low"] > 0.0)  # advantage must be positive
            beats_c = bool(adv_a["point"] > adv_c["point"])
            per_seed.append(
                {
                    "available": True,
                    "point_a": adv_a["point"],
                    "ci_a": [adv_a["ci_low"], adv_a["ci_high"]],
                    "point_c": adv_c["point"],
                    "ci_excludes_zero": ci_excludes_zero,
                    "beats_condition_c": beats_c,
                    "passes": bool(positive and ci_excludes_zero and beats_c),
                }
            )
        n = sum(s["passes"] for s in per_seed)
        per_probe[key] = {
            "per_seed": per_seed,
            "n_seeds_passing": n,
            "holds": n >= SEEDS_REQUIRED,
        }
    holds = any(p["holds"] for p in per_probe.values())
    return {"evaluated": True, "per_probe": per_probe, "second_clause_holds": holds}


def evaluate_trigger(
    gate: dict,
    a_metrics: list[dict],
    c_metrics: list[dict],
    b_metrics: list[dict] | None = None,
    a_fidelity: list[dict] | None = None,
    b_fidelity: list[dict] | None = None,
    a_probes: list[dict] | None = None,
    c_probes: list[dict] | None = None,
) -> dict:
    """Apply the frozen A18 two-route rule mechanically. Returns the full
    verdict record (inputs echoed) for reports/confirmatory_trigger.json."""
    gate_passed = bool(gate.get("passed"))
    nmse = nmse_route_condition(a_metrics, c_metrics)
    arch = architecture_route(a_metrics, b_metrics, a_fidelity, b_fidelity)
    repr_ = representation_route(a_probes, c_probes)

    architecture_triggers = bool(nmse["holds"] and arch["second_clause_holds"])
    representation_triggers = bool(nmse["holds"] and repr_["second_clause_holds"])
    proceed = bool(gate_passed and (architecture_triggers or representation_triggers))

    return {
        "rule": "A18 (v0.1.2): proceed iff §20.5 gate passed (A10 semantics) "
        "and (architecture route or representation route) holds; evaluated on "
        "the dev split only, evaluation split untouched",
        "gate_passed": gate_passed,
        "nmse_condition": nmse,
        "architecture_route": {**arch, "triggers": architecture_triggers},
        "representation_route": {**repr_, "triggers": representation_triggers},
        "confirmatory_run_proceeds": proceed,
        "verdict": (
            "TRIGGERED: the confirmatory run proceeds; record this verdict as "
            "a registration addendum BEFORE launch (A18)."
            if proceed
            else "NOT TRIGGERED: discovery is reported as final; the negative "
            "interpretation remains unclaimed under the power clause; "
            "confirmation is deferred to future work (A18 disclosed "
            "asymmetry: confirmation is reachable only from "
            "positive-trending discovery)."
        ),
    }


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gate", required=True, help="reports/sanity_gate.json")
    ap.add_argument("--a-metrics", nargs=3, required=True, help="A dev metrics.json, seeds 1-3")
    ap.add_argument("--c-metrics", nargs=3, required=True, help="C dev metrics.json, seeds 1-3")
    ap.add_argument("--b-metrics", nargs=3, default=None, help="B dev metrics.json, seeds 1-3")
    ap.add_argument("--a-fidelity", nargs=3, default=None, help="A matched fidelity json, seeds 1-3")
    ap.add_argument("--b-fidelity", nargs=3, default=None, help="B matched fidelity json, seeds 1-3")
    ap.add_argument("--a-probes", nargs=3, default=None, help="A probes.json with advantages_vs_router")
    ap.add_argument("--c-probes", nargs=3, default=None, help="C probes.json with advantages_vs_router")
    ap.add_argument("--out", default="reports/confirmatory_trigger.json")
    args = ap.parse_args(argv)

    def load_list(paths):
        if paths is None:
            return None
        out = []
        for p in paths:
            rec = _load(p)
            _check_dev_split(rec, p)
            out.append(rec)
        return out

    gate = _load(args.gate)
    verdict = evaluate_trigger(
        gate,
        load_list(args.a_metrics),
        load_list(args.c_metrics),
        load_list(args.b_metrics),
        load_list(args.a_fidelity),
        load_list(args.b_fidelity),
        load_list(args.a_probes),
        load_list(args.c_probes),
    )
    record = {
        "inputs": {
            "gate": args.gate,
            "a_metrics": list(args.a_metrics),
            "c_metrics": list(args.c_metrics),
            "b_metrics": list(args.b_metrics) if args.b_metrics else None,
            "a_fidelity": list(args.a_fidelity) if args.a_fidelity else None,
            "b_fidelity": list(args.b_fidelity) if args.b_fidelity else None,
            "a_probes": list(args.a_probes) if args.a_probes else None,
            "c_probes": list(args.c_probes) if args.c_probes else None,
        },
        **verdict,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2))
    print(record["verdict"])
    return record


if __name__ == "__main__":
    main()
