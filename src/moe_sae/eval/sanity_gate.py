"""A-vs-C gate (spec §20 step 5; v0.1.2 addendum A10 audit-first semantics).

At discovery seed 1, condition A must beat condition C on local micro-averaged
NMSE by ≥10% relative: NMSE_A ≤ 0.9 · NMSE_C.

Failure semantics (A10, committee-amended — Reviewer S; protocol Reviewer F):
failure does NOT mean "bug, by definition" and does NOT halt the study
outright. It prescribes the audit protocol:

1. an implementation audit runs — the full regression suite plus a
   cross-backend replication of the A-vs-C comparison on the second compute
   lane with the parity suite green;
2. if a fault is found: fix, document, re-run the gate;
3. if no fault is found and both independent backends agree: THE STUDY
   CONTINUES, the result is recorded as evidence against the
   stable-router-partition hypothesis, and the confirmatory trigger (A18) is
   evaluated normally at its appointed step.

Audit-first converts the gate from a tautology into a measurement (A10
reason); the report emitted here prescribes that protocol rather than a halt.
"""

from __future__ import annotations

import json
from pathlib import Path

GATE_RATIO = 0.9  # frozen: A must beat C by ≥10% relative (spec §20.5)
METRIC_KEY = "local_micro_nmse"

AUDIT_PROTOCOL = (
    "A10 audit protocol: (1) run the full regression suite AND a "
    "cross-backend replication of the A-vs-C comparison on the second "
    "compute lane with the 21-test parity suite green; (2) if a fault is "
    "found — fix, document, re-run the gate; (3) if no fault is found and "
    "both independent backends agree — the study CONTINUES, the result is "
    "recorded as evidence against the stable-router-partition hypothesis, "
    "and the A18 confirmatory trigger is evaluated normally at its "
    "appointed step."
)


def _find_metrics(run_dir: str | Path) -> Path:
    """Locate a run's eval metrics.json. Accepts a direct path to the file, or
    a directory searched at the conventional locations."""
    p = Path(run_dir)
    if p.is_file():
        return p
    for candidate in (p / "metrics.json", p / "eval" / "metrics.json"):
        if candidate.exists():
            return candidate
    hits = sorted(p.glob("**/metrics.json"))
    assert hits, f"no metrics.json found under {p}"
    return hits[0]


def compare_sanity(
    run_dir_A: str | Path,
    run_dir_C: str | Path,
    out_path: str | Path = "reports/sanity_gate.json",
) -> dict:
    """Read both runs' eval metrics and apply the frozen gate. Writes the
    verdict to `out_path` and returns it."""
    path_a = _find_metrics(run_dir_A)
    path_c = _find_metrics(run_dir_C)
    nmse_a = json.loads(path_a.read_text())[METRIC_KEY]
    nmse_c = json.loads(path_c.read_text())[METRIC_KEY]
    assert nmse_a is not None and nmse_c is not None, (
        f"{METRIC_KEY} missing — the gate is defined on router-tied conditions only"
    )

    passed = nmse_a <= GATE_RATIO * nmse_c
    verdict = {
        "gate": "A_vs_C_local_micro_nmse",
        "rule": f"PASS iff NMSE_A <= {GATE_RATIO} * NMSE_C (spec §20 step 5; A10)",
        "nmse_a": nmse_a,
        "nmse_c": nmse_c,
        "ratio_a_over_c": nmse_a / nmse_c if nmse_c else None,
        "passed": bool(passed),
        "metrics_a": str(path_a),
        "metrics_c": str(path_c),
        "interpretation": (
            "PASS: condition A beats the random-routing control by >=10% "
            "relative; the §20.5 gate is satisfied."
            if passed
            else "FAIL: an implementation audit is required before the study "
            "proceeds — audit-first, NOT halt-by-definition (A10, v0.1.2 "
            "addendum). " + AUDIT_PROTOCOL
        ),
    }
    if not passed:
        verdict["audit_protocol"] = AUDIT_PROTOCOL
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(verdict, indent=2))
    return verdict
