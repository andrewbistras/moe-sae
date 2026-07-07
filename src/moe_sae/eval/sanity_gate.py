"""A-vs-C implementation sanity gate (spec §20 step 5, changelog #14).

At seed 1, condition A must beat condition C on local micro-averaged NMSE by
≥10% relative: NMSE_A ≤ 0.9 · NMSE_C. This is an **implementation gate, not a
hypothesis test**: failure halts the study for an implementation audit and is
recorded as such — it is never reported as a scientific result.
"""

from __future__ import annotations

import json
from pathlib import Path

GATE_RATIO = 0.9  # frozen: A must beat C by ≥10% relative (spec §20.5)
METRIC_KEY = "local_micro_nmse"


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
        "rule": f"PASS iff NMSE_A <= {GATE_RATIO} * NMSE_C (spec §20 step 5)",
        "nmse_a": nmse_a,
        "nmse_c": nmse_c,
        "ratio_a_over_c": nmse_a / nmse_c if nmse_c else None,
        "passed": bool(passed),
        "metrics_a": str(path_a),
        "metrics_c": str(path_c),
        "interpretation": (
            "PASS: condition A beats the random-routing control decisively; "
            "the implementation gate is satisfied."
            if passed
            else "FAIL: the study is HALTED for an implementation audit. This is "
            "an implementation gate, not a hypothesis test — failure indicates "
            "an implementation fault, NOT a scientific result (spec §20 step 5, "
            "changelog #14)."
        ),
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(verdict, indent=2))
    return verdict
