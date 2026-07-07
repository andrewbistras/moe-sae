"""Cross-backend run comparison (smoke ladder rung 4).

Aligns two runs' metrics.jsonl by step and compares the loss trajectories and
frozen θ*. Both runs must have consumed identical batch sequences (the JAX
trainer reuses the torch BatchComposer precisely so this holds); remaining
differences are numerics (bf16 kernel order, MPS vs TPU) and should be small
and non-systematic.

Exit 0 iff: median rel diff of l_local ≤ --tol-median (default 2%), final
l_local rel diff ≤ --tol-final (default 5%), θ* rel diff ≤ --tol-theta (5%).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_metrics(run_dir: Path) -> tuple[dict[int, dict], float]:
    lines = {}
    for line in (run_dir / "metrics.jsonl").read_text().splitlines():
        rec = json.loads(line)
        lines[rec["step"]] = rec
    theta = json.loads((run_dir / "final" / "theta_star.json").read_text())
    theta_star = theta["theta_star"] if isinstance(theta, dict) else float(theta)
    return lines, theta_star


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    ap.add_argument("--tol-median", type=float, default=0.02)
    ap.add_argument("--tol-final", type=float, default=0.05)
    ap.add_argument("--tol-theta", type=float, default=0.05)
    args = ap.parse_args()

    ma, ta = load_metrics(Path(args.run_a))
    mb, tb = load_metrics(Path(args.run_b))
    steps = sorted(set(ma) & set(mb))
    assert steps, "no overlapping logged steps"

    rels = []
    for s in steps:
        a, b = ma[s]["l_local"], mb[s]["l_local"]
        rels.append(abs(a - b) / max(abs(b), 1e-12))
    rels.sort()
    median = rels[len(rels) // 2]
    final = abs(ma[steps[-1]]["l_local"] - mb[steps[-1]]["l_local"]) / max(
        abs(mb[steps[-1]]["l_local"]), 1e-12
    )
    theta_rel = abs(ta - tb) / max(abs(tb), 1e-12)

    print(f"steps compared: {len(steps)}")
    print(f"l_local rel diff: median {median:.4f}, final {final:.4f}")
    print(f"theta*: {ta:.6g} vs {tb:.6g} (rel {theta_rel:.4f})")

    ok = median <= args.tol_median and final <= args.tol_final and theta_rel <= args.tol_theta
    print("VERDICT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
