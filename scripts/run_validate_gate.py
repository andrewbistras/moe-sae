"""Spec §20 step 1 HARD GATE: hook validation (FP32 gate + BF16 report, spec §3).

Runs both phases sequentially in one process — the memory-streamed FP32
reference forward first, then (after explicit memory release inside
run_validation) the BF16 production-path report — and prints a human-readable
verdict. Exit code 0 iff the gate passed.

Usage:
    uv run python scripts/run_validate_gate.py \
        --inputs validation_inputs/ --out reports/hook_validation.json \
        --tokens 100000 --device mps
"""

import argparse
import logging
import sys

from moe_sae.capture.validate import run_validation

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def _fmt(x) -> str:
    return "n/a" if x is None else f"{x:.3e}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", default="validation_inputs", help="directory of examples_*.pt")
    parser.add_argument("--out", default="reports/hook_validation.json")
    parser.add_argument("--tokens", type=int, default=100_000)
    parser.add_argument("--device", default=None, choices=["mps", "cpu", "cuda"])
    parser.add_argument("--chunk-examples", type=int, default=8,
                        help="examples per streamed forward (amortizes checkpoint IO)")
    parser.add_argument("--skip-bf16", action="store_true",
                        help="engineering knob: FP32 gate only")
    args = parser.parse_args()

    report = run_validation(
        inputs_dir=args.inputs,
        out_path=args.out,
        n_tokens=args.tokens,
        device=args.device,
        chunk_examples=args.chunk_examples,
        bf16_report=not args.skip_bf16,
    )

    fp32 = report["fp32"]
    xchk = report["production_crosscheck"]
    print()
    print("=" * 72)
    print("HOOK VALIDATION (spec §3, §20 step 1 — HARD GATE)")
    print("=" * 72)
    print(f"FP32 reference gate      : {fp32['status']}"
          f"  (n={fp32['n_tokens']}, max={_fmt(fp32['max_rel'])},"
          f" median={_fmt(fp32['median_rel'])}, tol<{fp32['tol']:.0e},"
          f" violations={fp32['n_violations']})")
    zn = fp32["zero_norm_aggregates"]
    if zn["n"]:
        print(f"  zero-norm aggregates   : {zn['n']} tokens (reported, not divided;"
              f" max ‖Σc‖ = {_fmt(zn['max_contrib_norm'])})")
    print(f"Production cross-check   : {'PASS' if xchk['pass'] else 'FAIL'}"
          f"  (max={_fmt(xchk['max_rel'])}, tol≤{xchk['tol']:.0e})")
    if "bf16" in report:
        bf16 = report["bf16"]
        print(f"BF16 report (NOT a gate) : max={_fmt(bf16['max'])},"
              f" median={_fmt(bf16['median'])}, p99={_fmt(bf16['p99'])}")
        for name, c in bf16["correlations"].items():
            rho = "n/a" if c["rho"] is None else f"{c['rho']:+.3f}"
            flag = f"  << {c['flag']}" if "flag" in c else ""
            print(f"  spearman(err, {name:<15s}): {rho}{flag}")
    print("-" * 72)
    print(f"GATE VERDICT: {report['gate']}"
          + ("" if report["gate"] == "PASS"
             else "  — capture is INVALID; do not proceed to SAE training"))
    print(f"Report: {args.out}")
    print("=" * 72)
    return 0 if report["gate"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
