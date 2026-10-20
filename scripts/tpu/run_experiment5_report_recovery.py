#!/usr/bin/env python3
"""Build, validate, inspect, or seal Experiment-5 report recovery."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from moe_sae.experiment5_report_recovery import (
    Experiment5ReportRecoveryError,
    build_report_recovery_plan,
    load_report_recovery_plan,
    report_recovery_status,
    seal_report_recovery,
    verify_report_recovery,
    write_report_recovery_plan,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)

    build = subparsers.add_parser("build")
    build.add_argument("--evaluation-plan", required=True, type=Path)
    build.add_argument("--budget-ledger", required=True, type=Path)
    build.add_argument("--report-root", required=True, type=Path)
    build.add_argument("--working-directory", required=True, type=Path)
    build.add_argument("--reporting-source-revision", required=True)
    build.add_argument("--recovery-id")
    build.add_argument("--output", required=True, type=Path)

    for action in ("validate", "status", "seal", "verify"):
        selected = subparsers.add_parser(action)
        selected.add_argument("--plan", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.action == "build":
            raw = build_report_recovery_plan(
                evaluation_plan_path=args.evaluation_plan,
                budget_ledger_path=args.budget_ledger,
                report_root=args.report_root,
                working_directory=args.working_directory,
                reporting_source_revision=args.reporting_source_revision,
                recovery_id=args.recovery_id,
            )
            plan = write_report_recovery_plan(args.output, raw)
            value = {
                "recovery_id": plan.recovery_id,
                "plan": str(args.output.expanduser().resolve()),
                "plan_sha256": plan.plan_sha256,
            }
        else:
            plan = load_report_recovery_plan(args.plan)
            if args.action == "validate":
                # Parent verification remains part of sealing so validation
                # cannot accidentally create evidence or outputs.
                value = {
                    "recovery_id": plan.recovery_id,
                    "plan_sha256": plan.plan_sha256,
                    "report_root": str(plan.report_root),
                    "reporting_source_revision": (plan.reporting_source_revision),
                    "evaluation_source_revision": (
                        plan.parent.evaluation_source_revision
                    ),
                }
            elif args.action == "status":
                value = report_recovery_status(plan)
            elif args.action == "verify":
                value = verify_report_recovery(plan)
            else:
                value = seal_report_recovery(plan)
        print(json.dumps(value, indent=2, sort_keys=True))
        return 0
    except Experiment5ReportRecoveryError as error:
        print(f"Experiment-5 report recovery error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
