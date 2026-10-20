#!/usr/bin/env python3
"""Build, validate, run, or inspect Experiment-5 v3 primary evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from moe_sae.experiment5_routed_ladder_v3_campaign import (
    load_routed_ladder_v3_training_plan,
)
from moe_sae.experiment5_routed_ladder_v3_evaluation import (
    build_from_paths,
    load_v3_primary_evaluation_plan,
    primary_evaluation_result_paths,
    run_v3_primary_evaluation,
    write_v3_primary_evaluation_plan,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--training-plan", required=True, type=Path)
    build.add_argument("--v2-plan", required=True, type=Path)
    build.add_argument("--softplus-references", required=True, type=Path)
    build.add_argument("--evaluation-source-revision", required=True)
    build.add_argument("--output", required=True, type=Path)

    validate = commands.add_parser("validate")
    validate.add_argument("--plan", required=True, type=Path)

    run = commands.add_parser("run")
    run.add_argument("--plan", required=True, type=Path)
    run.add_argument("--training-plan", required=True, type=Path)
    run.add_argument("--max-attempts", type=int, default=3)

    status = commands.add_parser("status")
    status.add_argument("--plan", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "build":
            plan = build_from_paths(
                training_plan_path=args.training_plan,
                v2_plan_path=args.v2_plan,
                completed_manifest_path=args.softplus_references,
                evaluation_source_revision=args.evaluation_source_revision,
            )
            value = write_v3_primary_evaluation_plan(args.output, plan)
            result = {
                "status": "evaluation_plan_sealed",
                "plan": str(args.output.resolve()),
                "plan_sha256": value["plan_sha256"],
                "bundles": len(value["bundles"]),
                "cells": sum(
                    len(row["operator_manifest"]["cells"])
                    for row in value["bundles"]
                ),
            }
        elif args.command == "validate":
            value = load_v3_primary_evaluation_plan(args.plan)
            result = {
                "status": "evaluation_plan_valid",
                "plan_sha256": value["plan_sha256"],
                "bundles": len(value["bundles"]),
            }
        elif args.command == "run":
            value = load_v3_primary_evaluation_plan(args.plan)
            training = load_routed_ladder_v3_training_plan(
                args.training_plan
            )
            result = run_v3_primary_evaluation(
                value,
                training,
                max_attempts=args.max_attempts,
            )
        else:
            value = load_v3_primary_evaluation_plan(args.plan)
            paths = primary_evaluation_result_paths(value)
            complete = [str(path) for path in paths if path.is_file()]
            result = {
                "status": (
                    "primary_evaluation_completed"
                    if len(complete) == len(paths)
                    else "primary_evaluation_pending"
                ),
                "complete_cells": len(complete),
                "expected_cells": len(paths),
                "completion_path": str(
                    Path(value["state_root"])
                    / "primary-evaluation"
                    / "primary-evaluation-complete.json"
                ),
            }
    except Exception as error:
        print(f"Experiment-5 v3 evaluation error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
