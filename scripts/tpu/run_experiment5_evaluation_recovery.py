#!/usr/bin/env python3
"""Validate, inspect, run, or hand off Experiment-5 evaluation recovery."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
from pathlib import Path

from moe_sae.experiment5_evaluation_recovery import (
    EvaluationRecoveryCampaign,
    EvaluationRecoveryPlanError,
    EvaluationRecoveryStateError,
    evaluation_recovery_status,
    load_evaluation_recovery_plan,
    seal_recovery_handoff,
)
from moe_sae.experiment5_campaign import (
    Experiment5BudgetStop,
    Experiment5CampaignPlanError,
    Experiment5CampaignStateError,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    for action in ("validate", "status", "run", "handoff"):
        selected = subparsers.add_parser(action)
        selected.add_argument("--plan", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        plan = load_evaluation_recovery_plan(args.plan)
        if args.action == "validate":
            print(
                json.dumps(
                    {
                        "campaign_id": plan.campaign_id,
                        "checkpoint_parents": sum(
                            len(row.checkpoint_pointers)
                            for row in plan.parent.training_cells
                        ),
                        "cost_hard_limit_usd": (
                            plan.cost_policy.hard_limit_usd
                        ),
                        "evaluations": len(plan.evaluations),
                        "plan_sha256": plan.plan_sha256,
                        "resources": len(plan.resources),
                        "training_parents": len(
                            plan.parent.training_cells
                        ),
                        "trajectories": 0,
                    },
                    sort_keys=True,
                )
            )
            return 0
        if args.action == "status":
            print(
                json.dumps(
                    evaluation_recovery_status(plan),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.action == "handoff":
            print(
                json.dumps(
                    seal_recovery_handoff(plan),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0

        stop = threading.Event()

        def request_stop(_signal: int, _frame: object) -> None:
            stop.set()

        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
        result = EvaluationRecoveryCampaign(
            plan, stop_event=stop
        ).run()
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return 0
    except Experiment5BudgetStop as error:
        print(f"Experiment-5 recovery budget stop: {error}", file=sys.stderr)
        return 3
    except (
        EvaluationRecoveryPlanError,
        EvaluationRecoveryStateError,
        Experiment5CampaignPlanError,
        Experiment5CampaignStateError,
    ) as error:
        print(f"Experiment-5 recovery error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
