#!/usr/bin/env python3
"""Validate, inspect, or run the fixed Experiment-5 TPU campaign."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
from pathlib import Path

from moe_sae.experiment5_campaign import (
    Experiment5BudgetStop,
    Experiment5Campaign,
    Experiment5CampaignPlanError,
    Experiment5CampaignStateError,
    experiment5_campaign_status,
    load_experiment5_campaign_plan,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    for action in ("validate", "status", "run"):
        selected = subparsers.add_parser(action)
        selected.add_argument("--plan", required=True, type=Path)
        if action == "run":
            selected.add_argument(
                "--stop-after-training",
                action="store_true",
                help="seal all 24 trajectories, then stop before evaluation",
            )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        plan = load_experiment5_campaign_plan(args.plan)
        if args.action == "validate":
            policy = plan.cost_policy
            print(
                json.dumps(
                    {
                        "campaign_id": plan.campaign_id,
                        "cost_hard_limit_usd": policy.hard_limit_usd,
                        "cost_initial_launch_limit_usd": (
                            policy.initial_launch_limit_usd
                        ),
                        "evaluations": len(plan.evaluations),
                        "plan_sha256": plan.plan_sha256,
                        "resources": len(plan.resources),
                        "trajectories": len(plan.trajectories),
                    },
                    sort_keys=True,
                )
            )
            return 0
        if args.action == "status":
            print(
                json.dumps(
                    experiment5_campaign_status(plan),
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
        result = Experiment5Campaign(plan, stop_event=stop).run(
            stop_after_training=args.stop_after_training
        )
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return 0
    except Experiment5BudgetStop as error:
        print(f"Experiment-5 budget stop: {error}", file=sys.stderr)
        return 3
    except (Experiment5CampaignPlanError, Experiment5CampaignStateError) as error:
        print(f"Experiment-5 campaign error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
