#!/usr/bin/env python3
"""Validate, inspect, or run the isolated routed-activation campaign."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
from pathlib import Path

from moe_sae.experiment5_campaign import Experiment5BudgetStop
from moe_sae.experiment5_routed_ladder_campaign import (
    RoutedLadderCampaign,
    RoutedLadderCampaignPlanError,
    RoutedLadderCampaignStateError,
    load_routed_ladder_campaign_plan,
    routed_ladder_campaign_status,
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
                help="seal all 12 new trajectories, then stop before evaluation",
            )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        plan = load_routed_ladder_campaign_plan(args.plan)
        if args.action == "validate":
            print(
                json.dumps(
                    {
                        "campaign_id": plan.campaign_id,
                        "cost_hard_limit_usd": plan.cost_policy.hard_limit_usd,
                        "cost_initial_launch_limit_usd": (
                            plan.cost_policy.initial_launch_limit_usd
                        ),
                        "evaluations": len(plan.evaluations),
                        "h_references": len(plan.h_references),
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
                    routed_ladder_campaign_status(plan),
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
        result = RoutedLadderCampaign(plan, stop_event=stop).run(
            stop_after_training=args.stop_after_training
        )
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return 0
    except Experiment5BudgetStop as error:
        print(f"activation-ladder budget stop: {error}", file=sys.stderr)
        return 3
    except (
        RoutedLadderCampaignPlanError,
        RoutedLadderCampaignStateError,
    ) as error:
        print(f"activation-ladder campaign error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
