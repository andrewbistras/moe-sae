#!/usr/bin/env python3
"""Run or inspect the preallocated-TPU Gemma horizon campaign."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
from pathlib import Path

from moe_sae.direct_horizon_campaign import (
    DirectCampaign,
    DirectCampaignPlanError,
    DirectCampaignStateError,
    direct_campaign_status,
    load_direct_campaign_plan,
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
                help=(
                    "seal all eight training completions, then stop before "
                    "the six joint evaluations so regional control trees can "
                    "be reconciled"
                ),
            )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        plan = load_direct_campaign_plan(args.plan)
        if args.action == "validate":
            print(
                json.dumps(
                    {
                        "campaign_id": plan.campaign_id,
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
                    direct_campaign_status(plan),
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
        result = DirectCampaign(plan, stop_event=stop).run(
            stop_after_training=args.stop_after_training
        )
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return 0
    except (DirectCampaignPlanError, DirectCampaignStateError) as error:
        print(f"direct horizon error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
