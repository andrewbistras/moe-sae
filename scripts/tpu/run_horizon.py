#!/usr/bin/env python3
"""Run or inspect the cloud-resident Gemma 10M horizon controller."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import sys
import threading

from moe_sae.horizon_supervisor import (
    HorizonPlanError,
    HorizonStateError,
    HorizonSupervisor,
    horizon_status,
    load_horizon_plan,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the immutable capture-to-report horizon DAG"
    )
    subparsers = parser.add_subparsers(dest="action", required=True)
    for action in ("validate", "status", "run"):
        selected = subparsers.add_parser(action)
        selected.add_argument("--plan", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        plan = load_horizon_plan(args.plan)
        if args.action == "validate":
            print(
                json.dumps(
                    {
                        "campaign_id": plan.campaign_id,
                        "plan_sha256": plan.plan_sha256,
                        "resources": len(plan.resources),
                        "tasks": len(plan.tasks),
                    },
                    sort_keys=True,
                )
            )
            return 0
        if args.action == "status":
            print(json.dumps(horizon_status(plan), indent=2, sort_keys=True))
            return 0

        stop = threading.Event()
        supervisor = HorizonSupervisor(plan, stop_event=stop)

        def request_stop(_signal: int, _frame: object) -> None:
            stop.set()

        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
        result = supervisor.run()
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return 0 if result.get("status") == "completed" else 2
    except (HorizonPlanError, HorizonStateError) as error:
        print(f"horizon error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
