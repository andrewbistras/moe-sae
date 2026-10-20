#!/usr/bin/env python3
"""Build, validate, inspect, or durably execute the closure campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import yaml

from moe_sae.component_closure_campaign import (
    ComponentClosureTrainingCampaign,
    build_train_command,
    build_training_plan,
    load_training_campaign,
    load_training_plan,
    parse_training_campaign,
    status,
)
from moe_sae.component_closure_references import load_a_reference_manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--source", required=True, type=Path)
    build.add_argument("--a-references", required=True, type=Path)
    build.add_argument("--project-commit", required=True)
    build.add_argument("--output", required=True, type=Path)
    for name in ("validate", "status", "commands", "run"):
        selected = commands.add_parser(name)
        selected.add_argument("--plan", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build":
        source = yaml.safe_load(args.source.read_text(encoding="utf-8"))
        value = build_training_plan(
            source,
            project_commit=args.project_commit,
            a_reference_manifest=load_a_reference_manifest(args.a_references),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            yaml.safe_dump(value, handle, sort_keys=False)
        return 0
    plan = load_training_plan(args.plan)
    if args.command == "status":
        print(json.dumps(status(plan), indent=2, sort_keys=True))
    elif args.command == "commands":
        campaign = parse_training_campaign(plan)
        print(
            json.dumps(
                [
                    {
                        "task_id": task.task_id,
                        "eligible_resource_ids": list(task.eligible_resource_ids),
                        "command": build_train_command(
                            campaign,
                            task,
                            "scheduler-generated-attempt-id",
                        ),
                    }
                    for task in campaign.trajectories
                ],
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "run":
        result = ComponentClosureTrainingCampaign(
            load_training_campaign(args.plan)
        ).run()
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
