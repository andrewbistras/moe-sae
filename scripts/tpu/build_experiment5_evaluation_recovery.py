#!/usr/bin/env python3
"""Build a create-only Experiment-5 evaluation-recovery plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from moe_sae.experiment5_evaluation_recovery import (
    EvaluationRecoveryPlanError,
    build_evaluation_recovery_plan,
    write_evaluation_recovery_plan,
)
from moe_sae.experiment5_campaign import Experiment5CampaignPlanError


def _mapping_file(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise EvaluationRecoveryPlanError(f"cannot read {name}: {path}") from error
    if not isinstance(value, dict):
        raise EvaluationRecoveryPlanError(f"{name} is not a mapping")
    return value


def _region_roots(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        region, separator, raw_path = value.partition("=")
        if (
            not separator
            or not region
            or not raw_path
            or region in result
        ):
            raise EvaluationRecoveryPlanError(
                "--result-root-controller must be unique REGION=/absolute/path"
            )
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            raise EvaluationRecoveryPlanError(
                "--result-root-controller path must be absolute"
            )
        result[region] = path
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-plan", required=True, type=Path)
    parser.add_argument("--evaluation-source-revision", required=True)
    parser.add_argument(
        "--resource-plan",
        required=True,
        type=Path,
        help="YAML/JSON mapping containing the fresh `resources` list",
    )
    parser.add_argument("--state-root", required=True, type=Path)
    parser.add_argument(
        "--result-root-controller",
        required=True,
        action="append",
        metavar="REGION=/ABSOLUTE/PATH",
        help=(
            "repeat once per frozen regional NFS plane; worker-visible "
            "remote paths remain common"
        ),
    )
    parser.add_argument("--result-root-remote", required=True)
    parser.add_argument("--already-spent-usd", required=True, type=float)
    parser.add_argument("--campaign-id")
    parser.add_argument("--max-parallel-eval", type=int)
    parser.add_argument("--eligibility-map", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        resource_source = _mapping_file(
            args.resource_plan, name="resource plan"
        )
        resources = resource_source.get("resources")
        if not isinstance(resources, list):
            raise EvaluationRecoveryPlanError(
                "resource plan lacks a resources list"
            )
        eligibility = None
        if args.eligibility_map is not None:
            eligibility_raw = _mapping_file(
                args.eligibility_map, name="eligibility map"
            )
            eligibility = eligibility_raw.get("eligible_resource_ids")
            if not isinstance(eligibility, dict):
                raise EvaluationRecoveryPlanError(
                    "eligibility map lacks eligible_resource_ids"
                )
        raw = build_evaluation_recovery_plan(
            core_plan_path=args.core_plan,
            evaluation_source_revision=args.evaluation_source_revision,
            resources=resources,
            state_root=args.state_root,
            result_roots_controller=_region_roots(
                args.result_root_controller
            ),
            result_root_remote=args.result_root_remote,
            already_spent_usd=args.already_spent_usd,
            eligible_resource_ids=eligibility,
            campaign_id=args.campaign_id,
            max_parallel_eval=args.max_parallel_eval,
        )
        plan = write_evaluation_recovery_plan(args.output, raw)
        print(
            json.dumps(
                {
                    "campaign_id": plan.campaign_id,
                    "checkpoint_parents": sum(
                        len(row.checkpoint_pointers)
                        for row in plan.parent.training_cells
                    ),
                    "evaluations": len(plan.evaluations),
                    "output": str(args.output.resolve()),
                    "plan_sha256": plan.plan_sha256,
                    "resources": len(plan.resources),
                    "training_parents": len(plan.parent.training_cells),
                    "trajectories": 0,
                },
                sort_keys=True,
            )
        )
        return 0
    except (EvaluationRecoveryPlanError, Experiment5CampaignPlanError) as error:
        print(f"Experiment-5 recovery plan error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
