#!/usr/bin/env python3
"""Build one create-only Experiment-5 activation-policy v3 training plan."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from moe_sae.experiment5_plan_builder import (
    Experiment5PlanBuildError,
    load_r8_plan,
)
from moe_sae.experiment5_routed_ladder_campaign import (
    RoutedLadderCampaignPlanError,
)
from moe_sae.experiment5_routed_ladder_resources import (
    RoutedLadderResourcePoolError,
    load_routed_ladder_resource_pool,
)
from moe_sae.experiment5_routed_ladder_v3_campaign import (
    build_routed_ladder_v3_training_plan,
    write_routed_ladder_v3_training_plan,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r8-plan", required=True, type=Path)
    parser.add_argument("--project-commit", required=True)
    parser.add_argument("--resource-pool", required=True, type=Path)
    parser.add_argument("--already-spent-usd", required=True, type=float)
    parser.add_argument("--campaign-id")
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        source = load_r8_plan(args.r8_plan)
        resource_pool = load_routed_ladder_resource_pool(
            args.resource_pool,
            project_commit=args.project_commit,
        )
        value = build_routed_ladder_v3_training_plan(
            source,
            project_commit=args.project_commit,
            resource_pool=resource_pool.source,
            already_spent_usd=args.already_spent_usd,
            campaign_id=args.campaign_id,
        )
        plan = write_routed_ladder_v3_training_plan(args.output, value)
    except (
        Experiment5PlanBuildError,
        RoutedLadderCampaignPlanError,
        RoutedLadderResourcePoolError,
    ) as error:
        print(f"activation-policy v3 plan build failed: {error}", file=sys.stderr)
        return 2
    print(
        f"wrote {args.output} plan_sha256={plan.plan_sha256} "
        f"resources={len(plan.resources)} "
        f"trajectories={len(plan.trajectories)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
