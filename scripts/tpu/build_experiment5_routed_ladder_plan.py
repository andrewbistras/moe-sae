#!/usr/bin/env python3
"""Build one create-only Experiment-5 routed-activation campaign plan."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from moe_sae.experiment5_campaign import Experiment5CampaignPlanError
from moe_sae.experiment5_routed_ladder_campaign import (
    RoutedLadderCampaignPlanError,
    build_routed_ladder_campaign_plan,
    load_h_reference_manifest,
    write_routed_ladder_campaign_plan,
)
from moe_sae.experiment5_plan_builder import (
    EXECUTION_PROFILE,
    Experiment5PlanBuildError,
    load_r8_plan,
    load_resource_evidence_overrides,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r8-plan", required=True, type=Path)
    parser.add_argument("--project-commit", required=True)
    parser.add_argument(
        "--execution-profile",
        required=True,
        choices=(EXECUTION_PROFILE,),
    )
    parser.add_argument("--resource-evidence-overrides", required=True, type=Path)
    parser.add_argument("--h-reference-manifest", required=True, type=Path)
    parser.add_argument("--already-spent-usd", required=True, type=float)
    parser.add_argument("--campaign-id")
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        plan = build_routed_ladder_campaign_plan(
            load_r8_plan(args.r8_plan),
            project_commit=args.project_commit,
            execution_profile=args.execution_profile,
            h_reference_manifest=load_h_reference_manifest(
                args.h_reference_manifest
            ),
            already_spent_usd=args.already_spent_usd,
            resource_evidence_overrides=load_resource_evidence_overrides(
                args.resource_evidence_overrides
            ),
            campaign_id=args.campaign_id,
        )
        parsed = write_routed_ladder_campaign_plan(args.output, plan)
    except (
        Experiment5CampaignPlanError,
        Experiment5PlanBuildError,
        RoutedLadderCampaignPlanError,
    ) as error:
        print(f"activation-ladder plan build failed: {error}", file=sys.stderr)
        return 2
    print(
        f"wrote {args.output} plan_sha256={parsed.plan_sha256} "
        f"resources={len(parsed.resources)} "
        f"trajectories={len(parsed.trajectories)} "
        f"evaluations={len(parsed.evaluations)} "
        f"h_references={len(parsed.h_references)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
