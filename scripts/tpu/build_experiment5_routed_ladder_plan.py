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
    load_a256_reference_manifest,
    load_h_reference_manifest,
    write_routed_ladder_campaign_plan,
)
from moe_sae.experiment5_plan_builder import (
    Experiment5PlanBuildError,
    load_r8_plan,
)
from moe_sae.experiment5_routed_ladder_resources import (
    RoutedLadderResourcePoolError,
    load_routed_ladder_resource_pool,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r8-plan", required=True, type=Path)
    parser.add_argument("--project-commit", required=True)
    parser.add_argument("--resource-pool", type=Path)
    # Accepted only so the retired ready-three handoff fails with a precise
    # migration error instead of an opaque "unrecognized argument" message.
    parser.add_argument("--execution-profile", help=argparse.SUPPRESS)
    parser.add_argument(
        "--resource-evidence-overrides",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--h-reference-manifest", required=True, type=Path)
    parser.add_argument("--a256-reference-manifest", type=Path)
    parser.add_argument("--already-spent-usd", required=True, type=float)
    parser.add_argument("--campaign-id")
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.resource_pool is None:
        parser.error(
            "--resource-pool is required; the ready-three "
            "--resource-evidence-overrides handoff is v1-only and cannot "
            "construct an N-way ladder plan"
        )
    if (
        args.execution_profile is not None
        or args.resource_evidence_overrides is not None
    ):
        parser.error(
            "--execution-profile/--resource-evidence-overrides are retired "
            "for the ladder; provide only the immutable --resource-pool"
        )
    if args.a256_reference_manifest is None:
        parser.error(
            "--a256-reference-manifest is required for the historical "
            "SAE A256 seed-zero checkpoints"
        )
    try:
        plan = build_routed_ladder_campaign_plan(
            load_r8_plan(args.r8_plan),
            project_commit=args.project_commit,
            h_reference_manifest=load_h_reference_manifest(
                args.h_reference_manifest
            ),
            a256_reference_manifest=load_a256_reference_manifest(
                args.a256_reference_manifest
            ),
            already_spent_usd=args.already_spent_usd,
            resource_pool=load_routed_ladder_resource_pool(
                args.resource_pool,
                project_commit=args.project_commit,
            ).source,
            campaign_id=args.campaign_id,
        )
        parsed = write_routed_ladder_campaign_plan(args.output, plan)
    except (
        Experiment5CampaignPlanError,
        Experiment5PlanBuildError,
        RoutedLadderCampaignPlanError,
        RoutedLadderResourcePoolError,
    ) as error:
        print(f"activation-ladder plan build failed: {error}", file=sys.stderr)
        return 2
    print(
        f"wrote {args.output} plan_sha256={parsed.plan_sha256} "
        f"resources={len(parsed.resources)} "
        f"trajectories={len(parsed.trajectories)} "
        f"evaluations={len(parsed.evaluations)} "
        f"h_references={len(parsed.h_references)} "
        f"a256_references={len(parsed.a256_references)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
