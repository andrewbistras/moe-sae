#!/usr/bin/env python3
"""Build the create-only exact-six v2 Softplus import inventory for v3."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from moe_sae.experiment5_routed_ladder_campaign import (
    PANELS,
    SEEDS,
    RoutedLadderCampaignPlanError,
    RoutedLadderCampaignStateError,
    build_completed_trajectory_reference_manifest,
    load_routed_ladder_campaign_plan,
    write_completed_trajectory_reference_manifest,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    required_roles = {
        (panel, "A512Softplus", seed)
        for panel in PANELS
        for seed in SEEDS
    }
    try:
        manifest = build_completed_trajectory_reference_manifest(
            load_routed_ladder_campaign_plan(args.v2_plan),
            roles=required_roles,
        )
        parsed, cells = write_completed_trajectory_reference_manifest(
            args.output,
            manifest,
        )
    except (
        RoutedLadderCampaignPlanError,
        RoutedLadderCampaignStateError,
    ) as error:
        print(
            f"v3 Softplus checkpoint inventory failed: {error}",
            file=sys.stderr,
        )
        return 2
    print(
        f"wrote {args.output} "
        f"manifest_sha256={parsed['manifest_sha256']} "
        f"trajectories={len(cells) // 3} cells={len(cells)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
