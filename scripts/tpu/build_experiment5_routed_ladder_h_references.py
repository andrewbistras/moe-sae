#!/usr/bin/env python3
"""Build a create-only inventory of the 18 sealed Experiment-5 H checkpoints."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from moe_sae.experiment5_campaign import (
    Experiment5CampaignPlanError,
    Experiment5CampaignStateError,
    load_experiment5_campaign_plan,
)
from moe_sae.experiment5_routed_ladder_campaign import (
    RoutedLadderCampaignPlanError,
    RoutedLadderCampaignStateError,
    build_h_reference_manifest,
    write_h_reference_manifest,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment5-plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = build_h_reference_manifest(
            load_experiment5_campaign_plan(args.experiment5_plan)
        )
        parsed, cells = write_h_reference_manifest(args.output, manifest)
    except (
        Experiment5CampaignPlanError,
        Experiment5CampaignStateError,
        RoutedLadderCampaignPlanError,
        RoutedLadderCampaignStateError,
    ) as error:
        print(f"H checkpoint inventory failed: {error}", file=sys.stderr)
        return 2
    print(
        f"wrote {args.output} "
        f"manifest_sha256={parsed['manifest_sha256']} cells={len(cells)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
