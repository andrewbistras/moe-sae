#!/usr/bin/env python3
"""Build the create-only historical SAE A256 seed-zero checkpoint inventory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from moe_sae.direct_horizon_campaign import (
    DirectCampaignPlanError,
    DirectCampaignStateError,
    load_direct_campaign_plan,
)
from moe_sae.experiment5_routed_ladder_campaign import (
    RoutedLadderCampaignPlanError,
    RoutedLadderCampaignStateError,
    build_a256_reference_manifest,
    write_a256_reference_manifest,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon-plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = build_a256_reference_manifest(
            load_direct_campaign_plan(args.horizon_plan)
        )
        parsed, cells, cost = write_a256_reference_manifest(
            args.output,
            manifest,
        )
    except (
        DirectCampaignPlanError,
        DirectCampaignStateError,
        RoutedLadderCampaignPlanError,
        RoutedLadderCampaignStateError,
    ) as error:
        print(f"A256 checkpoint inventory failed: {error}", file=sys.stderr)
        return 2
    print(
        f"wrote {args.output} "
        f"manifest_sha256={parsed['manifest_sha256']} "
        f"cells={len(cells)} chip_seconds={cost.chip_seconds}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
