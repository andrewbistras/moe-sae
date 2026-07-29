#!/usr/bin/env python3
"""Build the create-only two-panel Experiment-4 A256Native inventory."""

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
)
from moe_sae.experiment5_routed_ladder_v3_references import (
    build_native_a256_reference_manifest,
    write_native_a256_reference_manifest,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon-plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = build_native_a256_reference_manifest(
            load_direct_campaign_plan(args.horizon_plan)
        )
        parsed, cells, costs = write_native_a256_reference_manifest(
            args.output,
            manifest,
        )
    except (
        DirectCampaignPlanError,
        DirectCampaignStateError,
        RoutedLadderCampaignPlanError,
        RoutedLadderCampaignStateError,
    ) as error:
        print(f"A256Native checkpoint inventory failed: {error}", file=sys.stderr)
        return 2
    print(
        f"wrote {args.output} "
        f"manifest_sha256={parsed['manifest_sha256']} "
        f"cells={len(cells)} panels={len(costs)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
