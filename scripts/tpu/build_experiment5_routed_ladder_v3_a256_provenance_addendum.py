#!/usr/bin/env python3
"""Seal the Experiment-5 v3 A256Native provenance correction."""

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
from moe_sae.experiment5_routed_ladder_v3_provenance import (
    build_a256_provenance_addendum,
    write_a256_provenance_addendum,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon-plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = build_a256_provenance_addendum(
            load_direct_campaign_plan(args.horizon_plan)
        )
        parsed, panels = write_a256_provenance_addendum(
            args.output,
            manifest,
        )
    except (
        DirectCampaignPlanError,
        DirectCampaignStateError,
        RoutedLadderCampaignPlanError,
        RoutedLadderCampaignStateError,
    ) as error:
        print(f"A256Native provenance addendum failed: {error}", file=sys.stderr)
        return 2
    by_panel = {panel.panel: panel for panel in panels}
    print(
        f"wrote {args.output} "
        f"manifest_sha256={parsed['manifest_sha256']} "
        f"sae={by_panel['sae'].declaration_status} "
        f"transcoder={by_panel['transcoder'].declaration_status} "
        "scope=contextual_only"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
