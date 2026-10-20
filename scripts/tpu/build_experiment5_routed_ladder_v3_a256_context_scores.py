#!/usr/bin/env python3
"""Seal the six historical Experiment-5 v3 A256Native context scores."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from moe_sae.direct_horizon_campaign import (
    DirectCampaignPlanError,
    DirectCampaignStateError,
    load_direct_campaign_plan,
)
from moe_sae.experiment5_routed_ladder_v3_a256_scores import (
    A256ContextScoreError,
    build_a256_context_score_manifest,
    write_a256_context_score_manifest,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon-plan", required=True, type=Path)
    parser.add_argument("--provenance-addendum", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = build_a256_context_score_manifest(
            load_direct_campaign_plan(args.horizon_plan),
            args.provenance_addendum,
        )
        parsed = write_a256_context_score_manifest(args.output, manifest)
    except (
        A256ContextScoreError,
        DirectCampaignPlanError,
        DirectCampaignStateError,
    ) as error:
        print(f"A256Native context-score sealing failed: {error}", file=sys.stderr)
        return 2
    print(
        f"wrote {args.output} "
        f"manifest_sha256={parsed['manifest_sha256']} "
        f"cells={len(parsed['cells'])} "
        "scope=panel_native_descriptive_only"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
