#!/usr/bin/env python
"""Statistics pass over a capture (spec §20 step 2).

Usage:
    uv run python scripts/run_stats.py --capture captures/<name> --out stats/<name>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--capture", required=True, help="capture root (captures/<name>)")
    ap.add_argument("--out", required=True, help="output directory (stats/<name>)")
    args = ap.parse_args()

    from moe_sae.train.stats_pass import compute_stats, save_stats

    stats = compute_stats(args.capture)
    path = save_stats(stats, args.out)
    summary = json.loads((Path(args.out) / "stats.json").read_text())
    print(f"wrote {path}")
    print(json.dumps({k: summary[k] for k in ("tokens", "bank_load")}, indent=2))


if __name__ == "__main__":
    main()
