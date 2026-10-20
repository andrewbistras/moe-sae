#!/usr/bin/env python3
"""Seal JSON and Markdown component-closure reports from a score grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from moe_sae.component_closure_report import (
    build_closure_report,
    load_score_grid,
    render_markdown,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-markdown", required=True, type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build_closure_report(
        load_score_grid(args.evidence),
        bootstrap_replicates=args.bootstrap_replicates,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("x", encoding="ascii") as handle:
        json.dump(report, handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
    with args.output_markdown.open("x", encoding="utf-8") as handle:
        handle.write(render_markdown(report))
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
