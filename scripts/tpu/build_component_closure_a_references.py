#!/usr/bin/env python3
"""Build the read-only A reference manifest from a completed v3 plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from moe_sae.component_closure_references import (
    build_a_reference_manifest_from_path,
    validate_a_reference_manifest,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3-training-plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--validate", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.validate:
        validate_a_reference_manifest(
            json.loads(args.output.read_text(encoding="ascii"))
        )
        return 0
    value = build_a_reference_manifest_from_path(args.v3_training_plan)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="ascii") as handle:
        json.dump(value, handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
