#!/usr/bin/env python3
"""Build the committed, allowlisted R1 image payload tar."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from moe_sae.r1.runtime_bundle import build_runtime_archive


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _, identity = build_runtime_archive(
        args.repository,
        commit=args.commit,
        destination=args.output,
    )
    print(json.dumps(asdict(identity), sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
