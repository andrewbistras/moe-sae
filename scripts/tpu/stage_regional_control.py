#!/usr/bin/env python3
"""Build or apply a create-only regional control-plane bundle.

``build`` runs on the cloud controller while its central Filestore is mounted.
Send the resulting small tar file to worker zero only after
``deploy_workers.sh`` has mounted the regional Filestore, then invoke ``apply``
there with the TPU checkout's production artifact root as the destination.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from moe_sae.contribution_artifacts import canonical_json_bytes
from moe_sae.regional_control_stage import (
    RegionalControlStageError,
    apply_control_stage_bundle,
    write_control_stage_bundle,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--source-root", required=True)
    build.add_argument("--relative-file", action="append", required=True)
    build.add_argument("--output", required=True)
    apply = commands.add_parser("apply")
    apply.add_argument("--bundle", required=True)
    apply.add_argument("--destination-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "build":
            result = write_control_stage_bundle(
                args.source_root,
                args.relative_file,
                args.output,
            )
        else:
            result = apply_control_stage_bundle(
                args.bundle,
                args.destination_root,
            )
        print(canonical_json_bytes(result).decode("ascii"), flush=True)
        return 0
    except (OSError, RegionalControlStageError) as error:
        print(
            json.dumps(
                {"status": "error", "detail": str(error)},
                sort_keys=True,
            ),
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
