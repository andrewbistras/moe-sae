#!/usr/bin/env python3
"""Build or validate the immutable scientific-result registry."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from moe_sae.result_registry import (
    ResultRegistryError,
    build_registry,
    file_sha256,
    fragment_from_experiment4_report,
    fragment_from_experiment5_core,
    fragment_from_v3_report,
    load_registry,
    read_json,
    validate_fragment,
    write_registry,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the fail-closed Gemma result registry from sealed reports "
            "and/or generic future-experiment fragments."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build", help="validate, merge, and seal inputs")
    build.add_argument(
        "--experiment4-report",
        action="append",
        default=[],
        type=Path,
        help="sealed Experiment-4 horizon report (repeatable)",
    )
    build.add_argument(
        "--experiment5-core-report",
        type=Path,
        help="sealed Experiment-5 H/R/B/F report",
    )
    build.add_argument(
        "--experiment5-core-result",
        action="append",
        default=[],
        type=Path,
        help="one parent Experiment-5 scientific result (exactly 18 required)",
    )
    build.add_argument(
        "--v3-report",
        action="append",
        default=[],
        type=Path,
        help="sealed Experiment-5 v3 activation-policy report (repeatable)",
    )
    build.add_argument(
        "--fragment",
        action="append",
        default=[],
        type=Path,
        help="generic registry fragment, including future closure reports",
    )
    build.add_argument("--output", required=True, type=Path)

    validate = commands.add_parser("validate", help="validate one sealed registry")
    validate.add_argument("registry", type=Path)
    return parser


def _summary(registry: dict[str, object], *, path: Path) -> str:
    return json.dumps(
        {
            "registry": str(path.expanduser().resolve()),
            "registry_sha256": registry["registry_sha256"],
            "sources": len(registry["sources"]),  # type: ignore[arg-type]
            "score_spaces": len(registry["score_spaces"]),  # type: ignore[arg-type]
            "cost_spaces": len(registry["cost_spaces"]),  # type: ignore[arg-type]
            "cells": len(registry["cells"]),  # type: ignore[arg-type]
            "contrasts": len(registry["contrasts"]),  # type: ignore[arg-type]
        },
        sort_keys=True,
    )


def _build(args: argparse.Namespace) -> int:
    fragments: list[dict[str, object]] = []
    for path in args.experiment4_report:
        fragments.append(
            fragment_from_experiment4_report(
                read_json(path),
                file_sha256=file_sha256(path),
            )
        )

    core_report = args.experiment5_core_report
    core_results: Sequence[Path] = args.experiment5_core_result
    if (core_report is None) != (not core_results):
        raise ResultRegistryError(
            "--experiment5-core-report and its 18 "
            "--experiment5-core-result parents must be supplied together"
        )
    if core_report is not None:
        if len(core_results) != 18:
            raise ResultRegistryError(
                "Experiment-5 core ingestion requires exactly 18 parent results"
            )
        fragments.append(
            fragment_from_experiment5_core(
                read_json(core_report),
                [read_json(path) for path in core_results],
                file_sha256=file_sha256(core_report),
            )
        )

    for path in args.v3_report:
        fragments.append(
            fragment_from_v3_report(
                read_json(path),
                file_sha256=file_sha256(path),
            )
        )
    for path in args.fragment:
        fragments.append(validate_fragment(read_json(path)))
    if not fragments:
        raise ResultRegistryError("no result artifacts were supplied")

    registry = build_registry(fragments)
    output = write_registry(args.output, registry)
    print(_summary(registry, path=output))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "validate":
            registry = load_registry(args.registry)
            print(_summary(registry, path=args.registry))
            return 0
        return _build(args)
    except (OSError, ResultRegistryError, TypeError, ValueError) as error:
        raise SystemExit(f"result registry failed closed: {error}") from error


if __name__ == "__main__":
    raise SystemExit(main())
