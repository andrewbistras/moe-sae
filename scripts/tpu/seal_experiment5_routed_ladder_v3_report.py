#!/usr/bin/env python3
"""Seal or verify the immutable Experiment-5 v3 primary report."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Sequence

from moe_sae.experiment5_routed_ladder_v3_report import (
    RoutedLadderV3ReportError,
    build_v3_primary_report_from_paths,
    load_v3_primary_report,
    render_v3_primary_report_markdown,
    verify_v3_primary_report_from_paths,
    write_v3_primary_report,
)


def _source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--evaluation-plan", required=True, type=Path)
    parser.add_argument("--training-plan", required=True, type=Path)
    parser.add_argument("--softplus-references", required=True, type=Path)
    parser.add_argument(
        "--a256-provenance-addendum",
        type=Path,
        help=(
            "Optional immutable two-panel A256Native provenance addendum. "
            "It attaches config, cost, and checkpoint context only; no A256 "
            "score enters primary A512 inference."
        ),
    )
    parser.add_argument(
        "--h-reference-manifest",
        type=Path,
        help=(
            "Optional exact 18-cell H checkpoint manifest. It is pinned as "
            "read-only context and never enters primary inference."
        ),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    seal = commands.add_parser("seal")
    _source_arguments(seal)
    seal.add_argument("--output-json", required=True, type=Path)
    seal.add_argument("--output-markdown", required=True, type=Path)

    validate = commands.add_parser(
        "validate",
        help=(
            "Integrity-only check of self-hashes, internal arithmetic, and "
            "Markdown. This is not evidence-backed scientific verification."
        ),
    )
    validate.add_argument("--report", required=True, type=Path)
    validate.add_argument("--markdown", required=True, type=Path)

    verify = commands.add_parser("verify")
    _source_arguments(verify)
    verify.add_argument("--report", required=True, type=Path)
    verify.add_argument("--markdown", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "seal":
            report = build_v3_primary_report_from_paths(
                evaluation_plan_path=args.evaluation_plan,
                training_plan_path=args.training_plan,
                completed_manifest_path=args.softplus_references,
                a256_provenance_addendum_path=(
                    args.a256_provenance_addendum
                ),
                h_reference_manifest_path=args.h_reference_manifest,
            )
            result = write_v3_primary_report(
                json_path=args.output_json,
                markdown_path=args.output_markdown,
                report=report,
            )
        elif args.command == "validate":
            report = load_v3_primary_report(args.report)
            expected = render_v3_primary_report_markdown(report).encode(
                "utf-8"
            )
            observed = args.markdown.read_bytes()
            if observed != expected:
                raise RoutedLadderV3ReportError(
                    "Markdown differs from deterministic report rendering"
                )
            result = {
                "status": (
                    "primary_report_integrity_valid_non_authoritative"
                ),
                "authoritative": False,
                "validation_scope": (
                    "self_hash_internal_arithmetic_and_markdown_only"
                ),
                "authoritative_command": "verify",
                "report_sha256": report["report_sha256"],
                "json_file_sha256": hashlib.sha256(
                    args.report.read_bytes()
                ).hexdigest(),
                "markdown_file_sha256": hashlib.sha256(
                    observed
                ).hexdigest(),
            }
        else:
            result = verify_v3_primary_report_from_paths(
                report_path=args.report,
                markdown_path=args.markdown,
                evaluation_plan_path=args.evaluation_plan,
                training_plan_path=args.training_plan,
                completed_manifest_path=args.softplus_references,
                a256_provenance_addendum_path=(
                    args.a256_provenance_addendum
                ),
                h_reference_manifest_path=args.h_reference_manifest,
            )
    except (
        OSError,
        RoutedLadderV3ReportError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        print(f"Experiment-5 v3 report error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
