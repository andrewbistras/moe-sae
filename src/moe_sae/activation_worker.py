"""Private fresh-process validator used by ``activation-farm``.

This is intentionally not an installed command.  The public operator surface
is kept to activation-farm/status/abort while every collective gate still runs
inside a newly initialized Python/JAX process.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import sys

from moe_sae.cli import DEFAULT_CONFIG, _command_activation_gate, _sha256_argument
from moe_sae.runtime import PreflightError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m moe_sae.activation_worker")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument(
        "--phase", choices=("benchmark", "canary", "train", "eval"), required=True
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--checkpoint-sha256", type=_sha256_argument, required=True
    )
    parser.add_argument("--grain-train", required=True)
    parser.add_argument("--grain-eval", required=True)
    parser.add_argument("--dataset-train-manifest", required=True)
    parser.add_argument("--dataset-eval-manifest", required=True)
    parser.set_defaults(handler=_command_activation_gate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (PreflightError, RuntimeError, ValueError, FileNotFoundError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
