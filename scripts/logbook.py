#!/usr/bin/env python
"""Manual entries in the OPT/BLOOM-style run logbook (logs/LOGBOOK.md).

Usage:
    uv run python scripts/logbook.py --event "phase 3 start" --note "LR sanity skipped (A9)"
    uv run python scripts/logbook.py --tail 5
    uv run python scripts/logbook.py --event "gate PASS" --tail 3   # append, then show

Ops tooling only — automated wiring into training loops is done by the
Integrate stage, not here.
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--event", help="event title for a new entry")
    ap.add_argument("--note", help="free-text note recorded with --event")
    ap.add_argument("--tail", type=int, metavar="N", help="print the last N entries")
    ap.add_argument(
        "--path", default="logs/LOGBOOK.md", help="logbook file (default: %(default)s)"
    )
    args = ap.parse_args()

    if args.event is None and args.tail is None:
        ap.error("nothing to do: pass --event and/or --tail N")
    if args.note is not None and args.event is None:
        ap.error("--note requires --event")

    from moe_sae.util.manifest import logbook_append, logbook_entries

    if args.event is not None:
        detail = {"user": getpass.getuser(), "cwd": os.getcwd()}
        if args.note is not None:
            detail["note"] = args.note
        entry = logbook_append(args.event, detail, path=args.path)
        sys.stdout.write(f"appended to {args.path}:\n{entry}")

    if args.tail is not None:
        entries = logbook_entries(args.path)
        if not entries:
            print(f"(no entries in {args.path})")
            return
        for e in entries[-args.tail :]:
            print(e["raw"])
            print()


if __name__ == "__main__":
    main()
