"""Adapter from the closure schema to the mature contribution worker.

The numerical worker is intentionally not forked.  This entry point installs
two narrow adapters before delegating:

* the strict closure loader supplies the existing duck-typed worker surface;
* supervised child commands point back to this module, preserving the loader
  across the worker's hard process boundary.

No Experiment-5 v3 module or result is modified.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from moe_sae.component_closure_config import (
    PHASE_NAME,
    ComponentClosureConfigError,
    load_component_closure_config,
)

WORKER_MODULE = "moe_sae.component_closure_worker"
MATURE_WORKER_MODULE = "moe_sae.contribution_training_worker"


def _load_config(path: str | Path, scientific_model: str) -> Any:
    try:
        return load_component_closure_config(
            path,
            expected_panel=scientific_model,  # type: ignore[arg-type]
        )
    except ComponentClosureConfigError as error:
        raise RuntimeError("closure worker requires its strict config") from error


def _replace_worker_module(command: Sequence[str]) -> list[str]:
    return [
        WORKER_MODULE if token == MATURE_WORKER_MODULE else str(token)
        for token in command
    ]


def install_worker_adapter() -> Any:
    """Install the process-local adapter and return the mature worker module."""

    from moe_sae import contribution_training_worker as worker

    worker.LONG_HORIZON_PHASES = frozenset({*worker.LONG_HORIZON_PHASES, PHASE_NAME})
    worker._load_worker_config = _load_config
    worker._finalized_horizon_config = lambda path: load_component_closure_config(path)
    original_supervise = worker._supervise

    def supervise(args: Any, *, child_command: Sequence[str], **kwargs: Any) -> Any:
        return original_supervise(
            args,
            child_command=_replace_worker_module(child_command),
            **kwargs,
        )

    worker._supervise = supervise
    return worker


def main(argv: Sequence[str] | None = None) -> int:
    return int(install_worker_adapter().main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
