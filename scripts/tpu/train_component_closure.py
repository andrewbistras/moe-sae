#!/usr/bin/env python3
"""Closure-aware launcher for the mature contribution training operator."""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

from moe_sae.component_closure_worker import (
    MATURE_WORKER_MODULE,
    WORKER_MODULE,
    install_worker_adapter,
)

MATURE_SCRIPT = "train_contribution.py"
CLOSURE_SCRIPT = "train_component_closure.py"


def _load_operator() -> ModuleType:
    path = Path(__file__).with_name(MATURE_SCRIPT)
    spec = importlib.util.spec_from_file_location(
        "_component_closure_mature_operator",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load mature contribution operator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _replace(command: Sequence[str]) -> list[str]:
    return [
        str(token)
        .replace(MATURE_WORKER_MODULE, WORKER_MODULE)
        .replace(f"/{MATURE_SCRIPT}", f"/{CLOSURE_SCRIPT}")
        for token in command
    ]


def install_operator_adapter() -> ModuleType:
    install_worker_adapter()
    operator = _load_operator()
    for name in ("_init_command", "_prefetch_command", "_action_command"):
        original = getattr(operator, name)

        def adapted(*args: Any, __original: Any = original, **kwargs: Any) -> list[str]:
            return _replace(__original(*args, **kwargs))

        setattr(operator, name, adapted)
    return operator


def main(argv: Sequence[str] | None = None) -> int:
    return int(install_operator_adapter().main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
