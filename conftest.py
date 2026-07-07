"""Repo-root pytest bootstrap.

Guarantees `import moe_sae` resolves to src/ regardless of the editable
install's health. Context: on this machine uv's editable-install .pth files
are created with the macOS UF_HIDDEN flag set, and CPython's site module
skips hidden .pth files — so after any `uv sync` the editable install
silently vanishes from sys.path. See RUNBOOK "Environment notes".
"""

import sys
from pathlib import Path

_SRC = str(Path(__file__).parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
