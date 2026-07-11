"""Run manifests + OPT/BLOOM-style logbook (ops tooling; spec §20 hygiene).

Provenance only — nothing here touches training math, data order, selection,
losses, or the metrics.jsonl format (all frozen, spec v0.1.1 §5–§11 and
v0.1.2 addendum A1–A10). A manifest is a *record* of a run's inputs and
environment; the logbook is an append-only markdown journal of operational
events (starts, restarts, gate outcomes, manual notes), in the spirit of the
OPT/BLOOM training chronicles.

Manifest (`manifest.json` in the run directory):
- ISO-8601 UTC timestamp
- git commit + dirty flag (+ branch) of this repo
- resolved config echo and its content hash (sha256 of canonical YAML)
- spec version strings parsed from `spec/` filenames (e.g. v0.1.1, v0.1.2)
- env fingerprint: python/torch/jax versions, platform, hostname
- accelerator inventory (CPU count; CUDA devices / MPS via torch if present)
- frozen-constants echo from the config (spec §5–§11 values, for eyeballing)

Logbook: `logbook_append` writes one `## <ts> — <event>` block per call using
a single `os.write` on an `O_APPEND` fd, so concurrent writers (threads or
processes) never interleave within an entry on a local POSIX filesystem.

Decisions:
- Config hash = sha256 of `yaml.safe_dump(asdict(cfg), sort_keys=True)`: the
  *resolved* config content, canonically key-sorted so dataclass field order
  and YAML styling cannot change the hash. Reported as "sha256:<hex>".
- Heavy imports are lazy: `torch` is imported only inside the accelerator
  probe (with a fallback if unavailable) and library versions come from
  `importlib.metadata`, not from importing the libraries — so the logbook
  path stays import-light for 20-way multiprocess appends and CLI use.
- jax devices are NOT probed (importing jax can initialize a TPU backend as a
  side effect); only the jax version string is recorded.
- Logbook detail values: containers and strings containing newlines are
  rendered via compact `json.dumps`; other scalars via `str()`. Keeps entries
  one-line-per-key and hence parseable by `logbook_entries`.
- `write_run_manifest` does not validate the config — it records what it was
  given. Freezing is enforced by `RunConfig.validate()` at run start.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import platform
import re
import socket
import subprocess
from datetime import datetime, timezone
from importlib import metadata as _importlib_metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:  # avoid importing torch via moe_sae.model_loading at runtime
    from moe_sae.train.config import RunConfig

MANIFEST_NAME = "manifest.json"
DEFAULT_LOGBOOK = "logs/LOGBOOK.md"

_REPO_ROOT = Path(__file__).resolve().parents[3]  # src/moe_sae/util -> repo
_SPEC_VERSION_RE = re.compile(r"v\d+(?:\.\d+)+")

# RunConfig fields that echo frozen constants (spec §5, §6, §8, §9, §11).
_FROZEN_ECHO_FIELDS = (
    "condition",
    "f",
    "k_bar",
    "batch_tokens",
    "lr",
    "lr_min",
    "warmup_steps",
    "adam_beta1",
    "adam_beta2",
    "grad_clip",
    "aux_alpha",
    "aux_token_budget",
    "dead_window",
    "theta_ema_decay",
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _config_dict(cfg: Any) -> dict:
    """Resolved config as a plain dict (dataclass instance or mapping)."""
    if dataclasses.is_dataclass(cfg) and not isinstance(cfg, type):
        return dataclasses.asdict(cfg)
    return dict(cfg)


def config_content_hash(cfg: Any) -> str:
    """sha256 of the canonical resolved YAML (see module Decisions)."""
    canonical = yaml.safe_dump(_config_dict(cfg), sort_keys=True)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _git_info(repo_root: Path = _REPO_ROOT) -> dict:
    """Commit / dirty flag / branch; every field None if git is unusable."""

    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ("git", "-C", str(repo_root), *args),
                capture_output=True,
                text=True,
                timeout=10,
            )
            return out.stdout.strip() if out.returncode == 0 else None
        except OSError:
            return None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    branch = run("rev-parse", "--abbrev-ref", "HEAD")
    return {
        "commit": commit,
        "dirty": None if status is None else bool(status),
        "branch": branch,
    }


def _spec_versions(spec_dir: Path | None = None) -> list[str]:
    """Sorted, de-duplicated version strings found in spec/ filenames."""
    spec_dir = _REPO_ROOT / "spec" if spec_dir is None else Path(spec_dir)
    if not spec_dir.is_dir():
        return []
    versions: set[str] = set()
    for p in sorted(spec_dir.iterdir()):
        versions.update(_SPEC_VERSION_RE.findall(p.name))
    return sorted(versions)


def _dist_version(name: str) -> str | None:
    try:
        return _importlib_metadata.version(name)
    except _importlib_metadata.PackageNotFoundError:
        return None


def _env_fingerprint() -> dict:
    return {
        "python": platform.python_version(),
        "torch": _dist_version("torch"),
        "jax": _dist_version("jax"),
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
    }


def _accelerator_inventory() -> dict:
    """CPU count always; CUDA/MPS via a lazy torch import (see Decisions)."""
    inv: dict[str, Any] = {"cpu_count": os.cpu_count()}
    try:
        import torch

        if torch.cuda.is_available():
            inv["cuda"] = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
        else:
            inv["cuda"] = []
        mps = getattr(torch.backends, "mps", None)
        inv["mps"] = bool(mps is not None and mps.is_available())
    except Exception as exc:  # torch absent/broken: record, don't fail the run
        inv["torch_probe_error"] = repr(exc)
    return inv


def write_run_manifest(
    run_dir: str | Path,
    cfg: "RunConfig",
    extra: dict | None = None,
    *,
    spec_dir: Path | None = None,
) -> Path:
    """Write `<run_dir>/manifest.json` and return its path.

    Deterministic given fixed inputs and environment, except `created_at`.
    `extra` is caller-provided context (e.g. orchestrator id, resume notes).
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    config = _config_dict(cfg)
    manifest = {
        "created_at": _utc_now_iso(),
        "run_dir": str(run_dir),
        "git": _git_info(),
        "config": config,
        "config_hash": config_content_hash(config),
        "spec_versions": _spec_versions(spec_dir),
        "env": _env_fingerprint(),
        "accelerators": _accelerator_inventory(),
        "frozen_constants": {
            k: config[k] for k in _FROZEN_ECHO_FIELDS if k in config
        },
        "extra": dict(extra or {}),
    }
    path = run_dir / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n")
    return path


def load_run_manifest(run_dir: str | Path) -> dict:
    """Read back `<run_dir>/manifest.json`."""
    return json.loads((Path(run_dir) / MANIFEST_NAME).read_text())


# ---------------------------------------------------------------------------
# Logbook
# ---------------------------------------------------------------------------


def _fmt_detail_value(v: Any) -> str:
    if isinstance(v, str):
        return json.dumps(v) if ("\n" in v or v != v.strip()) else v
    if isinstance(v, (dict, list, tuple)):
        return json.dumps(v, separators=(",", ":"), default=str)
    return str(v)


def logbook_append(
    event: str, detail: dict | None = None, path: str | Path = DEFAULT_LOGBOOK
) -> str:
    """Append one timestamped markdown block; returns the text written.

    Concurrency-safe: the whole entry is emitted in a single `os.write` on an
    `O_APPEND` descriptor, so parallel writers cannot interleave mid-entry.
    """
    lines = [f"## {_utc_now_iso()} — {event}", ""]
    for k, v in (detail or {}).items():
        lines.append(f"- {k}: {_fmt_detail_value(v)}")
    if detail:
        lines.append("")
    entry = "\n".join(lines) + "\n"

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(fd, entry.encode("utf-8"))
    finally:
        os.close(fd)
    return entry


def logbook_entries(path: str | Path = DEFAULT_LOGBOOK) -> list[dict]:
    """Parse the logbook into [{ts, event, detail, raw}], oldest first."""
    p = Path(path)
    if not p.exists():
        return []
    entries: list[dict] = []
    for block in re.split(r"(?m)^(?=## )", p.read_text()):
        if not block.startswith("## "):
            continue  # preamble/garbage before the first header
        header, _, body = block.partition("\n")
        ts, sep, event = header[3:].partition(" — ")
        detail: dict[str, str] = {}
        for line in body.splitlines():
            if line.startswith("- ") and ": " in line:
                k, _, v = line[2:].partition(": ")
                detail[k] = v
        entries.append(
            {
                "ts": ts if sep else None,
                "event": event if sep else header[3:],
                "detail": detail,
                "raw": block.rstrip("\n"),
            }
        )
    return entries
