"""Durable, operator-visible control records for activation harvesting.

The scientific capture ledger lives in :mod:`moe_sae.artifacts`.  This module
is deliberately smaller and orthogonal: it prevents two processes from owning
the same controller lane, emits a bounded heartbeat record, and provides one
shared abort marker which workers observe only at synchronized safe boundaries.

All records are self-hashed and replaced atomically on the qualified POSIX
control plane.  The held ``flock`` is the lease; JSON files are evidence and
status, never the source of exclusivity.
"""

from __future__ import annotations

from collections.abc import Mapping
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import tempfile
import time
from typing import Any
import uuid


CONTROL_SCHEMA = "gemma4-harvest-control-v1"
STATUS_SCHEMA = "gemma4-harvest-heartbeat-v1"
ABORT_SCHEMA = "gemma4-harvest-abort-v1"
DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 10.0

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_CAMPAIGN_RE = _SHA256_RE


class HarvestControlError(RuntimeError):
    """A campaign lease or durable control record is invalid."""


class HarvestLeaseHeld(HarvestControlError):
    """Another live process already owns this campaign/controller lane."""


class HarvestAbortRequested(HarvestControlError):
    """The operator requested a coordinated stop at the next safe boundary."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_self_hashed_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    hash_field: str,
) -> dict[str, Any]:
    value = dict(payload)
    value.pop(hash_field, None)
    value[hash_field] = _sha256_json(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(json.dumps(value, indent=2, sort_keys=True).encode("utf-8"))
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return value


def _load_self_hashed_json(path: Path, *, hash_field: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HarvestControlError(
            f"cannot load valid harvest control JSON: {path}"
        ) from exc
    if not isinstance(value, dict):
        raise HarvestControlError(f"harvest control record is not an object: {path}")
    claimed = value.get(hash_field)
    unsigned = {key: item for key, item in value.items() if key != hash_field}
    if (
        not isinstance(claimed, str)
        or not _SHA256_RE.fullmatch(claimed)
        or claimed != _sha256_json(unsigned)
    ):
        raise HarvestControlError(f"harvest control record SHA-256 mismatch: {path}")
    return value


def campaign_identity(
    metadata: Mapping[str, Any],
    *,
    split: str,
    total_tokens: int,
    stop_after_tokens: int,
    process_count: int,
) -> str:
    """Return the deterministic identity shared by every campaign controller."""

    if split not in {"train", "eval"}:
        raise ValueError("split must be 'train' or 'eval'")
    for name, value in (
        ("total_tokens", total_tokens),
        ("stop_after_tokens", stop_after_tokens),
        ("process_count", process_count),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if stop_after_tokens > total_tokens:
        raise ValueError("stop_after_tokens cannot exceed total_tokens")
    try:
        normalized_metadata = json.loads(_canonical_json(dict(metadata)))
    except (TypeError, ValueError) as exc:
        raise ValueError("campaign metadata must contain only JSON values") from exc
    return _sha256_json(
        {
            "schema_version": CONTROL_SCHEMA,
            "metadata_sha256": _sha256_json(normalized_metadata),
            "split": split,
            "total_tokens": total_tokens,
            "stop_after_tokens": stop_after_tokens,
            "process_count": process_count,
        }
    )


def campaign_control_directory(artifact_root: str | Path, campaign_id: str) -> Path:
    if not _CAMPAIGN_RE.fullmatch(campaign_id):
        raise ValueError("campaign_id must be a lowercase SHA-256")
    return Path(artifact_root) / "telemetry" / "campaign-control" / campaign_id


def harvest_lane_lease_held(
    artifact_root: str | Path,
    process_index: int,
) -> bool:
    """Return a point-in-time probe of the kernel-backed host-lane lease."""

    if isinstance(process_index, bool) or not isinstance(process_index, int):
        raise ValueError("process_index must be a nonnegative integer")
    if process_index < 0:
        raise ValueError("process_index must be a nonnegative integer")
    path = (
        Path(artifact_root)
        / "telemetry"
        / "harvest-leases"
        / f"host-p{process_index:05d}.lock"
    )
    try:
        handle = path.open("rb", buffering=0)
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise HarvestControlError(f"cannot probe harvest lane lease: {path}") from exc
    with handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return False


def request_campaign_abort(
    artifact_root: str | Path,
    campaign_id: str,
    *,
    reason: str,
    requested_by: str = "operator",
    clock: Any = time.time,
) -> Path:
    """Durably request that every controller stop at its next safe boundary."""

    reason = str(reason).strip()
    requested_by = str(requested_by).strip()
    if not reason or len(reason) > 2_000:
        raise ValueError("abort reason must contain 1..2000 characters")
    if not requested_by or len(requested_by) > 256:
        raise ValueError("requested_by must contain 1..256 characters")
    path = campaign_control_directory(artifact_root, campaign_id) / "abort.json"
    if path.exists():
        existing = _load_self_hashed_json(path, hash_field="abort_sha256")
        if (
            existing.get("schema_version") != ABORT_SCHEMA
            or existing.get("campaign_id") != campaign_id
        ):
            raise HarvestControlError("existing campaign abort marker is incompatible")
        return path
    now = float(clock())
    if not math.isfinite(now) or now <= 0.0:
        raise ValueError("abort clock must return a positive finite Unix time")
    _atomic_self_hashed_json(
        path,
        {
            "schema_version": ABORT_SCHEMA,
            "campaign_id": campaign_id,
            "requested_at_unix": now,
            "requested_by": requested_by,
            "reason": reason,
        },
        hash_field="abort_sha256",
    )
    return path


def clear_campaign_abort(artifact_root: str | Path, campaign_id: str) -> bool:
    """Remove a prior abort request before a deliberate all-controller resume."""

    directory = campaign_control_directory(artifact_root, campaign_id)
    path = directory / "abort.json"
    if not path.exists():
        return False
    _load_self_hashed_json(path, hash_field="abort_sha256")
    path.unlink()
    _fsync_directory(directory)
    return True


def load_campaign_control(
    artifact_root: str | Path, campaign_id: str
) -> dict[str, Any]:
    """Load every validated host heartbeat and the optional abort request."""

    directory = campaign_control_directory(artifact_root, campaign_id)
    hosts = []
    for path in sorted((directory / "hosts").glob("host-p*.json")):
        value = _load_self_hashed_json(path, hash_field="status_sha256")
        if (
            value.get("schema_version") != STATUS_SCHEMA
            or value.get("campaign_id") != campaign_id
        ):
            raise HarvestControlError(f"incompatible campaign heartbeat: {path}")
        hosts.append(value)
    abort = None
    abort_path = directory / "abort.json"
    if abort_path.exists():
        abort = _load_self_hashed_json(abort_path, hash_field="abort_sha256")
        if abort.get("campaign_id") != campaign_id:
            raise HarvestControlError("campaign abort marker has the wrong identity")
    return {
        "schema_version": CONTROL_SCHEMA,
        "campaign_id": campaign_id,
        "hosts": hosts,
        "abort": abort,
    }


class HarvestCampaignControl:
    """Lifetime per-controller lease plus durable heartbeat for one campaign."""

    def __init__(
        self,
        artifact_root: str | Path,
        metadata: Mapping[str, Any],
        *,
        split: str,
        total_tokens: int,
        stop_after_tokens: int,
        process_count: int,
        process_index: int,
        heartbeat_interval_seconds: float = DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
        clock: Any = time.time,
    ) -> None:
        if not 0 <= process_index < process_count:
            raise ValueError("invalid process index/count")
        if (
            not math.isfinite(heartbeat_interval_seconds)
            or heartbeat_interval_seconds < 0.0
        ):
            raise ValueError("heartbeat interval must be finite and nonnegative")
        self.artifact_root = Path(artifact_root)
        self.campaign_id = campaign_identity(
            metadata,
            split=split,
            total_tokens=total_tokens,
            stop_after_tokens=stop_after_tokens,
            process_count=process_count,
        )
        self.split = split
        self.total_tokens = total_tokens
        self.stop_after_tokens = stop_after_tokens
        self.process_count = process_count
        self.process_index = process_index
        self.heartbeat_interval_seconds = float(heartbeat_interval_seconds)
        self.clock = clock
        self.attempt_id = uuid.uuid4().hex
        self.directory = campaign_control_directory(
            self.artifact_root, self.campaign_id
        )
        # The lease is rooted above campaign identity on purpose: a 2M canary
        # and its 50M continuation share the same scientific host ledger and
        # must never overlap merely because their stop targets differ.
        self.lock_path = (
            self.artifact_root
            / "telemetry"
            / "harvest-leases"
            / f"host-p{process_index:05d}.lock"
        )
        self.status_path = self.directory / "hosts" / f"host-p{process_index:05d}.json"
        self._lock_handle: Any | None = None
        self._last_heartbeat = -math.inf
        self._last_progress: dict[str, Any] | None = None
        self._completed = False

    def __enter__(self) -> "HarvestCampaignControl":
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.lock_path.open("a+b", buffering=0)
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.close()
            raise HarvestLeaseHeld(
                "another live process owns harvest host lane "
                f"{self.process_index} under {self.artifact_root} "
                f"(requested campaign {self.campaign_id})"
            ) from exc
        self._lock_handle = handle
        try:
            self.heartbeat("starting", force=True)
        except BaseException:
            self._lock_handle = None
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()
            raise
        return self

    def _base_status(self, state: str, now: float) -> dict[str, Any]:
        return {
            "schema_version": STATUS_SCHEMA,
            "campaign_id": self.campaign_id,
            "attempt_id": self.attempt_id,
            "split": self.split,
            "total_tokens": self.total_tokens,
            "stop_after_tokens": self.stop_after_tokens,
            "process_count": self.process_count,
            "process_index": self.process_index,
            "pid": os.getpid(),
            "hostname": platform.node(),
            "state": state,
            "heartbeat_unix": now,
        }

    def heartbeat(
        self,
        state: str,
        *,
        progress: Mapping[str, Any] | None = None,
        force: bool = False,
        error: BaseException | None = None,
    ) -> bool:
        """Publish one throttled heartbeat; return whether bytes were written."""

        if self._lock_handle is None:
            raise HarvestControlError("campaign heartbeat requires the held lease")
        state = str(state).strip()
        if not state or len(state) > 64:
            raise ValueError("heartbeat state must contain 1..64 characters")
        now = float(self.clock())
        if not math.isfinite(now) or now <= 0.0:
            raise HarvestControlError("heartbeat clock returned an invalid Unix time")
        if not force and now - self._last_heartbeat < self.heartbeat_interval_seconds:
            return False
        payload = self._base_status(state, now)
        if progress is not None:
            try:
                normalized_progress = json.loads(_canonical_json(dict(progress)))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "heartbeat progress must contain only JSON values"
                ) from exc
            self._last_progress = normalized_progress
        if self._last_progress is not None:
            payload["progress"] = self._last_progress
        if error is not None:
            payload["error_type"] = type(error).__name__
            payload["error"] = str(error)[:2_000]
        _atomic_self_hashed_json(
            self.status_path, payload, hash_field="status_sha256"
        )
        self._last_heartbeat = now
        return True

    def abort_requested(self) -> bool:
        path = self.directory / "abort.json"
        if not path.exists():
            return False
        value = _load_self_hashed_json(path, hash_field="abort_sha256")
        if (
            value.get("schema_version") != ABORT_SCHEMA
            or value.get("campaign_id") != self.campaign_id
        ):
            raise HarvestControlError("campaign abort marker is incompatible")
        return True

    def raise_if_aborted(self, *, globally_requested: bool | None = None) -> None:
        requested = (
            self.abort_requested()
            if globally_requested is None
            else bool(globally_requested)
        )
        if requested:
            self.heartbeat("aborting", force=True)
            raise HarvestAbortRequested(
                f"campaign {self.campaign_id} was aborted at a safe boundary"
            )

    def complete(self, progress: Mapping[str, Any]) -> None:
        self.heartbeat("completed", progress=progress, force=True)
        self._completed = True

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        try:
            if exc is not None and self._lock_handle is not None:
                state = (
                    "aborted"
                    if isinstance(exc, HarvestAbortRequested)
                    else "failed"
                )
                self.heartbeat(state, force=True, error=exc)
            elif not self._completed and self._lock_handle is not None:
                self.heartbeat("released", force=True)
        finally:
            handle = self._lock_handle
            self._lock_handle = None
            if handle is not None:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                finally:
                    handle.close()


__all__ = [
    "ABORT_SCHEMA",
    "CONTROL_SCHEMA",
    "DEFAULT_HEARTBEAT_INTERVAL_SECONDS",
    "HarvestAbortRequested",
    "HarvestCampaignControl",
    "HarvestControlError",
    "HarvestLeaseHeld",
    "STATUS_SCHEMA",
    "campaign_control_directory",
    "campaign_identity",
    "clear_campaign_abort",
    "harvest_lane_lease_held",
    "load_campaign_control",
    "request_campaign_abort",
]
