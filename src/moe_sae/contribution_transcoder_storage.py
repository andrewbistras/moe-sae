"""Content-addressed storage for Experiment-3 routed-input overlays.

Each payload is uploaded directly from a host-local staging directory.  The
small shared control plane contains immutable receipts, finalized host
indexes, and one sealed manifest.  Publication requires a one-to-one match
with every range of the supplied sealed Experiment-2 contribution manifest;
bucket listing is never used as progress or identity.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
import contextlib
import fcntl
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

import numpy as np
from safetensors import safe_open
from safetensors.numpy import load_file, save_file

from moe_sae.contribution_artifacts import (
    canonical_json_bytes,
    canonical_sha256,
    token_identity_sha256,
    validate_contribution_manifest,
)
from moe_sae.contribution_transcoder_artifacts import (
    BF16_LOGICAL_DTYPE,
    MAX_SHARD_TOKENS,
    PROTOCOL_ID,
    SCHEMA_VERSION,
    RoutedInputBatch,
    RoutedInputIntegrityError,
    RoutedInputMetadata,
    RoutedInputRangeError,
    compute_routed_input_statistics,
    deterministic_overlay_range_id,
    merge_routed_input_statistics,
    routed_input_token_identity_sha256,
    seal_routed_input_manifest,
    target_manifest_identity,
    validate_routed_input_batch,
    validate_routed_input_manifest,
    validate_routed_input_metadata,
    validate_routed_input_shard_payload,
    validate_routed_input_statistics,
)
from moe_sae.object_store import (
    ImmutableObjectStore,
    ObjectIdentity,
    ShardCache,
    fingerprint_file,
    join_gcs_uri,
    object_identity_from_mapping,
    sha256_file,
)


MANIFEST_NAME = "manifest.json"
RECEIPT_SCHEMA_VERSION = "gemma4-routed-input-receipt-v1"
HOST_INDEX_SCHEMA_VERSION = "gemma4-routed-input-host-index-v1"

_HOST_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_PAYLOAD_FIELDS = ("routed_input", "sequence_id", "position")
_STAT_FIELDS = (
    "__stats_count",
    "__stats_sum",
    "__stats_squared_norm",
)
_FILE_TENSORS = frozenset((*_PAYLOAD_FIELDS, *_STAT_FIELDS))


class RoutedInputStorageError(RuntimeError):
    """A routed-input storage transaction failed closed."""


@dataclass(frozen=True, slots=True)
class RoutedInputShard:
    """One verified, materialized overlay shard."""

    range_id: str
    target_range_id: str
    range_start: int
    range_end: int
    host_id: str
    sha256: str
    object: ObjectIdentity
    statistics: Mapping[str, Any]
    batch: RoutedInputBatch


def _canonical_copy(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(canonical_json_bytes(dict(value)))


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RoutedInputStorageError(
            f"control record is absent, non-regular, or a symlink: {path}"
        )
    if path.stat().st_size > 2 * 1024 * 1024 * 1024:
        raise RoutedInputStorageError(f"control record is unexpectedly large: {path}")
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RoutedInputStorageError(f"cannot read control JSON: {path}") from error
    if not isinstance(value, dict):
        raise RoutedInputStorageError(f"control record is not an object: {path}")
    return value


def _write_once_json(path: Path, value: Mapping[str, Any]) -> None:
    expected = _canonical_copy(value)
    if path.exists():
        if _read_json(path) != expected:
            raise RoutedInputStorageError(
                f"immutable control record differs from replay: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(expected))
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            pass
        else:
            _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
    if _read_json(path) != expected:
        raise RoutedInputStorageError(
            f"control record failed read-after-write: {path}"
        )


@contextlib.contextmanager
def _control_lock(root: Path, host_id: str | None = None) -> Iterator[None]:
    locks = root / "locks"
    locks.mkdir(parents=True, exist_ok=True)
    name = ".seal.lock" if host_id is None else f".host-{host_id}.lock"
    with (locks / name).open("a+b", buffering=0) as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _self_hash(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    unsigned = dict(value)
    unsigned.pop(field, None)
    return {**unsigned, field: canonical_sha256(unsigned)}


def _validate_self_hash(
    value: Mapping[str, Any], *, field: str, label: str
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or field not in value:
        raise RoutedInputStorageError(f"{label} is missing its self-hash")
    unsigned = {name: item for name, item in value.items() if name != field}
    if value[field] != canonical_sha256(unsigned):
        raise RoutedInputStorageError(f"{label} self-hash mismatch")
    return dict(value)


def _receipt_path(root: Path, host_id: str, range_id: str) -> Path:
    digest = canonical_sha256(
        {"schema_version": SCHEMA_VERSION, "host_id": host_id, "range_id": range_id}
    )
    return root / "receipts" / f"host-{host_id}" / f"{digest}.json"


def _host_index_path(root: Path, host_id: str) -> Path:
    return root / "hosts" / f"host-{host_id}.json"


def _file_metadata(
    *,
    metadata_sha256: str,
    host_id: str,
    range_id: str,
    target_range_id: str,
    range_start: int,
    range_end: int,
    token_identity_sha256: str,
    target_manifest_sha256: str,
) -> dict[str, str]:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "metadata_sha256": metadata_sha256,
        "host_id": host_id,
        "range_id": range_id,
        "target_range_id": target_range_id,
        "range_start": str(range_start),
        "range_end": str(range_end),
        "token_identity_sha256": token_identity_sha256,
        "target_manifest_sha256": target_manifest_sha256,
        "routed_input_logical_dtype": BF16_LOGICAL_DTYPE,
        "statistics_reduction_dtype": "float64",
    }


def _statistics_to_tensors(value: Mapping[str, Any]) -> dict[str, np.ndarray]:
    checked = validate_routed_input_statistics(value)
    return {
        "__stats_count": np.asarray([checked["count"]], dtype=np.int64),
        "__stats_sum": np.ascontiguousarray(
            np.asarray(checked["sum"], dtype=np.float64)
        ),
        "__stats_squared_norm": np.asarray(
            [checked["squared_norm"]], dtype=np.float64
        ),
    }


def _statistics_from_tensors(
    tensors: Mapping[str, np.ndarray], *, expected_tokens: int
) -> dict[str, Any]:
    count = np.asarray(tensors["__stats_count"])
    summed = np.asarray(tensors["__stats_sum"])
    squared = np.asarray(tensors["__stats_squared_norm"])
    if count.dtype != np.int64 or count.shape != (1,):
        raise RoutedInputIntegrityError("stored statistics count has wrong dtype/shape")
    if summed.dtype != np.float64 or summed.shape != (2_816,):
        raise RoutedInputIntegrityError("stored statistics sum has wrong dtype/shape")
    if squared.dtype != np.float64 or squared.shape != (1,):
        raise RoutedInputIntegrityError(
            "stored statistics squared_norm has wrong dtype/shape"
        )
    return validate_routed_input_statistics(
        {
            "count": int(count[0]),
            "sum": summed.tolist(),
            "squared_norm": float(squared[0]),
        },
        expected_tokens=expected_tokens,
    )


def _validate_metadata_against_target(
    metadata: Mapping[str, Any],
    target: Mapping[str, Any],
) -> None:
    target_metadata = target["metadata"]
    required = {
        "checkpoint_sha256": target_metadata["checkpoint_sha256"],
        "maxtext_fork_commit": target_metadata["maxtext_fork_commit"],
        "dataset_fingerprint": target_metadata["dataset_fingerprint"],
        "split": target_metadata["split"],
        "model_id": target_metadata["model_id"],
        "target_layer_one_indexed": target_metadata["target_layer_one_indexed"],
        "target_manifest_sha256": target["manifest_sha256"],
    }
    for name, expected in required.items():
        if metadata[name] != expected:
            raise RoutedInputStorageError(
                f"overlay metadata {name} differs from its target manifest"
            )


def _validate_receipt(
    value: Mapping[str, Any],
    *,
    metadata_sha256: str,
    host_id: str,
    object_store: ImmutableObjectStore | None = None,
) -> dict[str, Any]:
    receipt = _validate_self_hash(
        value, field="receipt_sha256", label="routed-input receipt"
    )
    if set(receipt) != {
        "schema_version",
        "overlay_schema_version",
        "protocol_id",
        "metadata_sha256",
        "host_id",
        "entry",
        "entry_sha256",
        "receipt_sha256",
    }:
        raise RoutedInputStorageError("routed-input receipt fields changed")
    entry = receipt["entry"]
    if (
        receipt["schema_version"] != RECEIPT_SCHEMA_VERSION
        or receipt["overlay_schema_version"] != SCHEMA_VERSION
        or receipt["protocol_id"] != PROTOCOL_ID
        or receipt["metadata_sha256"] != metadata_sha256
        or receipt["host_id"] != host_id
        or not isinstance(entry, Mapping)
        or receipt["entry_sha256"] != canonical_sha256(entry)
        or entry.get("metadata_sha256") != metadata_sha256
        or entry.get("host_id") != host_id
    ):
        raise RoutedInputStorageError("routed-input receipt identity changed")
    identity = object_identity_from_mapping(dict(entry["object"]))
    if identity.sha256 != entry.get("sha256"):
        raise RoutedInputStorageError("receipt object differs from its shard")
    if object_store is not None:
        object_store.verify(identity)
    return receipt


class RoutedInputShardWriter:
    """Publish exact target-bound ranges for one capture host."""

    def __init__(
        self,
        root: str | Path,
        host_id: str,
        metadata: RoutedInputMetadata | Mapping[str, Any],
        *,
        target_manifest: Mapping[str, Any],
        local_buffer_root: str | Path,
        object_prefix: str,
        object_store: ImmutableObjectStore,
        max_tokens: int = MAX_SHARD_TOKENS,
    ) -> None:
        if not isinstance(host_id, str) or _HOST_RE.fullmatch(host_id) is None:
            raise RoutedInputStorageError(f"unsafe host_id {host_id!r}")
        if (
            isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or not 0 < max_tokens <= MAX_SHARD_TOKENS
        ):
            raise RoutedInputStorageError(
                f"max_tokens must be in [1,{MAX_SHARD_TOKENS}]"
            )
        join_gcs_uri(object_prefix, "identity-probe")
        self.root = Path(root).expanduser().resolve()
        self.host_id = host_id
        self.metadata = validate_routed_input_metadata(
            metadata.to_dict()
            if isinstance(metadata, RoutedInputMetadata)
            else metadata
        )
        self.metadata_sha256 = canonical_sha256(self.metadata)
        self.target_manifest = validate_contribution_manifest(target_manifest)
        _validate_metadata_against_target(self.metadata, self.target_manifest)
        self.target_by_id = {
            str(entry["range_id"]): entry
            for entry in self.target_manifest["shards"]
        }
        self.local_buffer_root = (
            Path(local_buffer_root).expanduser().resolve()
            / "routed-input-shards"
            / self.metadata_sha256
            / f"host-{host_id}"
        )
        self.local_buffer_root.mkdir(parents=True, exist_ok=True)
        self.object_prefix = object_prefix.rstrip("/")
        self.object_store = object_store
        self.max_tokens = max_tokens
        (self.root / "receipts" / f"host-{host_id}").mkdir(
            parents=True, exist_ok=True
        )
        (self.root / "hosts").mkdir(parents=True, exist_ok=True)
        if (self.root / MANIFEST_NAME).exists():
            raise RoutedInputStorageError("routed-input store is already sealed")

    def range_id(
        self,
        target_entry: Mapping[str, Any],
        sequence_id: Any,
        position: Any,
    ) -> str:
        checked_target = self._target_entry(target_entry)
        token_sha = token_identity_sha256(sequence_id, position)
        if token_sha != checked_target["token_identity_sha256"]:
            raise RoutedInputRangeError(
                "overlay token keys differ from the target range"
            )
        return deterministic_overlay_range_id(
            split=self.metadata["split"],
            range_start=checked_target["range_start"],
            range_end=checked_target["range_end"],
            target_manifest_sha256=self.target_manifest["manifest_sha256"],
            target_range_id=checked_target["range_id"],
            token_identity_sha256=token_sha,
        )

    def _target_entry(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        if not isinstance(value, Mapping) or not isinstance(
            value.get("range_id"), str
        ):
            raise RoutedInputRangeError("target entry lacks a range_id")
        authoritative = self.target_by_id.get(str(value["range_id"]))
        if authoritative is None or canonical_json_bytes(value) != canonical_json_bytes(
            authoritative
        ):
            raise RoutedInputRangeError(
                "target entry is not an exact shard of the sealed target manifest"
            )
        if authoritative["tokens"] > self.max_tokens:
            raise RoutedInputRangeError("target range exceeds writer max_tokens")
        return authoritative

    def _receipt(
        self, target_entry: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        target = self._target_entry(target_entry)
        # Derive the child range ID from the already sealed target token digest.
        range_id = deterministic_overlay_range_id(
            split=self.metadata["split"],
            range_start=target["range_start"],
            range_end=target["range_end"],
            target_manifest_sha256=self.target_manifest["manifest_sha256"],
            target_range_id=target["range_id"],
            token_identity_sha256=target["token_identity_sha256"],
        )
        path = _receipt_path(self.root, self.host_id, range_id)
        if not path.exists():
            return None
        return _validate_receipt(
            _read_json(path),
            metadata_sha256=self.metadata_sha256,
            host_id=self.host_id,
            object_store=self.object_store,
        )

    def has_target_range(self, target_entry: Mapping[str, Any]) -> bool:
        return self._receipt(target_entry) is not None

    def write_range(
        self,
        target_entry: Mapping[str, Any],
        batch: RoutedInputBatch | Mapping[str, Any],
    ) -> dict[str, Any]:
        """Upload and durably receipt one exact sealed-target range."""

        target = self._target_entry(target_entry)
        checked = validate_routed_input_batch(batch, max_tokens=self.max_tokens)
        if checked.tokens != target["tokens"]:
            raise RoutedInputRangeError(
                "overlay payload count differs from its target range"
            )
        token_sha = routed_input_token_identity_sha256(checked)
        if token_sha != target["token_identity_sha256"]:
            raise RoutedInputRangeError(
                "overlay token-key sequence differs from its target range"
            )
        range_id = deterministic_overlay_range_id(
            split=self.metadata["split"],
            range_start=target["range_start"],
            range_end=target["range_end"],
            target_manifest_sha256=self.target_manifest["manifest_sha256"],
            target_range_id=target["range_id"],
            token_identity_sha256=token_sha,
        )
        existing = self._receipt(target)
        if existing is not None:
            return dict(existing["entry"])

        statistics = compute_routed_input_statistics(checked)
        tensors = {
            name: np.ascontiguousarray(np.asarray(getattr(checked, name)))
            for name in _PAYLOAD_FIELDS
        }
        tensors.update(_statistics_to_tensors(statistics))
        file_metadata = _file_metadata(
            metadata_sha256=self.metadata_sha256,
            host_id=self.host_id,
            range_id=range_id,
            target_range_id=target["range_id"],
            range_start=target["range_start"],
            range_end=target["range_end"],
            token_identity_sha256=token_sha,
            target_manifest_sha256=self.target_manifest["manifest_sha256"],
        )
        descriptor, name = tempfile.mkstemp(
            prefix="routed-input-",
            suffix=".safetensors",
            dir=self.local_buffer_root,
        )
        os.close(descriptor)
        staged = Path(name)
        try:
            save_file(tensors, str(staged), metadata=file_metadata)
            with staged.open("rb") as handle:
                os.fsync(handle.fileno())
            fingerprint = fingerprint_file(staged)
            relative = (
                f"shards/host-{self.host_id}/"
                f"{fingerprint.sha256}.safetensors"
            )
            uri = join_gcs_uri(self.object_prefix, *relative.split("/"))
            identity = self.object_store.put_if_absent(
                staged,
                uri,
                sha256=fingerprint.sha256,
                source_fingerprint=fingerprint,
            )
            entry = {
                "range_id": range_id,
                "target_range_id": target["range_id"],
                "range_start": target["range_start"],
                "range_end": target["range_end"],
                "tokens": checked.tokens,
                "host_id": self.host_id,
                "path": relative,
                "sha256": fingerprint.sha256,
                "token_identity_sha256": token_sha,
                "metadata_sha256": self.metadata_sha256,
                "target_manifest_sha256": self.target_manifest[
                    "manifest_sha256"
                ],
                "statistics": statistics,
                "object": identity.as_dict(),
            }
            unsigned_receipt = {
                "schema_version": RECEIPT_SCHEMA_VERSION,
                "overlay_schema_version": SCHEMA_VERSION,
                "protocol_id": PROTOCOL_ID,
                "metadata_sha256": self.metadata_sha256,
                "host_id": self.host_id,
                "entry": entry,
                "entry_sha256": canonical_sha256(entry),
            }
            receipt = _self_hash(unsigned_receipt, "receipt_sha256")
            with _control_lock(self.root, self.host_id):
                if (self.root / MANIFEST_NAME).exists():
                    raise RoutedInputStorageError(
                        "routed-input store sealed before range commit"
                    )
                _write_once_json(
                    _receipt_path(self.root, self.host_id, range_id), receipt
                )
            return dict(entry)
        finally:
            staged.unlink(missing_ok=True)

    def finalize_host(self) -> Path:
        """Freeze this host's exact receipt inventory."""

        with _control_lock(self.root, self.host_id):
            receipts: list[dict[str, Any]] = []
            receipt_root = self.root / "receipts" / f"host-{self.host_id}"
            for path in sorted(receipt_root.glob("*.json")):
                receipts.append(
                    _validate_receipt(
                        _read_json(path),
                        metadata_sha256=self.metadata_sha256,
                        host_id=self.host_id,
                        object_store=self.object_store,
                    )
                )
            references = [
                {
                    "range_id": receipt["entry"]["range_id"],
                    "target_range_id": receipt["entry"]["target_range_id"],
                    "range_start": receipt["entry"]["range_start"],
                    "range_end": receipt["entry"]["range_end"],
                    "receipt_sha256": receipt["receipt_sha256"],
                }
                for receipt in receipts
            ]
            references.sort(key=lambda item: (item["range_start"], item["range_id"]))
            index = _self_hash(
                {
                    "schema_version": HOST_INDEX_SCHEMA_VERSION,
                    "overlay_schema_version": SCHEMA_VERSION,
                    "protocol_id": PROTOCOL_ID,
                    "metadata_sha256": self.metadata_sha256,
                    "host_id": self.host_id,
                    "complete": True,
                    "receipts": references,
                },
                "index_sha256",
            )
            path = _host_index_path(self.root, self.host_id)
            _write_once_json(path, index)
            return path


def _validate_host_index(
    value: Mapping[str, Any], *, metadata_sha256: str
) -> dict[str, Any]:
    index = _validate_self_hash(
        value, field="index_sha256", label="routed-input host index"
    )
    if set(index) != {
        "schema_version",
        "overlay_schema_version",
        "protocol_id",
        "metadata_sha256",
        "host_id",
        "complete",
        "receipts",
        "index_sha256",
    }:
        raise RoutedInputStorageError("routed-input host-index fields changed")
    if (
        index["schema_version"] != HOST_INDEX_SCHEMA_VERSION
        or index["overlay_schema_version"] != SCHEMA_VERSION
        or index["protocol_id"] != PROTOCOL_ID
        or index["metadata_sha256"] != metadata_sha256
        or index["complete"] is not True
        or not isinstance(index["host_id"], str)
        or _HOST_RE.fullmatch(index["host_id"]) is None
        or not isinstance(index["receipts"], list)
    ):
        raise RoutedInputStorageError("routed-input host-index identity changed")
    return index


def publish_routed_input_manifest(
    root: str | Path,
    metadata: RoutedInputMetadata | Mapping[str, Any],
    *,
    target_manifest: Mapping[str, Any],
    object_prefix: str,
    object_store: ImmutableObjectStore,
    expected_hosts: int,
) -> Path:
    """Verify every host/object/range, then atomically seal the overlay."""

    if (
        isinstance(expected_hosts, bool)
        or not isinstance(expected_hosts, int)
        or expected_hosts <= 0
    ):
        raise RoutedInputStorageError("expected_hosts must be positive")
    join_gcs_uri(object_prefix, "identity-probe")
    selected_object_prefix = object_prefix.rstrip("/")
    selected_root = Path(root).expanduser().resolve()
    selected_metadata = validate_routed_input_metadata(
        metadata.to_dict() if isinstance(metadata, RoutedInputMetadata) else metadata
    )
    metadata_sha = canonical_sha256(selected_metadata)
    target = validate_contribution_manifest(target_manifest)
    _validate_metadata_against_target(selected_metadata, target)
    manifest_path = selected_root / MANIFEST_NAME
    with _control_lock(selected_root):
        if manifest_path.exists():
            existing = load_routed_input_manifest(
                selected_root, target_manifest=target
            )
            if existing["metadata_sha256"] != metadata_sha:
                raise RoutedInputStorageError(
                    "sealed overlay differs from this publication"
                )
            return manifest_path
        indexes = [
            _validate_host_index(
                _read_json(path), metadata_sha256=metadata_sha
            )
            for path in sorted((selected_root / "hosts").glob("host-*.json"))
        ]
        if (
            len(indexes) != expected_hosts
            or len({index["host_id"] for index in indexes}) != expected_hosts
        ):
            raise RoutedInputStorageError(
                "not every expected capture host has finalized"
            )
        entries: list[dict[str, Any]] = []
        for index in indexes:
            for reference in index["receipts"]:
                path = _receipt_path(
                    selected_root, index["host_id"], reference["range_id"]
                )
                receipt = _validate_receipt(
                    _read_json(path),
                    metadata_sha256=metadata_sha,
                    host_id=index["host_id"],
                    object_store=object_store,
                )
                entry = receipt["entry"]
                identity = object_identity_from_mapping(dict(entry["object"]))
                expected_uri = join_gcs_uri(
                    selected_object_prefix, *str(entry["path"]).split("/")
                )
                if identity.uri != expected_uri:
                    raise RoutedInputStorageError(
                        "receipt object URI is outside the publication prefix"
                    )
                expected_reference = {
                    "range_id": entry["range_id"],
                    "target_range_id": entry["target_range_id"],
                    "range_start": entry["range_start"],
                    "range_end": entry["range_end"],
                    "receipt_sha256": receipt["receipt_sha256"],
                }
                if reference != expected_reference:
                    raise RoutedInputStorageError(
                        "host index differs from its immutable receipt"
                    )
                entries.append(dict(entry))
        entries.sort(key=lambda item: (item["range_start"], item["range_id"]))
        if len(entries) != len(target["shards"]):
            raise RoutedInputRangeError(
                "overlay does not contain exactly one shard per target range"
            )
        for entry, target_entry in zip(entries, target["shards"], strict=True):
            exact = {
                "target_range_id": target_entry["range_id"],
                "range_start": target_entry["range_start"],
                "range_end": target_entry["range_end"],
                "tokens": target_entry["tokens"],
                "token_identity_sha256": target_entry[
                    "token_identity_sha256"
                ],
            }
            for name, expected in exact.items():
                if entry.get(name) != expected:
                    raise RoutedInputRangeError(
                        f"overlay {name} differs from target range"
                    )
        unsigned = {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "sealed": True,
            "metadata": selected_metadata,
            "metadata_sha256": metadata_sha,
            "target_manifest": target_manifest_identity(target),
            "total_tokens": target["total_tokens"],
            "shard_count": target["shard_count"],
            "statistics": merge_routed_input_statistics(
                [entry["statistics"] for entry in entries]
            ),
            "shards": entries,
        }
        sealed = seal_routed_input_manifest(unsigned)
        validated = validate_routed_input_manifest(
            sealed, target_manifest=target
        )
        _write_once_json(manifest_path, validated)
        return manifest_path


def load_routed_input_manifest(
    root: str | Path,
    *,
    target_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    path = Path(root).expanduser().resolve() / MANIFEST_NAME
    if not path.exists():
        raise RoutedInputStorageError(f"missing routed-input manifest: {path}")
    return validate_routed_input_manifest(
        _read_json(path), target_manifest=target_manifest
    )


class RoutedInputDataset:
    """Hash-verifying reader for a sealed, target-bound input overlay."""

    def __init__(
        self,
        root: str | Path,
        *,
        object_prefix: str,
        cache: ShardCache,
        target_manifest: Mapping[str, Any],
    ) -> None:
        if not isinstance(cache, ShardCache):
            raise RoutedInputStorageError("overlay reader requires an exact ShardCache")
        join_gcs_uri(object_prefix, "identity-probe")
        self.root = Path(root).expanduser().resolve()
        self.object_prefix = object_prefix.rstrip("/")
        self.cache = cache
        self.target_manifest = validate_contribution_manifest(target_manifest)
        self.manifest = load_routed_input_manifest(
            self.root, target_manifest=self.target_manifest
        )

    @property
    def total_tokens(self) -> int:
        return int(self.manifest["total_tokens"])

    @property
    def manifest_sha256(self) -> str:
        return str(self.manifest["manifest_sha256"])

    def _load_entry(self, entry: Mapping[str, Any]) -> RoutedInputShard:
        identity = object_identity_from_mapping(dict(entry["object"]))
        expected_uri = join_gcs_uri(
            self.object_prefix, *str(entry["path"]).split("/")
        )
        if identity.uri != expected_uri:
            raise RoutedInputStorageError(
                "overlay object URI is outside its bound object prefix"
            )
        path = self.cache.materialize(identity)
        digest, size = sha256_file(path)
        if digest != identity.sha256 or size != identity.size_bytes:
            raise RoutedInputStorageError(
                "materialized overlay failed SHA-256/size verification"
            )
        expected_metadata = _file_metadata(
            metadata_sha256=self.manifest["metadata_sha256"],
            host_id=entry["host_id"],
            range_id=entry["range_id"],
            target_range_id=entry["target_range_id"],
            range_start=entry["range_start"],
            range_end=entry["range_end"],
            token_identity_sha256=entry["token_identity_sha256"],
            target_manifest_sha256=entry["target_manifest_sha256"],
        )
        try:
            with safe_open(path, framework="numpy") as handle:
                if frozenset(handle.keys()) != _FILE_TENSORS:
                    raise RoutedInputStorageError(
                        "overlay safetensors contains missing or extra fields"
                    )
                if (handle.metadata() or {}) != expected_metadata:
                    raise RoutedInputStorageError(
                        "overlay safetensors metadata changed"
                    )
            tensors = load_file(path)
        except RoutedInputStorageError:
            raise
        except Exception as error:
            raise RoutedInputStorageError(
                f"cannot read overlay safetensors: {path}"
            ) from error
        batch = validate_routed_input_shard_payload(
            RoutedInputBatch(
                routed_input=tensors["routed_input"],
                sequence_id=tensors["sequence_id"],
                position=tensors["position"],
            ),
            entry,
        )
        stored_statistics = _statistics_from_tensors(
            tensors, expected_tokens=batch.tokens
        )
        computed_statistics = compute_routed_input_statistics(batch)
        for observed in (stored_statistics, computed_statistics):
            if canonical_json_bytes(observed) != canonical_json_bytes(
                entry["statistics"]
            ):
                raise RoutedInputIntegrityError(
                    "overlay payload statistics differ from sealed manifest"
                )
        return RoutedInputShard(
            range_id=entry["range_id"],
            target_range_id=entry["target_range_id"],
            range_start=entry["range_start"],
            range_end=entry["range_end"],
            host_id=entry["host_id"],
            sha256=entry["sha256"],
            object=identity,
            statistics=entry["statistics"],
            batch=batch,
        )

    def iter_shards(self) -> Iterator[RoutedInputShard]:
        for entry in self.manifest["shards"]:
            yield self._load_entry(entry)

    def _entries_for_range(
        self, start: int, end: int
    ) -> list[Mapping[str, Any]]:
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
            or not 0 <= start <= end <= self.total_tokens
        ):
            raise RoutedInputRangeError(
                f"token range must satisfy 0 <= start <= end <= {self.total_tokens}"
            )
        return [
            entry
            for entry in self.manifest["shards"]
            if max(start, int(entry["range_start"]))
            < min(end, int(entry["range_end"]))
        ]

    def token_range_object_identities(
        self, start: int, end: int
    ) -> tuple[ObjectIdentity, ...]:
        """Return the exact generation-pinned objects intersecting a range."""

        return tuple(
            object_identity_from_mapping(dict(entry["object"]))
            for entry in self._entries_for_range(start, end)
        )

    def prefetch_token_range(self, start: int, end: int) -> dict[str, int]:
        """Materialize and then resident-only prove every overlapping shard."""

        identities = self.token_range_object_identities(start, end)
        paths = self.cache.materialize_many(identities, max_workers=4)
        resident = self.cache.require_resident_many(identities)
        if paths != resident:
            raise RoutedInputStorageError(
                "prefetched overlay differs from its resident-only proof"
            )
        return {
            "objects": len(paths),
            "bytes": sum(path.stat().st_size for path in paths),
        }

    def require_resident_token_range(
        self, start: int, end: int
    ) -> dict[str, int]:
        """Prove complete local residency without permitting a remote read."""

        identities = self.token_range_object_identities(start, end)
        paths = self.cache.require_resident_many(identities)
        return {
            "objects": len(paths),
            "bytes": sum(path.stat().st_size for path in paths),
        }

    def iter_token_range(
        self, start: int, end: int
    ) -> Iterator[RoutedInputBatch]:
        yielded = 0
        for entry in self._entries_for_range(start, end):
            overlap_start = max(start, int(entry["range_start"]))
            overlap_end = min(end, int(entry["range_end"]))
            shard = self._load_entry(entry)
            local_start = overlap_start - shard.range_start
            local_end = overlap_end - shard.range_start
            selected = RoutedInputBatch(
                routed_input=np.ascontiguousarray(
                    shard.batch.routed_input[local_start:local_end]
                ),
                sequence_id=np.ascontiguousarray(
                    shard.batch.sequence_id[local_start:local_end]
                ),
                position=np.ascontiguousarray(
                    shard.batch.position[local_start:local_end]
                ),
            )
            yielded += selected.tokens
            yield selected
        if yielded != end - start:
            raise RoutedInputRangeError(
                f"manifest yielded {yielded} routed-input tokens for [{start},{end})"
            )
