"""Deterministic offline C4 packing and Grain ArrayRecord input.

The pinned MaxText pretraining pipeline performs packing *after* host sharding
and makes the pack layout depend on the local batch size.  That is unsuitable
for a benchmark which compares several batch sizes.  This module therefore
owns the small data boundary: C4 documents are tokenized and packed once, and
the TPU runtime only batches already-complete records.

Optional data dependencies are imported lazily so importing :mod:`moe_sae.cli`
does not initialize Hugging Face, Grain, or JAX.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
import gzip
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
from typing import Any

import numpy as np

from moe_sae.config import C4_REPOSITORY, C4_REVISION
from moe_sae.model import (
    GEMMA4_REPOSITORY_FILES,
    expected_hf_repository_inventory_sha256,
)


PACKED_DATASET_SCHEMA = "gemma4-c4-packed-arrayrecord-v2"
PACKED_RECORD_FORMAT = "moe-sae-packed-sequence-v1"
PACKED_RECORD_MAGIC = b"MSAEPK01"
PACKED_RECORD_VERSION = 1
ALLENAI_C4_REPO_ID = C4_REPOSITORY
ALLENAI_C4_REVISION = C4_REVISION
DEFAULT_SEQUENCE_LENGTH = 1_024
DEFAULT_PROCESS_COUNT = 4
DEFAULT_PER_DEVICE_SEQUENCE_BATCHES = (1, 2, 4)
DEFAULT_LOCAL_CHIPS = 4
PAD_ID = 0
EOS_ID = 1
BOS_ID = 2
_HEADER = struct.Struct("<8sHHIQ")
_REVISION = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class StagingError(RuntimeError):
    """Raised when deterministic staging cannot be proven."""


@dataclass(frozen=True, slots=True)
class SourceFile:
    """One immutable compressed source object in the bounded C4 input."""

    path: str
    sha256: str
    size_bytes: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


ALLENAI_C4_SOURCE_FILES: Mapping[str, tuple[SourceFile, ...]] = {
    "train": (
        SourceFile(
            path="en/c4-train.00000-of-01024.json.gz",
            sha256="8ef8d75b0e045dec4aa5123a671b4564466b0707086a7ed1ba8721626dfffbc9",
            size_bytes=319_308_785,
        ),
    ),
    "validation": (
        SourceFile(
            path="en/c4-validation.00000-of-00008.json.gz",
            sha256="1f25b6af12da84115301d4ee93ea5246c8fea5bb4a2008472794d95b917cc97f",
            size_bytes=40_471_190,
        ),
    ),
}
GEMMA4_TOKENIZER_SOURCE_FILES = tuple(
    entry
    for entry in GEMMA4_REPOSITORY_FILES
    if entry.path in {"tokenizer.json", "tokenizer_config.json"}
)
GEMMA4_TOKENIZER_INVENTORY_SHA256 = expected_hf_repository_inventory_sha256(
    GEMMA4_TOKENIZER_SOURCE_FILES
)


@dataclass(frozen=True, slots=True)
class PackedRecord:
    """One validated fixed-length model input record."""

    record_index: int
    scientific_tokens: int
    token_ids: np.ndarray
    positions: np.ndarray
    segment_ids: np.ndarray


@dataclass(frozen=True, slots=True)
class StagingResult:
    """A sealed local or GCS staging result."""

    manifest_location: str
    grain_files: str
    scientific_tokens: int
    record_count: int
    source_examples: int
    resumed: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "manifest_location": self.manifest_location,
            "grain_files": self.grain_files,
            "scientific_tokens": self.scientific_tokens,
            "record_count": self.record_count,
            "source_examples": self.source_examples,
            "resumed": self.resumed,
        }


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(_canonical_json(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    try:
        descriptor = os.open(path.parent, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _validate_revision(value: str, label: str) -> None:
    if not _REVISION.fullmatch(value):
        raise ValueError(f"{label} must be an immutable 40-character commit SHA")


def _expected_record_bytes(sequence_length: int) -> int:
    return _HEADER.size + sequence_length * (4 + 2 + 2)


def _validate_arrays(
    token_ids: np.ndarray,
    positions: np.ndarray,
    segment_ids: np.ndarray,
    *,
    sequence_length: int,
) -> int:
    arrays = (token_ids, positions, segment_ids)
    if any(array.shape != (sequence_length,) for array in arrays):
        raise StagingError(
            f"packed arrays must all have shape ({sequence_length},)"
        )
    if not np.issubdtype(token_ids.dtype, np.integer):
        raise StagingError("packed token IDs must be integral")
    if not np.issubdtype(positions.dtype, np.integer):
        raise StagingError("packed positions must be integral")
    if not np.issubdtype(segment_ids.dtype, np.integer):
        raise StagingError("packed segment IDs must be integral")
    if token_ids.size and (int(token_ids.min()) < 0 or int(token_ids.max()) > 2**31 - 1):
        raise StagingError("token IDs must fit non-negative int32")
    if positions.size and (int(positions.min()) < 0 or int(positions.max()) >= sequence_length):
        raise StagingError("packed positions are outside the sequence")
    if segment_ids.size and (
        int(segment_ids.min()) < 0 or int(segment_ids.max()) > 2**16 - 1
    ):
        raise StagingError("packed segment IDs must fit uint16")

    padded = segment_ids == 0
    if np.any(token_ids[padded] != PAD_ID) or np.any(positions[padded] != 0):
        raise StagingError("padding must have token, position, and segment equal to zero")
    nonzero = segment_ids[~padded]
    if nonzero.size:
        if not np.array_equal(np.flatnonzero(~padded), np.arange(nonzero.size)):
            raise StagingError("padding is only permitted at the record tail")
        unique = np.unique(nonzero)
        if not np.array_equal(unique, np.arange(1, unique.size + 1)):
            raise StagingError("segment IDs must be contiguous and one-indexed")
        if nonzero[0] != 1 or np.any(np.diff(nonzero.astype(np.int64)) < 0):
            raise StagingError("packed segments must appear in increasing order")
        for segment in unique:
            indexes = np.flatnonzero(segment_ids == segment)
            if not np.array_equal(indexes, np.arange(indexes[0], indexes[-1] + 1)):
                raise StagingError("each packed segment must be contiguous")
            expected_positions = np.arange(indexes.size)
            if not np.array_equal(positions[indexes], expected_positions):
                raise StagingError("positions must restart and increase within each segment")
            if token_ids[indexes[0]] != BOS_ID or token_ids[indexes[-1]] != EOS_ID:
                raise StagingError("every packed segment must be BOS/content/EOS")

    scientific = (~padded) & ~np.isin(token_ids, (PAD_ID, EOS_ID, BOS_ID))
    return int(np.count_nonzero(scientific))


def encode_packed_record(
    record_index: int,
    token_ids: np.ndarray,
    positions: np.ndarray,
    segment_ids: np.ndarray,
    *,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
) -> bytes:
    """Encode one record using a compact, versioned, endian-fixed payload."""

    if record_index < 0 or record_index > 2**64 - 1:
        raise ValueError("record_index must fit non-negative uint64")
    token_ids = np.asarray(token_ids)
    positions = np.asarray(positions)
    segment_ids = np.asarray(segment_ids)
    scientific = _validate_arrays(
        token_ids, positions, segment_ids, sequence_length=sequence_length
    )
    header = _HEADER.pack(
        PACKED_RECORD_MAGIC,
        PACKED_RECORD_VERSION,
        sequence_length,
        scientific,
        record_index,
    )
    return b"".join(
        (
            header,
            np.asarray(token_ids, dtype="<i4").tobytes(order="C"),
            np.asarray(positions, dtype="<u2").tobytes(order="C"),
            np.asarray(segment_ids, dtype="<u2").tobytes(order="C"),
        )
    )


def decode_packed_record(
    payload: bytes,
    *,
    expected_index: int | None = None,
    expected_sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
) -> PackedRecord:
    """Decode and fully validate one packed ArrayRecord payload."""

    if len(payload) != _expected_record_bytes(expected_sequence_length):
        raise StagingError(
            "packed record byte length is incompatible with the fixed schema"
        )
    magic, version, sequence_length, claimed_scientific, record_index = _HEADER.unpack_from(
        payload
    )
    if (
        magic != PACKED_RECORD_MAGIC
        or version != PACKED_RECORD_VERSION
        or sequence_length != expected_sequence_length
    ):
        raise StagingError("packed record header is incompatible")
    if expected_index is not None and record_index != expected_index:
        raise StagingError(
            f"packed record index changed: {record_index} != {expected_index}"
        )
    cursor = _HEADER.size
    token_bytes = sequence_length * 4
    small_bytes = sequence_length * 2
    token_ids = np.frombuffer(payload, dtype="<i4", count=sequence_length, offset=cursor).copy()
    cursor += token_bytes
    positions = np.frombuffer(payload, dtype="<u2", count=sequence_length, offset=cursor).copy()
    cursor += small_bytes
    segment_ids = np.frombuffer(payload, dtype="<u2", count=sequence_length, offset=cursor).copy()
    if cursor + small_bytes != len(payload):
        raise StagingError("packed record contains trailing bytes")
    actual_scientific = _validate_arrays(
        token_ids, positions, segment_ids, sequence_length=sequence_length
    )
    if claimed_scientific != actual_scientific:
        raise StagingError("packed record scientific-token count is corrupt")
    return PackedRecord(
        record_index=int(record_index),
        scientific_tokens=actual_scientific,
        token_ids=token_ids,
        positions=positions,
        segment_ids=segment_ids,
    )


class OrderedPacker:
    """Streaming next-fit packer which preserves document and chunk order."""

    def __init__(self, sequence_length: int = DEFAULT_SEQUENCE_LENGTH):
        if sequence_length < 3 or sequence_length > 2**16 - 1:
            raise ValueError("sequence_length must be in [3, 65535]")
        self.sequence_length = sequence_length
        self._tokens = np.zeros(sequence_length, dtype=np.int32)
        self._positions = np.zeros(sequence_length, dtype=np.uint16)
        self._segments = np.zeros(sequence_length, dtype=np.uint16)
        self._cursor = 0
        self._next_segment = 1

    @property
    def empty(self) -> bool:
        return self._cursor == 0

    def _emit(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.empty:
            raise StagingError("cannot emit an empty pack")
        result = (
            self._tokens.copy(),
            self._positions.copy(),
            self._segments.copy(),
        )
        self._tokens.fill(0)
        self._positions.fill(0)
        self._segments.fill(0)
        self._cursor = 0
        self._next_segment = 1
        return result

    def add_content(
        self, content_ids: Sequence[int]
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Add one document, chunking it without dropping content tokens."""

        content = np.asarray(content_ids, dtype=np.int64)
        if content.ndim != 1:
            raise StagingError("tokenizer output must be a one-dimensional token list")
        if content.size and (int(content.min()) < 0 or int(content.max()) > 2**31 - 1):
            raise StagingError("tokenizer emitted an ID outside non-negative int32")
        if content.size == 0:
            return []
        emitted: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        content_per_segment = self.sequence_length - 2
        for start in range(0, int(content.size), content_per_segment):
            chunk = content[start : start + content_per_segment].astype(np.int32)
            segment = np.concatenate(
                (
                    np.asarray([BOS_ID], dtype=np.int32),
                    chunk,
                    np.asarray([EOS_ID], dtype=np.int32),
                )
            )
            if self._cursor and self._cursor + segment.size > self.sequence_length:
                emitted.append(self._emit())
            end = self._cursor + segment.size
            if end > self.sequence_length:
                raise AssertionError("chunked segment does not fit a fresh pack")
            self._tokens[self._cursor : end] = segment
            self._positions[self._cursor : end] = np.arange(
                segment.size, dtype=np.uint16
            )
            self._segments[self._cursor : end] = self._next_segment
            self._cursor = end
            self._next_segment += 1
            if self._cursor == self.sequence_length:
                emitted.append(self._emit())
        return emitted

    def flush(self) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        return [] if self.empty else [self._emit()]


def lane_for_record(record_index: int, *, process_count: int = DEFAULT_PROCESS_COUNT) -> int:
    """Host lane used by the fixed strided multi-host reader."""

    if record_index < 0 or process_count <= 0:
        raise ValueError("record_index must be non-negative and process_count positive")
    return record_index % process_count


def _lane_quota(total: int, process_count: int, process_index: int) -> int:
    return total // process_count + (1 if process_index < total % process_count else 0)


def _lanes_ready(lanes: Sequence[int], target: int) -> bool:
    count = len(lanes)
    return all(value >= _lane_quota(target, count, index) for index, value in enumerate(lanes))


def _array_record_writer(path: Path):
    try:
        from array_record.python.array_record_module import ArrayRecordWriter
    except ImportError as error:
        raise StagingError(
            "ArrayRecord writer is unavailable; run `uv sync --extra data`"
        ) from error
    return ArrayRecordWriter(str(path), "group_size:1")


def _array_record_source(files: Sequence[str]):
    try:
        import grain.python as grain
    except ImportError as error:
        raise StagingError(
            "Grain reader is unavailable; run `uv sync --extra data` or `--extra tpu`"
        ) from error
    return grain.ArrayRecordDataSource(list(files))


def _manifest_identity(
    *,
    split: str,
    source_identity: Mapping[str, Any],
    tokenizer_id: str,
    tokenizer_revision: str,
    sequence_length: int,
    process_count: int,
) -> dict[str, Any]:
    source = _validated_source_identity(source_identity)
    return {
        "schema_version": PACKED_DATASET_SCHEMA,
        "dataset": {
            "name": source["repo_id"],
            "config": "en",
            "revision": source["revision"],
            "split": split,
        },
        "source": source,
        "tokenizer": {
            "id": tokenizer_id,
            "revision": tokenizer_revision,
            "repository_inventory_sha256": GEMMA4_TOKENIZER_INVENTORY_SHA256,
            "files": [
                entry.as_dict() for entry in GEMMA4_TOKENIZER_SOURCE_FILES
            ],
            "padding_id": PAD_ID,
            "eos_id": EOS_ID,
            "bos_id": BOS_ID,
        },
        "sequence_length": sequence_length,
        "record_format": {
            "name": PACKED_RECORD_FORMAT,
            "arrayrecord_options": "group_size:1",
            "token_ids": f"int32[{sequence_length}]",
            "positions": f"uint16[{sequence_length}]",
            "segment_ids": f"uint16[{sequence_length}]",
        },
        "packing": {
            "strategy": "ordered_next_fit_offline",
            "exclude_from_scientific_count": ["padding", "bos", "eos"],
            "special_token_ids": {"padding": PAD_ID, "eos": EOS_ID, "bos": BOS_ID},
        },
        "consumer": {
            "process_count": process_count,
            "host_sharding": "record_index_mod_process_count",
            "per_device_sequence_batches": list(DEFAULT_PER_DEVICE_SEQUENCE_BATCHES),
        },
    }


def _validated_source_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return a canonical copy of an exact, content-bound source identity."""

    required = {"provider", "repo_type", "repo_id", "revision", "files"}
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError(
            "source_identity must contain provider, repo_type, repo_id, revision, and files"
        )
    provider = value.get("provider")
    repo_type = value.get("repo_type")
    repo_id = value.get("repo_id")
    revision = value.get("revision")
    files = value.get("files")
    if not isinstance(provider, str) or not provider:
        raise ValueError("source_identity provider must be a non-empty string")
    if repo_type != "dataset":
        raise ValueError("source_identity repo_type must be 'dataset'")
    if (
        not isinstance(repo_id, str)
        or not repo_id
        or repo_id.startswith("/")
        or repo_id.endswith("/")
    ):
        raise ValueError("source_identity repo_id is invalid")
    if not isinstance(revision, str):
        raise ValueError("source_identity revision must be a string")
    _validate_revision(revision, "source_identity revision")
    if not isinstance(files, Sequence) or isinstance(files, (str, bytes)) or not files:
        raise ValueError("source_identity files must be a non-empty sequence")

    canonical_files: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in files:
        if not isinstance(entry, Mapping) or set(entry) != {
            "path",
            "sha256",
            "size_bytes",
        }:
            raise ValueError(
                "each source_identity file must contain path, sha256, and size_bytes"
            )
        path = entry.get("path")
        digest = entry.get("sha256")
        size = entry.get("size_bytes")
        if (
            not isinstance(path, str)
            or not path
            or Path(path).is_absolute()
            or ".." in Path(path).parts
            or path in seen
        ):
            raise ValueError("source_identity contains an unsafe or duplicate path")
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise ValueError("source_identity contains an invalid SHA-256")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError("source_identity contains an invalid file size")
        seen.add(path)
        canonical_files.append(
            {"path": path, "sha256": digest, "size_bytes": size}
        )
    return {
        "provider": provider,
        "repo_type": repo_type,
        "repo_id": repo_id,
        "revision": revision,
        "files": canonical_files,
    }


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StagingError(f"cannot read staging state: {path}") from error
    if not isinstance(value, dict):
        raise StagingError(f"staging state must be an object: {path}")
    return value


def _validate_file_entries(root: Path, files: Sequence[Mapping[str, Any]]) -> None:
    expected_first = 0
    for entry in files:
        if set(entry) != {
            "uri",
            "sha256",
            "size_bytes",
            "first_record_index",
            "record_count",
            "scientific_tokens",
        }:
            raise StagingError("staging file inventory has incompatible fields")
        uri = entry.get("uri")
        digest = entry.get("sha256")
        size = entry.get("size_bytes")
        first = entry.get("first_record_index")
        records = entry.get("record_count")
        scientific = entry.get("scientific_tokens")
        if (
            not isinstance(uri, str)
            or not uri
            or Path(uri).name != uri
            or not uri.endswith(".array_record")
        ):
            raise StagingError("staging file inventory contains an unsafe URI")
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise StagingError("staging file inventory has an invalid SHA-256")
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
            or isinstance(first, bool)
            or not isinstance(first, int)
            or isinstance(records, bool)
            or not isinstance(records, int)
            or records <= 0
            or isinstance(scientific, bool)
            or not isinstance(scientific, int)
            or scientific < 0
        ):
            raise StagingError("staging file inventory has invalid counts")
        if first != expected_first:
            raise StagingError("staging file inventory has a record gap or overlap")
        path = root / uri
        actual_digest, actual_size = _sha256_file(path)
        if actual_digest != digest or actual_size != size:
            raise StagingError(f"completed staging shard changed: {path}")
        expected_first += records


def _validate_aggregate_state(
    value: Mapping[str, Any],
    *,
    files: Sequence[Mapping[str, Any]],
    target_scientific_tokens: int,
    process_count: int,
    require_padded_record_count: bool,
) -> tuple[int, int, int, list[int]]:
    """Validate resume/seal counters without trusting coercible JSON values."""

    source_examples = value.get("source_examples")
    record_index = value.get("next_record_index", value.get("record_count"))
    declared_scientific = value.get("scientific_tokens")
    lanes = value.get("lane_scientific_tokens")
    if (
        isinstance(source_examples, bool)
        or not isinstance(source_examples, int)
        or source_examples <= 0
        or isinstance(record_index, bool)
        or not isinstance(record_index, int)
        or record_index <= 0
        or not isinstance(lanes, list)
        or len(lanes) != process_count
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in lanes
        )
    ):
        raise StagingError("staging aggregate counters are invalid")
    file_records = sum(int(item["record_count"]) for item in files)
    file_scientific = sum(int(item["scientific_tokens"]) for item in files)
    if record_index != file_records or sum(lanes) != file_scientific:
        raise StagingError("staging aggregate counters differ from the shard inventory")
    if declared_scientific is not None and (
        isinstance(declared_scientific, bool)
        or not isinstance(declared_scientific, int)
        or declared_scientific != file_scientific
    ):
        raise StagingError("staging scientific-token total differs from its shards")
    if require_padded_record_count:
        record_multiple = (
            process_count
            * DEFAULT_LOCAL_CHIPS
            * max(DEFAULT_PER_DEVICE_SEQUENCE_BATCHES)
        )
        if record_index % record_multiple:
            raise StagingError("sealed staging record count is not batch-aligned")
        if not _lanes_ready(lanes, target_scientific_tokens):
            raise StagingError("sealed staging controller lanes do not meet their quotas")
    return source_examples, record_index, file_scientific, list(lanes)


def _iter_text_values(values: Iterable[Any]) -> Iterator[str]:
    for value in values:
        if isinstance(value, str):
            yield value
        elif isinstance(value, (bytes, np.bytes_)):
            yield bytes(value).decode("utf-8")
        elif isinstance(value, Mapping):
            text = value.get("text")
            if isinstance(text, str):
                yield text
            elif isinstance(text, (bytes, np.bytes_)):
                yield bytes(text).decode("utf-8")
            else:
                raise StagingError(
                    "C4 source yielded an element without string field 'text'"
                )
        else:
            raise StagingError("C4 source yielded an element without string field 'text'")


def _download_hf_source_file(
    *, repo_id: str, revision: str, filename: str, cache_dir: Path
) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise StagingError(
            "Hugging Face Hub is unavailable; run `uv sync --extra data`"
        ) from error
    try:
        downloaded = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            filename=filename,
            cache_dir=str(cache_dir),
            token=os.environ.get("HF_TOKEN"),
        )
    except Exception as error:
        raise StagingError(
            f"cannot materialize immutable C4 source file {filename!r}"
        ) from error
    return Path(downloaded)


def _download_hf_model_file(
    *, repo_id: str, revision: str, filename: str, cache_dir: Path
) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise StagingError(
            "Hugging Face Hub is unavailable; run `uv sync --extra data`"
        ) from error
    try:
        downloaded = hf_hub_download(
            repo_id=repo_id,
            repo_type="model",
            revision=revision,
            filename=filename,
            cache_dir=str(cache_dir),
            token=os.environ.get("HF_TOKEN"),
        )
    except Exception as error:
        raise StagingError(
            f"cannot materialize immutable Gemma tokenizer file {filename!r}"
        ) from error
    return Path(downloaded)


def _verify_gemma_tokenizer_view(path: Path) -> None:
    children = tuple(sorted(path.iterdir(), key=lambda item: item.name))
    actual = tuple(item.name for item in children)
    expected = tuple(entry.path for entry in GEMMA4_TOKENIZER_SOURCE_FILES)
    if actual != expected or any(not item.is_file() for item in children):
        raise StagingError(
            "local Gemma tokenizer view differs from its pinned two-file inventory"
        )
    for entry in GEMMA4_TOKENIZER_SOURCE_FILES:
        digest, size = _sha256_file(path / entry.path)
        if (digest, size) != (entry.content_sha256, entry.size_bytes):
            raise StagingError(
                f"local Gemma tokenizer content changed: {entry.path}"
            )


def materialize_gemma_tokenizer(
    tokenizer_id: str, revision: str, cache_dir: str | Path
) -> Path:
    """Build a verified two-file local tokenizer view at the pinned revision."""

    from moe_sae.config import MODEL_ID, MODEL_REVISION

    if tokenizer_id != MODEL_ID or revision != MODEL_REVISION:
        raise StagingError("Gemma tokenizer ID/revision differs from the pinned model")
    cache = Path(cache_dir).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    view = cache / "gemma-tokenizer" / revision
    view.mkdir(parents=True, exist_ok=True)
    for partial in view.glob(".*.partial"):
        partial.unlink()

    for entry in GEMMA4_TOKENIZER_SOURCE_FILES:
        source = _download_hf_model_file(
            repo_id=tokenizer_id,
            revision=revision,
            filename=entry.path,
            cache_dir=cache,
        )
        digest, size = _sha256_file(source)
        if (digest, size) != (entry.content_sha256, entry.size_bytes):
            raise StagingError(
                "immutable Gemma tokenizer hash/size mismatch for "
                f"{entry.path}: got {digest}/{size}, expected "
                f"{entry.content_sha256}/{entry.size_bytes}"
            )
        destination = view / entry.path
        if destination.exists():
            existing_digest, existing_size = _sha256_file(destination)
            if (existing_digest, existing_size) != (
                entry.content_sha256,
                entry.size_bytes,
            ):
                raise StagingError(
                    f"existing local Gemma tokenizer view is corrupt: {destination}"
                )
            continue
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
        temporary.unlink(missing_ok=True)
        try:
            shutil.copyfile(source, temporary)
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            copied_digest, copied_size = _sha256_file(temporary)
            if (copied_digest, copied_size) != (
                entry.content_sha256,
                entry.size_bytes,
            ):
                raise StagingError(
                    f"copied Gemma tokenizer content changed: {entry.path}"
                )
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    _verify_gemma_tokenizer_view(view)
    return view


def _json_gzip_text_stream(path: Path) -> Iterator[str]:
    """Yield strict JSONL ``text`` values from one verified gzip object."""

    try:
        with gzip.open(path, mode="rt", encoding="utf-8", errors="strict") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise StagingError(
                        f"C4 source contains a blank JSON line at {path}:{line_number}"
                    )
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise StagingError(
                        f"C4 source contains invalid JSON at {path}:{line_number}"
                    ) from error
                if not isinstance(value, Mapping) or not isinstance(
                    value.get("text"), str
                ):
                    raise StagingError(
                        f"C4 source lacks a string 'text' field at {path}:{line_number}"
                    )
                yield value["text"]
    except StagingError:
        raise
    except (OSError, EOFError, UnicodeError) as error:
        raise StagingError(f"cannot decode immutable C4 source file {path}") from error


def allenai_c4_stream(
    split: str, source_cache: str | Path
) -> tuple[Iterable[str], dict[str, Any]]:
    """Materialize, verify, and stream the bounded immutable AllenAI C4 source."""

    if split not in ALLENAI_C4_SOURCE_FILES:
        raise ValueError("split must be 'train' or 'validation'")
    cache = Path(source_cache).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    specifications = ALLENAI_C4_SOURCE_FILES[split]
    local_files: list[Path] = []
    for specification in specifications:
        local_path = _download_hf_source_file(
            repo_id=ALLENAI_C4_REPO_ID,
            revision=ALLENAI_C4_REVISION,
            filename=specification.path,
            cache_dir=cache,
        )
        actual_digest, actual_size = _sha256_file(local_path)
        if (actual_digest, actual_size) != (
            specification.sha256,
            specification.size_bytes,
        ):
            raise StagingError(
                "immutable C4 source hash/size mismatch for "
                f"{specification.path}: got {actual_digest}/{actual_size}, expected "
                f"{specification.sha256}/{specification.size_bytes}"
            )
        local_files.append(local_path)

    def stream() -> Iterator[str]:
        for local_path in local_files:
            yield from _json_gzip_text_stream(local_path)

    identity = {
        "provider": "huggingface_hub",
        "repo_type": "dataset",
        "repo_id": ALLENAI_C4_REPO_ID,
        "revision": ALLENAI_C4_REVISION,
        "files": [specification.as_dict() for specification in specifications],
    }
    return stream(), _validated_source_identity(identity)


def gemma_tokenizer(
    tokenizer_id: str, revision: str, cache_dir: str | Path
) -> Callable[[str], Sequence[int]]:
    """Load a revision-pinned Gemma tokenizer and return raw-content encoding."""

    _validate_revision(revision, "tokenizer_revision")
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise StagingError(
            "Transformers is unavailable; run `uv sync --extra data`"
        ) from error
    local_view = materialize_gemma_tokenizer(tokenizer_id, revision, cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        str(local_view),
        local_files_only=True,
    )
    _verify_gemma_tokenizer_view(local_view)
    actual = (
        tokenizer.pad_token_id,
        tokenizer.eos_token_id,
        tokenizer.bos_token_id,
    )
    if actual != (PAD_ID, EOS_ID, BOS_ID):
        raise StagingError(
            f"Gemma special token IDs changed: pad/eos/bos={actual}, expected (0, 1, 2)"
        )

    def encode(text: str) -> Sequence[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    return encode


def _close_writer(writer: Any, path: Path) -> None:
    ok = getattr(writer, "ok", None)
    if callable(ok) and not bool(ok()):
        raise StagingError(f"ArrayRecord writer failed before closing {path}")
    writer.close()


def stage_c4_split(
    *,
    split: str,
    target_scientific_tokens: int,
    output_directory: str | Path,
    source_cache: str | Path | None,
    tokenizer_id: str,
    tokenizer_revision: str,
    records_per_shard: int = 4_096,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    process_count: int = DEFAULT_PROCESS_COUNT,
    texts: Iterable[Any] | None = None,
    tokenize: Callable[[str], Sequence[int]] | None = None,
    source_identity: Mapping[str, Any] | None = None,
) -> StagingResult:
    """Stage and seal one deterministic C4 split, resuming at shard boundaries."""

    if split not in {"train", "validation"}:
        raise ValueError("split must be 'train' or 'validation'")
    if target_scientific_tokens <= 0 or records_per_shard <= 0:
        raise ValueError("token target and records_per_shard must be positive")
    if process_count != DEFAULT_PROCESS_COUNT:
        raise ValueError("the v4-32 staging contract requires exactly four controllers")
    _validate_revision(tokenizer_revision, "tokenizer_revision")
    if texts is None:
        if source_cache is None:
            raise ValueError("source_cache is required for production C4 staging")
        if source_identity is not None:
            raise ValueError("production C4 source identity is fixed and cannot be supplied")
        source, source_identity = allenai_c4_stream(split, source_cache)
    else:
        source = texts
        if source_cache is not None:
            raise ValueError("source_cache and injected texts are mutually exclusive")
        if source_identity is None:
            raise ValueError("injected staging input requires an immutable source_identity")
        source_identity = _validated_source_identity(source_identity)
    root = Path(output_directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    identity = _manifest_identity(
        split=split,
        source_identity=source_identity,
        tokenizer_id=tokenizer_id,
        tokenizer_revision=tokenizer_revision,
        sequence_length=sequence_length,
        process_count=process_count,
    )
    manifest_path = root / f"{split}.manifest.json"
    state_path = root / f".{split}.staging.json"
    if manifest_path.exists():
        manifest = _load_json(manifest_path)
        required = set(identity) | {
            "target_scientific_tokens",
            "source_examples",
            "scientific_tokens",
            "record_count",
            "lane_scientific_tokens",
            "files",
        }
        if set(manifest) != required:
            raise StagingError("sealed staging manifest has incompatible fields")
        for name, value in identity.items():
            if manifest.get(name) != value:
                raise StagingError("sealed staging manifest belongs to another input")
        declared_target = manifest.get("target_scientific_tokens")
        if (
            isinstance(declared_target, bool)
            or not isinstance(declared_target, int)
            or declared_target != target_scientific_tokens
        ):
            raise StagingError("sealed staging manifest has a different token target")
        files = manifest.get("files")
        if not isinstance(files, list) or not files:
            raise StagingError("sealed staging manifest has no files")
        _validate_file_entries(root, files)
        source_examples, record_count, scientific_tokens, _ = _validate_aggregate_state(
            manifest,
            files=files,
            target_scientific_tokens=target_scientific_tokens,
            process_count=process_count,
            require_padded_record_count=True,
        )
        return StagingResult(
            manifest_location=str(manifest_path),
            grain_files=str(root / f"{split}-*.array_record"),
            scientific_tokens=scientific_tokens,
            record_count=record_count,
            source_examples=source_examples,
            resumed=True,
        )

    for partial in root.glob(f".{split}-*.partial"):
        partial.unlink()
    if state_path.exists():
        state = _load_json(state_path)
        if set(state) != {
            "schema_version",
            "identity",
            "target_scientific_tokens",
            "source_examples",
            "next_record_index",
            "lane_scientific_tokens",
            "files",
        } or state.get("schema_version") != "gemma4-c4-staging-state-v2":
            raise StagingError("existing staging state has incompatible fields")
        declared_target = state.get("target_scientific_tokens")
        if state.get("identity") != identity or (
            isinstance(declared_target, bool)
            or not isinstance(declared_target, int)
            or declared_target != target_scientific_tokens
        ):
            raise StagingError("existing staging state belongs to another input")
        files = state.get("files")
        lanes = state.get("lane_scientific_tokens")
        if not isinstance(files, list) or not isinstance(lanes, list):
            raise StagingError("existing staging state is incomplete")
        _validate_file_entries(root, files)
        source_examples, record_index, _, lanes = _validate_aggregate_state(
            state,
            files=files,
            target_scientific_tokens=target_scientific_tokens,
            process_count=process_count,
            require_padded_record_count=False,
        )
        resumed = True
    else:
        files = []
        lanes = [0] * process_count
        source_examples = 0
        record_index = 0
        resumed = False

    if tokenize is None:
        if source_cache is None:
            raise ValueError(
                "source_cache is required when the pinned Gemma tokenizer is used"
            )
        encode = gemma_tokenizer(tokenizer_id, tokenizer_revision, source_cache)
    else:
        encode = tokenize
    iterator = iter(_iter_text_values(source))
    for _ in itertools.islice(iterator, source_examples):
        pass

    max_local_batch = DEFAULT_LOCAL_CHIPS * max(DEFAULT_PER_DEVICE_SEQUENCE_BATCHES)
    record_multiple = process_count * max_local_batch
    packer = OrderedPacker(sequence_length)
    writer: Any | None = None
    partial_path: Path | None = None
    shard_first = record_index
    shard_records = 0
    shard_scientific = 0
    shard_index = len(files)

    def open_writer() -> Any:
        nonlocal partial_path
        partial_path = root / f".{split}-{shard_index:05d}.array_record.partial"
        partial_path.unlink(missing_ok=True)
        return _array_record_writer(partial_path)

    def write_pack(arrays: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
        nonlocal writer, record_index, shard_records, shard_scientific
        if writer is None:
            writer = open_writer()
        payload = encode_packed_record(
            record_index, *arrays, sequence_length=sequence_length
        )
        record = decode_packed_record(
            payload,
            expected_index=record_index,
            expected_sequence_length=sequence_length,
        )
        writer.write(payload)
        lanes[lane_for_record(record_index, process_count=process_count)] += (
            record.scientific_tokens
        )
        shard_records += 1
        shard_scientific += record.scientific_tokens
        record_index += 1

    def seal_shard() -> None:
        nonlocal writer, partial_path, shard_first, shard_records
        nonlocal shard_scientific, shard_index
        if writer is None or partial_path is None or shard_records == 0:
            return
        _close_writer(writer, partial_path)
        digest, size = _sha256_file(partial_path)
        final_name = f"{split}-{shard_index:05d}-{digest[:16]}.array_record"
        final_path = root / final_name
        if final_path.exists():
            existing_digest, existing_size = _sha256_file(final_path)
            if (existing_digest, existing_size) != (digest, size):
                raise StagingError(f"content-addressed shard collision: {final_path}")
            partial_path.unlink()
        else:
            os.replace(partial_path, final_path)
        source_check = _array_record_source([str(final_path)])
        if len(source_check) != shard_records:
            raise StagingError("reopened ArrayRecord count differs from writer count")
        first = decode_packed_record(
            bytes(source_check[0]),
            expected_index=shard_first,
            expected_sequence_length=sequence_length,
        )
        last = decode_packed_record(
            bytes(source_check[shard_records - 1]),
            expected_index=shard_first + shard_records - 1,
            expected_sequence_length=sequence_length,
        )
        del first, last, source_check
        files.append(
            {
                "uri": final_name,
                "sha256": digest,
                "size_bytes": size,
                "first_record_index": shard_first,
                "record_count": shard_records,
                "scientific_tokens": shard_scientific,
            }
        )
        _atomic_write_json(
            state_path,
            {
                "schema_version": "gemma4-c4-staging-state-v2",
                "identity": identity,
                "target_scientific_tokens": target_scientific_tokens,
                "source_examples": source_examples,
                "next_record_index": record_index,
                "lane_scientific_tokens": lanes,
                "files": files,
            },
        )
        writer = None
        partial_path = None
        shard_first = record_index
        shard_records = 0
        shard_scientific = 0
        shard_index += 1

    try:
        while not _lanes_ready(lanes, target_scientific_tokens):
            try:
                text = next(iterator)
            except StopIteration as error:
                raise StagingError(
                    "C4 input exhausted before every TPU controller lane met its quota"
                ) from error
            content = encode(text)
            for arrays in packer.add_content(content):
                write_pack(arrays)
            source_examples += 1
            if shard_records >= records_per_shard:
                for arrays in packer.flush():
                    write_pack(arrays)
                seal_shard()

        for arrays in packer.flush():
            write_pack(arrays)
        while record_index % record_multiple:
            # A padding-only record is valid at the dataset tail.  It is never
            # counted or persisted by the scientific capture mask.
            zeros_i32 = np.zeros(sequence_length, dtype=np.int32)
            zeros_u16 = np.zeros(sequence_length, dtype=np.uint16)
            if writer is None:
                writer = open_writer()
            payload = _HEADER.pack(
                PACKED_RECORD_MAGIC,
                PACKED_RECORD_VERSION,
                sequence_length,
                0,
                record_index,
            ) + b"".join(
                (
                    np.asarray(zeros_i32, dtype="<i4").tobytes(),
                    np.asarray(zeros_u16, dtype="<u2").tobytes(),
                    np.asarray(zeros_u16, dtype="<u2").tobytes(),
                )
            )
            decode_packed_record(
                payload,
                expected_index=record_index,
                expected_sequence_length=sequence_length,
            )
            writer.write(payload)
            shard_records += 1
            record_index += 1
        seal_shard()
    except BaseException:
        if writer is not None:
            try:
                writer.close()
            finally:
                if partial_path is not None:
                    partial_path.unlink(missing_ok=True)
        raise

    _validate_file_entries(root, files)
    scientific_tokens = sum(int(item["scientific_tokens"]) for item in files)
    if scientific_tokens != sum(lanes) or not _lanes_ready(lanes, target_scientific_tokens):
        raise StagingError("sealed staging totals do not satisfy every controller lane")
    manifest = {
        **identity,
        "target_scientific_tokens": target_scientific_tokens,
        "source_examples": source_examples,
        "scientific_tokens": scientific_tokens,
        "record_count": record_index,
        "lane_scientific_tokens": lanes,
        "files": files,
    }
    _atomic_write_json(manifest_path, manifest)
    state_path.unlink(missing_ok=True)
    return StagingResult(
        manifest_location=str(manifest_path),
        grain_files=str(root / f"{split}-*.array_record"),
        scientific_tokens=scientific_tokens,
        record_count=record_index,
        source_examples=source_examples,
        resumed=resumed,
    )


def verify_packed_source(
    files: Sequence[str],
    *,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
) -> tuple[int, int, tuple[int, ...]]:
    """Reopen every record and verify its index, schema, and scientific count."""

    source = _array_record_source(files)
    lanes = [0] * DEFAULT_PROCESS_COUNT
    scientific = 0
    for index in range(len(source)):
        record = decode_packed_record(
            bytes(source[index]),
            expected_index=index,
            expected_sequence_length=sequence_length,
        )
        scientific += record.scientific_tokens
        lanes[index % DEFAULT_PROCESS_COUNT] += record.scientific_tokens
    return len(source), scientific, tuple(lanes)


class PackedBatchIterable:
    """Yield host-local batches from immutable packed records in strided order."""

    def __init__(
        self,
        files: Sequence[str],
        *,
        global_batch_size: int,
        process_index: int,
        process_count: int,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    ):
        if not files:
            raise ValueError("packed ArrayRecord file inventory cannot be empty")
        if process_count != DEFAULT_PROCESS_COUNT:
            raise ValueError("packed C4 input requires the four-controller v4-32 layout")
        if not 0 <= process_index < process_count:
            raise ValueError("process_index is outside process_count")
        if global_batch_size <= 0 or global_batch_size % process_count:
            raise ValueError("global batch size must divide evenly across controllers")
        supported = {
            process_count * DEFAULT_LOCAL_CHIPS * per_device
            for per_device in DEFAULT_PER_DEVICE_SEQUENCE_BATCHES
        }
        if global_batch_size not in supported:
            raise ValueError(
                "global batch size must match a qualified per-device sequence batch"
            )
        self.files = tuple(files)
        self.global_batch_size = global_batch_size
        self.process_index = process_index
        self.process_count = process_count
        self.sequence_length = sequence_length

    def __iter__(self) -> Iterator[dict[str, np.ndarray]]:
        source = _array_record_source(self.files)
        local_batch = self.global_batch_size // self.process_count
        lane_records = (
            len(source) - self.process_index + self.process_count - 1
        ) // self.process_count
        full_steps = lane_records // local_batch
        for step in range(full_steps):
            records = []
            lane_start = step * local_batch
            for offset in range(local_batch):
                index = self.process_index + (lane_start + offset) * self.process_count
                records.append(
                    decode_packed_record(
                        bytes(source[index]),
                        expected_index=index,
                        expected_sequence_length=self.sequence_length,
                    )
                )
            yield {
                "inputs": np.stack([record.token_ids for record in records]),
                "inputs_position": np.stack([record.positions for record in records]).astype(
                    np.int32
                ),
                "inputs_segmentation": np.stack(
                    [record.segment_ids for record in records]
                ).astype(np.int32),
            }


def build_maxtext_packed_iterator(
    files: Sequence[str],
    *,
    config: Any,
    mesh: Any,
    global_batch_size: int | None = None,
) -> Any:
    """Wrap the offline reader in MaxText's native global-array adapter."""

    try:
        import jax
        from maxtext.input_pipeline.multihost_dataloading import MultiHostDataLoadIterator
    except ImportError as error:
        raise StagingError("packed TPU input requires the pinned MaxText runtime") from error
    local = PackedBatchIterable(
        files,
        global_batch_size=int(global_batch_size or config.global_batch_size_to_load),
        process_index=int(jax.process_index()),
        process_count=int(jax.process_count()),
        sequence_length=int(config.max_target_length),
    )
    return MultiHostDataLoadIterator(local, mesh, generate_padding_batch=False)


def publish_staged_dataset(
    local_manifest: str | Path,
    destination: str,
    *,
    object_store: Any | None = None,
) -> StagingResult:
    """Conditionally upload immutable shards, then publish the manifest last."""

    from moe_sae.object_store import GCSObjectStore, join_gcs_uri

    if not destination.startswith("gs://"):
        raise ValueError("staged publication destination must be gs://")
    manifest_path = Path(local_manifest).expanduser().resolve()
    manifest = _load_json(manifest_path)
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise StagingError("cannot publish an unsealed staging manifest")
    prefix = destination.rstrip("/")
    store = GCSObjectStore() if object_store is None else object_store
    remote_files: list[dict[str, Any]] = []
    for shard_index, entry in enumerate(files):
        local = manifest_path.parent / str(entry["uri"])
        digest = str(entry["sha256"])
        remote = join_gcs_uri(
            prefix,
            "shards",
            f"{shard_index:05d}-{digest}.array_record",
        )
        identity = store.put_if_absent(local, remote, sha256=digest)
        if identity.size_bytes != int(entry["size_bytes"]):
            raise StagingError("published ArrayRecord size differs from its manifest")
        # The packed-data schema stays reader-agnostic.  The object is
        # generation-pinned and verified at publication, while consumers also
        # re-hash the exact URI against the sealed scientific inventory.
        store.verify(identity)
        remote_entry = dict(entry)
        remote_entry["uri"] = remote
        remote_entry["object"] = identity.as_dict()
        remote_files.append(remote_entry)
    remote_manifest = {**manifest, "files": remote_files}
    temporary = manifest_path.parent / f".{manifest_path.name}.gcs.tmp"
    _atomic_write_json(temporary, remote_manifest)
    manifest_digest, _ = _sha256_file(temporary)
    remote_manifest_uri = join_gcs_uri(
        prefix, "manifests", f"{manifest_digest}.json"
    )
    try:
        manifest_identity = store.put_if_absent(
            temporary,
            remote_manifest_uri,
            sha256=manifest_digest,
        )
        store.verify(manifest_identity)
    finally:
        temporary.unlink(missing_ok=True)
    return StagingResult(
        manifest_location=remote_manifest_uri,
        grain_files=f"{prefix}/shards/*.array_record",
        scientific_tokens=int(manifest["scientific_tokens"]),
        record_count=int(manifest["record_count"]),
        source_examples=int(manifest["source_examples"]),
        resumed=False,
    )


__all__ = [
    "ALLENAI_C4_REPO_ID",
    "ALLENAI_C4_REVISION",
    "ALLENAI_C4_SOURCE_FILES",
    "BOS_ID",
    "DEFAULT_PROCESS_COUNT",
    "EOS_ID",
    "GEMMA4_TOKENIZER_INVENTORY_SHA256",
    "GEMMA4_TOKENIZER_SOURCE_FILES",
    "PACKED_DATASET_SCHEMA",
    "PACKED_RECORD_FORMAT",
    "PAD_ID",
    "PackedBatchIterable",
    "PackedRecord",
    "SourceFile",
    "StagingError",
    "StagingResult",
    "build_maxtext_packed_iterator",
    "allenai_c4_stream",
    "decode_packed_record",
    "encode_packed_record",
    "gemma_tokenizer",
    "lane_for_record",
    "publish_staged_dataset",
    "materialize_gemma_tokenizer",
    "stage_c4_split",
    "verify_packed_source",
]
