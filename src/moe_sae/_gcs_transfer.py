"""JAX-cold process-pool transfers for large immutable GCS objects.

This module is launched with ``python -m`` by :class:`GCSObjectStore`.  It
must remain free of JAX imports: the point of the exec boundary is to keep
``ProcessPoolExecutor`` away from the initialized TPU runtime in its parent.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import re
import sys
from typing import Any, Sequence

from moe_sae.object_store import ObjectIdentity, parse_gcs_uri


_SHA256 = re.compile(r"[0-9a-f]{64}")
_CRC32C = re.compile(r"[A-Za-z0-9+/]{6}==")


def _client_and_blob(uri: str) -> tuple[Any, Any]:
    try:
        from google.cloud import storage
    except ImportError as error:  # pragma: no cover - optional production package
        raise RuntimeError("google-cloud-storage is unavailable") from error
    bucket_name, object_name = parse_gcs_uri(uri)
    client = storage.Client()
    return client, client.bucket(bucket_name).blob(object_name)


def _transfer_manager() -> Any:
    try:
        from google.cloud.storage import transfer_manager
    except ImportError as error:  # pragma: no cover - optional production package
        raise RuntimeError("GCS transfer manager is unavailable") from error
    return transfer_manager


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def _upload_temporary(args: argparse.Namespace) -> None:
    source = Path(args.source)
    if not source.is_file() or source.stat().st_size <= 0:
        raise RuntimeError("multipart upload source is not a nonempty file")
    if not _SHA256.fullmatch(args.sha256):
        raise RuntimeError("multipart upload SHA-256 is invalid")
    if not _CRC32C.fullmatch(args.crc32c):
        raise RuntimeError("multipart upload CRC32C is invalid")
    source_before = source.stat()
    _client, blob = _client_and_blob(args.uri)
    transfer_manager = _transfer_manager()
    blob.metadata = {"sha256": args.sha256}
    generation = 0
    try:
        transfer_manager.upload_chunks_concurrently(
            str(source),
            blob,
            chunk_size=args.chunk_bytes,
            worker_type=transfer_manager.PROCESS,
            max_workers=args.workers,
            checksum="crc32c",
            deadline=args.deadline_seconds,
        )
        source_after = source.stat()
        blob.reload()
        generation = int(blob.generation or 0)
        if (
            generation <= 0
            or int(blob.size or 0) != source_before.st_size
            or dict(blob.metadata or {}).get("sha256") != args.sha256
            or str(blob.crc32c or "") != args.crc32c
            or (
                source_before.st_dev,
                source_before.st_ino,
                source_before.st_size,
                source_before.st_mtime_ns,
                source_before.st_ctime_ns,
            )
            != (
                source_after.st_dev,
                source_after.st_ino,
                source_after.st_size,
                source_after.st_mtime_ns,
                source_after.st_ctime_ns,
            )
        ):
            raise RuntimeError("multipart temporary has an invalid GCS identity")
    except Exception:
        if generation <= 0:
            try:
                blob.reload()
                generation = int(blob.generation or 0)
            except Exception:
                generation = 0
        if generation > 0:
            try:
                blob.delete(if_generation_match=generation)
            except Exception:
                pass
        raise


def _download(args: argparse.Namespace) -> None:
    identity = ObjectIdentity(
        uri=args.uri,
        generation=args.generation,
        size_bytes=args.size_bytes,
        crc32c=args.crc32c,
        sha256=args.sha256,
    )
    destination = Path(args.destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _client, blob = _client_and_blob(identity.uri)
    transfer_manager = _transfer_manager()
    blob.reload(if_generation_match=identity.generation)
    if (
        int(blob.generation or 0) != identity.generation
        or int(blob.size or 0) != identity.size_bytes
        or str(blob.crc32c or "") != identity.crc32c
        or dict(blob.metadata or {}).get("sha256") != identity.sha256
    ):
        raise RuntimeError("generation-pinned download identity changed")
    try:
        transfer_manager.download_chunks_concurrently(
            blob,
            str(destination),
            chunk_size=args.chunk_bytes,
            download_kwargs={"if_generation_match": identity.generation},
            worker_type=transfer_manager.PROCESS,
            max_workers=args.workers,
            crc32c_checksum=True,
            deadline=args.deadline_seconds,
        )
    except Exception:
        destination.unlink(missing_ok=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)

    upload = commands.add_parser("upload-temp")
    upload.add_argument("--source", required=True)
    upload.add_argument("--uri", required=True)
    upload.add_argument("--sha256", required=True)
    upload.add_argument("--crc32c", required=True)
    upload.add_argument("--chunk-bytes", required=True, type=_positive)
    upload.add_argument("--workers", required=True, type=_positive)
    upload.add_argument("--deadline-seconds", required=True, type=_positive_float)
    upload.set_defaults(handler=_upload_temporary)

    download = commands.add_parser("download")
    download.add_argument("--uri", required=True)
    download.add_argument("--generation", required=True, type=_positive)
    download.add_argument("--size-bytes", required=True, type=_positive)
    download.add_argument("--crc32c", required=True)
    download.add_argument("--sha256", required=True)
    download.add_argument("--destination", required=True)
    download.add_argument("--chunk-bytes", required=True, type=_positive)
    download.add_argument("--workers", required=True, type=_positive)
    download.add_argument("--deadline-seconds", required=True, type=_positive_float)
    download.set_defaults(handler=_download)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        args.handler(args)
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess
    raise SystemExit(main())
