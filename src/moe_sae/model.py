"""Gemma 4 prefix capture without a hard MaxText import dependency.

The production adapter is deliberately duck typed against the pinned MaxText
NNX model.  Importing :mod:`moe_sae.model` therefore works on laptops and in
unit tests that do not have MaxText or TPU libraries installed.  The explicit
``from_maxtext`` constructor performs dependency and checkout validation.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
import hashlib
from importlib import import_module
import json
import os
from pathlib import Path
from typing import Any, Mapping, NamedTuple, Sequence
from urllib.parse import urlparse

import numpy as np

from moe_sae.config import MAXTEXT_COMMIT, MODEL_ID, MODEL_REVISION, Gemma4Config


class MaxTextUnavailableError(ImportError):
    """Raised when a production adapter is requested without MaxText/JAX."""


class CaptureBatch(NamedTuple):
    """Host-ready rows captured at the Gemma 4 routed-branch site.

    A named tuple is intentional: JAX treats it as a pytree without this
    package importing JAX or registering global pytree handlers.
    """

    activation: Any
    expert_ids: Any
    router_weights: Any
    sequence_id: Any
    position: Any

    @property
    def rows(self) -> int:
        return int(self.activation.shape[0])


@dataclass(frozen=True, slots=True)
class CaptureSummary:
    rows: int
    hidden_size: int
    experts_per_token: int
    min_expert_id: int
    max_expert_id: int
    max_router_sum_error: float


@dataclass(frozen=True, slots=True)
class HuggingFaceRepoFile:
    """One file in the immutable Hugging Face repository tree.

    ``git_blob_oid`` is the ordinary Git blob identity returned by the Hub
    tree API.  LFS snapshot files contain the resolved payload rather than the
    pointer blob, so those entries additionally pin the payload SHA-256.
    ``content_sha256`` also pins ordinary-file bytes, making the final snapshot
    inventory independently comparable without trusting the local Hub cache.
    """

    path: str
    size_bytes: int
    git_blob_oid: str
    content_sha256: str
    lfs_sha256: str | None = None

    def as_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "git_blob_oid": self.git_blob_oid,
            "content_sha256": self.content_sha256,
            "storage": "lfs" if self.lfs_sha256 is not None else "git",
        }
        if self.lfs_sha256 is not None:
            value["lfs_sha256"] = self.lfs_sha256
        return value


# Derived 2026-07-17 from
# https://huggingface.co/api/models/google/gemma-4-26B-A4B/tree/
# 24548b62aa021d562695c04aaf7758a1ea47990b?recursive=true&expand=true
# (shown across two comment lines only for readability).  The three LFS
# hashes/sizes are provider-published payload identities.  The seven ordinary
# file SHA-256s were computed from their small revision-addressed blobs after
# independently confirming each Git blob OID.
GEMMA4_REPOSITORY_FILES: tuple[HuggingFaceRepoFile, ...] = (
    HuggingFaceRepoFile(
        path=".gitattributes",
        size_bytes=1_570,
        git_blob_oid="52373fe24473b1aa44333d318f578ae6bf04b49b",
        content_sha256="34448b82c17d60fec9b65b1f093c115ddbaadc04beb1b0140b6bfed2e012a930",
    ),
    HuggingFaceRepoFile(
        path="README.md",
        size_bytes=27_830,
        git_blob_oid="f6fa2c9771907998658ecf59378910300c474288",
        content_sha256="2d87a2d936e3e03077589114bd1702cf480647fb145a41fe973eba49f934d0ce",
    ),
    HuggingFaceRepoFile(
        path="config.json",
        size_bytes=3_775,
        git_blob_oid="29512a0d8108fc06ea4cec3cf2be250cfe0b960a",
        content_sha256="0364256c5de7863a22b5dd666043d542d42b681c02fd83eb67dc7b483d859e96",
    ),
    HuggingFaceRepoFile(
        path="generation_config.json",
        size_bytes=181,
        git_blob_oid="a07eb97486e34532ffd3cf64e44cba33af40127c",
        content_sha256="de16fa1d777387d5fe6df54a9565d9c0f1213f7b31f2a2cf44805e27686d9c7c",
    ),
    HuggingFaceRepoFile(
        path="model-00001-of-00002.safetensors",
        size_bytes=49_907_246_508,
        git_blob_oid="15df541ca6c85580c13b75290565817ba0783f1e",
        content_sha256="a0e671c5e81ce9a8fd151e60676e7b3a7f2e9ec685a1e8b6b504de2082453f96",
        lfs_sha256="a0e671c5e81ce9a8fd151e60676e7b3a7f2e9ec685a1e8b6b504de2082453f96",
    ),
    HuggingFaceRepoFile(
        path="model-00002-of-00002.safetensors",
        size_bytes=1_704_763_408,
        git_blob_oid="2db4c35287ffa2aa85226dd53adb3ba3cee9010b",
        content_sha256="dda6797de622b19c46320bb31073323d06a3444a149d4e0e028a3dbd682e0364",
        lfs_sha256="dda6797de622b19c46320bb31073323d06a3444a149d4e0e028a3dbd682e0364",
    ),
    HuggingFaceRepoFile(
        path="model.safetensors.index.json",
        size_bytes=103_196,
        git_blob_oid="889ab10e12aca96c43c9804a041fb1780de1e5ca",
        content_sha256="907826a6e46ff454272bd6db1fee629d5531a2303be22986d825a0871d7dc7a7",
    ),
    HuggingFaceRepoFile(
        path="processor_config.json",
        size_bytes=1_689,
        git_blob_oid="5465974d23e1eca2c46c2809b26c997946ce0d90",
        content_sha256="32bdf45d2ad4cc29a0822ddd157a182de76644f0419a6228d151495256e9813c",
    ),
    HuggingFaceRepoFile(
        path="tokenizer.json",
        size_bytes=32_170_070,
        git_blob_oid="890ddaefca8201dd023c4f013db5ce492d12e7e0",
        content_sha256="12bac982b793c44b03d52a250a9f0d0b666813da566b910c24a6da0695fd11e6",
        lfs_sha256="12bac982b793c44b03d52a250a9f0d0b666813da566b910c24a6da0695fd11e6",
    ),
    HuggingFaceRepoFile(
        path="tokenizer_config.json",
        size_bytes=881,
        git_blob_oid="54a4569882af13f7bf57fdf459f2db243ff29680",
        content_sha256="6a9383197f000d2723684efd2210f5bf217bc29fa2cf360ae1574201b47af060",
    ),
)


@dataclass(frozen=True, slots=True)
class HuggingFaceSnapshot:
    """Content-attested, commit-addressed Hugging Face model snapshot."""

    model_id: str
    revision: str
    path: Path
    inventory_sha256: str
    repository_inventory_sha256: str


@dataclass(frozen=True, slots=True)
class CheckpointConversionSpec:
    """Secret-free description of a pinned Hugging Face -> Orbax conversion."""

    maxtext_root: Path
    hf_model_id: str
    hf_revision: str
    hf_snapshot_path: Path
    hf_snapshot_inventory_sha256: str
    hf_repository_inventory_sha256: str
    output_directory: str
    python_executable: str = sys.executable
    save_dtype: str = "bfloat16"
    lazy_load_tensors: bool = True

    @property
    def hf_model_path(self) -> str:
        """Local model/tokenizer path consumed by MaxText and Transformers."""

        return str(self.hf_snapshot_path)

    @property
    def base_config(self) -> Path:
        return self.maxtext_root / "src" / "maxtext" / "configs" / "base.yml"

    def maxtext_overrides(self) -> dict[str, str]:
        return {
            "model_name": "gemma4-26b",
            "base_output_directory": self.output_directory,
            "use_multimodal": "false",
            "scan_layers": "false",
            "pure_nnx_decoder": "true",
            "hardware": "cpu",
            "skip_jax_distributed_system": "true",
        }


def verify_maxtext_checkout(maxtext_root: str | Path) -> str:
    """Return HEAD after verifying an exact, clean MaxText source pin."""

    root = Path(maxtext_root).expanduser().resolve()
    if not (root / "src" / "maxtext" / "configs" / "base.yml").is_file():
        raise FileNotFoundError(f"not a MaxText checkout: {root}")
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"could not determine MaxText commit at {root}") from exc
    actual = result.stdout.strip().lower()
    if actual != MAXTEXT_COMMIT:
        raise RuntimeError(
            f"MaxText must be at {MAXTEXT_COMMIT}; checkout is at {actual or '<unknown>'}"
        )
    try:
        status = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"could not verify MaxText source cleanliness at {root}") from exc
    if status.strip():
        raise RuntimeError(
            "MaxText checkout must be clean before conversion/parity validation"
        )
    return actual


def maxtext_checkout_environment(
    maxtext_root: str | Path,
    environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Force conversion subprocesses to import code from the verified checkout.

    The caller's credentials and ordinary dependency environment are retained,
    while Python package search is restricted to the checkout's ``src`` tree
    and test utilities.  This prevents an installed or user-site MaxText copy
    from satisfying a parity run attributed to different source code.
    """

    root = Path(maxtext_root).expanduser().resolve()
    verify_maxtext_checkout(root)
    result = dict(os.environ if environment is None else environment)
    result.pop("PYTHONHOME", None)
    result["PYTHONPATH"] = os.pathsep.join((str(root / "src"), str(root)))
    result["PYTHONNOUSERSITE"] = "1"
    result["PYTHONSAFEPATH"] = "1"
    return result


def require_maxtext(maxtext_root: str | Path | None = None) -> tuple[Any, Any]:
    """Lazily import MaxText and JAX, optionally verifying the source checkout."""

    if maxtext_root is not None:
        verify_maxtext_checkout(maxtext_root)
    try:
        maxtext = import_module("maxtext")
        jax = import_module("jax")
    except (ImportError, ModuleNotFoundError) as exc:
        raise MaxTextUnavailableError(
            "Gemma 4 capture requires the pinned MaxText checkout and its TPU dependencies; "
            "install MaxText from commit " + MAXTEXT_COMMIT
        ) from exc
    return maxtext, jax


def _validate_hf_revision(revision: str) -> str:
    normalized = str(revision).strip()
    if (
        len(normalized) != 40
        or normalized != normalized.lower()
        or any(character not in "0123456789abcdef" for character in normalized)
    ):
        raise ValueError(
            "Hugging Face revision must be an exact lowercase 40-character commit SHA"
        )
    return normalized


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _validate_repository_files(
    files: Sequence[HuggingFaceRepoFile],
) -> tuple[HuggingFaceRepoFile, ...]:
    result = tuple(files)
    if not result:
        raise ValueError("Hugging Face repository inventory cannot be empty")
    seen: set[str] = set()
    for entry in result:
        path = Path(entry.path)
        if (
            not entry.path
            or path.is_absolute()
            or ".." in path.parts
            or entry.path in seen
        ):
            raise ValueError("Hugging Face repository inventory has an unsafe path")
        if entry.size_bytes <= 0:
            raise ValueError("Hugging Face repository inventory has an invalid size")
        if len(entry.git_blob_oid) != 40 or any(
            character not in "0123456789abcdef" for character in entry.git_blob_oid
        ):
            raise ValueError("Hugging Face repository inventory has an invalid Git OID")
        if len(entry.content_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in entry.content_sha256
        ):
            raise ValueError(
                "Hugging Face repository inventory has an invalid content SHA-256"
            )
        if entry.lfs_sha256 is not None:
            if entry.lfs_sha256 != entry.content_sha256:
                raise ValueError("Hugging Face LFS and content SHA-256 values differ")
            pointer = (
                "version https://git-lfs.github.com/spec/v1\n"
                f"oid sha256:{entry.lfs_sha256}\n"
                f"size {entry.size_bytes}\n"
            ).encode("ascii")
            git_blob = hashlib.sha1(  # noqa: S324 - Git's specified blob identity
                f"blob {len(pointer)}\0".encode("ascii") + pointer,
                usedforsecurity=False,
            ).hexdigest()
            if git_blob != entry.git_blob_oid:
                raise ValueError(
                    "Hugging Face LFS pointer does not match its pinned Git blob OID"
                )
        seen.add(entry.path)
    if tuple(sorted(seen)) != tuple(entry.path for entry in result):
        raise ValueError("Hugging Face repository inventory must be path-sorted")
    return result


def expected_hf_repository_inventory_sha256(
    files: Sequence[HuggingFaceRepoFile] = GEMMA4_REPOSITORY_FILES,
) -> str:
    """Return the pinned Hub tree-metadata inventory digest."""

    validated = _validate_repository_files(files)
    payload = {
        "format": "huggingface-repository-file-inventory-v1",
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": [entry.as_dict() for entry in validated],
    }
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def expected_hf_snapshot_inventory_sha256(
    files: Sequence[HuggingFaceRepoFile] = GEMMA4_REPOSITORY_FILES,
) -> str:
    """Return the expected byte-level snapshot digest without reading weights."""

    validated = _validate_repository_files(files)
    payload = {
        "format": "huggingface-snapshot-content-inventory-v1",
        "files": [
            {
                "path": entry.path,
                "size": entry.size_bytes,
                "sha256": entry.content_sha256,
            }
            for entry in validated
        ],
    }
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def hf_snapshot_inventory_sha256(
    snapshot_path: str | Path,
    *,
    expected_repo_files: Sequence[HuggingFaceRepoFile] | None = None,
) -> str:
    """Hash every file in a downloaded Hugging Face snapshot.

    Hugging Face's cache normally represents snapshot files as symbolic links
    to immutable blobs.  File links are followed deliberately, while directory
    links and special files are rejected.  Byte-level hashes make the result
    useful as both provenance and a before/after mutation check.
    """

    root = Path(snapshot_path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Hugging Face snapshot is not a directory: {root}")

    paths = sorted(
        root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()
    )
    expected = (
        None
        if expected_repo_files is None
        else _validate_repository_files(expected_repo_files)
    )
    expected_by_path = (
        {} if expected is None else {entry.path: entry for entry in expected}
    )
    actual_file_paths: list[str] = []
    for path in paths:
        relative = path.relative_to(root).as_posix()
        if path.is_symlink() and path.resolve().is_dir():
            raise ValueError(
                f"Hugging Face snapshot refuses symbolic-link directories: {relative}"
            )
        if path.is_dir():
            continue
        actual_file_paths.append(relative)
    if expected is not None and tuple(actual_file_paths) != tuple(expected_by_path):
        missing = sorted(set(expected_by_path) - set(actual_file_paths))
        extra = sorted(set(actual_file_paths) - set(expected_by_path))
        raise RuntimeError(
            "Hugging Face snapshot file set differs from the pinned repository "
            f"inventory (missing={missing}, extra={extra})"
        )

    entries: list[dict[str, Any]] = []
    for path in paths:
        relative = path.relative_to(root).as_posix()
        if path.is_symlink() and path.resolve().is_dir():
            raise ValueError(
                f"Hugging Face snapshot refuses symbolic-link directories: {relative}"
            )
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(
                f"Hugging Face snapshot contains a non-regular file: {relative}"
            )

        link_target = os.readlink(path) if path.is_symlink() else None
        resolved = path.resolve(strict=True)
        before = resolved.stat()
        expected_entry = expected_by_path.get(relative)
        if expected_entry is not None and before.st_size != expected_entry.size_bytes:
            raise RuntimeError(
                f"Hugging Face snapshot content changed or has the wrong size: {relative}"
            )
        file_digest = hashlib.sha256()
        git_blob_digest = None
        if expected_entry is not None and expected_entry.lfs_sha256 is None:
            git_blob_digest = hashlib.sha1(  # noqa: S324 - Git blob identity
                usedforsecurity=False
            )
            git_blob_digest.update(f"blob {before.st_size}\0".encode("ascii"))
        with resolved.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                file_digest.update(chunk)
                if git_blob_digest is not None:
                    git_blob_digest.update(chunk)
        after = resolved.stat()
        if (
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or (path.is_symlink() and os.readlink(path) != link_target)
        ):
            raise RuntimeError(
                f"Hugging Face snapshot changed while it was being hashed: {relative}"
            )
        content_digest = file_digest.hexdigest()
        if expected_entry is not None and (
            content_digest != expected_entry.content_sha256
            or (
                git_blob_digest is not None
                and git_blob_digest.hexdigest() != expected_entry.git_blob_oid
            )
        ):
            raise RuntimeError(
                "Hugging Face snapshot content changed or differs from the pinned "
                f"repository blob: {relative}"
            )
        entries.append(
            {
                "path": relative,
                "size": int(after.st_size),
                "sha256": content_digest,
            }
        )

    if not entries:
        raise FileNotFoundError(f"Hugging Face snapshot contains no files: {root}")
    payload = {
        "format": "huggingface-snapshot-content-inventory-v1",
        "files": entries,
    }
    inventory = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    if expected is not None:
        expected_inventory = expected_hf_snapshot_inventory_sha256(expected)
        if inventory != expected_inventory:
            raise RuntimeError(
                "Hugging Face snapshot aggregate differs from its pinned inventory"
            )
    return inventory


def verify_hf_snapshot(
    snapshot_path: str | Path,
    *,
    model_id: str,
    revision: str,
    expected_inventory_sha256: str | None = None,
    expected_repo_files: Sequence[HuggingFaceRepoFile] | None = None,
) -> HuggingFaceSnapshot:
    """Verify a Hub-cache snapshot and return its complete source identity."""

    if not model_id or model_id.strip() != model_id:
        raise ValueError("Hugging Face model_id must be a non-empty canonical repo ID")
    pinned_revision = _validate_hf_revision(revision)
    path = Path(snapshot_path).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Hugging Face snapshot is not a directory: {path}")
    expected_repo_directory = "models--" + model_id.replace("/", "--")
    if (
        path.parent.name != "snapshots"
        or path.parent.parent.name != expected_repo_directory
        or path.name != pinned_revision
    ):
        raise ValueError(
            "Hugging Face snapshot must be the repo- and commit-addressed directory "
            f"returned by snapshot_download({model_id!r}, revision={pinned_revision!r})"
        )
    repository_files = (
        GEMMA4_REPOSITORY_FILES
        if expected_repo_files is None
        else tuple(expected_repo_files)
    )
    if expected_repo_files is None and (
        model_id != MODEL_ID or pinned_revision != MODEL_REVISION
    ):
        raise ValueError(
            "no pinned repository inventory exists for this Hugging Face model revision"
        )
    repository_inventory = expected_hf_repository_inventory_sha256(repository_files)
    inventory = hf_snapshot_inventory_sha256(
        path, expected_repo_files=repository_files
    )
    if expected_inventory_sha256 is not None:
        expected = str(expected_inventory_sha256).lower()
        if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
            raise ValueError("expected snapshot inventory must be a SHA-256 hex digest")
        if inventory != expected:
            raise RuntimeError(
                "Hugging Face snapshot content changed after it was materialized"
            )
    return HuggingFaceSnapshot(
        model_id=model_id,
        revision=pinned_revision,
        path=path,
        inventory_sha256=inventory,
        repository_inventory_sha256=repository_inventory,
    )


def materialize_hf_snapshot(
    model_id: str,
    revision: str,
    *,
    cache_dir: str | Path | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
    expected_repo_files: Sequence[HuggingFaceRepoFile] | None = None,
) -> HuggingFaceSnapshot:
    """Download an exact model commit and content-attest its local snapshot."""

    pinned_revision = _validate_hf_revision(revision)
    try:
        huggingface_hub = import_module("huggingface_hub")
    except (ImportError, ModuleNotFoundError) as exc:
        raise MaxTextUnavailableError(
            "checkpoint conversion requires huggingface-hub in the conversion environment"
        ) from exc
    kwargs: dict[str, Any] = {
        "repo_id": model_id,
        "repo_type": "model",
        "revision": pinned_revision,
        "token": token,
        "local_files_only": local_files_only,
    }
    if cache_dir is not None:
        kwargs["cache_dir"] = str(Path(cache_dir).expanduser())
    snapshot_path = huggingface_hub.snapshot_download(**kwargs)
    return verify_hf_snapshot(
        snapshot_path,
        model_id=model_id,
        revision=pinned_revision,
        expected_repo_files=expected_repo_files,
    )


def assert_hf_snapshot_unchanged(
    snapshot: HuggingFaceSnapshot,
    *,
    expected_repo_files: Sequence[HuggingFaceRepoFile] | None = None,
) -> str:
    """Re-hash a snapshot and fail if its contents changed; return the digest."""

    verified = verify_hf_snapshot(
        snapshot.path,
        model_id=snapshot.model_id,
        revision=snapshot.revision,
        expected_inventory_sha256=snapshot.inventory_sha256,
        expected_repo_files=expected_repo_files,
    )
    if (
        verified.repository_inventory_sha256
        != snapshot.repository_inventory_sha256
    ):
        raise RuntimeError("Hugging Face repository inventory identity changed")
    return verified.inventory_sha256


def checkpoint_conversion_spec(
    maxtext_root: str | Path,
    output_directory: str,
    *,
    hf_snapshot: HuggingFaceSnapshot,
    python_executable: str = sys.executable,
    config: Gemma4Config | None = None,
) -> CheckpointConversionSpec:
    """Build and validate a secret-free conversion specification."""

    cfg = config or Gemma4Config()
    cfg.validate()
    configured_revision = cfg.model_revision
    if hf_snapshot.model_id != cfg.model_id:
        raise ValueError(
            f"snapshot model ID must be {cfg.model_id}, got {hf_snapshot.model_id}"
        )
    if hf_snapshot.revision != configured_revision:
        raise ValueError(
            "snapshot revision does not match the configured Gemma model revision"
        )
    _validate_hf_revision(hf_snapshot.revision)
    if len(hf_snapshot.inventory_sha256) != 64 or any(
        character not in "0123456789abcdef"
        for character in hf_snapshot.inventory_sha256
    ):
        raise ValueError("snapshot inventory must be a SHA-256 hex digest")
    expected_repository_inventory = expected_hf_repository_inventory_sha256()
    expected_snapshot_inventory = expected_hf_snapshot_inventory_sha256()
    if hf_snapshot.repository_inventory_sha256 != expected_repository_inventory:
        raise ValueError(
            "snapshot repository inventory does not match the pinned Gemma revision"
        )
    if hf_snapshot.inventory_sha256 != expected_snapshot_inventory:
        raise ValueError(
            "snapshot content inventory does not match the pinned Gemma revision"
        )
    snapshot_path = Path(hf_snapshot.path).expanduser().resolve()
    if not snapshot_path.is_dir():
        raise FileNotFoundError(f"Hugging Face snapshot is missing: {snapshot_path}")
    return CheckpointConversionSpec(
        maxtext_root=Path(maxtext_root),
        hf_model_id=hf_snapshot.model_id,
        hf_revision=hf_snapshot.revision,
        hf_snapshot_path=snapshot_path,
        hf_snapshot_inventory_sha256=hf_snapshot.inventory_sha256,
        hf_repository_inventory_sha256=hf_snapshot.repository_inventory_sha256,
        output_directory=output_directory,
        python_executable=python_executable,
    )


def build_checkpoint_conversion_command(
    spec: CheckpointConversionSpec,
) -> tuple[str, ...]:
    """Return argv for MaxText's official converter.

    The Hub download and authentication happen before this command is built.
    MaxText receives only a local, content-attested snapshot path, keeping
    credentials and mutable repository references out of argv and provenance.
    """

    if not spec.base_config.is_file():
        raise FileNotFoundError(
            f"MaxText base config does not exist: {spec.base_config}"
        )
    overrides = tuple(
        f"{key}={value}" for key, value in spec.maxtext_overrides().items()
    )
    return (
        spec.python_executable,
        "-m",
        "maxtext.checkpoint_conversion.to_maxtext",
        str(spec.base_config),
        *overrides,
        f"--hf_model_path={spec.hf_model_path}",
        f"--save_dtype={spec.save_dtype}",
        f"--lazy_load_tensors={'true' if spec.lazy_load_tensors else 'false'}",
    )


def checkpoint_validation_command(
    spec: CheckpointConversionSpec,
    checkpoint_items_path: str,
    *,
    max_kl_div: float = 0.03,
) -> tuple[str, ...]:
    """Return argv for MaxText's Hugging Face logit-parity checker."""

    if not (0.0 < max_kl_div <= 0.03):
        raise ValueError("max_kl_div must be in (0, 0.03]")
    return (
        spec.python_executable,
        "-m",
        "tests.utils.forward_pass_logit_checker",
        str(spec.base_config),
        f"tokenizer_path={spec.hf_model_path}",
        f"load_parameters_path={checkpoint_items_path}",
        "model_name=gemma4-26b",
        "use_multimodal=false",
        "scan_layers=false",
        "pure_nnx_decoder=true",
        "hardware=cpu",
        "skip_jax_distributed_system=true",
        # Gemma 4's model config enables the Pallas-backed MegaBlox GMM.  JAX
        # cannot lower that kernel on CPU outside Pallas interpret mode.  The
        # portable MaxText branch uses jax.lax.ragged_dot and computes the same
        # routed MoE function, which is what the CPU parity checker needs.
        "megablox=false",
        "attention=dot_product",
        "per_device_batch_size=1",
        "dtype=float32",
        "wi_tile_fwd_embed_dim=512",
        "wi_tile_fwd_mlp_dim=512",
        "wo_tile_fwd_embed_dim=512",
        "wo_tile_fwd_mlp_dim=512",
        f"--max_kl_div={max_kl_div}",
        "--run_hf_model=true",
        f"--hf_model_path={spec.hf_model_path}",
    )


def checkpoint_inventory_sha256(checkpoint_items_path: str) -> str:
    """Return a deterministic content-inventory digest for an Orbax checkpoint.

    Local checkpoints are hashed byte-for-byte.  For GCS checkpoints the digest
    covers every object's relative name, size, generation, and provider-reported
    CRC32C/MD5 checksum, avoiding a second multi-gigabyte download after MaxText's
    parity checker has already read the checkpoint.
    """

    if checkpoint_items_path.startswith("gs://"):
        parsed = urlparse(checkpoint_items_path)
        prefix = parsed.path.lstrip("/").rstrip("/")
        if not parsed.netloc or not prefix:
            raise ValueError(
                "GCS checkpoint path must include a bucket and object prefix"
            )
        try:
            storage = import_module("google.cloud.storage")
        except (ImportError, ModuleNotFoundError) as exc:
            raise MaxTextUnavailableError(
                "GCS checkpoint hashing requires google-cloud-storage in the conversion environment"
            ) from exc
        client = storage.Client()
        entries: list[dict[str, Any]] = []
        for blob in client.list_blobs(parsed.netloc, prefix=prefix + "/"):
            if blob.name.endswith("/"):
                continue
            relative = blob.name[len(prefix) :].lstrip("/")
            entries.append(
                {
                    "path": relative,
                    "size": int(blob.size),
                    "generation": str(blob.generation),
                    "crc32c": blob.crc32c,
                    "md5": blob.md5_hash,
                }
            )
        if not entries:
            raise FileNotFoundError(
                f"checkpoint prefix contains no objects: {checkpoint_items_path}"
            )
        payload = {
            "format": "orbax-gcs-content-inventory-v1",
            "objects": sorted(entries, key=lambda item: item["path"]),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    root = Path(checkpoint_items_path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"checkpoint does not exist: {root}")
    files = (
        [root]
        if root.is_file()
        else sorted(path for path in root.rglob("*") if path.is_file())
    )
    if not files:
        raise FileNotFoundError(f"checkpoint contains no files: {root}")
    digest = hashlib.sha256()
    digest.update(b"orbax-local-content-inventory-v1\0")
    for path in files:
        if path.is_symlink():
            raise ValueError(f"checkpoint inventory refuses symbolic links: {path}")
        relative = path.name if root.is_file() else path.relative_to(root).as_posix()
        size = path.stat().st_size
        file_digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                file_digest.update(chunk)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(file_digest.digest())
    return digest.hexdigest()


class Gemma4PrefixAdapter:
    """Execute blocks 1--14, then capture block 15's routed branch.

    ``model`` must be an unscanned, pure-NNX MaxText model or a compatible fake.
    No MaxText symbol is imported by the ordinary constructor, which keeps
    offline validation and CPU tests lightweight.
    """

    def __init__(self, model: Any, config: Gemma4Config | None = None):
        self.model = model
        self.config = config or Gemma4Config()
        self.config.validate()
        self.decoder = _require_attribute(model, "decoder", "model")
        self.layers = _resolve_layers(self.decoder)
        validate_model_architecture(model, self.config, layers=self.layers)

    @classmethod
    def from_maxtext(
        cls,
        model: Any,
        config: Gemma4Config | None = None,
        *,
        maxtext_root: str | Path | None = None,
    ) -> "Gemma4PrefixAdapter":
        require_maxtext(maxtext_root)
        return cls(model, config)

    def __call__(
        self,
        token_ids: Any,
        decoder_positions: Any,
        decoder_segment_ids: Any,
        sequence_ids: Any,
        *,
        valid_token_indices: Any | None = None,
        model_mode: str = "train",
    ) -> CaptureBatch:
        """Capture routed-branch rows.

        Inputs have shape ``[batch, sequence]``.  ``sequence_ids`` may be
        ``[batch]`` or ``[batch, sequence]``.  ``valid_token_indices`` indexes
        the flattened token dimension and lets the data pipeline remove BOS,
        EOS, and padding with a static-shape gather before device-to-host copy.
        """

        _validate_input_shapes(
            token_ids, decoder_positions, decoder_segment_ids, sequence_ids
        )
        hidden = self._run_prefix(
            token_ids,
            decoder_positions,
            decoder_segment_ids,
            model_mode,
        )

        target = self.layers[self.config.target_layer_index]
        activation, expert_ids, router_weights = self._capture_target(
            target,
            hidden,
            decoder_positions,
            decoder_segment_ids,
            model_mode,
        )

        batch, sequence = token_ids.shape
        flat_activation = activation.reshape(
            (batch * sequence, self.config.hidden_size)
        )
        flat_ids = expert_ids.reshape((batch * sequence, self.config.experts_per_token))
        flat_weights = router_weights.reshape(
            (batch * sequence, self.config.experts_per_token)
        )
        flat_positions = decoder_positions.reshape((batch * sequence,))
        if len(sequence_ids.shape) == 1:
            flat_sequence_ids = sequence_ids.repeat(sequence)
        else:
            flat_sequence_ids = sequence_ids.reshape((batch * sequence,))

        if valid_token_indices is not None:
            indices = valid_token_indices.reshape((-1,))
            flat_activation = flat_activation[indices]
            flat_ids = flat_ids[indices]
            flat_weights = flat_weights[indices]
            flat_positions = flat_positions[indices]
            flat_sequence_ids = flat_sequence_ids[indices]

        if hasattr(flat_ids, "astype"):
            flat_ids = flat_ids.astype("uint8")
        return CaptureBatch(
            activation=flat_activation,
            expert_ids=flat_ids,
            router_weights=flat_weights,
            sequence_id=flat_sequence_ids,
            position=flat_positions,
        )

    def instrumentation_parity(
        self,
        token_ids: Any,
        decoder_positions: Any,
        decoder_segment_ids: Any,
        sequence_ids: Any,
        *,
        model_mode: str = "train",
    ) -> Any:
        """Prove that target instrumentation leaves the full block unchanged.

        The production benchmark calls this before timing.  It compares an
        untouched block-15 invocation with a manual recomposition built from
        the same attention residual, captured routed aggregate, shared expert,
        outer FFN normalization, residual addition, and layer scalar.  It also
        independently repeats the small router projection so the IDs and
        normalized weights persisted by capture must equal MaxText's routing
        primitives exactly.

        A JAX boolean scalar is returned so the check itself can be compiled
        under the same mesh and logical-axis contexts as harvesting.
        """

        _validate_input_shapes(
            token_ids,
            decoder_positions,
            decoder_segment_ids,
            sequence_ids,
        )
        hidden = self._run_prefix(
            token_ids,
            decoder_positions,
            decoder_segment_ids,
            model_mode,
        )
        target = self.layers[self.config.target_layer_index]

        reference = _first(
            target(
                hidden,
                decoder_segment_ids,
                decoder_positions,
                True,
                model_mode,
                bidirectional_mask=None,
                kv_cache=None,
                attention_metadata=None,
            )
        )

        residual = self._target_attention_residual(
            target,
            hidden,
            decoder_positions,
            decoder_segment_ids,
            model_mode,
        )
        activation, expert_ids, router_weights = self._capture_routed(
            target.mlp, residual
        )

        shared_inputs = target.pre_ffw_norm(residual)
        shared_output = target.mlp.moe_block.shared_experts(
            shared_inputs,
            intermediate_sharding=None,
            out_sharding=None,
        )
        shared_output = target.mlp.post_feedforward_layernorm_1(shared_output)
        reconstructed = activation + shared_output
        if getattr(target, "post_ffw_norm", None) is not None:
            reconstructed = target.post_ffw_norm(reconstructed)
        reconstructed = (reconstructed + residual) * _array_value(
            target.layer_scalar,
            dtype=getattr(target.config, "dtype", None),
        )

        reference_ids, reference_weights = self._route(target.mlp, residual)
        reference_weights = _as_bfloat16(reference_weights)
        jnp = import_module("jax.numpy")
        return jnp.logical_and(
            jnp.array_equal(reference, reconstructed),
            jnp.logical_and(
                jnp.array_equal(expert_ids, reference_ids),
                jnp.array_equal(router_weights, reference_weights),
            ),
        )

    def _run_prefix(
        self,
        token_ids: Any,
        positions: Any,
        segment_ids: Any,
        model_mode: str,
    ) -> Any:
        hidden = self._embed(token_ids, positions, model_mode)
        for layer in self.layers[: self.config.target_layer_index]:
            hidden = _first(
                layer(
                    hidden,
                    segment_ids,
                    positions,
                    True,
                    model_mode,
                    bidirectional_mask=None,
                    kv_cache=None,
                    attention_metadata=None,
                )
            )
        return hidden

    def _embed(self, token_ids: Any, positions: Any, model_mode: str) -> Any:
        embed = getattr(self.decoder, "_apply_embedding", None)
        token_embedder = getattr(self.model, "token_embedder", None)
        if callable(embed) and token_embedder is not None:
            return embed(token_embedder, token_ids, positions, True, model_mode)
        fallback = getattr(self.model, "embed_tokens", None)
        if callable(fallback):
            return fallback(token_ids, positions)
        raise TypeError(
            "model must expose token_embedder + decoder._apply_embedding or embed_tokens"
        )

    def _capture_target(
        self,
        layer: Any,
        hidden: Any,
        positions: Any,
        segment_ids: Any,
        model_mode: str,
    ) -> tuple[Any, Any, Any]:
        residual = self._target_attention_residual(
            layer,
            hidden,
            positions,
            segment_ids,
            model_mode,
        )
        return self._capture_routed(layer.mlp, residual)

    def _target_attention_residual(
        self,
        layer: Any,
        hidden: Any,
        positions: Any,
        segment_ids: Any,
        model_mode: str,
    ) -> Any:
        # This is the attention half of Gemma4DecoderLayer.__call__.
        normalized = layer.pre_self_attention_norm(hidden)
        attention_result = layer.self_attention(
            normalized,
            normalized,
            positions,
            decoder_segment_ids=segment_ids,
            deterministic=True,
            model_mode=model_mode,
            bidirectional_mask=None,
            kv_cache=None,
            attention_metadata=None,
        )
        attention = _first(attention_result)
        if getattr(layer, "post_self_attention_norm", None) is not None:
            attention = layer.post_self_attention_norm(attention)
        return attention + hidden

    def _capture_routed(self, mlp: Any, residual: Any) -> tuple[Any, Any, Any]:
        """Run the routed branch and cast its storage boundary to BF16."""

        # Gemma4MoE applies this path independently from the shared expert.
        routed_inputs = mlp.pre_feedforward_layernorm_2(residual)
        unscaled_gate_inputs = mlp.gate_norm(residual)
        router_scale = _value(mlp.pre_forward_scale_2)
        gate_inputs = (
            unscaled_gate_inputs * (self.config.hidden_size**-0.5) * router_scale
        )

        routed_moe = mlp.moe_block.routed_moe
        routed_result = routed_moe(
            routed_inputs, gate_inputs=gate_inputs, out_sharding=None
        )
        routed_output = _first(routed_result)
        activation = _as_bfloat16(mlp.post_feedforward_layernorm_2(routed_output))

        # Re-run only the small router projection.  get_topk is the same method
        # used by the routed kernel, including Gemma 4 softmax + top-k renorm.
        expert_ids, router_weights = self._route(mlp, residual)
        return activation, expert_ids, _as_bfloat16(router_weights)

    def _route(self, mlp: Any, residual: Any) -> tuple[Any, Any]:
        unscaled_gate_inputs = mlp.gate_norm(residual)
        router_scale = _value(mlp.pre_forward_scale_2)
        gate_inputs = (
            unscaled_gate_inputs * (self.config.hidden_size**-0.5) * router_scale
        )
        routed_moe = mlp.moe_block.routed_moe
        logits_result = routed_moe.gate(gate_inputs)
        if not isinstance(logits_result, tuple) or len(logits_result) != 2:
            raise TypeError(
                "MaxText routed gate must return (gate_logits, pre_bias_logits)"
            )
        gate_logits, pre_bias_logits = logits_result
        router_weights, expert_ids = routed_moe.get_topk(
            gate_logits,
            pre_bias_logits,
            getattr(routed_moe, "rngs", None),
        )
        return expert_ids, router_weights


def validate_model_architecture(
    model: Any,
    config: Gemma4Config,
    *,
    layers: Sequence[Any] | None = None,
) -> None:
    """Fail before compilation if the checkpoint/model shape has drifted."""

    decoder = _require_attribute(model, "decoder", "model")
    resolved_layers = list(layers) if layers is not None else _resolve_layers(decoder)
    if len(resolved_layers) != config.num_decoder_layers:
        raise ValueError(
            f"expected {config.num_decoder_layers} unscanned decoder layers, found {len(resolved_layers)}"
        )
    model_config = getattr(model, "config", getattr(decoder, "config", None))
    if model_config is None:
        raise TypeError("model or decoder must expose a MaxText config")
    model_name = getattr(model_config, "model_name", None)
    if model_name != config.maxtext_model_name:
        raise ValueError(
            f"model_name must be {config.maxtext_model_name!r}, got {model_name!r}"
        )
    expected_aliases = (
        (("num_decoder_layers", "base_num_decoder_layers"), config.num_decoder_layers),
        (("emb_dim", "base_emb_dim", "hidden_size"), config.hidden_size),
        (("num_query_heads", "base_num_query_heads"), config.num_query_heads),
        (("num_kv_heads", "base_num_kv_heads"), config.num_kv_heads),
        (("global_num_kv_heads",), config.global_num_kv_heads),
        (("head_dim",), config.head_dim),
        (("global_head_dim",), config.global_head_dim),
        (("mlp_dim", "base_mlp_dim"), config.mlp_dim),
        (("moe_mlp_dim", "base_moe_mlp_dim"), config.moe_mlp_dim),
        (("num_experts",), config.num_routed_experts),
        (("num_experts_per_tok",), config.experts_per_token),
        (("shared_experts",), config.num_shared_experts),
        (("sliding_window_size",), config.sliding_window_size),
    )
    for aliases, expected in expected_aliases:
        actual = _first_attribute(model_config, aliases)
        if actual != expected:
            raise ValueError(f"model {aliases[0]} must be {expected}, got {actual!r}")
    for name, expected in (
        ("scan_layers", False),
        ("pure_nnx_decoder", True),
        ("use_multimodal", False),
        ("norm_topk_prob", True),
        ("use_random_routing", False),
        ("share_kv_projections", config.share_kv_projections),
        ("use_post_attn_norm", config.use_post_attn_norm),
        ("use_post_ffw_norm", config.use_post_ffw_norm),
        ("float32_gate_logits", config.float32_gate_logits),
    ):
        actual = getattr(model_config, name, None)
        if actual != expected:
            raise ValueError(f"model {name} must be {expected}, got {actual!r}")
    for name, expected in (("dropout_rate", 0.0), ("load_balance_loss_weight", 0.0)):
        actual = getattr(model_config, name, None)
        if actual != expected:
            raise ValueError(f"model {name} must be {expected}, got {actual!r}")
    for name, expected in (
        ("dtype", config.dtype),
        ("weight_dtype", config.weight_dtype),
    ):
        actual = getattr(model_config, name, None)
        if _dtype_name(actual) != expected:
            raise ValueError(f"model {name} must be {expected}, got {actual!r}")
    capacity_factor = getattr(model_config, "capacity_factor", None)
    if capacity_factor is None or capacity_factor > 0.0:
        raise ValueError(
            f"model capacity_factor must disable token dropping, got {capacity_factor!r}"
        )

    for index, layer in enumerate(resolved_layers):
        named_layer = _require_attribute(
            decoder,
            f"layers_{index}",
            "pure NNX decoder",
        )
        if named_layer is not layer:
            raise ValueError(
                f"decoder layer order drift at block {index + 1}: "
                f"decoder.layers[{index}] does not match decoder.layers_{index}"
            )
        actual_attention = str(getattr(layer, "attention_type", "")).lower()
        expected_attention = "global" if (index + 1) % 6 == 0 else "local"
        if expected_attention not in actual_attention:
            raise ValueError(
                f"block {index + 1} must use {expected_attention} attention, "
                f"got {getattr(layer, 'attention_type', None)!r}"
            )

    target = resolved_layers[config.target_layer_index]
    if "local" not in str(getattr(target, "attention_type", "")).lower():
        raise ValueError("one-indexed block 15 must use local sliding attention")
    mlp = _require_attribute(target, "mlp", "target layer")
    for name in (
        "post_feedforward_layernorm_1",
        "pre_feedforward_layernorm_2",
        "gate_norm",
        "pre_forward_scale_2",
        "post_feedforward_layernorm_2",
        "moe_block",
    ):
        _require_attribute(mlp, name, "target Gemma4MoE")
    routed_moe = _require_attribute(
        mlp.moe_block, "routed_moe", "target RoutedAndSharedMoE"
    )
    if not callable(getattr(mlp.moe_block, "shared_experts", None)):
        raise TypeError("target routed/shared MoE lacks callable shared_experts")
    for name in ("gate", "get_topk"):
        if not callable(getattr(routed_moe, name, None)):
            raise TypeError(f"target routed MoE lacks callable {name}")


def validate_capture_batch(
    batch: CaptureBatch,
    config: Gemma4Config | None = None,
    *,
    require_storage_dtypes: bool = False,
) -> CaptureSummary:
    """Validate a host shard before it can be published in a manifest."""

    cfg = config or Gemma4Config()
    activation = np.asarray(batch.activation)
    ids = np.asarray(batch.expert_ids)
    weights = np.asarray(batch.router_weights)
    sequence_ids = np.asarray(batch.sequence_id)
    positions = np.asarray(batch.position)

    if activation.ndim != 2 or activation.shape[1] != cfg.hidden_size:
        raise ValueError(
            f"activation must have shape [N, {cfg.hidden_size}], got {activation.shape}"
        )
    rows = activation.shape[0]
    expected_route_shape = (rows, cfg.experts_per_token)
    if ids.shape != expected_route_shape:
        raise ValueError(
            f"expert_ids must have shape {expected_route_shape}, got {ids.shape}"
        )
    if weights.shape != expected_route_shape:
        raise ValueError(
            f"router_weights must have shape {expected_route_shape}, got {weights.shape}"
        )
    if sequence_ids.shape != (rows,) or positions.shape != (rows,):
        raise ValueError("sequence_id and position must each have shape [N]")
    if rows == 0:
        raise ValueError("capture batches cannot be empty")
    if not np.issubdtype(ids.dtype, np.integer):
        raise TypeError("expert_ids must have an integer dtype")
    if not np.issubdtype(sequence_ids.dtype, np.integer) or not np.issubdtype(
        positions.dtype, np.integer
    ):
        raise TypeError("sequence_id and position must have integer dtypes")
    if not _all_finite_in_row_chunks(activation) or not _all_finite_in_row_chunks(
        weights
    ):
        raise ValueError("capture contains NaN or infinite values")
    if np.any(ids < 0) or np.any(ids >= cfg.num_routed_experts):
        raise ValueError("expert IDs must be in [0, 127]")
    if np.any(np.diff(np.sort(ids.astype(np.int16), axis=1), axis=1) == 0):
        raise ValueError("each token must route to eight unique experts")
    if np.any(weights <= 0.0):
        raise ValueError("router weights must be strictly positive")
    router_sums = weights.astype(np.float32).sum(axis=1)
    max_sum_error = float(np.max(np.abs(router_sums - 1.0)))
    if max_sum_error > 0.02:
        raise ValueError(
            f"router weights must sum to one (max error {max_sum_error:.6f})"
        )
    if np.any(positions < 0) or np.any(positions >= cfg.sequence_length):
        raise ValueError(f"positions must lie in [0, {cfg.sequence_length - 1}]")
    if require_storage_dtypes:
        if str(activation.dtype) != "bfloat16" or str(weights.dtype) != "bfloat16":
            raise TypeError(
                "activation and router_weights must be bfloat16 for storage"
            )
        if ids.dtype != np.dtype("uint8"):
            raise TypeError("expert_ids must be uint8 for storage")

    return CaptureSummary(
        rows=rows,
        hidden_size=cfg.hidden_size,
        experts_per_token=cfg.experts_per_token,
        min_expert_id=int(ids.min()),
        max_expert_id=int(ids.max()),
        max_router_sum_error=max_sum_error,
    )


def _resolve_layers(decoder: Any) -> list[Any]:
    layers = getattr(decoder, "layers", None)
    if layers is None:
        raise TypeError("pure NNX decoder must expose an unscanned layers collection")
    try:
        return [layers[index] for index in range(len(layers))]
    except (TypeError, KeyError, IndexError) as exc:
        raise TypeError(
            "decoder.layers must be an indexable unscanned collection"
        ) from exc


def _validate_input_shapes(
    token_ids: Any, positions: Any, segment_ids: Any, sequence_ids: Any
) -> None:
    if len(token_ids.shape) != 2:
        raise ValueError(
            f"token_ids must have shape [batch, sequence], got {token_ids.shape}"
        )
    if positions.shape != token_ids.shape or segment_ids.shape != token_ids.shape:
        raise ValueError(
            "decoder_positions and decoder_segment_ids must match token_ids"
        )
    if len(sequence_ids.shape) not in (1, 2):
        raise ValueError("sequence_ids must have shape [batch] or [batch, sequence]")
    if len(sequence_ids.shape) == 1 and sequence_ids.shape[0] != token_ids.shape[0]:
        raise ValueError(
            "one-dimensional sequence_ids must have one entry per batch row"
        )
    if len(sequence_ids.shape) == 2 and sequence_ids.shape != token_ids.shape:
        raise ValueError("two-dimensional sequence_ids must match token_ids")


def _first(result: Any) -> Any:
    return result[0] if isinstance(result, tuple) else result


def _value(value: Any) -> Any:
    return getattr(value, "value", value)


def _as_bfloat16(value: Any) -> Any:
    """Cast a JAX/NumPy-compatible array at the capture storage boundary."""

    jnp = import_module("jax.numpy")
    return value.astype(jnp.bfloat16)


def _array_value(value: Any, *, dtype: Any | None = None) -> Any:
    """Materialize an NNX parameter using the model's compute dtype."""

    jnp = import_module("jax.numpy")
    return jnp.asarray(_value(value), dtype=dtype)


def _require_attribute(value: Any, name: str, owner: str) -> Any:
    if not hasattr(value, name):
        raise TypeError(f"{owner} lacks required attribute {name}")
    return getattr(value, name)


def _first_attribute(value: Any, names: Sequence[str]) -> Any:
    for name in names:
        if hasattr(value, name):
            return getattr(value, name)
    raise TypeError(f"model config lacks all of: {', '.join(names)}")


def _dtype_name(value: Any) -> str:
    try:
        return str(np.dtype(value)).lower()
    except (TypeError, ValueError):
        text = str(value).lower()
        return "bfloat16" if "bfloat16" in text else text


def _all_finite_in_row_chunks(array: np.ndarray, chunk_rows: int = 4096) -> bool:
    """Check large BF16 activation shards without a shard-sized bool temporary."""

    for start in range(0, array.shape[0], chunk_rows):
        if not bool(np.all(np.isfinite(array[start : start + chunk_rows]))):
            return False
    return True
