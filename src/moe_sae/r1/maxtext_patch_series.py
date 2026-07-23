"""Fail-closed application of the two pinned R1 MaxText source patches."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import fcntl
import hashlib
import os
from pathlib import Path
import stat
import subprocess
from typing import Iterator, Mapping


MAXTEXT_BASE_COMMIT = "c03f8bd307d18049fa814fc8c104bd9ec8c292aa"
PATCH_SERIES = (
    (
        "0001-routed-moe-prefix-tap.patch",
        "f5b4f810dd981add4f13baad91d5811cf632ec32d17266b67b84644783ae850b",
    ),
    (
        "0002-b128-routed-prefix.patch",
        "d78349f83d08993afbaa1b2b346a9ac34f1b034b5015fa02572f71b79b6859fe",
    ),
)
POST_APPLY_SHA256: Mapping[str, str] = {
    "src/maxtext/layers/moe.py": (
        "f6fe4ace7538f542747d20b3167e3424219aed4562552bd5a51b178d4f0cf527"
    ),
    "src/maxtext/layers/nnx_decoders.py": (
        "e51522f3249fb7d1bcf8e39ca6f55b3c51c5471fc5adf0ac7b1a5de94e3c90f5"
    ),
    "src/maxtext/models/deepseek_prefix.py": (
        "11fdb7bb64797822c336759df42dafa9524a1c1af50f82b72dc7af8edc8048c0"
    ),
}
EXPECTED_STATUS = (
    " M src/maxtext/layers/moe.py",
    " M src/maxtext/layers/nnx_decoders.py",
    "?? src/maxtext/models/deepseek_prefix.py",
)
_MAX_PATCH_BYTES = 1024 * 1024


class MaxTextPatchSeriesError(RuntimeError):
    """The checkout or sealed patch-series contract failed."""


@dataclass(frozen=True, slots=True)
class MaxTextPatchSeriesAttestation:
    schema: str
    base_commit: str
    patch_sha256: tuple[str, ...]
    post_apply_sha256: tuple[tuple[str, str], ...]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_patch_root() -> Path:
    return _project_root() / "patches" / "r1" / "maxtext"


def _absolute_owned_directory(value: str | os.PathLike[str], label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or path.resolve(strict=True) != path:
        raise MaxTextPatchSeriesError(f"{label} must be an absolute non-symlink path")
    metadata = path.stat(follow_symlinks=False)
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.geteuid():
        raise MaxTextPatchSeriesError(f"{label} must be an owned directory")
    return path


def _read_exact_file(path: Path, expected_sha256: str, *, label: str) -> bytes:
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or not 1 <= before.st_size <= _MAX_PATCH_BYTES
        ):
            raise MaxTextPatchSeriesError(
                f"{label} is not one bounded regular file: {path}"
            )
        payload = bytearray()
        while len(payload) <= _MAX_PATCH_BYTES:
            chunk = os.read(
                descriptor,
                min(1024 * 1024, _MAX_PATCH_BYTES + 1 - len(payload)),
            )
            if not chunk:
                break
            payload.extend(chunk)
        after = os.fstat(descriptor)
    except OSError as error:
        raise MaxTextPatchSeriesError(
            f"cannot securely read {label}: {path}"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ) or len(payload) != before.st_size:
        raise MaxTextPatchSeriesError(f"{label} changed while being read: {path}")
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected_sha256:
        raise MaxTextPatchSeriesError(
            f"{label} SHA-256 drift for {path.name}: expected "
            f"{expected_sha256}, got {actual}"
        )
    return bytes(payload)


def _git(
    root: Path,
    *arguments: str,
    input_bytes: bytes | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("GIT_")
    }
    environment.update(
        {
            "LC_ALL": "C",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    result = subprocess.run(
        ("git", "-C", str(root), *arguments),
        input=input_bytes,
        capture_output=True,
        check=False,
        env=environment,
    )
    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout).decode("utf-8", "replace").strip()
        raise MaxTextPatchSeriesError(
            f"git {' '.join(arguments)} failed: {detail or 'no diagnostic'}"
        )
    return result


def _stdout(root: Path, *arguments: str) -> str:
    return _git(root, *arguments).stdout.decode("utf-8", "strict").rstrip("\n")


def _require_clean_base(root: Path) -> None:
    git_directory = root / ".git"
    metadata = git_directory.stat(follow_symlinks=False)
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.geteuid():
        raise MaxTextPatchSeriesError("MaxText checkout must own a real .git directory")
    if Path(_stdout(root, "rev-parse", "--show-toplevel")) != root:
        raise MaxTextPatchSeriesError("MaxText root is not the Git top level")
    actual = _stdout(root, "rev-parse", "--verify", "HEAD^{commit}")
    if actual != MAXTEXT_BASE_COMMIT:
        raise MaxTextPatchSeriesError(
            f"MaxText HEAD is {actual}; expected {MAXTEXT_BASE_COMMIT}"
        )
    status = _stdout(root, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise MaxTextPatchSeriesError(f"MaxText checkout is not clean: {status}")


@contextmanager
def _exclusive_checkout(root: Path) -> Iterator[None]:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open(root, flags)
    try:
        if os.fstat(descriptor).st_uid != os.geteuid():
            raise MaxTextPatchSeriesError("MaxText checkout owner changed")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise MaxTextPatchSeriesError(
                "MaxText checkout is already locked"
            ) from error
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def apply_pinned_maxtext_patch_series(
    maxtext_root: str | os.PathLike[str],
    *,
    patch_root: str | os.PathLike[str] | None = None,
) -> MaxTextPatchSeriesAttestation:
    """Apply the exact ordered series to one clean, owned pinned checkout."""

    root = _absolute_owned_directory(maxtext_root, "MaxText root")
    patches = _absolute_owned_directory(
        default_patch_root() if patch_root is None else patch_root,
        "patch root",
    )
    payloads = tuple(
        _read_exact_file(
            patches / name,
            expected_sha256,
            label=f"patch {name}",
        )
        for name, expected_sha256 in PATCH_SERIES
    )
    series = b"".join(payloads)
    with _exclusive_checkout(root):
        _require_clean_base(root)
        _git(
            root,
            "apply",
            "--check",
            "--whitespace=error-all",
            input_bytes=series,
        )
        _require_clean_base(root)
        _git(root, "apply", "--whitespace=error-all", input_bytes=series)
        if (
            _stdout(root, "rev-parse", "--verify", "HEAD^{commit}")
            != MAXTEXT_BASE_COMMIT
        ):
            raise MaxTextPatchSeriesError(
                "MaxText HEAD changed during patch application"
            )
        status = tuple(
            _stdout(
                root, "status", "--porcelain=v1", "--untracked-files=all"
            ).splitlines()
        )
        if status != EXPECTED_STATUS:
            raise MaxTextPatchSeriesError(
                f"post-apply path set changed: expected {EXPECTED_STATUS}, got {status}"
            )
        _git(root, "diff", "--check")
        observed = tuple(
            (
                relative_path,
                hashlib.sha256(
                    _read_exact_file(
                        root / relative_path,
                        expected_sha256,
                        label=f"post-apply file {relative_path}",
                    )
                ).hexdigest(),
            )
            for relative_path, expected_sha256 in POST_APPLY_SHA256.items()
        )
        if dict(observed) != dict(POST_APPLY_SHA256):
            raise MaxTextPatchSeriesError(
                f"post-apply hashes changed: expected {POST_APPLY_SHA256}, "
                f"got {dict(observed)}"
            )
    return MaxTextPatchSeriesAttestation(
        schema="maxtext-r1-patch-series-attestation-v1",
        base_commit=MAXTEXT_BASE_COMMIT,
        patch_sha256=tuple(digest for _name, digest in PATCH_SERIES),
        post_apply_sha256=observed,
    )
