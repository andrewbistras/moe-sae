"""Fail-closed application and attestation for the pinned MaxText MoE tap.

The patch is deliberately narrower than a general patching utility.  It accepts
one exact MaxText ``moe.py`` base, one exact patch artifact, and one exact
patched result.  Anything else is source drift and is rejected before mutation.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import fcntl
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import stat
from typing import Iterator, Literal


MAXTEXT_BASE_COMMIT = "c03f8bd307d18049fa814fc8c104bd9ec8c292aa"
TARGET_RELATIVE_PATH = PurePosixPath("src/maxtext/layers/moe.py")
PATCH_RELATIVE_PATH = PurePosixPath(
    "patches/r1/maxtext/0001-routed-moe-prefix-tap.patch"
)

BASE_MOE_GIT_BLOB = "7737734b09c8be5dc332f9b29748ff93b7c96cff"
PATCHED_MOE_GIT_BLOB = "c84c56b28bdd31b2e16001d5028db47de2b274ae"
BASE_MOE_SHA256 = "b1a56ed4f74c7a072e944381f04fa0f23541dcdf73e9f10e298c4010a487c94d"
PATCHED_MOE_SHA256 = "f6fe4ace7538f542747d20b3167e3424219aed4562552bd5a51b178d4f0cf527"
PATCH_SHA256 = "f5b4f810dd981add4f13baad91d5811cf632ec32d17266b67b84644783ae850b"

BASE_MOE_SIZE_BYTES = 135_495
PATCHED_MOE_SIZE_BYTES = 141_152
PATCH_SIZE_BYTES = 11_641
_GENERAL_FILE_MAX_BYTES = 16 * 1024 * 1024
_READ_CHUNK_BYTES = 1024 * 1024

_TARGET = TARGET_RELATIVE_PATH.as_posix()
_PATCH_HEADER = (
    f"diff --git a/{_TARGET} b/{_TARGET}\n",
    f"index {BASE_MOE_GIT_BLOB}..{PATCHED_MOE_GIT_BLOB} 100644\n",
    f"--- a/{_TARGET}\n",
    f"+++ b/{_TARGET}\n",
)
_HUNK_HEADER = re.compile(rb"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(?: .*)?\n$")

TapState = Literal["base", "patched"]


class MaxTextTapPatchError(RuntimeError):
    """The pinned source, patch, or atomic application contract was violated."""


@dataclass(frozen=True, slots=True)
class MaxTextTapAttestationV1:
    """Content identity for an accepted MaxText source state."""

    schema: Literal["maxtext-routed-moe-prefix-tap-attestation-v1"]
    expected_upstream_commit: str
    target_relative_path: str
    source_state: TapState
    source_sha256: str
    patch_sha256: str

    def as_dict(self) -> dict[str, str]:
        """Return a stable, JSON-serializable mapping."""

        return asdict(self)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_patch_path() -> Path:
    """Return the canonical patch artifact in this source checkout."""

    return _project_root().joinpath(*PATCH_RELATIVE_PATH.parts)


@dataclass(frozen=True, slots=True)
class _DirectoryIdentity:
    device: int
    inode: int
    mode: int


@dataclass(slots=True)
class _DirectoryChain:
    display_path: Path
    descriptors: list[int]
    links: list[tuple[int, str, int, _DirectoryIdentity]]
    root_identity: _DirectoryIdentity

    @property
    def parent_fd(self) -> int:
        return self.descriptors[-1]

    def revalidate(self) -> None:
        root_now = os.fstat(self.descriptors[0])
        if _directory_identity(root_now, label="filesystem root") != self.root_identity:
            raise MaxTextTapPatchError("filesystem root identity changed")
        for parent_fd, name, child_fd, expected in self.links:
            try:
                open_state = os.fstat(child_fd)
                path_state = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            except OSError as exc:
                raise MaxTextTapPatchError(
                    f"directory chain changed while accessing {self.display_path}"
                ) from exc
            if (
                _directory_identity(open_state, label=f"directory component {name!r}")
                != expected
                or _directory_identity(
                    path_state, label=f"directory component {name!r}"
                )
                != expected
            ):
                raise MaxTextTapPatchError(
                    f"directory chain changed while accessing {self.display_path}"
                )

    def close(self) -> None:
        for descriptor in reversed(self.descriptors):
            os.close(descriptor)
        self.descriptors.clear()


def _directory_identity(metadata: os.stat_result, *, label: str) -> _DirectoryIdentity:
    if not stat.S_ISDIR(metadata.st_mode):
        raise MaxTextTapPatchError(f"{label} is not a directory")
    return _DirectoryIdentity(metadata.st_dev, metadata.st_ino, metadata.st_mode)


def _require_absolute_path(path: Path, *, label: str) -> Path:
    if not path.is_absolute() or path.anchor != os.sep:
        raise MaxTextTapPatchError(f"{label} must be an absolute POSIX path")
    if any(component in ("", ".", "..") for component in path.parts[1:]):
        raise MaxTextTapPatchError(f"{label} contains a forbidden path component")
    return path


def _open_directory_chain(directory: Path) -> _DirectoryChain:
    directory = _require_absolute_path(directory, label="directory")
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise MaxTextTapPatchError(
            "O_NOFOLLOW and O_DIRECTORY are required for sealed path traversal"
        )
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    descriptors: list[int] = []
    links: list[tuple[int, str, int, _DirectoryIdentity]] = []
    try:
        root_fd = os.open(os.sep, flags)
        descriptors.append(root_fd)
        root_identity = _directory_identity(os.fstat(root_fd), label="filesystem root")
        for component in directory.parts[1:]:
            parent_fd = descriptors[-1]
            child_fd = os.open(component, flags, dir_fd=parent_fd)
            descriptors.append(child_fd)
            identity = _directory_identity(
                os.fstat(child_fd), label=f"directory component {component!r}"
            )
            path_state = os.stat(component, dir_fd=parent_fd, follow_symlinks=False)
            if (
                _directory_identity(
                    path_state, label=f"directory component {component!r}"
                )
                != identity
            ):
                raise MaxTextTapPatchError(
                    f"directory component changed while opening {directory}"
                )
            links.append((parent_fd, component, child_fd, identity))
    except OSError as exc:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
        raise MaxTextTapPatchError(
            f"cannot securely traverse directory without symlinks: {directory}"
        ) from exc
    except Exception:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
        raise
    chain = _DirectoryChain(directory, descriptors, links, root_identity)
    chain.revalidate()
    return chain


@contextmanager
def _opened_directory_chain(directory: Path) -> Iterator[_DirectoryChain]:
    chain = _open_directory_chain(directory)
    try:
        yield chain
    finally:
        chain.close()


@contextmanager
def _cooperative_directory_lock(
    chain: _DirectoryChain, *, exclusive: bool
) -> Iterator[None]:
    operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
    try:
        fcntl.flock(chain.parent_fd, operation | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise MaxTextTapPatchError(
            f"another cooperative patch operation holds {chain.display_path}"
        ) from exc
    except OSError as exc:
        raise MaxTextTapPatchError(
            f"cannot acquire cooperative directory lock: {chain.display_path}"
        ) from exc
    try:
        chain.revalidate()
        yield
    finally:
        fcntl.flock(chain.parent_fd, fcntl.LOCK_UN)


def _same_open_file(before: os.stat_result, after: os.stat_result) -> bool:
    return (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_nlink,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) == (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_nlink,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )


def _read_regular_at(
    parent_fd: int,
    name: str,
    display_path: Path,
    *,
    label: str,
    max_bytes: int = _GENERAL_FILE_MAX_BYTES,
    allowed_sizes: frozenset[int] | None = None,
    required_mode: int | None = None,
) -> tuple[bytes, os.stat_result]:
    if max_bytes <= 0:
        raise MaxTextTapPatchError("internal file-size bound must be positive")
    if not hasattr(os, "O_NOFOLLOW"):
        raise MaxTextTapPatchError("O_NOFOLLOW is required for sealed file reads")
    if not name or name in (".", "..") or os.sep in name:
        raise MaxTextTapPatchError("sealed file name must be one path component")
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(name, flags, dir_fd=parent_fd)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise MaxTextTapPatchError(f"{label} is not a regular file: {display_path}")
        if before.st_nlink != 1:
            raise MaxTextTapPatchError(
                f"{label} must have exactly one hard link: {display_path}"
            )
        if required_mode is not None and stat.S_IMODE(before.st_mode) != required_mode:
            raise MaxTextTapPatchError(
                f"{label} mode is {stat.S_IMODE(before.st_mode):04o}; "
                f"expected {required_mode:04o}: {display_path}"
            )
        if before.st_size > max_bytes:
            raise MaxTextTapPatchError(
                f"{label} exceeds the {max_bytes}-byte read bound: {display_path}"
            )
        if allowed_sizes is not None and before.st_size not in allowed_sizes:
            expected = ", ".join(str(size) for size in sorted(allowed_sizes))
            raise MaxTextTapPatchError(
                f"{label} has unexpected size {before.st_size}; expected {expected}"
            )

        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(
                descriptor,
                min(_READ_CHUNK_BYTES, max_bytes + 1 - total),
            )
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > max_bytes:
                raise MaxTextTapPatchError(
                    f"{label} exceeds the {max_bytes}-byte read bound: {display_path}"
                )
        after = os.fstat(descriptor)
    except OSError as exc:
        raise MaxTextTapPatchError(
            f"cannot securely read {label}: {display_path}"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)

    if not _same_open_file(before, after):
        raise MaxTextTapPatchError(
            f"{label} changed while it was being read: {display_path}"
        )
    payload = b"".join(chunks)
    if len(payload) != before.st_size:
        raise MaxTextTapPatchError(
            f"{label} size changed while it was being read: {display_path}"
        )
    try:
        path_after = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as exc:
        raise MaxTextTapPatchError(f"cannot restat {label}: {display_path}") from exc
    if not _same_open_file(after, path_after):
        raise MaxTextTapPatchError(
            f"{label} path changed while it was being read: {display_path}"
        )
    return payload, before


def _read_regular_file(
    path: Path,
    *,
    label: str,
    max_bytes: int = _GENERAL_FILE_MAX_BYTES,
    allowed_sizes: frozenset[int] | None = None,
    required_mode: int | None = None,
) -> tuple[bytes, os.stat_result]:
    path = _require_absolute_path(path, label=label)
    with _opened_directory_chain(path.parent) as chain:
        chain.revalidate()
        result = _read_regular_at(
            chain.parent_fd,
            path.name,
            path,
            label=label,
            max_bytes=max_bytes,
            allowed_sizes=allowed_sizes,
            required_mode=required_mode,
        )
        chain.revalidate()
        return result


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Hash a regular non-symlink file using SHA-256."""

    payload, _ = _read_regular_file(Path(path), label="file")
    return _sha256(payload)


def _verified_patch(path: Path) -> bytes:
    payload, _ = _read_regular_file(
        path,
        label="MaxText tap patch",
        max_bytes=PATCH_SIZE_BYTES,
        allowed_sizes=frozenset((PATCH_SIZE_BYTES,)),
        required_mode=0o644,
    )
    actual = _sha256(payload)
    if actual != PATCH_SHA256:
        raise MaxTextTapPatchError(
            f"MaxText tap patch hash drift: expected {PATCH_SHA256}, got {actual}"
        )
    return payload


def _target_path(maxtext_root: str | os.PathLike[str]) -> Path:
    root = _require_absolute_path(Path(maxtext_root), label="MaxText root")
    return root.joinpath(*TARGET_RELATIVE_PATH.parts)


def _source_state(payload: bytes) -> TapState:
    actual = _sha256(payload)
    if actual == BASE_MOE_SHA256:
        return "base"
    if actual == PATCHED_MOE_SHA256:
        return "patched"
    raise MaxTextTapPatchError(
        "MaxText moe.py source drift: expected base "
        f"{BASE_MOE_SHA256} or patched {PATCHED_MOE_SHA256}, got {actual}"
    )


def _attestation(state: TapState) -> MaxTextTapAttestationV1:
    source_sha256 = BASE_MOE_SHA256 if state == "base" else PATCHED_MOE_SHA256
    return MaxTextTapAttestationV1(
        schema="maxtext-routed-moe-prefix-tap-attestation-v1",
        expected_upstream_commit=MAXTEXT_BASE_COMMIT,
        target_relative_path=_TARGET,
        source_state=state,
        source_sha256=source_sha256,
        patch_sha256=PATCH_SHA256,
    )


def attest_maxtext_tree(
    maxtext_root: str | os.PathLike[str],
    *,
    expected_state: TapState | None = None,
    patch_path: str | os.PathLike[str] | None = None,
) -> MaxTextTapAttestationV1:
    """Attest the exact target file and patch, not the complete checkout tree."""

    if expected_state not in (None, "base", "patched"):
        raise MaxTextTapPatchError("expected_state must be 'base', 'patched', or None")
    _verified_patch(
        Path(patch_path) if patch_path is not None else default_patch_path()
    )
    target = _target_path(maxtext_root)
    with _opened_directory_chain(target.parent) as chain:
        with _cooperative_directory_lock(chain, exclusive=False):
            payload, _ = _read_regular_at(
                chain.parent_fd,
                target.name,
                target,
                label="MaxText moe.py",
                max_bytes=PATCHED_MOE_SIZE_BYTES,
                allowed_sizes=frozenset((BASE_MOE_SIZE_BYTES, PATCHED_MOE_SIZE_BYTES)),
                required_mode=0o644,
            )
            chain.revalidate()
            state = _source_state(payload)
            if expected_state is not None and state != expected_state:
                raise MaxTextTapPatchError(
                    f"MaxText moe.py is {state}, expected {expected_state}"
                )
            return _attestation(state)


def _parse_hunk_header(line: bytes) -> tuple[int, int, int, int]:
    match = _HUNK_HEADER.fullmatch(line)
    if match is None:
        raise MaxTextTapPatchError(f"malformed unified-diff hunk header: {line!r}")
    old_start = int(match.group(1))
    old_count = int(match.group(2) or b"1")
    new_start = int(match.group(3))
    new_count = int(match.group(4) or b"1")
    if old_start <= 0 or new_start <= 0 or old_count < 0 or new_count < 0:
        raise MaxTextTapPatchError("unified-diff hunk coordinates are invalid")
    return old_start, old_count, new_start, new_count


def _apply_exact_unified_diff(source: bytes, patch: bytes) -> bytes:
    """Apply the one-file pinned patch in memory with exact context checks."""

    patch_lines = patch.splitlines(keepends=True)
    expected_header = tuple(line.encode("utf-8") for line in _PATCH_HEADER)
    if tuple(patch_lines[:4]) != expected_header:
        raise MaxTextTapPatchError(
            "patch does not have the exact pinned one-file header"
        )
    source_lines = source.splitlines(keepends=True)
    output: list[bytes] = []
    source_cursor = 0
    patch_cursor = 4
    hunk_count = 0

    while patch_cursor < len(patch_lines):
        header = patch_lines[patch_cursor]
        if not header.startswith(b"@@ "):
            raise MaxTextTapPatchError("patch contains data outside a hunk")
        old_start, old_count, new_start, new_count = _parse_hunk_header(header)
        patch_cursor += 1
        hunk_count += 1

        old_index = old_start - 1
        if old_index < source_cursor or old_index > len(source_lines):
            raise MaxTextTapPatchError("patch hunks overlap or exceed the source")
        output.extend(source_lines[source_cursor:old_index])
        source_cursor = old_index
        actual_old_count = 0
        actual_new_count = 0

        while patch_cursor < len(patch_lines) and not patch_lines[
            patch_cursor
        ].startswith(b"@@ "):
            diff_line = patch_lines[patch_cursor]
            patch_cursor += 1
            # Blank context lines are normalized from canonical ``b" \n"`` to
            # ``b"\n"`` so the patch artifact is itself whitespace-clean when
            # staged. Git accepts this representation; exact base/result hashes
            # keep application fail closed.
            if diff_line == b"\n":
                marker = b" "
                content = b"\n"
            else:
                marker = diff_line[:1]
                content = diff_line[1:]
            if diff_line.startswith(b"\\"):
                raise MaxTextTapPatchError("no-newline diff markers are not supported")
            if not diff_line or marker not in (b" ", b"-", b"+"):
                raise MaxTextTapPatchError("patch contains a malformed hunk line")
            if marker in (b" ", b"-"):
                if (
                    source_cursor >= len(source_lines)
                    or source_lines[source_cursor] != content
                ):
                    raise MaxTextTapPatchError(
                        f"patch context mismatch at source line {source_cursor + 1}"
                    )
                actual_old_count += 1
                if marker == b" ":
                    output.append(content)
                    actual_new_count += 1
                source_cursor += 1
            else:
                output.append(content)
                actual_new_count += 1

        if actual_old_count != old_count or actual_new_count != new_count:
            raise MaxTextTapPatchError("patch hunk line counts do not match its header")
        if len(output) != new_start - 1 + new_count:
            raise MaxTextTapPatchError(
                "patch hunk new-source coordinates are inconsistent"
            )

    if hunk_count == 0:
        raise MaxTextTapPatchError("patch has no hunks")
    output.extend(source_lines[source_cursor:])
    return b"".join(output)


def _create_temporary_at(parent_fd: int, target_name: str) -> tuple[int, str]:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )
    for _ in range(128):
        temporary_name = f".{target_name}.tap-{secrets.token_hex(12)}.tmp"
        try:
            descriptor = os.open(
                temporary_name,
                flags,
                0o600,
                dir_fd=parent_fd,
            )
        except FileExistsError:
            continue
        return descriptor, temporary_name
    raise MaxTextTapPatchError("cannot allocate a unique patch temporary file")


def _revalidate_regular_at(
    parent_fd: int,
    name: str,
    expected: os.stat_result,
    *,
    label: str,
) -> None:
    try:
        current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as exc:
        raise MaxTextTapPatchError(f"cannot revalidate {label}") from exc
    if not _same_open_file(expected, current):
        raise MaxTextTapPatchError(f"{label} changed immediately before replace")


def apply_maxtext_tap_patch(
    maxtext_root: str | os.PathLike[str],
    *,
    patch_path: str | os.PathLike[str] | None = None,
) -> MaxTextTapAttestationV1:
    """Atomically apply the exact patch, or attest its exact target file."""

    patch = _verified_patch(
        Path(patch_path) if patch_path is not None else default_patch_path()
    )
    target = _target_path(maxtext_root)
    target_sizes = frozenset((BASE_MOE_SIZE_BYTES, PATCHED_MOE_SIZE_BYTES))
    with _opened_directory_chain(target.parent) as chain:
        with _cooperative_directory_lock(chain, exclusive=True):
            source, source_metadata = _read_regular_at(
                chain.parent_fd,
                target.name,
                target,
                label="MaxText moe.py",
                max_bytes=PATCHED_MOE_SIZE_BYTES,
                allowed_sizes=target_sizes,
                required_mode=0o644,
            )
            chain.revalidate()
            state = _source_state(source)
            if state == "patched":
                return _attestation("patched")

            patched = _apply_exact_unified_diff(source, patch)
            patched_hash = _sha256(patched)
            if patched_hash != PATCHED_MOE_SHA256:
                raise MaxTextTapPatchError(
                    "in-memory patch result is not the sealed patched source: "
                    f"expected {PATCHED_MOE_SHA256}, got {patched_hash}"
                )

            temporary_name: str | None = None
            descriptor: int | None = None
            try:
                descriptor, temporary_name = _create_temporary_at(
                    chain.parent_fd, target.name
                )
                os.fchmod(descriptor, stat.S_IMODE(source_metadata.st_mode))
                with os.fdopen(descriptor, "wb", closefd=True) as stream:
                    descriptor = None
                    stream.write(patched)
                    stream.flush()
                    os.fsync(stream.fileno())

                current, current_metadata = _read_regular_at(
                    chain.parent_fd,
                    target.name,
                    target,
                    label="MaxText moe.py",
                    max_bytes=PATCHED_MOE_SIZE_BYTES,
                    allowed_sizes=target_sizes,
                    required_mode=0o644,
                )
                current_state = _source_state(current)
                if current_state == "patched":
                    return _attestation("patched")
                if current != source:
                    raise MaxTextTapPatchError(
                        "MaxText moe.py changed during patch application"
                    )

                _revalidate_regular_at(
                    chain.parent_fd,
                    target.name,
                    current_metadata,
                    label="MaxText moe.py",
                )
                chain.revalidate()
                os.replace(
                    temporary_name,
                    target.name,
                    src_dir_fd=chain.parent_fd,
                    dst_dir_fd=chain.parent_fd,
                )
                temporary_name = None
                os.fsync(chain.parent_fd)
                chain.revalidate()
            except OSError as exc:
                raise MaxTextTapPatchError(
                    f"atomic MaxText patch application failed: {exc}"
                ) from exc
            finally:
                if descriptor is not None:
                    os.close(descriptor)
                if temporary_name is not None:
                    try:
                        os.unlink(temporary_name, dir_fd=chain.parent_fd)
                    except FileNotFoundError:
                        pass

            final, _ = _read_regular_at(
                chain.parent_fd,
                target.name,
                target,
                label="MaxText moe.py",
                max_bytes=PATCHED_MOE_SIZE_BYTES,
                allowed_sizes=target_sizes,
                required_mode=0o644,
            )
            chain.revalidate()
            if _source_state(final) != "patched":
                raise MaxTextTapPatchError(
                    "atomic replacement did not produce patched source"
                )
            return _attestation("patched")
