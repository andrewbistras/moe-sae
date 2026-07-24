#!/usr/bin/env python3
"""Fail-closed verifier for the Experiment-2 MaxText patch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
MANIFEST_PATH = HERE / "selected_outputs_manifest.json"


class VerificationError(RuntimeError):
    """The patch or its pinned MaxText source does not match its manifest."""


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "maxtext-selected-outputs-patch-v1":
        raise VerificationError("unsupported patch manifest schema")
    for name, length in (("base_commit", 40), ("fork_commit", 40)):
        value = payload.get(name)
        if (
            not isinstance(value, str)
            or len(value) != length
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise VerificationError(f"manifest {name} is not a lowercase Git SHA")
    if payload["fork_commit"] == payload["base_commit"]:
        raise VerificationError("fork commit cannot equal the unpatched base")
    if not isinstance(payload.get("fork_repository"), str):
        raise VerificationError("manifest fork_repository is missing")
    if not isinstance(payload.get("fork_branch"), str):
        raise VerificationError("manifest fork_branch is missing")
    if payload.get("performance_optimizations") != [
        "linear_sender_to_expert_ragged_permutation",
        "linear_inverse_permutation",
    ]:
        raise VerificationError("manifest performance optimizations changed")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_git(root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        stderr = getattr(exc, "stderr", "")
        detail = stderr.strip() if isinstance(stderr, str) else ""
        raise VerificationError(
            f"git {' '.join(args)} failed: {detail or exc}"
        ) from exc
    return result.stdout.strip()


def verify_patch_content(manifest: dict[str, Any]) -> None:
    patch_path = HERE / manifest["patch"]
    if sha256_file(patch_path) != manifest["patch_sha256"]:
        raise VerificationError("patch SHA-256 does not match the manifest")
    patch_text = patch_path.read_text(encoding="utf-8")
    expected_files = set(manifest["files"])
    touched_files = {
        line.removeprefix("+++ b/")
        for line in patch_text.splitlines()
        if line.startswith("+++ b/")
    }
    if touched_files != expected_files:
        raise VerificationError(
            f"patch touches {sorted(touched_files)}, expected {sorted(expected_files)}"
        )

    required_anchors = (
        "return_selected_outputs: bool = False",
        "return output, routing.lb_loss, routing.bias_updates",
        "return output, routing.lb_loss, routing.bias_updates, selected_outputs",
        "RoutedMoE._invert_permutation(sorted_selected_experts)",
        "def _linear_local_sort_indices(",
        "def _invert_permutation(",
        "RoutedMoE._linear_local_sort_indices(",
        ".at[indices].set(positions)",
        "out_specs=sparse_out_specs",
        '(batch_logical_axis, "activation_norm_length", None, "activation_embed")',
        "decoder_block is not Gemma 4",
        "fused-vLLM attention",
        "dense MoE",
        "ring of experts",
        "potentially truncating ragged buffer",
        "tensor parallelism != 1",
        "hash routing",
        "serving quantization",
    )
    missing = [anchor for anchor in required_anchors if anchor not in patch_text]
    if missing:
        raise VerificationError(f"patch is missing required anchors: {missing}")

    for relative_path, identity in manifest["files"].items():
        index_line = (
            f"index {identity['base_git_blob_sha1']}.."
            f"{identity['patched_git_blob_sha1']} 100644"
        )
        if index_line not in patch_text:
            raise VerificationError(f"patch is not blob-anchored for {relative_path}")


def verify_checkout(root: Path, manifest: dict[str, Any]) -> None:
    root = root.expanduser().resolve()
    if run_git(root, "rev-parse", "HEAD") != manifest["base_commit"]:
        raise VerificationError("MaxText checkout is not at the pinned base commit")
    if run_git(root, "status", "--porcelain"):
        raise VerificationError(
            "MaxText checkout must be clean before applying the patch"
        )

    for relative_path, identity in manifest["files"].items():
        source_path = root / relative_path
        if not source_path.is_file():
            raise VerificationError(f"missing pinned source file: {relative_path}")
        if (
            run_git(root, "hash-object", relative_path)
            != identity["base_git_blob_sha1"]
        ):
            raise VerificationError(f"Git blob mismatch for {relative_path}")
        if sha256_file(source_path) != identity["base_sha256"]:
            raise VerificationError(f"SHA-256 mismatch for {relative_path}")

    patch_path = HERE / manifest["patch"]
    try:
        subprocess.run(
            ["git", "-C", str(root), "apply", "--check", str(patch_path)],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        stderr = getattr(exc, "stderr", "")
        detail = stderr.strip() if isinstance(stderr, str) else ""
        raise VerificationError(
            f"patch does not apply cleanly: {detail or exc}"
        ) from exc

    # Prove the post-image without mutating the supplied checkout. A temporary
    # Git index applies the patch to HEAD's index entries only; the worktree and
    # the real index remain untouched.
    with tempfile.TemporaryDirectory(prefix="maxtext-selected-outputs-") as tmp:
        environment = dict(os.environ)
        environment["GIT_INDEX_FILE"] = str(Path(tmp) / "index")

        def indexed_git(
            *args: str, text: bool = True
        ) -> subprocess.CompletedProcess[Any]:
            try:
                return subprocess.run(
                    ["git", "-C", str(root), *args],
                    check=True,
                    capture_output=True,
                    text=text,
                    timeout=30,
                    env=environment,
                )
            except (
                OSError,
                subprocess.CalledProcessError,
                subprocess.TimeoutExpired,
            ) as exc:
                stderr = getattr(exc, "stderr", "")
                if isinstance(stderr, bytes):
                    detail = stderr.decode("utf-8", errors="replace").strip()
                elif isinstance(stderr, str):
                    detail = stderr.strip()
                else:
                    detail = ""
                raise VerificationError(
                    f"temporary-index git {' '.join(args)} failed: {detail or exc}"
                ) from exc

        indexed_git("read-tree", "HEAD")
        indexed_git("apply", "--cached", "--check", str(patch_path))
        indexed_git("apply", "--cached", str(patch_path))
        for relative_path, identity in manifest["files"].items():
            staged = indexed_git(
                "ls-files",
                "--stage",
                "--",
                relative_path,
            ).stdout.strip()
            fields = staged.split()
            if len(fields) < 2 or fields[1] != identity["patched_git_blob_sha1"]:
                actual = fields[1] if len(fields) >= 2 else "<missing>"
                raise VerificationError(
                    f"patched Git blob mismatch for {relative_path}: {actual}"
                )
            payload = indexed_git(
                "show",
                f":{relative_path}",
                text=False,
            ).stdout
            if hashlib.sha256(payload).hexdigest() != identity["patched_sha256"]:
                raise VerificationError(f"patched SHA-256 mismatch for {relative_path}")
            if relative_path.endswith(".py"):
                try:
                    compile(
                        payload.decode("utf-8"),
                        relative_path,
                        "exec",
                    )
                except (SyntaxError, UnicodeDecodeError) as exc:
                    raise VerificationError(
                        f"patched Python source does not compile: {relative_path}"
                    ) from exc


def verify_fork_checkout(root: Path, manifest: dict[str, Any]) -> None:
    """Verify the published commit, its parent, complete delta, and post-image."""

    root = root.expanduser().resolve()
    if run_git(root, "rev-parse", "HEAD") != manifest["fork_commit"]:
        raise VerificationError("MaxText fork checkout is not at the pinned commit")
    if run_git(root, "status", "--porcelain"):
        raise VerificationError("MaxText fork checkout must be clean")
    ancestry = run_git(root, "rev-list", "--parents", "-n", "1", "HEAD").split()
    if ancestry != [manifest["fork_commit"], manifest["base_commit"]]:
        raise VerificationError(
            "MaxText fork commit must have exactly the pinned base as its parent"
        )
    changed = set(
        filter(
            None,
            run_git(
                root,
                "diff-tree",
                "--no-commit-id",
                "--name-only",
                "-r",
                "HEAD",
            ).splitlines(),
        )
    )
    expected = set(manifest["files"])
    if changed != expected:
        raise VerificationError(
            f"fork commit changes {sorted(changed)}, expected {sorted(expected)}"
        )
    for relative_path, identity in manifest["files"].items():
        source_path = root / relative_path
        if not source_path.is_file():
            raise VerificationError(f"fork is missing patched file: {relative_path}")
        if (
            run_git(root, "hash-object", relative_path)
            != identity["patched_git_blob_sha1"]
        ):
            raise VerificationError(f"fork Git blob mismatch for {relative_path}")
        if sha256_file(source_path) != identity["patched_sha256"]:
            raise VerificationError(f"fork SHA-256 mismatch for {relative_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--maxtext-root",
        type=Path,
        help="Clean checkout of the pinned MaxText base commit to verify",
    )
    parser.add_argument(
        "--maxtext-fork-root",
        type=Path,
        help="Clean checkout of the pinned patched MaxText fork commit to verify",
    )
    args = parser.parse_args(argv)
    try:
        manifest = load_manifest()
        verify_patch_content(manifest)
        if args.maxtext_root is not None:
            verify_checkout(args.maxtext_root, manifest)
        if args.maxtext_fork_root is not None:
            verify_fork_checkout(args.maxtext_fork_root, manifest)
    except VerificationError as exc:
        print(f"verification failed: {exc}", file=sys.stderr)
        return 1
    print("MaxText selected-output patch verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
