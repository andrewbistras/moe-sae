from __future__ import annotations

import copy
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import pytest

import moe_sae.r1.runtime_bundle as runtime_bundle_module
from moe_sae.r1.checkpoint_conversion import (
    CheckpointConversionError,
    attest_r1_runtime_lock,
)
from moe_sae.r1.checkpoint_inventory import MAXTEXT_REVISION
from moe_sae.r1.runtime_bundle import (
    CONTEXT_MANIFEST_SCHEMA,
    RuntimeBundleError,
    build_runtime_archive,
    validate_runtime_bundle,
)


REPOSITORY = Path(__file__).resolve().parents[2]


def _copy_bundle(tmp_path: Path) -> Path:
    root = tmp_path / "repository"
    shutil.copytree(REPOSITORY / "deploy/r1", root / "deploy/r1")
    destination = root / "src/moe_sae/r1"
    destination.mkdir(parents=True)
    for name in ("runtime_entrypoint.py", "runtime_bundle.py"):
        shutil.copy2(REPOSITORY / "src/moe_sae/r1" / name, destination / name)
    return root


def _rehash_bundle_file(root: Path, relative: str) -> None:
    bundle = root / "deploy/r1/bundle.toml"
    path = root / relative
    text = bundle.read_text(encoding="utf-8")
    marker = f'path = "{relative}"\n'
    start = text.index(marker) + len(marker)
    digest_start = text.index('sha256 = "', start) + len('sha256 = "')
    digest_stop = text.index('"', digest_start)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    bundle.write_text(
        text[:digest_start] + digest + text[digest_stop:], encoding="utf-8"
    )


def test_real_isolated_bundle_and_conversion_lock_validate_offline() -> None:
    identity = validate_runtime_bundle(REPOSITORY)

    assert identity.production_qualified is False
    assert identity.file_count == 11
    assert identity.lock_identity == attest_r1_runtime_lock(
        REPOSITORY / "deploy/r1/runtime"
    )
    assert identity.lock_identity.maxtext_revision == MAXTEXT_REVISION

    with pytest.raises(CheckpointConversionError, match="audited R1 revision"):
        attest_r1_runtime_lock(REPOSITORY)


def test_bundle_rejects_false_qualification_and_unlisted_files(tmp_path: Path) -> None:
    root = _copy_bundle(tmp_path)
    bundle = root / "deploy/r1/bundle.toml"
    bundle.write_text(
        bundle.read_text(encoding="utf-8").replace(
            'qualification_state = "UNBUILT"',
            'qualification_state = "QUALIFIED"',
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeBundleError, match="qualification_state"):
        validate_runtime_bundle(root)

    root = _copy_bundle(tmp_path / "extra")
    (root / "deploy/r1/runtime/ambient.env").write_text("VALUE=1\n", encoding="utf-8")
    with pytest.raises(RuntimeBundleError, match="extra, missing, or generated"):
        validate_runtime_bundle(root)


def test_bundle_rejects_hash_mode_and_maxtext_drift(tmp_path: Path) -> None:
    root = _copy_bundle(tmp_path / "hash")
    service = root / "deploy/r1/systemd/moe-sae-r1.service"
    service.write_text(service.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(RuntimeBundleError, match="hash drifted for service-unit"):
        validate_runtime_bundle(root)

    root = _copy_bundle(tmp_path / "mode")
    provisioner = root / "deploy/r1/image/provision.sh"
    provisioner.chmod(0o644)
    with pytest.raises(RuntimeBundleError, match="mode drifted for provisioner"):
        validate_runtime_bundle(root)

    root = _copy_bundle(tmp_path / "maxtext")
    pyproject = root / "deploy/r1/runtime/pyproject.toml"
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8").replace(MAXTEXT_REVISION, "a" * 40),
        encoding="utf-8",
    )
    _rehash_bundle_file(root, "deploy/r1/runtime/pyproject.toml")
    with pytest.raises(CheckpointConversionError, match="audited R1 revision"):
        validate_runtime_bundle(root)


def test_semantic_checks_survive_attacker_rehash(tmp_path: Path) -> None:
    root = _copy_bundle(tmp_path / "service")
    service = root / "deploy/r1/systemd/moe-sae-r1.service"
    service.write_text(
        service.read_text(encoding="utf-8")
        + "\nEnvironmentFile=/var/lib/moe-sae-r1/operator.env\n",
        encoding="utf-8",
    )
    _rehash_bundle_file(root, "deploy/r1/systemd/moe-sae-r1.service")
    with pytest.raises(RuntimeBundleError, match="mutable/unsafe"):
        validate_runtime_bundle(root)

    root = _copy_bundle(tmp_path / "entrypoint")
    entrypoint = root / "src/moe_sae/r1/runtime_entrypoint.py"
    entrypoint.write_text(
        "import jax\n" + entrypoint.read_text(encoding="utf-8"), encoding="utf-8"
    )
    _rehash_bundle_file(root, "src/moe_sae/r1/runtime_entrypoint.py")
    with pytest.raises(RuntimeBundleError, match="forbidden import"):
        validate_runtime_bundle(root)

    root = _copy_bundle(tmp_path / "nested-entrypoint-import")
    entrypoint = root / "src/moe_sae/r1/runtime_entrypoint.py"
    entrypoint.write_text(
        entrypoint.read_text(encoding="utf-8")
        + "\ndef delayed_runtime_import():\n    import subprocess\n",
        encoding="utf-8",
    )
    _rehash_bundle_file(root, "src/moe_sae/r1/runtime_entrypoint.py")
    with pytest.raises(RuntimeBundleError, match="forbidden import"):
        validate_runtime_bundle(root)

    root = _copy_bundle(tmp_path / "context")
    context = root / "deploy/r1/context-files.json"
    value = json.loads(context.read_text(encoding="utf-8"))
    value["files"].append(
        {"source": "pyproject.toml", "archive": "root-project.toml", "mode": "0644"}
    )
    context.write_text(json.dumps(value), encoding="utf-8")
    _rehash_bundle_file(root, "deploy/r1/context-files.json")
    with pytest.raises(RuntimeBundleError, match="exact R1 allowlist"):
        validate_runtime_bundle(root)


def test_runtime_lock_has_hashed_registry_artifacts_and_no_torch() -> None:
    import tomllib

    lock = tomllib.loads(
        (REPOSITORY / "deploy/r1/runtime/uv.lock").read_text(encoding="utf-8")
    )
    packages = lock["package"]
    assert not any(package["name"] == "torch" for package in packages)
    for package in packages:
        if "registry" not in package["source"]:
            continue
        artifacts = ([package["sdist"]] if "sdist" in package else []) + package.get(
            "wheels", []
        )
        assert artifacts
        assert all(artifact["hash"].startswith("sha256:") for artifact in artifacts)


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repository), *arguments], text=True
    ).strip()


def _context_repository(tmp_path: Path) -> tuple[Path, str]:
    repository = tmp_path / "source"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    _git(repository, "config", "user.name", "Runtime Test")
    _git(repository, "config", "user.email", "runtime@example.invalid")
    (repository / "payload").mkdir()
    (repository / "deploy/r1").mkdir(parents=True)
    (repository / "payload/data.txt").write_text("immutable\n", encoding="utf-8")
    executable = repository / "payload/run"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o755)
    specification = {
        "schema": "moe-sae.r1.runtime-context.v1",
        "files": [
            {"source": "payload/data.txt", "archive": "data.txt", "mode": "0644"},
            {"source": "payload/run", "archive": "bin/run", "mode": "0755"},
        ],
    }
    (repository / "deploy/r1/context-files.json").write_text(
        json.dumps(specification), encoding="utf-8"
    )
    _git(repository, "add", ".")
    _git(repository, "commit", "-qm", "fixture")
    return repository, _git(repository, "rev-parse", "HEAD")


def test_context_archive_is_commit_only_and_byte_deterministic(tmp_path: Path) -> None:
    repository, commit = _context_repository(tmp_path)
    first, first_identity = (
        runtime_bundle_module._build_runtime_archive_from_committed_spec(
            repository, commit=commit
        )
    )
    (repository / "payload/data.txt").write_text("dirty worktree\n", encoding="utf-8")
    old_umask = os.umask(0o022)
    try:
        second, second_identity = (
            runtime_bundle_module._build_runtime_archive_from_committed_spec(
                repository, commit=commit
            )
        )
    finally:
        os.umask(old_umask)

    assert first == second
    assert first_identity == second_identity
    with tarfile.open(fileobj=io.BytesIO(first), mode="r:") as archive:
        members = archive.getmembers()
        assert [member.name for member in members] == [
            "CONTEXT-MANIFEST.json",
            "bin/run",
            "data.txt",
        ]
        assert all(
            member.uid == 0 and member.gid == 0 and member.mtime == 0
            for member in members
        )
        assert all(member.isfile() for member in members)
        assert archive.extractfile("data.txt").read() == b"immutable\n"  # type: ignore[union-attr]
        manifest_raw = archive.extractfile("CONTEXT-MANIFEST.json").read()  # type: ignore[union-attr]
    manifest = json.loads(manifest_raw)
    assert manifest["schema"] == CONTEXT_MANIFEST_SCHEMA
    assert manifest["production_qualified"] is False
    assert manifest["commit"] == commit


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("traversal", "safe relative path"),
        ("secret", "token-shaped"),
        ("symlink", "allowed regular blob"),
    ],
)
def test_context_archive_rejects_unsafe_committed_inputs(
    tmp_path: Path, mutation: str, message: str
) -> None:
    repository, _ = _context_repository(tmp_path)
    spec_path = repository / "deploy/r1/context-files.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if mutation == "traversal":
        spec["files"][0]["archive"] = "../escape"
        spec_path.write_text(json.dumps(spec), encoding="utf-8")
    elif mutation == "secret":
        (repository / "payload/data.txt").write_text(
            "-----BEGIN PRIVATE KEY-----\n", encoding="utf-8"
        )
    else:
        link = repository / "payload/link"
        link.symlink_to("data.txt")
        spec["files"][0] = {
            "source": "payload/link",
            "archive": "link",
            "mode": "0644",
        }
        spec_path.write_text(json.dumps(spec), encoding="utf-8")
    _git(repository, "add", ".")
    _git(repository, "commit", "-qm", mutation)
    commit = _git(repository, "rev-parse", "HEAD")
    with pytest.raises(RuntimeBundleError, match=message):
        runtime_bundle_module._build_runtime_archive_from_committed_spec(
            repository, commit=commit
        )


def test_real_context_spec_excludes_gemma_and_shared_project() -> None:
    spec = json.loads(
        (REPOSITORY / "deploy/r1/context-files.json").read_text(encoding="utf-8")
    )
    sources = {item["source"] for item in spec["files"]}
    assert "pyproject.toml" not in sources
    assert "uv.lock" not in sources
    assert not any("gemma" in source.lower() for source in sources)
    assert not any(source.startswith("configs/") for source in sources)
    assert not any(source.startswith("scripts/tpu/") for source in sources)
    assert not any(source.startswith("tests/") for source in sources)


def _production_context_repository(tmp_path: Path) -> tuple[Path, str]:
    repository = tmp_path / "production-source"
    shutil.copytree(REPOSITORY / "deploy/r1", repository / "deploy/r1")
    specification = json.loads(
        (repository / "deploy/r1/context-files.json").read_text(encoding="utf-8")
    )
    for item in specification["files"]:
        source = item["source"]
        destination = repository / source
        if destination.exists():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPOSITORY / source, destination)
    # The bundle validator itself is authority but is not a runtime payload.
    validator = repository / "src/moe_sae/r1/runtime_bundle.py"
    if not validator.exists():
        validator.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPOSITORY / "src/moe_sae/r1/runtime_bundle.py", validator)
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    _git(repository, "config", "user.name", "Runtime Test")
    _git(repository, "config", "user.email", "runtime@example.invalid")
    _git(repository, "add", ".")
    _git(repository, "commit", "-qm", "production fixture")
    return repository, _git(repository, "rev-parse", "HEAD")


def test_public_context_builder_validates_exact_head_and_ignores_unrelated_dirty_file(
    tmp_path: Path,
) -> None:
    repository, commit = _production_context_repository(tmp_path)
    first, identity = build_runtime_archive(repository, commit=commit)
    (repository / "unrelated-notes.txt").write_text("dirty\n", encoding="utf-8")
    second, repeated = build_runtime_archive(repository, commit=commit)

    assert first == second
    assert identity == repeated


def test_public_context_builder_rejects_valid_worktree_over_malicious_head(
    tmp_path: Path,
) -> None:
    repository, _ = _production_context_repository(tmp_path)
    entrypoint = repository / "src/moe_sae/r1/runtime_entrypoint.py"
    bundle = repository / "deploy/r1/bundle.toml"
    clean_entrypoint = entrypoint.read_bytes()
    clean_bundle = bundle.read_bytes()
    entrypoint.write_text(
        "import jax\n" + entrypoint.read_text(encoding="utf-8"), encoding="utf-8"
    )
    _rehash_bundle_file(repository, "src/moe_sae/r1/runtime_entrypoint.py")
    _git(repository, "add", ".")
    _git(repository, "commit", "-qm", "malicious committed entrypoint")
    malicious_commit = _git(repository, "rev-parse", "HEAD")
    entrypoint.write_bytes(clean_entrypoint)
    bundle.write_bytes(clean_bundle)

    with pytest.raises(RuntimeBundleError, match="differs from target commit"):
        build_runtime_archive(repository, commit=malicious_commit)


def _bake_archive_verifier() -> str:
    provisioner = (REPOSITORY / "deploy/r1/image/provision.sh").read_text(
        encoding="utf-8"
    )
    start_marker = "/usr/bin/python3 -I - /tmp/moe-sae-r1-runtime.tar <<'PY'\n"
    start = provisioner.index(start_marker) + len(start_marker)
    stop = provisioner.index("\nPY\ntar \\", start)
    return provisioner[start:stop]


def _mutate_archive_member(archive_raw: bytes, mutation: str) -> bytes:
    entries: list[tuple[tarfile.TarInfo, bytes | None]] = []
    with tarfile.open(fileobj=io.BytesIO(archive_raw), mode="r:") as source:
        for member in source.getmembers():
            stream = source.extractfile(member)
            entries.append(
                (copy.copy(member), None if stream is None else stream.read())
            )
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w", format=tarfile.USTAR_FORMAT) as target:
        for member, raw in entries:
            if member.name == "uv.lock":
                if mutation == "traversal":
                    member.name = "../uv.lock"
                elif mutation == "absolute":
                    member.name = "/uv.lock"
                elif mutation == "unexpected":
                    member.name = "unexpected.lock"
                elif mutation == "symlink":
                    member.type = tarfile.SYMTYPE
                    member.linkname = "../../etc/shadow"
                    member.size = 0
                    raw = None
                elif mutation == "device":
                    member.type = tarfile.BLKTYPE
                    member.devmajor = 1
                    member.devminor = 1
                    member.size = 0
                    raw = None
                elif mutation == "mode":
                    member.mode = 0o777
                else:  # pragma: no cover - private test helper contract
                    raise AssertionError(f"unknown mutation {mutation}")
            target.addfile(member, None if raw is None else io.BytesIO(raw))
    return output.getvalue()


@pytest.mark.parametrize(
    "mutation",
    ["traversal", "absolute", "unexpected", "symlink", "device", "mode"],
)
def test_bake_verifier_accepts_real_context_and_rejects_unsafe_tar_members(
    tmp_path: Path, mutation: str
) -> None:
    repository, commit = _production_context_repository(tmp_path)
    archive_raw, _ = build_runtime_archive(repository, commit=commit)
    verifier = _bake_archive_verifier()
    archive_path = tmp_path / "runtime.tar"
    archive_path.write_bytes(archive_raw)
    accepted = subprocess.run(
        [sys.executable, "-I", "-", str(archive_path)],
        input=verifier,
        text=True,
        capture_output=True,
        check=False,
    )
    assert accepted.returncode == 0, accepted.stderr

    archive_path.write_bytes(_mutate_archive_member(archive_raw, mutation))
    rejected = subprocess.run(
        [sys.executable, "-I", "-", str(archive_path)],
        input=verifier,
        text=True,
        capture_output=True,
        check=False,
    )
    assert rejected.returncode != 0
    assert "runtime archive" in rejected.stderr
