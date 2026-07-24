from __future__ import annotations

import base64
import csv
import hashlib
import importlib.util
import io
import json
import os
import stat
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/r1/repack_legacy_runtime.py"
BOOTSTRAP = ROOT / "scripts/r1/bootstrap_legacy_v6e_worker.sh"
SPEC = importlib.util.spec_from_file_location("r1_runtime_repacker", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
repacker = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = repacker
SPEC.loader.exec_module(repacker)


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
        + b"\n"
    )


def _context_archive(
    path: Path,
    *,
    commit: str = "a" * 40,
    worker: bytes = b"VALUE = 1\n",
    pyproject: bytes = b"[project]\nname='moe-sae-r1-worker'\n",
    provision: bytes = b"#!/bin/sh\nexit 0\n",
    measure: bytes = b"print('base')\n",
    extra_payloads: dict[str, tuple[str, int, bytes]] | None = None,
    omitted: frozenset[str] = frozenset(),
) -> str:
    payloads = {
        "pyproject.toml": (
            "deploy/r1/runtime/pyproject.toml",
            0o644,
            pyproject,
        ),
        "src/moe_sae/r1/worker.py": (
            "src/moe_sae/r1/worker.py",
            0o644,
            worker,
        ),
        "image/provision.sh": (
            "deploy/r1/image/provision.sh",
            0o755,
            provision,
        ),
        "image/measure_image.py": (
            "deploy/r1/image/measure_image.py",
            0o755,
            measure,
        ),
    }
    if extra_payloads is not None:
        payloads.update(extra_payloads)
    payloads = {name: value for name, value in payloads.items() if name not in omitted}
    records = [
        {
            "source": source,
            "archive": archive,
            "mode": f"{mode:04o}",
            "size": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "git_blob": hashlib.sha1(
                f"blob {len(raw)}\0".encode() + raw,
                usedforsecurity=False,
            ).hexdigest(),
        }
        for archive, (source, mode, raw) in sorted(payloads.items())
    ]
    manifest = _canonical(
        {
            "schema": repacker.CONTEXT_MANIFEST_SCHEMA,
            "commit": commit,
            "files": records,
            "production_qualified": False,
        }
    )
    members = {
        "CONTEXT-MANIFEST.json": (0o644, manifest),
        **{name: (mode, raw) for name, (_, mode, raw) in payloads.items()},
    }
    with tarfile.open(path, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for name, (mode, raw) in sorted(members.items()):
            info = tarfile.TarInfo(name)
            info.mode = mode
            info.size = len(raw)
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            archive.addfile(info, io.BytesIO(raw))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _wheel(
    path: Path,
    *,
    foreign: bool = False,
    absolute: bool = False,
) -> None:
    dist_info = "moe_sae_r1_worker-0.1.0.dist-info"
    payloads = {
        "moe_sae/__init__.py": b"",
        "moe_sae/r1/worker.py": b"VALUE = 2\n",
        f"{dist_info}/METADATA": (
            b"Metadata-Version: 2.4\nName: moe-sae-r1-worker\nVersion: 0.1.0\n\n"
        ),
        f"{dist_info}/WHEEL": (
            b"Wheel-Version: 1.0\n"
            b"Generator: test\n"
            b"Root-Is-Purelib: true\n"
            b"Tag: py3-none-any\n\n"
        ),
    }
    if foreign:
        payloads["maxtext/overwrite.py"] = b"bad\n"
    if absolute:
        payloads["/etc/passwd"] = b"bad\n"
    record_name = f"{dist_info}/RECORD"
    rows: list[list[str]] = []
    for name, raw in sorted(payloads.items()):
        encoded = base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).rstrip(b"=")
        rows.append([name, f"sha256={encoded.decode()}", str(len(raw))])
    rows.append([record_name, "", ""])
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerows(rows)
    payloads[record_name] = output.getvalue().encode()
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED) as wheel:
        for name, raw in sorted(payloads.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            wheel.writestr(info, raw)


def test_script_is_executable_and_offline_by_construction() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert SCRIPT.stat().st_mode & 0o111
    assert source.startswith("#!/usr/bin/env python3\n")
    assert '"--offline"' in source
    assert '"--no-deps"' in source
    assert "RENAME_EXCHANGE" in source
    assert "refusing non-atomic fallback" in source
    assert "gcloud" not in source
    assert "google.cloud" not in source
    assert "storage.googleapis.com" not in source


def test_context_archive_is_exact_and_worker_only_delta_is_allowed(
    tmp_path: Path,
) -> None:
    base_archive = tmp_path / "base.tar"
    target_archive = tmp_path / "target.tar"
    base_sha = _context_archive(base_archive)
    target_sha = _context_archive(
        target_archive,
        commit="b" * 40,
        worker=b"VALUE = 2\n",
    )
    base = tmp_path / "base"
    target = tmp_path / "target"
    base_identity = repacker.validate_and_extract_context(
        base_archive,
        expected_sha256=base_sha,
        destination=base,
    )
    target_identity = repacker.validate_and_extract_context(
        target_archive,
        expected_sha256=target_sha,
        destination=target,
    )

    repacker.require_worker_only_context_delta(base, target)
    assert base_identity.commit == "a" * 40
    assert target_identity.commit == "b" * 40
    assert target_identity.archive_sha256 == target_sha
    assert repacker.validate_context_directory(target).manifest_sha256 == (
        target_identity.manifest_sha256
    )


def test_context_archive_uses_one_nofollow_descriptor_and_rejects_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "context.tar"
    digest = _context_archive(archive)
    linked = tmp_path / "linked.tar"
    linked.symlink_to(archive)
    with pytest.raises(repacker.RepackError, match="without following links"):
        repacker.validate_and_extract_context(
            linked,
            expected_sha256=digest,
            destination=tmp_path / "linked-out",
        )

    real_fstat = os.fstat
    calls = 0

    def raced_fstat(descriptor: int) -> object:
        nonlocal calls
        calls += 1
        observed = real_fstat(descriptor)
        if calls == 2:
            values = {
                name: getattr(observed, name)
                for name in (
                    "st_mode",
                    "st_size",
                    "st_dev",
                    "st_ino",
                    "st_mtime_ns",
                    "st_ctime_ns",
                )
            }
            values["st_ctime_ns"] += 1
            return SimpleNamespace(**values)
        return observed

    monkeypatch.setattr(repacker.os, "fstat", raced_fstat)
    with pytest.raises(repacker.RepackError, match="changed while it was read"):
        repacker.validate_and_extract_context(
            archive,
            expected_sha256=digest,
            destination=tmp_path / "raced-out",
        )


def test_context_delta_rejects_dependency_or_unowned_file_change(
    tmp_path: Path,
) -> None:
    base_archive = tmp_path / "base.tar"
    target_archive = tmp_path / "target.tar"
    base_sha = _context_archive(base_archive)
    target_sha = _context_archive(
        target_archive,
        commit="b" * 40,
        pyproject=b"[project]\nname='attacker'\n",
    )
    base = tmp_path / "base"
    target = tmp_path / "target"
    repacker.validate_and_extract_context(
        base_archive, expected_sha256=base_sha, destination=base
    )
    repacker.validate_and_extract_context(
        target_archive, expected_sha256=target_sha, destination=target
    )

    with pytest.raises(repacker.RepackError, match="non-worker"):
        repacker.require_worker_only_context_delta(base, target)

    (base / "ambient.py").write_text("ambient = True\n", encoding="utf-8")
    with pytest.raises(repacker.RepackError, match="unowned"):
        repacker.validate_context_directory(base)


def test_context_delta_allows_additive_worker_module_only(
    tmp_path: Path,
) -> None:
    base_archive = tmp_path / "base.tar"
    target_archive = tmp_path / "target.tar"
    base_sha = _context_archive(base_archive)
    target_sha = _context_archive(
        target_archive,
        commit="b" * 40,
        extra_payloads={
            "src/moe_sae/r1/legacy_slice_runtime.py": (
                "src/moe_sae/r1/legacy_slice_runtime.py",
                0o644,
                b"READY = True\n",
            )
        },
    )
    base = tmp_path / "base"
    target = tmp_path / "target"
    repacker.validate_and_extract_context(
        base_archive, expected_sha256=base_sha, destination=base
    )
    repacker.validate_and_extract_context(
        target_archive, expected_sha256=target_sha, destination=target
    )

    repacker.require_worker_only_context_delta(base, target)


def test_context_delta_binds_two_exact_inert_image_validator_changes(
    tmp_path: Path,
) -> None:
    base_archive = tmp_path / "base.tar"
    target_archive = tmp_path / "target.tar"
    base_sha = _context_archive(base_archive)
    target_sha = _context_archive(
        target_archive,
        commit="b" * 40,
        provision=b"#!/bin/sh\nexit 1\n",
        measure=b"print('target')\n",
    )
    base = tmp_path / "base"
    target = tmp_path / "target"
    repacker.validate_and_extract_context(
        base_archive, expected_sha256=base_sha, destination=base
    )
    repacker.validate_and_extract_context(
        target_archive, expected_sha256=target_sha, destination=target
    )

    deltas = repacker.require_worker_only_context_delta(base, target)

    assert [item["path"] for item in deltas] == [
        "image/measure_image.py",
        "image/provision.sh",
    ]
    assert all(item["base_sha256"] != item["target_sha256"] for item in deltas)
    assert all(item["mode"] == "0755" for item in deltas)


@pytest.mark.parametrize("operation", ["add", "remove"])
def test_context_delta_rejects_non_worker_addition_or_removal(
    tmp_path: Path,
    operation: str,
) -> None:
    base_archive = tmp_path / "base.tar"
    target_archive = tmp_path / "target.tar"
    base_kwargs: dict[str, object] = {}
    target_kwargs: dict[str, object] = {"commit": "b" * 40}
    extra = {
        "systemd/new.service": (
            "deploy/r1/systemd/new.service",
            0o644,
            b"[Service]\n",
        )
    }
    if operation == "add":
        target_kwargs["extra_payloads"] = extra
    else:
        base_kwargs["extra_payloads"] = extra
    base_sha = _context_archive(base_archive, **base_kwargs)
    target_sha = _context_archive(target_archive, **target_kwargs)
    base = tmp_path / "base"
    target = tmp_path / "target"
    repacker.validate_and_extract_context(
        base_archive, expected_sha256=base_sha, destination=base
    )
    repacker.validate_and_extract_context(
        target_archive, expected_sha256=target_sha, destination=target
    )

    with pytest.raises(repacker.RepackError, match="added or removed"):
        repacker.require_worker_only_context_delta(base, target)


def test_context_tar_rejects_symlink_and_duplicate_json_key(
    tmp_path: Path,
) -> None:
    malicious = tmp_path / "malicious.tar"
    manifest = (
        b'{"commit":"'
        + b"a" * 40
        + b'","commit":"'
        + b"a" * 40
        + b'","files":[],"production_qualified":false,'
        + b'"schema":"moe-sae.r1.runtime-context-manifest.v1"}\n'
    )
    with tarfile.open(malicious, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        info = tarfile.TarInfo("CONTEXT-MANIFEST.json")
        info.mode = 0o644
        info.size = len(manifest)
        archive.addfile(info, io.BytesIO(manifest))
        link = tarfile.TarInfo("src/moe_sae/escape.py")
        link.type = tarfile.SYMTYPE
        link.linkname = "../../etc/passwd"
        link.mode = 0o644
        archive.addfile(link)
    digest = hashlib.sha256(malicious.read_bytes()).hexdigest()

    with pytest.raises(repacker.RepackError, match="unsafe context tar metadata"):
        repacker.validate_and_extract_context(
            malicious,
            expected_sha256=digest,
            destination=tmp_path / "out",
        )


def test_context_tar_rejects_absolute_member_before_any_extraction(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "absolute.tar"
    with tarfile.open(archive_path, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for name, raw in (
            ("/etc/passwd", b"bad\n"),
            ("CONTEXT-MANIFEST.json", b"{}\n"),
        ):
            info = tarfile.TarInfo(name)
            info.mode = 0o644
            info.size = len(raw)
            archive.addfile(info, io.BytesIO(raw))
    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    destination = tmp_path / "out"
    with pytest.raises(repacker.RepackError, match="normalized relative path"):
        repacker.validate_and_extract_context(
            archive_path,
            expected_sha256=digest,
            destination=destination,
        )
    assert destination.is_dir()
    assert list(destination.iterdir()) == []


def test_context_manifest_recomputes_git_blob_identity(tmp_path: Path) -> None:
    original = tmp_path / "original.tar"
    _context_archive(original)
    payloads: dict[str, tuple[int, bytes]] = {}
    with tarfile.open(original, mode="r:") as archive:
        for member in archive.getmembers():
            stream = archive.extractfile(member)
            assert stream is not None
            payloads[member.name] = (member.mode, stream.read())
    manifest = json.loads(payloads["CONTEXT-MANIFEST.json"][1])
    manifest["files"][0]["git_blob"] = "0" * 40
    payloads["CONTEXT-MANIFEST.json"] = (0o644, _canonical(manifest))
    tampered = tmp_path / "tampered.tar"
    with tarfile.open(tampered, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for name, (mode, raw) in sorted(payloads.items()):
            info = tarfile.TarInfo(name)
            info.mode = mode
            info.size = len(raw)
            archive.addfile(info, io.BytesIO(raw))
    digest = hashlib.sha256(tampered.read_bytes()).hexdigest()

    with pytest.raises(repacker.RepackError, match="does not match payload"):
        repacker.validate_and_extract_context(
            tampered,
            expected_sha256=digest,
            destination=tmp_path / "out",
        )


def test_pure_worker_wheel_and_offline_double_build(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture.whl"
    _wheel(fixture)
    calls: list[tuple[tuple[str, ...], dict[str, str] | None]] = []

    def runner(argv: object, environment: object) -> subprocess.CompletedProcess[str]:
        command = tuple(argv)  # type: ignore[arg-type]
        env = dict(environment)  # type: ignore[arg-type]
        calls.append((command, env))
        output = Path(command[command.index("--out-dir") + 1])
        (output / "moe_sae_r1_worker-0.1.0-py3-none-any.whl").write_bytes(
            fixture.read_bytes()
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    work = tmp_path / "work"
    work.mkdir()
    identity = repacker.build_worker_wheel(
        tmp_path / "source",
        runtime_python=Path("/sealed/python"),
        uv=Path("/pinned/uv"),
        work_root=work,
        runner=runner,
    )

    assert identity.sha256 == hashlib.sha256(fixture.read_bytes()).hexdigest()
    assert len(calls) == 2
    for command, environment in calls:
        assert command[0:2] == ("/pinned/uv", "build")
        assert "--offline" in command
        assert "--no-build-isolation" in command
        assert "--no-python-downloads" in command
        assert environment["UV_OFFLINE"] == "1"
        assert environment["SOURCE_DATE_EPOCH"] == str(repacker.SOURCE_DATE_EPOCH)


def test_worker_wheel_exactly_covers_validated_context_source(
    tmp_path: Path,
) -> None:
    context_archive = tmp_path / "context.tar"
    context_sha = _context_archive(
        context_archive,
        worker=b"VALUE = 2\n",
        extra_payloads={
            "src/moe_sae/__init__.py": (
                "src/moe_sae/__init__.py",
                0o644,
                b"",
            )
        },
    )
    context = repacker.validate_and_extract_context(
        context_archive,
        expected_sha256=context_sha,
        destination=tmp_path / "context",
    )
    wheel_path = tmp_path / "moe_sae_r1_worker-0.1.0-py3-none-any.whl"
    _wheel(wheel_path)
    wheel = repacker.validate_worker_wheel(wheel_path)

    assert repacker._context_worker_package_payload_sha256(context) == (
        wheel.package_payload_sha256
    )


def test_worker_wheel_rejects_foreign_maxtext_payload(tmp_path: Path) -> None:
    wheel = tmp_path / "moe_sae_r1_worker-0.1.0-py3-none-any.whl"
    _wheel(wheel, foreign=True)
    with pytest.raises(repacker.RepackError, match="foreign payload"):
        repacker.validate_worker_wheel(wheel)


def test_worker_wheel_rejects_absolute_member(tmp_path: Path) -> None:
    wheel = tmp_path / "moe_sae_r1_worker-0.1.0-py3-none-any.whl"
    _wheel(wheel, absolute=True)
    with pytest.raises(repacker.RepackError, match="normalized relative path"):
        repacker.validate_worker_wheel(wheel)


def test_install_command_can_only_reinstall_local_worker_without_deps(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, ...]] = []

    def runner(argv: object, environment: object) -> subprocess.CompletedProcess[str]:
        del environment
        calls.append(tuple(argv))  # type: ignore[arg-type]
        return subprocess.CompletedProcess(tuple(argv), 0, "", "")  # type: ignore[arg-type]

    work = tmp_path / "work"
    work.mkdir()
    repacker.install_worker_wheel(
        Path("/opt/moe-sae-r1/derived/wheels/exact.whl"),
        runtime_python=Path("/opt/moe-sae-r1/runtime/bin/python"),
        uv=Path("/usr/local/bin/uv"),
        work_root=work,
        runner=runner,
    )

    assert len(calls) == 1
    command = calls[0]
    assert command[:3] == ("/usr/local/bin/uv", "pip", "install")
    assert "--offline" in command
    assert "--no-deps" in command
    assert "--reinstall" in command
    assert command[-1] == "/opt/moe-sae-r1/derived/wheels/exact.whl"


def test_installed_worker_tree_rejects_unrecorded_stale_file(
    tmp_path: Path,
) -> None:
    environment = tmp_path / "runtime"
    package = environment / "lib/python3.12/site-packages/moe_sae"
    package.mkdir(parents=True)
    installed = package / "__init__.py"
    installed.write_bytes(b"")
    worker = {
        "files": [
            {
                "logical_path": "moe_sae/__init__.py",
                "installed_path": ("lib/python3.12/site-packages/moe_sae/__init__.py"),
            }
        ]
    }
    repacker._require_worker_filesystem_exact(environment, worker)
    (package / "stale.py").write_bytes(b"STALE = True\n")
    with pytest.raises(repacker.RepackError, match="differs from its RECORD"):
        repacker._require_worker_filesystem_exact(environment, worker)


def test_release_boundary_rejects_import_side_effect_drift() -> None:
    post_install = [
        {
            "name": "maxtext",
            "version": "0.2.3",
            "record_sha256": "1" * 64,
            "files": [],
        },
        {
            "name": "moe-sae-r1-worker",
            "version": "0.1.0",
            "record_sha256": "2" * 64,
            "files": [],
        },
    ]
    release_boundary = [
        dict(post_install[0], record_sha256="3" * 64),
        dict(post_install[1]),
    ]
    with pytest.raises(repacker.RepackError, match="import probes mutated"):
        repacker._require_release_inventory_unchanged(
            post_install,
            release_boundary,
        )
    repacker._require_release_inventory_unchanged(post_install, post_install)


def test_runtime_archive_is_byte_deterministic_and_safe(tmp_path: Path) -> None:
    prefix = tmp_path / "opt/moe-sae-r1"
    (prefix / "runtime/bin").mkdir(parents=True)
    (prefix / "runtime/python/bin").mkdir(parents=True)
    (prefix / "source").mkdir()
    python = prefix / "runtime/python/bin/python"
    python.write_bytes(b"python\n")
    python.chmod(0o755)
    os.symlink("../python/bin/python", prefix / "runtime/bin/python")
    (prefix / "source/worker.py").write_bytes(b"VALUE = 1\n")

    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"
    first_identity = repacker.write_deterministic_runtime_archive(prefix, first)
    second_identity = repacker.write_deterministic_runtime_archive(prefix, second)

    assert first.read_bytes() == second.read_bytes()
    assert first_identity == second_identity
    assert first.read_bytes()[4:8] == b"\x00\x00\x00\x00"
    with tarfile.open(first, mode="r:gz") as archive:
        members = archive.getmembers()
    assert [member.name for member in members] == [
        "opt",
        "opt/moe-sae-r1",
        "opt/moe-sae-r1/runtime",
        "opt/moe-sae-r1/runtime/bin",
        "opt/moe-sae-r1/runtime/bin/python",
        "opt/moe-sae-r1/runtime/python",
        "opt/moe-sae-r1/runtime/python/bin",
        "opt/moe-sae-r1/runtime/python/bin/python",
        "opt/moe-sae-r1/source",
        "opt/moe-sae-r1/source/worker.py",
    ]
    assert all(member.uid == 0 and member.gid == 0 for member in members)
    assert all(member.mtime == 0 for member in members)
    assert members[4].issym()


def test_generated_gnu_archive_passes_actual_bootstrap_validator_with_long_name(
    tmp_path: Path,
) -> None:
    prefix = tmp_path / "opt/moe-sae-r1"
    (prefix / "runtime/bin").mkdir(parents=True)
    (prefix / "runtime/python/bin").mkdir(parents=True)
    executable = prefix / "runtime/python/bin/python"
    executable.write_bytes(b"python\n")
    executable.chmod(0o755)
    os.symlink("../python/bin/python", prefix / "runtime/bin/python")
    long_directory = prefix
    for index in range(5):
        long_directory /= f"long-runtime-component-{index}-" + "x" * 45
    long_directory.mkdir(parents=True)
    (long_directory / "payload.py").write_bytes(b"VALUE = 1\n")
    archive_path = tmp_path / "runtime.tar.gz"
    identity = repacker.write_deterministic_runtime_archive(prefix, archive_path)
    assert identity.max_member_name_bytes > 255

    bootstrap = BOOTSTRAP.read_text(encoding="utf-8")
    function_start = bootstrap.index("validate_and_extract_archive()")
    code_start = bootstrap.index("<<'PY'\n", function_start) + len("<<'PY'\n")
    code_end = bootstrap.index("\nPY\n}", code_start)
    validator = bootstrap[code_start:code_end]
    staging = tmp_path / "staging"
    staging.mkdir()
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            validator,
            str(archive_path),
            str(staging),
            str(identity.member_count),
            str(repacker.MAX_RUNTIME_BYTES),
        ],
        check=True,
    )
    assert (staging / "opt/moe-sae-r1/runtime/python/bin/python").is_file()


def test_runtime_archive_rejects_escape_symlink_and_hardlink(
    tmp_path: Path,
) -> None:
    prefix = tmp_path / "escape/opt/moe-sae-r1"
    prefix.mkdir(parents=True)
    os.symlink("/etc/passwd", prefix / "escape")
    with pytest.raises(repacker.RepackError, match="escapes /opt"):
        repacker.write_deterministic_runtime_archive(prefix, tmp_path / "escape.tar.gz")

    prefix = tmp_path / "hardlink/opt/moe-sae-r1"
    prefix.mkdir(parents=True)
    original = prefix / "one"
    original.write_bytes(b"same")
    os.link(original, prefix / "two")
    with pytest.raises(repacker.RepackError, match="hard-linked"):
        repacker.write_deterministic_runtime_archive(
            prefix, tmp_path / "hardlink.tar.gz"
        )


def test_publication_never_replaces_or_deletes_an_existing_name(
    tmp_path: Path,
) -> None:
    existing = tmp_path / "result.json"
    existing.write_bytes(b"unowned\n")
    with pytest.raises(repacker.RepackError, match="refusing to replace"):
        repacker._atomic_write(existing, b"ours\n", mode=0o444)
    assert existing.read_bytes() == b"unowned\n"

    prefix = tmp_path / "opt/moe-sae-r1"
    prefix.mkdir(parents=True)
    archive = tmp_path / "runtime.tar.gz"
    archive.write_bytes(b"unowned archive\n")
    with pytest.raises(repacker.RepackError, match="refusing to replace"):
        repacker.write_deterministic_runtime_archive(prefix, archive)
    assert archive.read_bytes() == b"unowned archive\n"


def _request(tmp_path: Path, **changes: object) -> object:
    values: dict[str, object] = {
        "context_archive": tmp_path / "context.tar",
        "context_archive_sha256": "1" * 64,
        "expected_repacker_sha256": hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
        "expected_boot_image": (
            "projects/655343590563/global/images/moe-sae-r1-v6e-native-0b01bc2-20260724"
        ),
        "controller_admitted_boot_source_image_id": 123,
        "expected_image_recipe_sha256": "2" * 64,
        "expected_image_observation_sha256": "3" * 64,
        "controller_admitted_base_runtime_archive_sha256": "4" * 64,
        "expected_base_context_commit": "5" * 40,
        "prefix": tmp_path / "opt/moe-sae-r1",
        "identity_root": tmp_path / "usr/local/lib/moe-sae-r1",
        "uv": tmp_path / "usr/local/bin/uv",
        "output": tmp_path / "runtime.tar.gz",
        "result": tmp_path / "result.json",
        "work_root": tmp_path / "work",
    }
    values.update(changes)
    return repacker.RepackRequest(**values)


def test_request_paths_are_distinct_and_outside_work_root(
    tmp_path: Path,
) -> None:
    same = tmp_path / "same"
    with pytest.raises(repacker.RepackError, match="pairwise distinct"):
        repacker._validate_request(_request(tmp_path, output=same, result=same))
    work = tmp_path / "work"
    with pytest.raises(repacker.RepackError, match="inside the work root"):
        repacker._validate_request(
            _request(tmp_path, work_root=work, output=work / "runtime.tar.gz")
        )


def test_request_rejects_symlinked_output_parent(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    with pytest.raises(repacker.RepackError, match="existing real directory"):
        repacker._validate_request(_request(tmp_path, output=linked / "runtime.tar.gz"))


def test_metadata_and_context_paths_use_closed_ascii_grammar() -> None:
    for image in (
        "projects/neural-dragnet/global/images/r1-image",
        "projects/655343590563/global/images/r1-image",
    ):
        assert repacker._IMAGE_RESOURCE.fullmatch(image)
    for unsafe in (
        "/",
        "/etc/passwd",
        "line\nbreak",
        "back\\slash",
        "é.py",
        "x" * 241,
    ):
        with pytest.raises(
            repacker.RepackError,
            match="(?:safe|normalized) relative path",
        ):
            repacker._safe_archive_name(unsafe, name="test")


@pytest.mark.parametrize("phase", ["build", "install", "import"])
def test_failure_cleanup_and_fail_dirty_rerun_boundary(
    tmp_path: Path,
    phase: str,
) -> None:
    work = tmp_path / f"work-{phase}"
    prefix = tmp_path / f"prefix-{phase}"
    prefix.mkdir()
    request = SimpleNamespace(work_root=work)

    @repacker._cleanup_owned_work_root
    def fail(_request: object) -> object:
        work.mkdir()
        repacker._atomic_write(
            work / ".repacker-owned",
            repacker._WORK_ROOT_MARKER,
            mode=0o400,
        )
        if phase != "build":
            derived = prefix / "derived"
            derived.mkdir()
            (derived / "OVERLAY-INCOMPLETE.json").write_bytes(b"incomplete\n")
        raise repacker.RepackError(f"{phase} failed")

    with pytest.raises(repacker.RepackError, match=f"{phase} failed"):
        fail(request)
    assert not work.exists()
    if phase == "build":
        repacker.require_pristine_derived_state(prefix)
    else:
        with pytest.raises(repacker.RepackError, match="mutation began"):
            repacker.require_pristine_derived_state(prefix)


def test_cleanup_never_removes_an_unowned_work_root(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    (work / "sentinel").write_bytes(b"unowned\n")
    repacker._remove_owned_work_root(work)
    assert (work / "sentinel").read_bytes() == b"unowned\n"
