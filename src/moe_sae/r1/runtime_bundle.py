"""Offline authority for the isolated DeepSeek-R1 image build context.

The validator in this module can prove only that checked-in build inputs obey
the static R1 contract.  It cannot publish or qualify an image.  A production
image still needs the existing recipe/observation/publication chain, a trusted
Compute placement receipt, and a complete 16-host TPU/JAX attestation.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import tarfile
import tomllib
from typing import Any, ClassVar, Mapping, Sequence

from .checkpoint_conversion import (
    R1RuntimeLockIdentityV1,
    attest_r1_runtime_lock,
)
from .checkpoint_inventory import MAXTEXT_REVISION


BUNDLE_SCHEMA = "moe-sae.r1.runtime-bundle.v1"
CONTEXT_SCHEMA = "moe-sae.r1.runtime-context.v1"
CONTEXT_MANIFEST_SCHEMA = "moe-sae.r1.runtime-context-manifest.v1"
PYTHON_VERSION = "3.12.11"
UV_VERSION = "0.11.14"
UV_ARTIFACT_URL = "https://github.com/astral-sh/uv/releases/download/0.11.14/uv-x86_64-unknown-linux-gnu.tar.gz"
UV_ARTIFACT_SHA256 = "f3b623eb0e6141a7053d571d59a0bdc341e0f238ea8f5f0b4815ddbec9a2a296"
UV_BINARY_SHA256 = "b5cbc3a3f35debad0b4770811efd190bcf460b654114d6a3f71e0ce298468e5d"
UV_ARTIFACT_MEMBER = "uv-x86_64-unknown-linux-gnu/uv"
PACKER_VERSION = "1.15.4"
GOOGLECOMPUTE_PLUGIN_VERSION = "1.2.7"
JAX_VERSION = "0.10.2"
JAXLIB_VERSION = "0.10.2"
LIBTPU_VERSION = "0.0.43"
RUNTIME_USER = "moe_sae"
RUNTIME_UID = 20001
RUNTIME_GID = 20001
SERVICE_UNIT = "moe-sae-r1.service"
ENTRYPOINT_MODULE = "moe_sae.r1.runtime_entrypoint"
ENTRYPOINT_METADATA_KEY = "moe-sae-r1-launch-ref"
BASE_IMAGE_SOURCE_FAMILY = (
    "projects/ubuntu-os-accelerator-images/global/images/family/"
    "ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e"
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_REVISION = re.compile(r"[0-9a-f]{40}")
_SAFE_LOGICAL_NAME = re.compile(r"[a-z][a-z0-9_-]{0,63}")
_ALLOWED_MODES = {"0644": 0o644, "0755": 0o755}
_MAX_MANIFEST_BYTES = 1024 * 1024
_MAX_CONTEXT_FILE_BYTES = 2 * 1024 * 1024
_MAX_CONTEXT_BYTES = 8 * 1024 * 1024
_MAX_CONTEXT_FILES = 64
_CONTEXT_SPEC_PATH = "deploy/r1/context-files.json"

_EXPECTED_DIRECT_DEPENDENCIES = frozenset(
    {
        "array-record==0.8.3",
        "google-cloud-storage==3.13.0",
        "google-crc32c==1.8.0",
        "grain==0.2.18",
        "hatch-requirements-txt==0.4.1",
        "hatchling==1.27.0",
        "jax==0.10.2",
        "jaxlib==0.10.2",
        "libtpu==0.0.43 ; sys_platform == 'linux' and platform_machine == 'x86_64'",
        "maxtext[tpu]==0.2.3",
        "ml-dtypes==0.5.4",
        "numpy==2.5.1",
        "orbax-checkpoint==0.12.1",
        "safetensors==0.8.0",
        "tensorstore==0.1.84",
    }
)
_CRITICAL_LOCKED_VERSIONS = {
    "array-record": "0.8.3",
    "google-cloud-storage": "3.13.0",
    "google-crc32c": "1.8.0",
    "grain": "0.2.18",
    "hatch-requirements-txt": "0.4.1",
    "hatchling": "1.27.0",
    "jax": JAX_VERSION,
    "jaxlib": JAXLIB_VERSION,
    "libtpu": LIBTPU_VERSION,
    "maxtext": "0.2.3",
    "ml-dtypes": "0.5.4",
    "numpy": "2.5.1",
    "orbax-checkpoint": "0.12.1",
    "safetensors": "0.8.0",
    "tensorstore": "0.1.84",
}
_REQUIRED_FILE_NAMES = frozenset(
    {
        "python-version",
        "runtime-pyproject",
        "runtime-lock",
        "context-spec",
        "context-builder",
        "bundle-validator",
        "packer-template",
        "provisioner",
        "image-measurer",
        "service-unit",
        "entrypoint",
    }
)
_EXPECTED_CONTEXT_FILES = frozenset(
    {
        ("deploy/r1/runtime/.python-version", ".python-version", "0644"),
        ("deploy/r1/runtime/pyproject.toml", "pyproject.toml", "0644"),
        ("deploy/r1/runtime/uv.lock", "uv.lock", "0644"),
        (
            "deploy/r1/image/r1-v6e.pkr.hcl",
            "image/r1-v6e.pkr.hcl",
            "0644",
        ),
        ("deploy/r1/image/provision.sh", "image/provision.sh", "0755"),
        ("deploy/r1/image/measure_image.py", "image/measure_image.py", "0755"),
        (
            "deploy/r1/systemd/moe-sae-r1.service",
            "systemd/moe-sae-r1.service",
            "0644",
        ),
        (
            "patches/r1/maxtext/0001-routed-moe-prefix-tap.patch",
            "patches/r1/maxtext/0001-routed-moe-prefix-tap.patch",
            "0644",
        ),
        (
            "patches/r1/maxtext/0002-b128-routed-prefix.patch",
            "patches/r1/maxtext/0002-b128-routed-prefix.patch",
            "0644",
        ),
        (
            "patches/r1/maxtext/0003-native-ragged-r1-prefix.patch",
            "patches/r1/maxtext/0003-native-ragged-r1-prefix.patch",
            "0644",
        ),
        ("src/moe_sae/__init__.py", "src/moe_sae/__init__.py", "0644"),
        ("src/moe_sae/r1/__init__.py", "src/moe_sae/r1/__init__.py", "0644"),
        (
            "src/moe_sae/r1/checkpoint_conversion.py",
            "src/moe_sae/r1/checkpoint_conversion.py",
            "0644",
        ),
        (
            "src/moe_sae/r1/checkpoint_inventory.py",
            "src/moe_sae/r1/checkpoint_inventory.py",
            "0644",
        ),
        ("src/moe_sae/r1/protocol.py", "src/moe_sae/r1/protocol.py", "0644"),
        (
            "src/moe_sae/r1/control_store.py",
            "src/moe_sae/r1/control_store.py",
            "0644",
        ),
        (
            "src/moe_sae/r1/gcs_control_store.py",
            "src/moe_sae/r1/gcs_control_store.py",
            "0644",
        ),
        (
            "src/moe_sae/r1/gcs_immutable_byte_store.py",
            "src/moe_sae/r1/gcs_immutable_byte_store.py",
            "0644",
        ),
        (
            "src/moe_sae/r1/controller.py",
            "src/moe_sae/r1/controller.py",
            "0644",
        ),
        (
            "src/moe_sae/r1/image_protocol.py",
            "src/moe_sae/r1/image_protocol.py",
            "0644",
        ),
        (
            "src/moe_sae/r1/preemption.py",
            "src/moe_sae/r1/preemption.py",
            "0644",
        ),
        (
            "src/moe_sae/r1/jax_prefix_restore.py",
            "src/moe_sae/r1/jax_prefix_restore.py",
            "0644",
        ),
        (
            "src/moe_sae/r1/maxtext_patch_series.py",
            "src/moe_sae/r1/maxtext_patch_series.py",
            "0644",
        ),
        (
            "src/moe_sae/r1/maxtext_prefix_runtime.py",
            "src/moe_sae/r1/maxtext_prefix_runtime.py",
            "0644",
        ),
        (
            "src/moe_sae/r1/prefix_batchsplit_contract.py",
            "src/moe_sae/r1/prefix_batchsplit_contract.py",
            "0644",
        ),
        (
            "src/moe_sae/r1/prefix_benchmark.py",
            "src/moe_sae/r1/prefix_benchmark.py",
            "0644",
        ),
        (
            "src/moe_sae/r1/prefix_benchmark_runner.py",
            "src/moe_sae/r1/prefix_benchmark_runner.py",
            "0644",
        ),
        (
            "src/moe_sae/r1/prefix_checkpoint.py",
            "src/moe_sae/r1/prefix_checkpoint.py",
            "0644",
        ),
        (
            "src/moe_sae/r1/prefix_checkpoint_v2.py",
            "src/moe_sae/r1/prefix_checkpoint_v2.py",
            "0644",
        ),
        (
            "src/moe_sae/r1/prefix_restore_authority.py",
            "src/moe_sae/r1/prefix_restore_authority.py",
            "0644",
        ),
        ("src/moe_sae/r1/flops.py", "src/moe_sae/r1/flops.py", "0644"),
        ("src/moe_sae/r1/mfu.py", "src/moe_sae/r1/mfu.py", "0644"),
        ("src/moe_sae/r1/mfu_v2.py", "src/moe_sae/r1/mfu_v2.py", "0644"),
        (
            "src/moe_sae/r1/mfu_authority.py",
            "src/moe_sae/r1/mfu_authority.py",
            "0644",
        ),
        (
            "src/moe_sae/r1/placement_v2.py",
            "src/moe_sae/r1/placement_v2.py",
            "0644",
        ),
        (
            "src/moe_sae/r1/preemption_v2.py",
            "src/moe_sae/r1/preemption_v2.py",
            "0644",
        ),
        (
            "src/moe_sae/r1/runtime_supervisor.py",
            "src/moe_sae/r1/runtime_supervisor.py",
            "0644",
        ),
        (
            "src/moe_sae/r1/runtime_entrypoint.py",
            "src/moe_sae/r1/runtime_entrypoint.py",
            "0644",
        ),
    }
)
_FORBIDDEN_RUNTIME_TOKENS = (
    "apt-get",
    "apt ",
    "pip install",
    "uv sync",
    "uv pip",
    "uv python",
    "git clone",
    "git pull",
    "reboot",
    "shutdown",
    "curl |",
    "wget |",
    "ssh-agent",
    "ssh-add",
)
_SECRET_PATTERNS = (
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(rb"\bhf_[A-Za-z0-9]{20,}\b"),
    re.compile(rb"\bgh[opusr]_[A-Za-z0-9]{30,}\b"),
    re.compile(rb"\bAIza[0-9A-Za-z_-]{30,}\b"),
    re.compile(rb'"private_key"\s*:\s*"-----BEGIN'),
)


class RuntimeBundleError(RuntimeError):
    """A static runtime input or deterministic context failed closed."""


def _strict_json(raw: bytes, *, name: str, maximum: int) -> Any:
    if len(raw) > maximum:
        raise RuntimeBundleError(f"{name} exceeds the bounded size")

    def pairs(items: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise RuntimeBundleError(f"{name} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeBundleError(f"{name} is not strict UTF-8 JSON") from error


def _strict_object(
    value: Any, *, name: str, fields: frozenset[str]
) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise RuntimeBundleError(f"{name} must contain exactly {sorted(fields)}")
    return value


def _read_bounded(path: Path, *, maximum: int = _MAX_MANIFEST_BYTES) -> bytes:
    try:
        if path.is_symlink() or not path.is_file():
            raise RuntimeBundleError(f"runtime input is not a regular file: {path}")
        size = path.stat().st_size
        if size > maximum:
            raise RuntimeBundleError(f"runtime input exceeds bounded size: {path}")
        return path.read_bytes()
    except OSError as error:
        raise RuntimeBundleError(f"cannot read runtime input: {path}") from error


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise RuntimeBundleError("runtime identity is not canonical JSON") from error


def _safe_relative_path(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise RuntimeBundleError(f"{name} must be a non-empty normalized path")
    if (
        any(ord(character) < 0x20 for character in value)
        or "\\" in value
        or "//" in value
        or re.fullmatch(r"[A-Za-z0-9._/-]+", value) is None
    ):
        raise RuntimeBundleError(f"{name} contains unsafe characters")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise RuntimeBundleError(f"{name} must be a safe relative path")
    if len(value.encode("utf-8")) > 240:
        raise RuntimeBundleError(f"{name} is too long")
    return value


def _file_mode(path: Path) -> str:
    mode = stat.S_IMODE(path.stat().st_mode)
    for text, expected in _ALLOWED_MODES.items():
        if mode == expected:
            return text
    raise RuntimeBundleError(f"runtime input has forbidden mode {mode:o}: {path}")


def _assert_no_secret(raw: bytes, name: str) -> None:
    for pattern in _SECRET_PATTERNS:
        if pattern.search(raw) is not None:
            raise RuntimeBundleError(f"token-shaped or private-key content in {name}")


@dataclass(frozen=True, slots=True)
class RuntimeBundleFileV1:
    name: str
    path: str
    mode: str
    sha256: str

    def __post_init__(self) -> None:
        if _SAFE_LOGICAL_NAME.fullmatch(self.name) is None:
            raise RuntimeBundleError("runtime file logical name is malformed")
        _safe_relative_path(self.path, "runtime file path")
        if self.mode not in _ALLOWED_MODES:
            raise RuntimeBundleError("runtime file mode is not 0644 or 0755")
        if _SHA256.fullmatch(self.sha256) is None:
            raise RuntimeBundleError("runtime file sha256 is malformed")

    def as_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "path": self.path,
            "mode": self.mode,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class RuntimeBundleIdentityV1:
    """Static build identity; never an image qualification."""

    SCHEMA: ClassVar[str] = "moe-sae.r1.runtime-bundle-identity.v1"

    manifest_sha256: str
    files_sha256: str
    lock_identity: R1RuntimeLockIdentityV1
    file_count: int
    production_qualified: bool = False

    def __post_init__(self) -> None:
        if _SHA256.fullmatch(self.manifest_sha256) is None:
            raise RuntimeBundleError("manifest_sha256 is malformed")
        if _SHA256.fullmatch(self.files_sha256) is None:
            raise RuntimeBundleError("files_sha256 is malformed")
        if type(self.lock_identity) is not R1RuntimeLockIdentityV1:
            raise RuntimeBundleError("lock_identity has the wrong type")
        if type(self.file_count) is not int or self.file_count < 1:
            raise RuntimeBundleError("file_count must be positive")
        if self.production_qualified is not False:
            raise RuntimeBundleError("static runtime identity cannot be qualified")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "manifest_sha256": self.manifest_sha256,
            "files_sha256": self.files_sha256,
            "lock_identity": self.lock_identity.as_dict(),
            "file_count": self.file_count,
            "production_qualified": False,
        }

    @property
    def digest(self) -> str:
        return _sha256_bytes(_canonical_bytes(self.as_dict()))


def _parse_bundle_files(value: Any) -> tuple[RuntimeBundleFileV1, ...]:
    if not isinstance(value, list) or not value:
        raise RuntimeBundleError("bundle files must be a non-empty array")
    result: list[RuntimeBundleFileV1] = []
    for index, raw in enumerate(value):
        obj = _strict_object(
            raw,
            name=f"files[{index}]",
            fields=frozenset({"name", "path", "mode", "sha256"}),
        )
        if not all(isinstance(obj[key], str) for key in obj):
            raise RuntimeBundleError(f"files[{index}] fields must be strings")
        result.append(
            RuntimeBundleFileV1(
                name=obj["name"],
                path=obj["path"],
                mode=obj["mode"],
                sha256=obj["sha256"],
            )
        )
    if len({item.name for item in result}) != len(result):
        raise RuntimeBundleError("bundle file logical names must be unique")
    if len({item.path for item in result}) != len(result):
        raise RuntimeBundleError("bundle file paths must be unique")
    if len({item.path.casefold() for item in result}) != len(result):
        raise RuntimeBundleError("bundle file paths must not case-fold collide")
    if {item.name for item in result} != _REQUIRED_FILE_NAMES:
        raise RuntimeBundleError("bundle does not name the exact required file set")
    return tuple(sorted(result, key=lambda item: item.name))


def _validate_pyproject_and_lock(runtime_root: Path) -> R1RuntimeLockIdentityV1:
    lock_identity = attest_r1_runtime_lock(runtime_root)
    pyproject_raw = _read_bounded(runtime_root / "pyproject.toml")
    lock_raw = _read_bounded(runtime_root / "uv.lock", maximum=2 * 1024 * 1024)
    try:
        pyproject = tomllib.loads(pyproject_raw.decode("utf-8"))
        lock = tomllib.loads(lock_raw.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise RuntimeBundleError("R1 project or lock is malformed") from error
    project = pyproject.get("project")
    if not isinstance(project, dict):
        raise RuntimeBundleError("runtime pyproject has no project table")
    if project.get("name") != "moe-sae-r1-worker":
        raise RuntimeBundleError("runtime project has the wrong package name")
    if project.get("requires-python") != f"=={PYTHON_VERSION}":
        raise RuntimeBundleError("runtime Python patch is not exact")
    dependencies = project.get("dependencies")
    if not isinstance(dependencies, list) or not all(
        isinstance(item, str) for item in dependencies
    ):
        raise RuntimeBundleError("runtime direct dependencies are malformed")
    if frozenset(dependencies) != _EXPECTED_DIRECT_DEPENDENCIES or len(
        dependencies
    ) != len(_EXPECTED_DIRECT_DEPENDENCIES):
        raise RuntimeBundleError("runtime direct dependency set drifted")
    build_system = pyproject.get("build-system")
    if build_system != {
        "requires": ["hatchling==1.27.0"],
        "build-backend": "hatchling.build",
    }:
        raise RuntimeBundleError("runtime build backend is not the allowlisted pin")
    uv = pyproject.get("tool", {}).get("uv", {})
    if not isinstance(uv, dict) or uv.get("package") is not True:
        raise RuntimeBundleError("runtime project must build its isolated package")
    if uv.get("default-groups") != []:
        raise RuntimeBundleError(
            "runtime project must have no default dependency groups"
        )
    if set(uv) != {"package", "default-groups", "sources"}:
        raise RuntimeBundleError("runtime uv configuration contains unknown controls")
    if "workspace" in pyproject or "dependency-groups" in pyproject:
        raise RuntimeBundleError("runtime project must not inherit workspace/groups")
    if lock.get("version") != 1 or lock.get("revision") != 3:
        raise RuntimeBundleError("runtime uv.lock schema/revision is unsupported")
    if lock.get("requires-python") != f"=={PYTHON_VERSION}":
        raise RuntimeBundleError("runtime lock Python patch differs from project")
    packages = lock.get("package")
    if not isinstance(packages, list):
        raise RuntimeBundleError("runtime lock package inventory is absent")
    by_name: dict[str, list[Mapping[str, Any]]] = {}
    for package in packages:
        if not isinstance(package, dict) or not isinstance(package.get("name"), str):
            raise RuntimeBundleError("runtime lock contains a malformed package")
        by_name.setdefault(package["name"], []).append(package)
        source = package.get("source")
        if not isinstance(source, dict):
            raise RuntimeBundleError("runtime lock package has no structured source")
        if "registry" in source:
            if source != {"registry": "https://pypi.org/simple"}:
                raise RuntimeBundleError("runtime lock uses an untrusted package index")
            artifacts = []
            if "sdist" in package:
                artifacts.append(package["sdist"])
            wheels = package.get("wheels", [])
            if not isinstance(wheels, list):
                raise RuntimeBundleError("runtime lock wheels are malformed")
            artifacts.extend(wheels)
            if not artifacts:
                raise RuntimeBundleError("registry package has no hashed artifact")
            for artifact in artifacts:
                if not isinstance(artifact, dict):
                    raise RuntimeBundleError("runtime artifact is malformed")
                digest = artifact.get("hash")
                if not isinstance(digest, str) or not digest.startswith("sha256:"):
                    raise RuntimeBundleError("runtime artifact lacks a SHA-256")
                if _SHA256.fullmatch(digest[7:]) is None:
                    raise RuntimeBundleError("runtime artifact SHA-256 is malformed")
        elif package["name"] == "maxtext":
            expected = (
                "https://github.com/AI-Hypercomputer/maxtext.git"
                f"?rev={MAXTEXT_REVISION}#{MAXTEXT_REVISION}"
            )
            if source != {"git": expected}:
                raise RuntimeBundleError("MaxText lock source is not the exact commit")
        elif package["name"] == "moe-sae-r1-worker":
            if source != {"editable": "."} and source != {"virtual": "."}:
                raise RuntimeBundleError("runtime package source is not isolated root")
        else:
            raise RuntimeBundleError("runtime lock contains an unapproved source type")
    if "torch" in by_name:
        raise RuntimeBundleError("CPU/GPU Torch must not enter the TPU runtime")
    for name, version in _CRITICAL_LOCKED_VERSIONS.items():
        candidates = by_name.get(name, [])
        if len(candidates) != 1 or candidates[0].get("version") != version:
            raise RuntimeBundleError(f"runtime lock pin drifted for {name}")
    return lock_identity


def _validate_service(text: str) -> None:
    required = {
        "User=moe_sae",
        "Group=moe_sae",
        "ExecStart=/opt/moe-sae-r1/runtime/bin/python -I -m "
        "moe_sae.r1.runtime_entrypoint --metadata-key=moe-sae-r1-launch-ref",
        "StandardInput=null",
        "Restart=no",
        "NoNewPrivileges=yes",
        "ProtectSystem=strict",
        "ProtectHome=yes",
        "UMask=0077",
        "CapabilityBoundingSet=",
        "RuntimeDirectory=moe-sae-r1",
        "RuntimeDirectoryMode=0700",
        "Environment=TMPDIR=/run/moe-sae-r1",
        "ReadWritePaths=/var/lib/moe-sae-r1 /var/cache/moe-sae-r1 "
        "/run/moe-sae-r1 /dev/shm",
    }
    lines = {line.strip() for line in text.splitlines() if line.strip()}
    missing = sorted(required - lines)
    if missing:
        raise RuntimeBundleError(f"systemd service is missing {missing}")
    forbidden = (
        "EnvironmentFile=",
        "PassEnvironment=",
        "Restart=on",
        "ExecStartPre=",
        "ExecStartPost=",
        "PrivateDevices=",
        "/bin/sh",
        "/bin/bash",
        "%i",
        "%I",
        "${",
        "/var/tmp/moe-sae-r1",
    )
    if any(token in text for token in forbidden):
        raise RuntimeBundleError("systemd service contains a mutable/unsafe control")
    for line in lines:
        if line.startswith("ExecStart=") and line != next(
            item for item in required if item.startswith("ExecStart=")
        ):
            raise RuntimeBundleError("systemd service has non-canonical ExecStart")


def _validate_entrypoint(raw: bytes) -> None:
    text = raw.decode("utf-8")
    lowered = text.lower()
    for token in _FORBIDDEN_RUNTIME_TOKENS:
        if token in lowered:
            raise RuntimeBundleError(
                f"runtime entrypoint contains mutation token {token}"
            )
    try:
        tree = ast.parse(text)
    except SyntaxError as error:
        raise RuntimeBundleError("runtime entrypoint is not valid Python") from error
    forbidden_import_roots = {"importlib", "jax", "maxtext", "subprocess"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(
                alias.name.split(".", 1)[0] in forbidden_import_roots
                for alias in node.names
            ):
                raise RuntimeBundleError("entrypoint has a forbidden import")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".", 1)[0] in forbidden_import_roots:
                raise RuntimeBundleError("entrypoint has a forbidden import")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in {"eval", "exec", "compile", "__import__"}:
                raise RuntimeBundleError("entrypoint has dynamic code execution")
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "os"
            and (
                node.func.attr in {"system", "popen"}
                or node.func.attr.startswith("exec")
                or node.func.attr.startswith("spawn")
            )
        ):
            raise RuntimeBundleError("entrypoint has process execution")


def _validate_packer(text: str) -> None:
    required = (
        f'required_version = "= {PACKER_VERSION}"',
        f'version = "= {GOOGLECOMPUTE_PLUGIN_VERSION}"',
        'variable "base_image_self_link"',
        'variable "base_image_id"',
        "projects/ubuntu-os-accelerator-images/global/images/"
        "ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e-v[0-9]{8}",
        "source_image            = local.base_image_name",
        "source_image_project_id = [local.base_image_project_id]",
        "use_internal_ip         = true",
        "omit_external_ip        = false",
        'state    = "unqualified"',
        '"BUILD_TIME_ONLY=1"',
        '"BASE_IMAGE_ID=${var.base_image_id}"',
        '"BASE_IMAGE_SELF_LINK=${var.base_image_self_link}"',
        '"IMAGE_RECIPE_SHA256=${var.image_recipe_sha256}"',
        '"PROVISIONER_SHA256=${var.provisioner_sha256}"',
        "/usr/bin/sudo /usr/bin/env -i PATH=/usr/sbin:/usr/bin:/sbin:/bin",
    )
    if any(token not in text for token in required):
        raise RuntimeBundleError(
            "Packer template does not bind required versions/state"
        )
    forbidden = (
        "source_image_family",
        "image_family",
        "startup-script",
        "startup_script",
        "metadata_startup_script",
        "latest",
        'variable "base_image_name"',
        'variable "base_image_project_id"',
        "sudo -e",
    )
    if any(token in text.lower() for token in forbidden):
        raise RuntimeBundleError("Packer template contains mutable image/boot controls")


def _validate_context_spec(raw: bytes) -> None:
    value = _strict_json(raw, name="context-files.json", maximum=_MAX_MANIFEST_BYTES)
    obj = _strict_object(
        value,
        name="context-files.json",
        fields=frozenset({"schema", "files"}),
    )
    if obj["schema"] != CONTEXT_SCHEMA or not isinstance(obj["files"], list):
        raise RuntimeBundleError("context specification schema/files are invalid")
    observed: list[tuple[str, str, str]] = []
    for index, value in enumerate(obj["files"]):
        item = _strict_object(
            value,
            name=f"context files[{index}]",
            fields=frozenset({"source", "archive", "mode"}),
        )
        source = _safe_relative_path(item["source"], "context source")
        archive = _safe_relative_path(item["archive"], "context archive path")
        mode = item["mode"]
        if mode not in _ALLOWED_MODES:
            raise RuntimeBundleError("context file mode is not allowed")
        observed.append((source, archive, mode))
    if len(observed) != len(set(observed)):
        raise RuntimeBundleError("context specification contains duplicates")
    if frozenset(observed) != _EXPECTED_CONTEXT_FILES:
        raise RuntimeBundleError("context specification is not the exact R1 allowlist")


def validate_runtime_bundle(repository_root: str | Path) -> RuntimeBundleIdentityV1:
    """Validate tracked static inputs without claiming an image was built."""

    candidate = Path(repository_root).resolve()
    if candidate.name == "r1" and candidate.parent.name == "deploy":
        bundle_root = candidate
        root = candidate.parents[1]
    else:
        root = candidate
        bundle_root = root / "deploy/r1"
    manifest_path = bundle_root / "bundle.toml"
    manifest_raw = _read_bounded(manifest_path)
    try:
        manifest = tomllib.loads(manifest_raw.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise RuntimeBundleError("bundle.toml is malformed") from error
    required_fields = frozenset(
        {
            "schema",
            "qualification_state",
            "production_admission_enabled",
            "python_version",
            "uv_version",
            "uv_artifact_url",
            "uv_artifact_sha256",
            "uv_binary_sha256",
            "uv_artifact_member",
            "packer_version",
            "googlecompute_plugin_version",
            "maxtext_commit",
            "jax_version",
            "jaxlib_version",
            "libtpu_version",
            "target_os_release_id",
            "target_os_release_version_id",
            "target_architecture",
            "base_image_source_family",
            "runtime_user",
            "runtime_uid",
            "runtime_gid",
            "service_unit",
            "entrypoint_module",
            "entrypoint_metadata_key",
            "unmet_gates",
            "files",
        }
    )
    obj = _strict_object(manifest, name="bundle.toml", fields=required_fields)
    expected_scalars = {
        "schema": BUNDLE_SCHEMA,
        "qualification_state": "UNBUILT",
        "production_admission_enabled": False,
        "python_version": PYTHON_VERSION,
        "uv_version": UV_VERSION,
        "uv_artifact_url": UV_ARTIFACT_URL,
        "uv_artifact_sha256": UV_ARTIFACT_SHA256,
        "uv_binary_sha256": UV_BINARY_SHA256,
        "uv_artifact_member": UV_ARTIFACT_MEMBER,
        "packer_version": PACKER_VERSION,
        "googlecompute_plugin_version": GOOGLECOMPUTE_PLUGIN_VERSION,
        "maxtext_commit": MAXTEXT_REVISION,
        "jax_version": JAX_VERSION,
        "jaxlib_version": JAXLIB_VERSION,
        "libtpu_version": LIBTPU_VERSION,
        "target_os_release_id": "ubuntu",
        "target_os_release_version_id": "22.04",
        "target_architecture": "x86_64",
        "base_image_source_family": BASE_IMAGE_SOURCE_FAMILY,
        "runtime_user": RUNTIME_USER,
        "runtime_uid": RUNTIME_UID,
        "runtime_gid": RUNTIME_GID,
        "service_unit": SERVICE_UNIT,
        "entrypoint_module": ENTRYPOINT_MODULE,
        "entrypoint_metadata_key": ENTRYPOINT_METADATA_KEY,
    }
    for field, expected in expected_scalars.items():
        if obj[field] != expected or type(obj[field]) is not type(expected):
            raise RuntimeBundleError(f"bundle field {field} drifted")
    expected_gates = [
        "controller-base-image-family-resolution-receipt",
        "v6e-base-image-prerequisite-service-attestation",
        "published-numeric-gce-image-identity",
        "observed-image-manifest-and-sbom",
        "controller-compute-placement-receipt",
        "atomic-16-host-jax-vfio-admission",
        "exact-jax-libtpu-xla-loader-proxy-environment",
        "ssh-disconnect-liveness",
        "checkpoint-restore-and-parity",
        "v6e-kernel-and-mfu-qualification",
    ]
    if obj["unmet_gates"] != expected_gates:
        raise RuntimeBundleError("bundle unmet-gate declaration drifted")
    files = _parse_bundle_files(obj["files"])
    expected_deploy_paths = {
        item.path for item in files if item.path.startswith("deploy/r1/")
    } | {"deploy/r1/bundle.toml"}
    actual_paths: set[str] = set()
    for path in bundle_root.rglob("*"):
        if path.is_dir():
            continue
        relative = path.relative_to(root).as_posix()
        actual_paths.add(relative)
    if actual_paths != expected_deploy_paths:
        raise RuntimeBundleError("bundle tree has extra, missing, or generated files")
    observed: list[dict[str, str]] = []
    by_name: dict[str, bytes] = {}
    for item in files:
        path = root / item.path
        raw = _read_bounded(
            path,
            maximum=2 * 1024 * 1024
            if item.name == "runtime-lock"
            else _MAX_MANIFEST_BYTES,
        )
        if _file_mode(path) != item.mode:
            raise RuntimeBundleError(f"runtime file mode drifted for {item.name}")
        digest = _sha256_bytes(raw)
        if digest != item.sha256:
            raise RuntimeBundleError(f"runtime file hash drifted for {item.name}")
        _assert_no_secret(raw, item.name)
        observed.append(item.as_dict())
        by_name[item.name] = raw
    lock_identity = _validate_pyproject_and_lock(bundle_root / "runtime")
    if lock_identity.pyproject_sha256 != next(
        item.sha256 for item in files if item.name == "runtime-pyproject"
    ):
        raise RuntimeBundleError("lock attestation and bundle pyproject disagree")
    if lock_identity.uv_lock_sha256 != next(
        item.sha256 for item in files if item.name == "runtime-lock"
    ):
        raise RuntimeBundleError("lock attestation and bundle uv.lock disagree")
    if by_name["python-version"] != f"{PYTHON_VERSION}\n".encode():
        raise RuntimeBundleError(".python-version does not bind the Python patch")
    _validate_service(by_name["service-unit"].decode("utf-8"))
    _validate_entrypoint(by_name["entrypoint"])
    _validate_packer(by_name["packer-template"].decode("utf-8"))
    _validate_context_spec(by_name["context-spec"])
    provisioner = by_name["provisioner"].decode("utf-8").lower()
    required_provisioner = (
        "build_time_only",
        "systemctl enable",
        "systemd-analyze verify",
        UV_ARTIFACT_URL.lower(),
        UV_ARTIFACT_SHA256,
        UV_BINARY_SHA256,
        UV_ARTIFACT_MEMBER,
        "base_image_id",
        "base_image_self_link",
        "/usr/bin/python3",
        "runtime archive does not contain the exact member allowlist",
        "--no-same-owner",
        "--no-same-permissions",
        "chmod -r u=rwx,go=rx",
        "/usr/sbin/runuser --user",
    )
    if any(token not in provisioner for token in required_provisioner):
        raise RuntimeBundleError(
            "provisioner lacks build-only/service ownership boundary"
        )
    files_sha256 = _sha256_bytes(_canonical_bytes(observed))
    return RuntimeBundleIdentityV1(
        manifest_sha256=_sha256_bytes(manifest_raw),
        files_sha256=files_sha256,
        lock_identity=lock_identity,
        file_count=len(files),
    )


@dataclass(frozen=True, slots=True)
class RuntimeArchiveIdentityV1:
    commit: str
    manifest_sha256: str
    archive_sha256: str
    file_count: int
    total_payload_bytes: int

    def __post_init__(self) -> None:
        if _REVISION.fullmatch(self.commit) is None:
            raise RuntimeBundleError("archive commit is not a full Git revision")
        if _SHA256.fullmatch(self.manifest_sha256) is None:
            raise RuntimeBundleError("archive manifest SHA-256 is malformed")
        if _SHA256.fullmatch(self.archive_sha256) is None:
            raise RuntimeBundleError("archive SHA-256 is malformed")
        if not (1 <= self.file_count <= _MAX_CONTEXT_FILES):
            raise RuntimeBundleError("archive file count is out of bounds")
        if not (1 <= self.total_payload_bytes <= _MAX_CONTEXT_BYTES):
            raise RuntimeBundleError("archive payload size is out of bounds")


def _git(
    repository: Path,
    arguments: Sequence[str],
    *,
    maximum: int = _MAX_CONTEXT_FILE_BYTES,
) -> bytes:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as error:
        raise RuntimeBundleError(
            "git is unavailable for context construction"
        ) from error
    if result.returncode != 0:
        raise RuntimeBundleError(
            "git context read failed: "
            + result.stderr.decode("utf-8", errors="replace")[:512]
        )
    if len(result.stdout) > maximum:
        raise RuntimeBundleError("git context read exceeded its bound")
    return result.stdout


def _git_entry(repository: Path, commit: str, path: str) -> tuple[str, str, bytes]:
    tree = _git(repository, ["ls-tree", "-z", commit, "--", path])
    entries = [entry for entry in tree.split(b"\0") if entry]
    if len(entries) != 1:
        raise RuntimeBundleError(f"context source is missing or ambiguous: {path}")
    try:
        header, returned_path = entries[0].split(b"\t", 1)
        mode, kind, object_id = header.decode("ascii").split(" ")
        decoded_path = returned_path.decode("utf-8")
    except (ValueError, UnicodeDecodeError) as error:
        raise RuntimeBundleError("git tree entry is malformed") from error
    if decoded_path != path or kind != "blob" or mode not in {"100644", "100755"}:
        raise RuntimeBundleError(
            f"context source is not an allowed regular blob: {path}"
        )
    raw = _git(repository, ["show", f"{commit}:{path}"])
    if raw.startswith(b"version https://git-lfs.github.com/spec/v1\n"):
        raise RuntimeBundleError(f"Git LFS pointer is not runtime content: {path}")
    return mode, object_id, raw


def _resolved_commit(repository: Path, commit: str) -> None:
    if _REVISION.fullmatch(commit) is None:
        raise RuntimeBundleError("context commit must be a full lowercase Git revision")
    resolved = _git(repository, ["rev-parse", "--verify", f"{commit}^{{commit}}"])
    if resolved.decode("ascii").strip() != commit:
        raise RuntimeBundleError("context revision did not resolve to the exact commit")


def _context_source_modes(spec_raw: bytes) -> dict[str, str]:
    spec = _strict_json(
        spec_raw, name="context-files.json", maximum=_MAX_MANIFEST_BYTES
    )
    obj = _strict_object(
        spec,
        name="context-files.json",
        fields=frozenset({"schema", "files"}),
    )
    if obj["schema"] != CONTEXT_SCHEMA or not isinstance(obj["files"], list):
        raise RuntimeBundleError("context specification schema/files are invalid")
    result: dict[str, str] = {}
    for index, item in enumerate(obj["files"]):
        record = _strict_object(
            item,
            name=f"context files[{index}]",
            fields=frozenset({"source", "archive", "mode"}),
        )
        source = _safe_relative_path(record["source"], "context source")
        _safe_relative_path(record["archive"], "context archive path")
        mode = record["mode"]
        if mode not in _ALLOWED_MODES:
            raise RuntimeBundleError("context file mode is not allowed")
        existing = result.setdefault(source, mode)
        if existing != mode:
            raise RuntimeBundleError("context source has conflicting modes")
    return result


def _require_head_bundle_matches_worktree(repository: Path, commit: str) -> bytes:
    head = _git(repository, ["rev-parse", "HEAD"]).decode("ascii").strip()
    if head != commit:
        raise RuntimeBundleError("production context commit must equal repository HEAD")
    validate_runtime_bundle(repository)
    bundle_mode, _, committed_bundle = _git_entry(
        repository, commit, "deploy/r1/bundle.toml"
    )
    if bundle_mode != "100644":
        raise RuntimeBundleError("committed bundle.toml mode is not 0644")
    try:
        bundle = tomllib.loads(committed_bundle.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise RuntimeBundleError("committed bundle.toml is malformed") from error
    files = _parse_bundle_files(bundle.get("files"))
    _, _, committed_spec = _git_entry(repository, commit, _CONTEXT_SPEC_PATH)
    _validate_context_spec(committed_spec)
    required_modes = {"deploy/r1/bundle.toml": "0644"}
    for item in files:
        existing = required_modes.setdefault(item.path, item.mode)
        if existing != item.mode:
            raise RuntimeBundleError("bundle input has conflicting modes")
    for path, mode in _context_source_modes(committed_spec).items():
        existing = required_modes.setdefault(path, mode)
        if existing != mode:
            raise RuntimeBundleError("context input has conflicting modes")
    for path, mode in sorted(required_modes.items()):
        git_mode, _, committed_raw = _git_entry(repository, commit, path)
        expected_git_mode = "100755" if mode == "0755" else "100644"
        if git_mode != expected_git_mode:
            raise RuntimeBundleError(f"committed mode drifted for {path}")
        worktree_path = repository / path
        worktree_raw = _read_bounded(worktree_path, maximum=_MAX_CONTEXT_FILE_BYTES)
        if _file_mode(worktree_path) != mode or worktree_raw != committed_raw:
            raise RuntimeBundleError(
                f"validated worktree input differs from target commit: {path}"
            )
    return committed_spec


def _build_runtime_archive_from_committed_spec(
    repository_root: str | Path,
    *,
    commit: str,
    destination: str | Path | None = None,
    context_spec_path: str = "deploy/r1/context-files.json",
) -> tuple[bytes, RuntimeArchiveIdentityV1]:
    """Build one deterministic uncompressed tar from an exact committed allowlist."""

    repository = Path(repository_root).resolve()
    _resolved_commit(repository, commit)
    _, _, spec_raw = _git_entry(repository, commit, context_spec_path)
    spec = _strict_json(
        spec_raw, name="context-files.json", maximum=_MAX_MANIFEST_BYTES
    )
    obj = _strict_object(
        spec,
        name="context-files.json",
        fields=frozenset({"schema", "files"}),
    )
    if obj["schema"] != CONTEXT_SCHEMA or not isinstance(obj["files"], list):
        raise RuntimeBundleError("context specification schema/files are invalid")
    if not (1 <= len(obj["files"]) <= _MAX_CONTEXT_FILES):
        raise RuntimeBundleError("context file count is out of bounds")
    entries: list[dict[str, Any]] = []
    payloads: dict[str, bytes] = {}
    seen_sources: set[str] = set()
    seen_archives: set[str] = set()
    total = 0
    for index, item in enumerate(obj["files"]):
        record = _strict_object(
            item,
            name=f"context files[{index}]",
            fields=frozenset({"source", "archive", "mode"}),
        )
        source = _safe_relative_path(record["source"], "context source")
        archive = _safe_relative_path(record["archive"], "context archive path")
        mode_text = record["mode"]
        if mode_text not in _ALLOWED_MODES:
            raise RuntimeBundleError("context file mode is not allowed")
        if source in seen_sources or archive in seen_archives:
            raise RuntimeBundleError("context contains duplicate source/archive path")
        if archive.casefold() in {item.casefold() for item in seen_archives}:
            raise RuntimeBundleError("context archive paths case-fold collide")
        seen_sources.add(source)
        seen_archives.add(archive)
        git_mode, blob_id, raw = _git_entry(repository, commit, source)
        expected_git_mode = "100755" if mode_text == "0755" else "100644"
        if git_mode != expected_git_mode:
            raise RuntimeBundleError(f"Git executable mode drifted for {source}")
        _assert_no_secret(raw, source)
        total += len(raw)
        if total > _MAX_CONTEXT_BYTES:
            raise RuntimeBundleError("context payload exceeds its total bound")
        payloads[archive] = raw
        entries.append(
            {
                "source": source,
                "archive": archive,
                "mode": mode_text,
                "size": len(raw),
                "sha256": _sha256_bytes(raw),
                "git_blob": blob_id,
            }
        )
    manifest = {
        "schema": CONTEXT_MANIFEST_SCHEMA,
        "commit": commit,
        "files": sorted(entries, key=lambda item: item["archive"]),
        "production_qualified": False,
    }
    manifest_raw = _canonical_bytes(manifest) + b"\n"
    if "CONTEXT-MANIFEST.json" in payloads:
        raise RuntimeBundleError("context reserves CONTEXT-MANIFEST.json")
    payloads["CONTEXT-MANIFEST.json"] = manifest_raw
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for name in sorted(payloads):
            raw = payloads[name]
            info = tarfile.TarInfo(name=name)
            entry = next((item for item in entries if item["archive"] == name), None)
            info.mode = 0o644 if entry is None else _ALLOWED_MODES[entry["mode"]]
            info.size = len(raw)
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            archive.addfile(info, io.BytesIO(raw))
    archive_raw = buffer.getvalue()
    identity = RuntimeArchiveIdentityV1(
        commit=commit,
        manifest_sha256=_sha256_bytes(manifest_raw),
        archive_sha256=_sha256_bytes(archive_raw),
        file_count=len(entries),
        total_payload_bytes=total,
    )
    if destination is not None:
        output = Path(destination)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(archive_raw)
    return archive_raw, identity


def build_runtime_archive(
    repository_root: str | Path,
    *,
    commit: str,
    destination: str | Path | None = None,
) -> tuple[bytes, RuntimeArchiveIdentityV1]:
    """Build the sole production context from validated bytes at exact HEAD.

    Unrelated dirty files are irrelevant. Every static bundle authority and
    every allowlisted payload file must, however, be byte/mode-identical to the
    target commit before the private deterministic tar writer is reached.
    """

    repository = Path(repository_root).resolve()
    _resolved_commit(repository, commit)
    committed_spec = _require_head_bundle_matches_worktree(repository, commit)
    _validate_context_spec(committed_spec)
    return _build_runtime_archive_from_committed_spec(
        repository,
        commit=commit,
        destination=destination,
        context_spec_path=_CONTEXT_SPEC_PATH,
    )


__all__ = [
    "BUNDLE_SCHEMA",
    "CONTEXT_MANIFEST_SCHEMA",
    "CONTEXT_SCHEMA",
    "RuntimeArchiveIdentityV1",
    "RuntimeBundleError",
    "RuntimeBundleFileV1",
    "RuntimeBundleIdentityV1",
    "build_runtime_archive",
    "validate_runtime_bundle",
]
