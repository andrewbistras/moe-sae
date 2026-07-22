#!/usr/bin/env python3
"""Emit deterministic diagnostic package/OS observations inside a baked image."""

from __future__ import annotations

import base64
import hashlib
from importlib import import_module
from importlib.metadata import distributions
import json
from pathlib import Path
import platform
import sys
import sysconfig


PREFIX = Path("/opt/moe-sae-r1")
SOURCE = PREFIX / "source"
ENVIRONMENT = PREFIX / "runtime"
OUTPUT = Path("/usr/local/lib/moe-sae-r1/image-observation.json")
MAXTEXT_REVISION = "c03f8bd307d18049fa814fc8c104bd9ec8c292aa"
UV_BINARY_SHA256 = "b5cbc3a3f35debad0b4770811efd190bcf460b654114d6a3f71e0ce298468e5d"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def package_inventory() -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    digest_cache: dict[Path, tuple[int, str]] = {}
    for distribution in distributions():
        distribution_name = distribution.metadata["Name"]
        if not isinstance(distribution_name, str) or not distribution_name:
            raise RuntimeError("installed distribution has no normalized name")
        name = distribution_name.lower().replace("_", "-")
        record_text = distribution.read_text("RECORD")
        if record_text is None:
            raise RuntimeError(f"installed distribution lacks RECORD: {name}")
        direct_url = distribution.read_text("direct_url.json")
        files = distribution.files
        if files is None:
            raise RuntimeError(f"installed distribution has no file inventory: {name}")
        installed_files: list[dict[str, object]] = []
        logical_paths: set[str] = set()
        for item in files:
            logical_path = str(item)
            if logical_path in logical_paths:
                raise RuntimeError(f"duplicate RECORD path for {name}: {logical_path}")
            logical_paths.add(logical_path)
            located = Path(distribution.locate_file(item))
            if not located.is_file():
                raise RuntimeError(f"installed RECORD file is absent: {located}")
            resolved = located.resolve(strict=True)
            try:
                installed_path = resolved.relative_to(ENVIRONMENT).as_posix()
            except ValueError as error:
                raise RuntimeError(
                    f"installed distribution file escapes runtime: {resolved}"
                ) from error
            measured = digest_cache.get(resolved)
            if measured is None:
                measured = (resolved.stat().st_size, sha256(resolved))
                digest_cache[resolved] = measured
            size, digest = measured
            if item.size is not None and item.size != size:
                raise RuntimeError(f"RECORD size mismatch for {name}: {logical_path}")
            if item.hash is not None:
                if item.hash.mode != "sha256":
                    raise RuntimeError(f"non-SHA256 RECORD hash for {name}")
                encoded = base64.urlsafe_b64encode(bytes.fromhex(digest)).rstrip(b"=")
                if encoded.decode("ascii") != item.hash.value:
                    raise RuntimeError(
                        f"RECORD content hash mismatch for {name}: {logical_path}"
                    )
            installed_files.append(
                {
                    "logical_path": logical_path,
                    "installed_path": installed_path,
                    "size": size,
                    "sha256": digest,
                    "record_sha256": (None if item.hash is None else item.hash.value),
                }
            )
        result.append(
            {
                "name": name,
                "version": distribution.version,
                "record_sha256": hashlib.sha256(
                    record_text.encode("utf-8")
                ).hexdigest(),
                "direct_url_sha256": (
                    None
                    if direct_url is None
                    else hashlib.sha256(direct_url.encode("utf-8")).hexdigest()
                ),
                "files": sorted(
                    installed_files, key=lambda item: str(item["logical_path"])
                ),
            }
        )
    result.sort(key=lambda item: (str(item["name"]), str(item["version"])))
    if len({str(item["name"]) for item in result}) != len(result):
        raise RuntimeError("installed distribution names are not unique")
    return result


def main() -> None:
    if Path(sys.prefix).resolve() != ENVIRONMENT:
        raise RuntimeError("image measurement is not running in the sealed environment")
    os_release = Path("/etc/os-release")
    dpkg_status = Path("/var/lib/dpkg/status")
    os_values: dict[str, str] = {}
    for line in os_release.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            os_values[key] = value.strip().strip('"')
    if os_values.get("ID") != "ubuntu" or os_values.get("VERSION_ID") != "22.04":
        raise RuntimeError("base OS does not match ubuntu 22.04 contract")
    inventory = package_inventory()
    versions = {str(item["name"]): str(item["version"]) for item in inventory}
    expected = {
        "jax": "0.10.2",
        "jaxlib": "0.10.2",
        "libtpu": "0.0.43",
        "maxtext": "0.2.3",
        "moe-sae-r1-worker": "0.1.0",
    }
    if {name: versions.get(name) for name in expected} != expected:
        raise RuntimeError("critical installed distribution version mismatch")
    maxtext_direct = next(
        distribution.read_text("direct_url.json")
        for distribution in distributions()
        if distribution.metadata["Name"].lower().replace("_", "-") == "maxtext"
    )
    if maxtext_direct is None:
        raise RuntimeError("installed MaxText has no direct_url identity")
    maxtext_source = json.loads(maxtext_direct)
    if (
        maxtext_source.get("url") != "https://github.com/AI-Hypercomputer/maxtext.git"
        or maxtext_source.get("vcs_info", {}).get("vcs") != "git"
        or maxtext_source.get("vcs_info", {}).get("commit_id") != MAXTEXT_REVISION
        or maxtext_source.get("vcs_info", {}).get("requested_revision")
        != MAXTEXT_REVISION
    ):
        raise RuntimeError("installed MaxText direct_url lacks the audited commit")
    runtime_module = import_module("moe_sae.r1.runtime_entrypoint")
    module_file = getattr(runtime_module, "__file__", None)
    if not isinstance(module_file, str):
        raise RuntimeError("imported runtime entrypoint has no file identity")
    installed_entrypoint = Path(module_file).resolve(strict=True)
    try:
        installed_entrypoint_path = installed_entrypoint.relative_to(
            ENVIRONMENT
        ).as_posix()
    except ValueError as error:
        raise RuntimeError(
            "imported runtime entrypoint escapes the sealed environment"
        ) from error
    source_entrypoint = SOURCE / "src/moe_sae/r1/runtime_entrypoint.py"
    if sha256(installed_entrypoint) != sha256(source_entrypoint):
        raise RuntimeError("installed runtime entrypoint differs from committed source")
    python_executable = Path(sys.executable).resolve()
    library_name = sysconfig.get_config_var("LDLIBRARY")
    library_dir = sysconfig.get_config_var("LIBDIR")
    if not isinstance(library_name, str) or not isinstance(library_dir, str):
        raise RuntimeError("managed Python shared-library identity is unavailable")
    python_library = Path(library_dir) / library_name
    if not python_library.is_file():
        raise RuntimeError("managed Python shared library is unavailable")
    uv_executable_sha256 = sha256(Path("/usr/local/bin/uv"))
    if uv_executable_sha256 != UV_BINARY_SHA256:
        raise RuntimeError("installed uv executable differs from pinned artifact")
    installed_files_raw = json.dumps(
        inventory, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    record = {
        "schema": "moe-sae.r1.image-observation-diagnostic.v1",
        "production_qualified": False,
        "python_version": platform.python_version(),
        "architecture": platform.machine(),
        "kernel_release_at_bake": platform.release(),
        "os_release_sha256": sha256(os_release),
        "dpkg_status_sha256": sha256(dpkg_status),
        "python_executable_sha256": sha256(python_executable),
        "python_library_sha256": sha256(python_library),
        "uv_executable_sha256": uv_executable_sha256,
        "context_manifest_sha256": sha256(SOURCE / "CONTEXT-MANIFEST.json"),
        "image_recipe_sha256": sha256(
            Path("/usr/local/lib/moe-sae-r1/image-recipe.json")
        ),
        "pyproject_sha256": sha256(SOURCE / "pyproject.toml"),
        "uv_lock_sha256": sha256(SOURCE / "uv.lock"),
        "service_unit_sha256": sha256(Path("/etc/systemd/system/moe-sae-r1.service")),
        "entrypoint_source_sha256": sha256(source_entrypoint),
        "installed_entrypoint_path": installed_entrypoint_path,
        "installed_entrypoint_sha256": sha256(installed_entrypoint),
        "installed_distribution_files_sha256": hashlib.sha256(
            installed_files_raw
        ).hexdigest(),
        "packages": inventory,
        "unmet_gates": [
            "controller-base-image-family-resolution-receipt",
            "v6e-base-image-prerequisite-service-attestation",
            "published-numeric-gce-image-identity",
            "runtime-kernel-and-tpu-observation",
            "controller-compute-placement-receipt",
            "atomic-16-host-jax-vfio-admission",
            "exact-jax-libtpu-xla-loader-proxy-environment",
            "ssh-disconnect-liveness",
            "checkpoint-restore-and-parity",
            "v6e-kernel-and-mfu-qualification",
        ],
    }
    if record["python_version"] != "3.12.11" or record["architecture"] != "x86_64":
        raise RuntimeError("Python/architecture does not match the baked contract")
    raw = (
        json.dumps(
            record, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        + b"\n"
    )
    OUTPUT.write_bytes(raw)
    OUTPUT.chmod(0o444)
    print(hashlib.sha256(raw).hexdigest())


if __name__ == "__main__":
    main()
