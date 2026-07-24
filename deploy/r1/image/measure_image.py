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
MAXTEXT_PATCH_SHA256 = (
    "f5b4f810dd981add4f13baad91d5811cf632ec32d17266b67b84644783ae850b",
    "d78349f83d08993afbaa1b2b346a9ac34f1b034b5015fa02572f71b79b6859fe",
)
MAXTEXT_POST_APPLY_SHA256 = (
    (
        "src/maxtext/layers/moe.py",
        "f6fe4ace7538f542747d20b3167e3424219aed4562552bd5a51b178d4f0cf527",
    ),
    (
        "src/maxtext/layers/nnx_decoders.py",
        "e51522f3249fb7d1bcf8e39ca6f55b3c51c5471fc5adf0ac7b1a5de94e3c90f5",
    ),
    (
        "src/maxtext/models/deepseek_prefix.py",
        "11fdb7bb64797822c336759df42dafa9524a1c1af50f82b72dc7af8edc8048c0",
    ),
)
MAXTEXT_WHEEL = (
    Path("/usr/local/lib/moe-sae-r1")
    / "maxtext-0.2.3-py3-none-any.whl"
)
MAXTEXT_PATCH_ATTESTATION = (
    Path("/usr/local/lib/moe-sae-r1") / "maxtext-patch-attestation.json"
)
MAXTEXT_WHEEL_ATTESTATION = (
    Path("/usr/local/lib/moe-sae-r1") / "maxtext-wheel-attestation.json"
)
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
        "hatch-requirements-txt": "0.4.1",
        "hatchling": "1.27.0",
        "maxtext": "0.2.3",
        "moe-sae-r1-worker": "0.1.0",
    }
    if {name: versions.get(name) for name in expected} != expected:
        raise RuntimeError("critical installed distribution version mismatch")
    maxtext_distribution = next(
        item
        for item in distributions()
        if item.metadata["Name"].lower().replace("_", "-") == "maxtext"
    )
    maxtext_direct = maxtext_distribution.read_text("direct_url.json")
    if maxtext_direct is None:
        raise RuntimeError("installed MaxText has no direct_url identity")
    maxtext_source = json.loads(maxtext_direct)
    expected_direct_url = {
        "archive_info": {},
        "url": "file:///usr/local/lib/moe-sae-r1/"
        "maxtext-0.2.3-py3-none-any.whl",
    }
    if maxtext_source != expected_direct_url:
        raise RuntimeError("installed MaxText is not the retained patched wheel")
    if not MAXTEXT_WHEEL.is_file():
        raise RuntimeError("retained patched MaxText wheel is absent")
    wheel_attestation_raw = MAXTEXT_WHEEL_ATTESTATION.read_bytes()
    wheel_attestation = json.loads(wheel_attestation_raw)
    expected_wheel_attestation = {
        "schema": "moe-sae.r1.maxtext-wheel-attestation.v1",
        "source_date_epoch": 1_784_735_802,
        "wheel_name": MAXTEXT_WHEEL.name,
        "wheel_sha256": sha256(MAXTEXT_WHEEL),
        "wheel_size": MAXTEXT_WHEEL.stat().st_size,
    }
    expected_wheel_raw = (
        json.dumps(
            expected_wheel_attestation,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        + b"\n"
    )
    if (
        wheel_attestation != expected_wheel_attestation
        or wheel_attestation_raw != expected_wheel_raw
    ):
        raise RuntimeError("retained patched MaxText wheel identity changed")
    patch_attestation_raw = MAXTEXT_PATCH_ATTESTATION.read_bytes()
    patch_attestation = json.loads(patch_attestation_raw)
    expected_patch_attestation = {
        "base_commit": MAXTEXT_REVISION,
        "patch_sha256": list(MAXTEXT_PATCH_SHA256),
        "post_apply_sha256": [
            list(item) for item in MAXTEXT_POST_APPLY_SHA256
        ],
        "schema": "maxtext-r1-patch-series-attestation-v1",
    }
    expected_patch_raw = (
        json.dumps(
            expected_patch_attestation,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        + b"\n"
    )
    if (
        patch_attestation != expected_patch_attestation
        or patch_attestation_raw != expected_patch_raw
    ):
        raise RuntimeError("MaxText patch attestation changed")
    for source_path, expected_sha256 in MAXTEXT_POST_APPLY_SHA256:
        installed_relative = source_path.removeprefix("src/")
        installed_path = Path(
            maxtext_distribution.locate_file(installed_relative)
        ).resolve(strict=True)
        try:
            installed_path.relative_to(ENVIRONMENT)
        except ValueError as error:
            raise RuntimeError(
                "patched MaxText module escapes the sealed environment"
            ) from error
        if sha256(installed_path) != expected_sha256:
            raise RuntimeError(
                f"installed patched MaxText hash changed: {installed_relative}"
            )
    maxtext_base_yml = Path(
        maxtext_distribution.locate_file("maxtext/configs/base.yml")
    ).resolve(strict=True)
    try:
        maxtext_base_yml.relative_to(ENVIRONMENT)
    except ValueError as error:
        raise RuntimeError(
            "installed MaxText base configuration escapes the runtime"
        ) from error

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
    for module_name in (
        "jax_prefix_restore",
        "maxtext_patch_series",
        "maxtext_prefix_runtime",
        "prefix_benchmark",
        "prefix_benchmark_runner",
    ):
        module = import_module(f"moe_sae.r1.{module_name}")
        module_file = getattr(module, "__file__", None)
        if not isinstance(module_file, str):
            raise RuntimeError(f"runtime module has no file identity: {module_name}")
        installed_module = Path(module_file).resolve(strict=True)
        source_module = SOURCE / f"src/moe_sae/r1/{module_name}.py"
        try:
            installed_module.relative_to(ENVIRONMENT)
        except ValueError as error:
            raise RuntimeError(
                f"runtime module escapes sealed environment: {module_name}"
            ) from error
        if sha256(installed_module) != sha256(source_module):
            raise RuntimeError(
                f"installed runtime module differs from source: {module_name}"
            )
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
        "maxtext_base_yml_sha256": sha256(maxtext_base_yml),
        "maxtext_patch_attestation_sha256": sha256(MAXTEXT_PATCH_ATTESTATION),
        "maxtext_patched_files": {
            path: digest for path, digest in MAXTEXT_POST_APPLY_SHA256
        },
        "maxtext_wheel_sha256": sha256(MAXTEXT_WHEEL),
        "maxtext_wheel_attestation_sha256": sha256(MAXTEXT_WHEEL_ATTESTATION),
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
