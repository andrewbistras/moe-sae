#!/usr/bin/env bash
# Image-bake mutation only. This file is never installed as a boot hook.

set -euo pipefail
IFS=$'\n\t'
umask 077

readonly PREFIX=/opt/moe-sae-r1
readonly SOURCE_DIR="$PREFIX/source"
readonly ENV_DIR="$PREFIX/runtime"
readonly UV_VERSION=0.11.14
readonly UV_ARCHIVE_URL=https://github.com/astral-sh/uv/releases/download/0.11.14/uv-x86_64-unknown-linux-gnu.tar.gz
readonly UV_ARCHIVE_SHA256=f3b623eb0e6141a7053d571d59a0bdc341e0f238ea8f5f0b4815ddbec9a2a296
readonly UV_BINARY_SHA256=b5cbc3a3f35debad0b4770811efd190bcf460b654114d6a3f71e0ce298468e5d
readonly UV_ARCHIVE_MEMBER=uv-x86_64-unknown-linux-gnu/uv
readonly PYTHON_VERSION=3.12.11
readonly RUNTIME_USER=moe_sae
readonly RUNTIME_UID=20001
readonly RUNTIME_GID=20001
readonly PATH=/usr/sbin:/usr/bin:/sbin:/bin
export PATH

die() {
  printf '[r1-image-bake] ERROR: %s\n' "$*" >&2
  exit 1
}

sha256_file() {
  sha256sum "$1" | awk '{print $1}'
}

[[ "${BUILD_TIME_ONLY:-}" == 1 ]] || die "provisioner is build-time only"
[[ "$EUID" == 0 ]] || die "provisioner must run as root"
[[ "${RUNTIME_ARCHIVE_SHA256:-}" =~ ^[0-9a-f]{64}$ ]] || \
  die "RUNTIME_ARCHIVE_SHA256 is invalid"
[[ "${IMAGE_RECIPE_SHA256:-}" =~ ^[0-9a-f]{64}$ ]] || \
  die "IMAGE_RECIPE_SHA256 is invalid"
[[ "${PROVISIONER_SHA256:-}" =~ ^[0-9a-f]{64}$ ]] || \
  die "PROVISIONER_SHA256 is invalid"
[[ "${BASE_IMAGE_ID:-}" =~ ^[1-9][0-9]*$ ]] || die "BASE_IMAGE_ID is invalid"
[[ "${BASE_IMAGE_SELF_LINK:-}" =~ ^https://www\.googleapis\.com/compute/v1/projects/ubuntu-os-accelerator-images/global/images/ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e-v[0-9]{8}$ ]] || \
  die "BASE_IMAGE_SELF_LINK is not an exact dated v6e Ubuntu 22.04 image"

# The exact base image must already contain this small build tool surface. The
# provisioner never consults a moving OS package repository.
for command in awk chmod chown curl find git groupadd install mktemp sed sha256sum systemctl systemd-analyze tar useradd; do
  command -v "$command" >/dev/null 2>&1 || die "base image lacks $command"
done
[[ -x /usr/bin/python3 ]] || die "base image lacks /usr/bin/python3"
[[ -r /etc/ssl/certs/ca-certificates.crt ]] || die "base image lacks CA roots"

# Packer uploads these as its temporary SSH user. Transfer ownership before
# measuring them so that user cannot race the root-only validation/extraction.
chown root:root \
  /tmp/moe-sae-r1-runtime.tar \
  /tmp/moe-sae-r1-image-recipe.json
chmod 0400 \
  /tmp/moe-sae-r1-runtime.tar \
  /tmp/moe-sae-r1-image-recipe.json
[[ "$(sha256_file "$0")" == "$PROVISIONER_SHA256" ]] || \
  die "executed provisioner hash mismatch"
[[ "$(sha256_file /tmp/moe-sae-r1-runtime.tar)" == "$RUNTIME_ARCHIVE_SHA256" ]] || \
  die "runtime archive hash mismatch"
[[ "$(sha256_file /tmp/moe-sae-r1-image-recipe.json)" == \
  "$IMAGE_RECIPE_SHA256" ]] || die "image recipe hash mismatch"

groupadd --system --gid "$RUNTIME_GID" "$RUNTIME_USER"
useradd \
  --system \
  --uid "$RUNTIME_UID" \
  --gid "$RUNTIME_GID" \
  --home-dir /var/lib/moe-sae-r1 \
  --shell /usr/sbin/nologin \
  "$RUNTIME_USER"

install -d -o root -g root -m 0755 "$PREFIX" "$SOURCE_DIR" /usr/local/lib/moe-sae-r1
install -d -o "$RUNTIME_USER" -g "$RUNTIME_USER" -m 0700 \
  /var/lib/moe-sae-r1 /var/cache/moe-sae-r1 /var/tmp/moe-sae-r1
[[ "$(find "$SOURCE_DIR" -mindepth 1 -print -quit)" == "" ]] || \
  die "runtime source destination is not empty"
/usr/bin/python3 -I - /tmp/moe-sae-r1-runtime.tar <<'PY'
from __future__ import annotations

import hashlib
import json
from pathlib import PurePosixPath
import re
import sys
import tarfile


MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_TOTAL_BYTES = 8 * 1024 * 1024
EXPECTED = {
    ".python-version": ("deploy/r1/runtime/.python-version", 0o644),
    "image/measure_image.py": ("deploy/r1/image/measure_image.py", 0o755),
    "image/provision.sh": ("deploy/r1/image/provision.sh", 0o755),
    "image/r1-v6e.pkr.hcl": ("deploy/r1/image/r1-v6e.pkr.hcl", 0o644),
    "pyproject.toml": ("deploy/r1/runtime/pyproject.toml", 0o644),
    "src/moe_sae/__init__.py": ("src/moe_sae/__init__.py", 0o644),
    "src/moe_sae/r1/__init__.py": ("src/moe_sae/r1/__init__.py", 0o644),
    "src/moe_sae/r1/control_store.py": ("src/moe_sae/r1/control_store.py", 0o644),
    "src/moe_sae/r1/controller.py": ("src/moe_sae/r1/controller.py", 0o644),
    "src/moe_sae/r1/flops.py": ("src/moe_sae/r1/flops.py", 0o644),
    "src/moe_sae/r1/gcs_control_store.py": (
        "src/moe_sae/r1/gcs_control_store.py",
        0o644,
    ),
    "src/moe_sae/r1/image_protocol.py": ("src/moe_sae/r1/image_protocol.py", 0o644),
    "src/moe_sae/r1/mfu.py": ("src/moe_sae/r1/mfu.py", 0o644),
    "src/moe_sae/r1/mfu_authority.py": (
        "src/moe_sae/r1/mfu_authority.py",
        0o644,
    ),
    "src/moe_sae/r1/mfu_v2.py": ("src/moe_sae/r1/mfu_v2.py", 0o644),
    "src/moe_sae/r1/preemption.py": ("src/moe_sae/r1/preemption.py", 0o644),
    "src/moe_sae/r1/protocol.py": ("src/moe_sae/r1/protocol.py", 0o644),
    "src/moe_sae/r1/runtime_entrypoint.py": (
        "src/moe_sae/r1/runtime_entrypoint.py",
        0o644,
    ),
    "systemd/moe-sae-r1.service": (
        "deploy/r1/systemd/moe-sae-r1.service",
        0o644,
    ),
    "uv.lock": ("deploy/r1/runtime/uv.lock", 0o644),
}


def strict_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


archive_path = sys.argv[1]
payloads: dict[str, bytes] = {}
with tarfile.open(archive_path, mode="r:") as archive:
    members = archive.getmembers()
    expected_names = set(EXPECTED) | {"CONTEXT-MANIFEST.json"}
    if len(members) != len(expected_names) or {item.name for item in members} != expected_names:
        raise RuntimeError("runtime archive does not contain the exact member allowlist")
    total_size = 0
    for member in members:
        if (
            not member.isreg()
            or member.linkname
            or member.pax_headers
            or member.uid != 0
            or member.gid != 0
            or member.uname
            or member.gname
            or member.mtime != 0
        ):
            raise RuntimeError(f"unsafe runtime archive member metadata: {member.name}")
        path = PurePosixPath(member.name)
        if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
            raise RuntimeError(f"unsafe runtime archive member path: {member.name}")
        expected_mode = 0o644 if member.name == "CONTEXT-MANIFEST.json" else EXPECTED[member.name][1]
        if member.mode != expected_mode:
            raise RuntimeError(f"runtime archive member mode drifted: {member.name}")
        if member.size < 0 or member.size > MAX_FILE_BYTES:
            raise RuntimeError(f"runtime archive member size is out of bounds: {member.name}")
        total_size += member.size
        if total_size > MAX_TOTAL_BYTES:
            raise RuntimeError("runtime archive payload exceeds its bound")
        stream = archive.extractfile(member)
        if stream is None:
            raise RuntimeError(f"runtime archive member is unreadable: {member.name}")
        raw = stream.read(MAX_FILE_BYTES + 1)
        if len(raw) != member.size:
            raise RuntimeError(f"runtime archive member size disagrees: {member.name}")
        payloads[member.name] = raw

manifest_raw = payloads["CONTEXT-MANIFEST.json"]
manifest = json.loads(manifest_raw.decode("utf-8"), object_pairs_hook=strict_pairs)
if set(manifest) != {"schema", "commit", "files", "production_qualified"}:
    raise RuntimeError("runtime context manifest fields drifted")
if manifest["schema"] != "moe-sae.r1.runtime-context-manifest.v1":
    raise RuntimeError("runtime context manifest schema drifted")
if re.fullmatch(r"[0-9a-f]{40}", str(manifest["commit"])) is None:
    raise RuntimeError("runtime context commit is malformed")
if manifest["production_qualified"] is not False:
    raise RuntimeError("runtime context cannot claim production qualification")
records = manifest["files"]
if not isinstance(records, list) or len(records) != len(EXPECTED):
    raise RuntimeError("runtime context manifest file inventory drifted")
seen: set[str] = set()
for record in records:
    if not isinstance(record, dict) or set(record) != {
        "source", "archive", "mode", "size", "sha256", "git_blob"
    }:
        raise RuntimeError("runtime context manifest record is malformed")
    name = record["archive"]
    if not isinstance(name, str) or name not in EXPECTED or name in seen:
        raise RuntimeError("runtime context manifest archive identity drifted")
    seen.add(name)
    source, mode = EXPECTED[name]
    raw = payloads[name]
    if (
        record["source"] != source
        or record["mode"] != f"{mode:04o}"
        or record["size"] != len(raw)
        or record["sha256"] != hashlib.sha256(raw).hexdigest()
        or re.fullmatch(r"[0-9a-f]{40}", str(record["git_blob"])) is None
    ):
        raise RuntimeError(f"runtime context manifest/content disagreement: {name}")
canonical = json.dumps(
    manifest,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=False,
).encode("utf-8") + b"\n"
if manifest_raw != canonical:
    raise RuntimeError("runtime context manifest is not canonical JSON")
PY
tar \
  --extract \
  --file /tmp/moe-sae-r1-runtime.tar \
  --directory "$SOURCE_DIR" \
  --no-same-owner \
  --no-same-permissions \
  --no-overwrite-dir
[[ "$(sha256_file /tmp/moe-sae-r1-runtime.tar)" == "$RUNTIME_ARCHIVE_SHA256" ]] || \
  die "runtime archive changed during guarded extraction"

uv_tmp=$(mktemp -d)
trap 'find "$uv_tmp" -depth -delete' EXIT
curl \
  --fail \
  --location \
  --proto '=https' \
  --show-error \
  --silent \
  --tlsv1.2 \
  "$UV_ARCHIVE_URL" \
  --output "$uv_tmp/uv.tar.gz"
[[ "$(sha256_file "$uv_tmp/uv.tar.gz")" == "$UV_ARCHIVE_SHA256" ]] || \
  die "pinned uv archive hash mismatch"
[[ "$(tar --list --gzip --file "$uv_tmp/uv.tar.gz" | sed -n '1p')" == \
  'uv-x86_64-unknown-linux-gnu/' ]] || die "uv archive root is unexpected"
[[ "$(tar --list --gzip --file "$uv_tmp/uv.tar.gz" | sed -n '2p')" == \
  'uv-x86_64-unknown-linux-gnu/uvx' ]] || die "uv archive uvx member is unexpected"
[[ "$(tar --list --gzip --file "$uv_tmp/uv.tar.gz" | sed -n '3p')" == \
  "$UV_ARCHIVE_MEMBER" ]] || die "uv archive uv member is unexpected"
[[ "$(tar --list --gzip --file "$uv_tmp/uv.tar.gz" | sed -n '4p')" == '' ]] || \
  die "uv archive contains extra members"
tar \
  --extract \
  --gzip \
  --file "$uv_tmp/uv.tar.gz" \
  --directory "$uv_tmp" \
  --no-same-owner \
  --no-same-permissions \
  "$UV_ARCHIVE_MEMBER"
[[ "$(sha256_file "$uv_tmp/$UV_ARCHIVE_MEMBER")" == "$UV_BINARY_SHA256" ]] || \
  die "pinned uv executable hash mismatch"
install -o root -g root -m 0755 "$uv_tmp/$UV_ARCHIVE_MEMBER" /usr/local/bin/uv
[[ "$(/usr/local/bin/uv --version)" == "uv $UV_VERSION" ]] || \
  die "uv version mismatch"

export UV_PYTHON_INSTALL_DIR="$PREFIX/python"
export UV_PYTHON_PREFERENCE=only-managed
export UV_PROJECT_ENVIRONMENT="$ENV_DIR"
export UV_LINK_MODE=copy
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
/usr/local/bin/uv python install "$PYTHON_VERSION"
/usr/local/bin/uv sync \
  --frozen \
  --no-dev \
  --no-editable \
  --project "$SOURCE_DIR" \
  --python "$PYTHON_VERSION"

[[ "$($ENV_DIR/bin/python -c 'import platform; print(platform.python_version())')" == \
  "$PYTHON_VERSION" ]] || die "installed Python version mismatch"
[[ "$($ENV_DIR/bin/python -c 'import jax; print(jax.__version__)')" == 0.10.2 ]] || \
  die "installed JAX version mismatch"
[[ "$($ENV_DIR/bin/python -c 'import importlib.metadata as m; print(m.version("libtpu"))')" == \
  0.0.43 ]] || die "installed libtpu version mismatch"

export BASE_IMAGE_ID BASE_IMAGE_SELF_LINK PROVISIONER_SHA256
export RUNTIME_ARCHIVE_SHA256 UV_VERSION PYTHON_VERSION
"$ENV_DIR/bin/python" - <<'PY'
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from moe_sae.r1.image_protocol import ImageRecipeV1

source = Path("/opt/moe-sae-r1/source")
recipe_path = Path("/tmp/moe-sae-r1-image-recipe.json")
raw = recipe_path.read_bytes()
recipe = ImageRecipeV1.from_json(raw)
if recipe.canonical_bytes() != raw:
    raise RuntimeError("image recipe must be exact canonical JSON bytes")
context = json.loads((source / "CONTEXT-MANIFEST.json").read_bytes())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


expected = {
    "base_image_id": int(os.environ["BASE_IMAGE_ID"]),
    "base_image_self_link": os.environ["BASE_IMAGE_SELF_LINK"],
    "repository_commit": context["commit"],
    "repository_archive_sha256": os.environ["RUNTIME_ARCHIVE_SHA256"],
    "pyproject_sha256": sha256(source / "pyproject.toml"),
    "uv_lock_sha256": sha256(source / "uv.lock"),
    "maxtext_commit": "c03f8bd307d18049fa814fc8c104bd9ec8c292aa",
    "python_version": os.environ["PYTHON_VERSION"],
    "uv_version": os.environ["UV_VERSION"],
    "packer_version": "1.15.4",
    "packer_template_sha256": sha256(source / "image/r1-v6e.pkr.hcl"),
    "provisioner_sha256": os.environ["PROVISIONER_SHA256"],
    "service_unit_name": "moe-sae-r1.service",
    "service_unit_sha256": sha256(source / "systemd/moe-sae-r1.service"),
}
for field, value in expected.items():
    if getattr(recipe, field) != value:
        raise RuntimeError(f"image recipe mismatch for {field}")
if recipe.provisioner_sha256 != sha256(source / "image/provision.sh"):
    raise RuntimeError("executed and archived provisioner identities disagree")
PY

install -o root -g root -m 0444 \
  /tmp/moe-sae-r1-image-recipe.json \
  /usr/local/lib/moe-sae-r1/image-recipe.json

install -o root -g root -m 0644 \
  "$SOURCE_DIR/systemd/moe-sae-r1.service" \
  /etc/systemd/system/moe-sae-r1.service
systemd-analyze verify /etc/systemd/system/moe-sae-r1.service
systemctl enable moe-sae-r1.service

chown -R root:root "$PREFIX" /usr/local/lib/moe-sae-r1
chmod -R go-w "$PREFIX" /usr/local/lib/moe-sae-r1
"$ENV_DIR/bin/python" "$SOURCE_DIR/image/measure_image.py"
find /tmp -maxdepth 1 -type f \
  \( -name 'moe-sae-r1-runtime.tar' -o \
     -name 'moe-sae-r1-image-recipe.json' \) -delete
