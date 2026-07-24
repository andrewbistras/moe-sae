#!/usr/bin/env bash
#
# Install one exact, region-local DeepSeek-R1 runtime on a legacy Cloud TPU VM.
#
# This is a qualification-only bridge for v6e-64 capacity obtained through the
# Cloud TPU queued-resource API.  It performs one generation-pinned GCS GET on
# a fresh worker, verifies the complete compressed object before extraction,
# validates every tar member, atomically installs only /opt/moe-sae-r1, and
# publishes a host-specific receipt last.  A valid receipt makes reruns local
# and idempotent: no GCS request is made.

set -Eeuo pipefail
IFS=$'\n\t'
umask 077

readonly EUW4_RUNTIME_URI="gs://neural-dragnet-r1-euw4-655343590563/runtime/6129738baad8d2bb4060c6e19b8a9227341afe9c795304f70fdf3206155f45c9.tar.gz"
readonly EUW4_RUNTIME_GENERATION="1784862130885035"
readonly USE1_RUNTIME_URI="gs://neural-dragnet-r1-use1-655343590563/runtime/6129738baad8d2bb4060c6e19b8a9227341afe9c795304f70fdf3206155f45c9.tar.gz"
readonly USE1_RUNTIME_GENERATION="1784865436897245"
readonly RUNTIME_SHA256="6129738baad8d2bb4060c6e19b8a9227341afe9c795304f70fdf3206155f45c9"
readonly RUNTIME_SIZE_BYTES="1378895492"
readonly RUNTIME_MEMBER_COUNT="69566"
readonly RUNTIME_COMMIT="0b01bc2b76e375271249eb6f7e0499d5b016a1da"
readonly EXPECTED_SERVICE_ACCOUNT="655343590563-compute@developer.gserviceaccount.com"
readonly METADATA_ROOT="http://metadata.google.internal/computeMetadata/v1"
readonly RUNTIME_ROOT="/opt/moe-sae-r1"
readonly STATE_ROOT="/var/lib/moe-sae-r1"
readonly CACHE_ROOT="/var/cache/moe-sae-r1"
readonly RECEIPT_PATH="$STATE_ROOT/runtime-install.json"
readonly PENDING_PATH="$STATE_ROOT/runtime-install.pending.json"
readonly RUNTIME_USER="moe_sae"
readonly RUNTIME_UID="20001"
readonly RUNTIME_GID="20001"
readonly MAX_EXPANDED_BYTES="$((12 * 1024 * 1024 * 1024))"

archive_path=''
staging_root=''
RUNTIME_URI=''
RUNTIME_GENERATION=''

log() {
  printf '%s [legacy-v6e-bootstrap] %s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

die() {
  log "ERROR: $*" >&2
  exit 1
}

cleanup() {
  if [[ -n "$archive_path" && -f "$archive_path" ]]; then
    rm -f -- "$archive_path"
  fi
  if [[ -n "$staging_root" && -d "$staging_root" ]]; then
    rm -rf --one-file-system -- "$staging_root"
  fi
  if [[ -e "$PENDING_PATH" && ! -e "$RUNTIME_ROOT" ]]; then
    rm -f -- "$PENDING_PATH"
    sync -f "$STATE_ROOT" 2>/dev/null || true
  fi
}

trap cleanup EXIT

require_command() {
  command -v "$1" >/dev/null 2>&1 ||
    die "required command is unavailable: $1"
}

cleanup_abandoned_staging() {
  local candidate
  for candidate in /opt/.moe-sae-r1-runtime.*.tar.gz; do
    [[ -e "$candidate" || -L "$candidate" ]] || continue
    rm -f -- "$candidate"
  done
  for candidate in /opt/.moe-sae-r1-stage.*; do
    [[ -e "$candidate" || -L "$candidate" ]] || continue
    rm -rf --one-file-system -- "$candidate"
  done
}

sha256_file() {
  sha256sum "$1" | awk '{print $1}'
}

ensure_runtime_identity() {
  local passwd_record
  local group_record

  passwd_record="$(getent passwd "$RUNTIME_USER" || true)"
  group_record="$(getent group "$RUNTIME_USER" || true)"
  if [[ -z "$group_record" ]]; then
    getent group "$RUNTIME_GID" >/dev/null 2>&1 &&
      die "GID $RUNTIME_GID is already owned by another group"
    groupadd --system --gid "$RUNTIME_GID" "$RUNTIME_USER"
  else
    [[ "$(cut -d: -f3 <<<"$group_record")" == "$RUNTIME_GID" ]] ||
      die "$RUNTIME_USER group has an unexpected GID"
  fi
  if [[ -z "$passwd_record" ]]; then
    getent passwd "$RUNTIME_UID" >/dev/null 2>&1 &&
      die "UID $RUNTIME_UID is already owned by another user"
    useradd \
      --system \
      --uid "$RUNTIME_UID" \
      --gid "$RUNTIME_GID" \
      --home-dir "$STATE_ROOT" \
      --shell /usr/sbin/nologin \
      "$RUNTIME_USER"
  else
    [[ "$(cut -d: -f3 <<<"$passwd_record")" == "$RUNTIME_UID" ]] ||
      die "$RUNTIME_USER has an unexpected UID"
    [[ "$(cut -d: -f4 <<<"$passwd_record")" == "$RUNTIME_GID" ]] ||
      die "$RUNTIME_USER has an unexpected primary GID"
    [[ "$(cut -d: -f6 <<<"$passwd_record")" == "$STATE_ROOT" ]] ||
      die "$RUNTIME_USER has an unexpected home directory"
    [[ "$(cut -d: -f7 <<<"$passwd_record")" == /usr/sbin/nologin ]] ||
      die "$RUNTIME_USER has an unexpected login shell"
  fi
  [[ "$(id -u "$RUNTIME_USER")" == "$RUNTIME_UID" ]] ||
    die "runtime user lookup changed"
}

verify_metadata_identity() {
  local actual_service_account
  local actual_zone
  local scopes

  actual_service_account="$(
    curl \
      --fail \
      --silent \
      --show-error \
      --connect-timeout 3 \
      --max-time 10 \
      -H 'Metadata-Flavor: Google' \
      "$METADATA_ROOT/instance/service-accounts/default/email"
  )"
  [[ "$actual_service_account" == "$EXPECTED_SERVICE_ACCOUNT" ]] ||
    die "metadata service account is not the admitted default Compute identity"
  scopes="$(
    curl \
      --fail \
      --silent \
      --show-error \
      --connect-timeout 3 \
      --max-time 10 \
      -H 'Metadata-Flavor: Google' \
      "$METADATA_ROOT/instance/service-accounts/default/scopes"
  )"
  grep -Fxq 'https://www.googleapis.com/auth/cloud-platform' <<<"$scopes" ||
    die "worker does not have the required cloud-platform OAuth scope"
  actual_zone="$(
    curl \
      --fail \
      --silent \
      --show-error \
      --connect-timeout 3 \
      --max-time 10 \
      -H 'Metadata-Flavor: Google' \
      "$METADATA_ROOT/instance/zone"
  )"
  case "${actual_zone##*/}" in
    europe-west4-a)
      RUNTIME_URI="$EUW4_RUNTIME_URI"
      RUNTIME_GENERATION="$EUW4_RUNTIME_GENERATION"
      ;;
    us-east1-d)
      RUNTIME_URI="$USE1_RUNTIME_URI"
      RUNTIME_GENERATION="$USE1_RUNTIME_GENERATION"
      ;;
    *) die "worker zone has no exact regional runtime replica" ;;
  esac
}

verify_runtime_tree() {
  [[ -d "$RUNTIME_ROOT" && ! -L "$RUNTIME_ROOT" ]] ||
    die "runtime root is missing or is a symbolic link"
  [[ -x "$RUNTIME_ROOT/runtime/bin/python" ]] ||
    die "runtime Python is absent"
  /usr/sbin/runuser --user "$RUNTIME_USER" -- \
    "$RUNTIME_ROOT/runtime/bin/python" -I - "$RUNTIME_COMMIT" <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import sys

import jax
import maxtext
from moe_sae.r1.runtime_entrypoint import EXPECTED_UID

expected_commit = sys.argv[1]
if os.getuid() != EXPECTED_UID or EXPECTED_UID != 20001:
    raise RuntimeError("runtime UID boundary changed")
context_path = Path("/opt/moe-sae-r1/source/CONTEXT-MANIFEST.json")
context = json.loads(context_path.read_bytes())
if (
    not isinstance(context, dict)
    or context.get("schema") != "moe-sae.r1.runtime-context-manifest.v1"
    or context.get("commit") != expected_commit
    or context.get("production_qualified") is not False
):
    raise RuntimeError("installed runtime context identity changed")
if re.fullmatch(r"0\.10\.2", jax.__version__) is None:
    raise RuntimeError("installed JAX version changed")
if maxtext.__file__ is None:
    raise RuntimeError("installed MaxText module has no origin")
PY
}

receipt_is_exact() {
  [[ -f "$RECEIPT_PATH" && ! -L "$RECEIPT_PATH" ]] || return 1
  local os_release_sha256
  local kernel_release
  os_release_sha256="$(sha256_file /etc/os-release)"
  kernel_release="$(uname -r)"
  /usr/bin/python3 -I - \
    "$RECEIPT_PATH" \
    "$RUNTIME_URI" \
    "$RUNTIME_GENERATION" \
    "$RUNTIME_SHA256" \
    "$RUNTIME_SIZE_BYTES" \
    "$RUNTIME_MEMBER_COUNT" \
    "$RUNTIME_COMMIT" \
    "$os_release_sha256" \
    "$kernel_release" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import re
import sys


def strict_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise RuntimeError("duplicate receipt key")
        value[key] = item
    return value


(
    receipt_path,
    source_uri,
    generation,
    archive_sha256,
    archive_size,
    member_count,
    repository_commit,
    os_release_sha256,
    kernel_release,
) = sys.argv[1:]
raw = Path(receipt_path).read_bytes()
value = json.loads(raw, object_pairs_hook=strict_pairs)
expected_keys = {
    "schema",
    "source_uri",
    "source_generation",
    "archive_sha256",
    "archive_size_bytes",
    "archive_member_count",
    "repository_commit",
    "runtime_root",
    "runtime_uid",
    "host_os_release_sha256",
    "host_kernel_release",
    "installed_at_unix_ns",
}
if not isinstance(value, dict) or set(value) != expected_keys:
    raise RuntimeError("runtime receipt fields changed")
expected = {
    "schema": "moe-sae.r1.legacy-runtime-install.v1",
    "source_uri": source_uri,
    "source_generation": int(generation),
    "archive_sha256": archive_sha256,
    "archive_size_bytes": int(archive_size),
    "archive_member_count": int(member_count),
    "repository_commit": repository_commit,
    "runtime_root": "/opt/moe-sae-r1",
    "runtime_uid": 20001,
    "host_os_release_sha256": os_release_sha256,
    "host_kernel_release": kernel_release,
}
for key, expected_value in expected.items():
    if value.get(key) != expected_value:
        raise RuntimeError(f"runtime receipt mismatch: {key}")
installed_at = value.get("installed_at_unix_ns")
if type(installed_at) is not int or installed_at <= 0:
    raise RuntimeError("runtime receipt timestamp is invalid")
canonical = (
    json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    + b"\n"
)
if raw != canonical:
    raise RuntimeError("runtime receipt is not canonical JSON")
if re.fullmatch(r"[0-9a-f]{64}", archive_sha256) is None:
    raise RuntimeError("runtime receipt SHA-256 is malformed")
PY
}

pending_is_exact() {
  [[ -f "$PENDING_PATH" && ! -L "$PENDING_PATH" ]] || return 1
  /usr/bin/python3 -I - \
    "$PENDING_PATH" \
    "$RUNTIME_URI" \
    "$RUNTIME_GENERATION" \
    "$RUNTIME_SHA256" \
    "$RUNTIME_SIZE_BYTES" \
    "$RUNTIME_MEMBER_COUNT" \
    "$RUNTIME_COMMIT" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sys


def strict_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise RuntimeError("duplicate pending-install key")
        value[key] = item
    return value


(
    pending_path,
    source_uri,
    generation,
    archive_sha256,
    archive_size,
    member_count,
    repository_commit,
) = sys.argv[1:]
raw = Path(pending_path).read_bytes()
value = json.loads(raw, object_pairs_hook=strict_pairs)
expected = {
    "schema": "moe-sae.r1.legacy-runtime-install-pending.v1",
    "phase": "validated-before-rename",
    "source_uri": source_uri,
    "source_generation": int(generation),
    "archive_sha256": archive_sha256,
    "archive_size_bytes": int(archive_size),
    "archive_member_count": int(member_count),
    "repository_commit": repository_commit,
    "runtime_root": "/opt/moe-sae-r1",
    "runtime_uid": 20001,
}
if type(value) is not dict or value != expected:
    raise RuntimeError("pending-install journal identity changed")
canonical = (
    json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    + b"\n"
)
if raw != canonical:
    raise RuntimeError("pending-install journal is not canonical JSON")
PY
}

publish_pending() {
  local temporary_pending
  temporary_pending="$(mktemp "$STATE_ROOT/.runtime-install.pending.XXXXXX")"
  /usr/bin/python3 -I - \
    "$temporary_pending" \
    "$PENDING_PATH" \
    "$RUNTIME_URI" \
    "$RUNTIME_GENERATION" \
    "$RUNTIME_SHA256" \
    "$RUNTIME_SIZE_BYTES" \
    "$RUNTIME_MEMBER_COUNT" \
    "$RUNTIME_COMMIT" <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

(
    temporary,
    destination,
    source_uri,
    generation,
    archive_sha256,
    archive_size,
    member_count,
    repository_commit,
) = sys.argv[1:]
value = {
    "schema": "moe-sae.r1.legacy-runtime-install-pending.v1",
    "phase": "validated-before-rename",
    "source_uri": source_uri,
    "source_generation": int(generation),
    "archive_sha256": archive_sha256,
    "archive_size_bytes": int(archive_size),
    "archive_member_count": int(member_count),
    "repository_commit": repository_commit,
    "runtime_root": "/opt/moe-sae-r1",
    "runtime_uid": 20001,
}
raw = (
    json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    + b"\n"
)
path = Path(temporary)
with path.open("wb") as stream:
    stream.write(raw)
    stream.flush()
    os.fsync(stream.fileno())
os.chmod(path, 0o400)
os.replace(path, destination)
directory = os.open(str(Path(destination).parent), os.O_RDONLY | os.O_DIRECTORY)
try:
    os.fsync(directory)
finally:
    os.close(directory)
PY
}

remove_exact_pending() {
  pending_is_exact || die "pending-install journal changed before removal"
  rm -f -- "$PENDING_PATH"
  sync -f "$STATE_ROOT"
}

validate_and_extract_archive() {
  local source_archive="$1"
  local destination="$2"
  /usr/bin/python3 -I - \
    "$source_archive" \
    "$destination" \
    "$RUNTIME_MEMBER_COUNT" \
    "$MAX_EXPANDED_BYTES" <<'PY'
from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
import posixpath
import stat
import sys
import tarfile


archive_path = Path(sys.argv[1])
destination = Path(sys.argv[2])
expected_count = int(sys.argv[3])
maximum_total = int(sys.argv[4])
allowed_prefixes = (
    "opt/moe-sae-r1",
    "usr/local/lib/moe-sae-r1",
)
allowed_parents = {
    "opt",
    "usr",
    "usr/local",
    "usr/local/lib",
}


def normalized_member_path(name: str) -> str:
    if "\x00" in name or name.startswith("/"):
        raise RuntimeError(f"unsafe absolute archive member: {name!r}")
    path = PurePosixPath(name)
    if not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise RuntimeError(f"unsafe archive member path: {name!r}")
    normalized = posixpath.normpath(name)
    if normalized != name.rstrip("/"):
        raise RuntimeError(f"non-canonical archive member path: {name!r}")
    if (
        normalized not in allowed_parents
        and not any(
            normalized == prefix or normalized.startswith(prefix + "/")
            for prefix in allowed_prefixes
        )
    ):
        raise RuntimeError(f"archive member escapes the allowlist: {name!r}")
    return normalized


with tarfile.open(archive_path, mode="r:gz") as archive:
    members = archive.getmembers()
    if len(members) != expected_count:
        raise RuntimeError(
            f"archive member count is {len(members)}, expected {expected_count}"
        )
    normalized_names: set[str] = set()
    symlink_names: set[str] = set()
    total_size = 0
    selected: list[tarfile.TarInfo] = []
    for member in members:
        name = normalized_member_path(member.name)
        if name in normalized_names:
            raise RuntimeError(f"duplicate archive member: {name!r}")
        normalized_names.add(name)
        if member.islnk() or member.ischr() or member.isblk() or member.isfifo():
            raise RuntimeError(f"unsafe archive member type: {name!r}")
        if not (member.isdir() or member.isreg() or member.issym()):
            raise RuntimeError(f"unsupported archive member type: {name!r}")
        if getattr(member, "sparse", None):
            raise RuntimeError(f"sparse archive member is forbidden: {name!r}")
        if member.mode & (stat.S_ISUID | stat.S_ISGID | stat.S_ISVTX):
            raise RuntimeError(f"privileged archive mode: {name!r}")
        if member.isreg():
            if member.size < 0 or member.size > maximum_total:
                raise RuntimeError(f"archive file size is out of bounds: {name!r}")
            total_size += member.size
            if total_size > maximum_total:
                raise RuntimeError("expanded archive exceeds its byte bound")
        elif member.size != 0:
            raise RuntimeError(f"non-file archive member has a payload: {name!r}")
        if member.issym():
            linkname = member.linkname
            if not linkname or "\x00" in linkname:
                raise RuntimeError(f"empty or malformed symlink: {name!r}")
            if linkname.startswith("/"):
                resolved = posixpath.normpath(linkname)
                if not (
                    resolved == "/opt/moe-sae-r1"
                    or resolved.startswith("/opt/moe-sae-r1/")
                ):
                    raise RuntimeError(f"absolute symlink escapes runtime: {name!r}")
            else:
                resolved = posixpath.normpath(
                    posixpath.join(posixpath.dirname(name), linkname)
                )
                if not (
                    resolved == "opt/moe-sae-r1"
                    or resolved.startswith("opt/moe-sae-r1/")
                ):
                    raise RuntimeError(f"relative symlink escapes runtime: {name!r}")
            symlink_names.add(name)
        if name == "opt" or name == "opt/moe-sae-r1" or name.startswith(
            "opt/moe-sae-r1/"
        ):
            selected.append(member)

    for name in normalized_names:
        parent = PurePosixPath(name).parent
        while parent != PurePosixPath("."):
            if str(parent) in symlink_names:
                raise RuntimeError(
                    f"archive member descends through symlink {str(parent)!r}"
                )
            parent = parent.parent
    if "opt/moe-sae-r1" not in normalized_names:
        raise RuntimeError("archive has no runtime root")

    def root_owned_filter(
        member: tarfile.TarInfo,
        path: str,
    ) -> tarfile.TarInfo:
        del path
        return member.replace(
            uid=None,
            gid=None,
            uname=None,
            gname=None,
            deep=False,
        )

    archive.extractall(
        destination,
        members=selected,
        filter=root_owned_filter,
    )

runtime = destination / "opt/moe-sae-r1"
if not runtime.is_dir() or runtime.is_symlink():
    raise RuntimeError("extracted runtime root is invalid")
python = runtime / "runtime/bin/python"
if python.is_symlink():
    linkname = os.readlink(python)
    if linkname.startswith("/opt/moe-sae-r1/"):
        python_payload = runtime / linkname.removeprefix("/opt/moe-sae-r1/")
    else:
        python_payload = python.parent / linkname
else:
    python_payload = python
if not python_payload.is_file():
    raise RuntimeError("extracted runtime Python is absent")
PY
}

publish_receipt() {
  local os_release_sha256
  local kernel_release
  local temporary_receipt
  os_release_sha256="$(sha256_file /etc/os-release)"
  kernel_release="$(uname -r)"
  temporary_receipt="$(mktemp "$STATE_ROOT/.runtime-install.XXXXXX")"
  /usr/bin/python3 -I - \
    "$temporary_receipt" \
    "$RECEIPT_PATH" \
    "$RUNTIME_URI" \
    "$RUNTIME_GENERATION" \
    "$RUNTIME_SHA256" \
    "$RUNTIME_SIZE_BYTES" \
    "$RUNTIME_MEMBER_COUNT" \
    "$RUNTIME_COMMIT" \
    "$os_release_sha256" \
    "$kernel_release" <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

(
    temporary,
    destination,
    source_uri,
    generation,
    archive_sha256,
    archive_size,
    member_count,
    repository_commit,
    os_release_sha256,
    kernel_release,
) = sys.argv[1:]
value = {
    "schema": "moe-sae.r1.legacy-runtime-install.v1",
    "source_uri": source_uri,
    "source_generation": int(generation),
    "archive_sha256": archive_sha256,
    "archive_size_bytes": int(archive_size),
    "archive_member_count": int(member_count),
    "repository_commit": repository_commit,
    "runtime_root": "/opt/moe-sae-r1",
    "runtime_uid": 20001,
    "host_os_release_sha256": os_release_sha256,
    "host_kernel_release": kernel_release,
    "installed_at_unix_ns": time.time_ns(),
}
raw = (
    json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    + b"\n"
)
path = Path(temporary)
with path.open("wb") as stream:
    stream.write(raw)
    stream.flush()
    os.fsync(stream.fileno())
os.chmod(path, 0o444)
os.replace(path, destination)
directory = os.open(str(Path(destination).parent), os.O_RDONLY | os.O_DIRECTORY)
try:
    os.fsync(directory)
finally:
    os.close(directory)
PY
}

main() {
  [[ "$#" == 0 ]] || die "this fixed bootstrap accepts no arguments"
  [[ "$EUID" == 0 ]] || die "bootstrap must run as root"
  for command in \
    awk chown chmod cut curl date flock getent gcloud grep groupadd gzip \
    install mktemp mv rm sha256sum stat sync uname useradd; do
    require_command "$command"
  done
  [[ -x /usr/bin/python3 ]] || die "system Python is unavailable"
  [[ -x /usr/sbin/runuser ]] || die "runuser is unavailable"
  [[ -r /etc/os-release ]] || die "OS release identity is unavailable"

  install -d -o root -g root -m 0755 /run/lock
  exec 9>/run/lock/moe-sae-r1-runtime-install.lock
  flock -n 9 || die "another runtime bootstrap owns this worker"

  verify_metadata_identity
  ensure_runtime_identity
  install -d -o "$RUNTIME_USER" -g "$RUNTIME_USER" -m 0700 \
    "$STATE_ROOT" "$CACHE_ROOT" "$STATE_ROOT/qualification"
  cleanup_abandoned_staging

  if receipt_is_exact; then
    verify_runtime_tree
    if [[ -e "$PENDING_PATH" ]]; then
      remove_exact_pending
    fi
    log "exact runtime receipt and local import probe already pass; no GCS read"
    exit 0
  fi
  [[ ! -e "$RECEIPT_PATH" ]] ||
    die "an invalid runtime receipt already exists"
  if pending_is_exact; then
    if [[ -e "$RUNTIME_ROOT" ]]; then
      verify_runtime_tree
      publish_receipt
      receipt_is_exact || die "recovered runtime receipt did not revalidate"
      remove_exact_pending
      log "recovered exact runtime after interrupted publish-last receipt"
      exit 0
    fi
    remove_exact_pending
    log "discarded an exact pre-rename journal and restarted cleanly"
  fi
  [[ ! -e "$PENDING_PATH" ]] ||
    die "an invalid pending-install journal already exists"
  [[ ! -e "$RUNTIME_ROOT" ]] ||
    die "runtime root exists without an exact publish-last receipt"

  archive_path="$(mktemp /opt/.moe-sae-r1-runtime.XXXXXX.tar.gz)"
  staging_root="$(mktemp -d /opt/.moe-sae-r1-stage.XXXXXX)"
  log "downloading one exact region-local runtime generation"
  gcloud storage cp \
    --quiet \
    "${RUNTIME_URI}#${RUNTIME_GENERATION}" \
    "$archive_path"
  [[ "$(stat -c %s "$archive_path")" == "$RUNTIME_SIZE_BYTES" ]] ||
    die "downloaded runtime size mismatch"
  [[ "$(sha256_file "$archive_path")" == "$RUNTIME_SHA256" ]] ||
    die "downloaded runtime SHA-256 mismatch"
  gzip --test "$archive_path" || die "downloaded runtime gzip stream is invalid"

  log "validating all archive members before extraction"
  validate_and_extract_archive "$archive_path" "$staging_root"
  [[ -d "$staging_root/opt/moe-sae-r1" ]] ||
    die "validated staging runtime is absent"
  chown -R root:root "$staging_root/opt/moe-sae-r1"
  chmod -R u=rwX,go=rX "$staging_root/opt/moe-sae-r1"
  publish_pending
  pending_is_exact || die "pending-install journal did not revalidate"
  mv "$staging_root/opt/moe-sae-r1" "$RUNTIME_ROOT"
  sync -f "$RUNTIME_ROOT"

  verify_runtime_tree
  publish_receipt
  receipt_is_exact || die "publish-last runtime receipt did not revalidate"
  remove_exact_pending
  rm -f -- "$archive_path"
  archive_path=''
  rm -rf --one-file-system -- "$staging_root"
  staging_root=''
  log "exact runtime installed and receipt published last"
}

main "$@"
