#!/usr/bin/env bash
# Read-only inventory of every controller in an existing TPU VM slice.
# Safe to invoke from zsh; the remote payload performs no writes.

set -euo pipefail
IFS=$'\n\t'

usage() {
  cat <<'EOF'
Usage: scripts/tpu/inspect_workers.sh

Required environment variables:
  GCP_PROJECT   GCP project containing the TPU VM
  TPU_NAME      existing TPU VM resource name
  TPU_ZONE      TPU VM zone

Optional environment variable:
  SSH_KEY_FILE  gcloud SSH key (default: ~/.ssh/google_compute_engine)

This command is read-only. Use its lsblk/df/findmnt output to choose an
existing host-local filesystem and set an explicit LOCAL_BUFFER_ROOT beneath
it. The dedicated /dev/shm tmpfs is also eligible when its capacity is large
enough; the production qualifier measures its real write path before use.
It intentionally does not recommend, partition, format, mount, or create a
scratch device. The later collective qualify-storage gate proves that the
same path is actually private on every worker.
EOF
}

die() {
  printf '[inspect] ERROR: %s\n' "$*" >&2
  exit 1
}

require_env() {
  local name=$1
  [[ -n "${!name:-}" ]] || die "required environment variable is unset: $name"
}

ensure_gcloud_ssh_key() {
  local key_file=$1
  gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
    --project="$GCP_PROJECT" \
    --zone="$TPU_ZONE" \
    --worker=0 \
    --ssh-key-file="$key_file" \
    --command=true \
    --quiet >/dev/null
  [[ -f "$key_file" ]] || die "gcloud did not create the expected SSH key: $key_file"
  ssh-add "$key_file" >/dev/null 2>&1 || die \
    "could not add $key_file to an SSH agent; run 'eval \"\$(ssh-agent -s)\"' and retry"
}

main() {
  if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
  fi
  (( $# == 0 )) || die "this command accepts no arguments; use environment variables"
  require_env GCP_PROJECT
  require_env TPU_NAME
  require_env TPU_ZONE
  SSH_KEY_FILE=${SSH_KEY_FILE:-$HOME/.ssh/google_compute_engine}

  command -v gcloud >/dev/null 2>&1 || die "gcloud is unavailable"
  command -v ssh-add >/dev/null 2>&1 || die "ssh-add is unavailable"
  [[ "$GCP_PROJECT" =~ ^[a-z][a-z0-9-]{4,61}[a-z0-9]$ ]] || die "invalid GCP_PROJECT"
  [[ "$TPU_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9-]*$ ]] || die "invalid TPU_NAME"
  [[ "$TPU_ZONE" =~ ^[a-z0-9-]+$ ]] || die "invalid TPU_ZONE"

  local remote_command
  remote_command=$(cat <<'REMOTE_BASH'
set -eu
metadata='http://metadata.google.internal/computeMetadata/v1'
header='Metadata-Flavor: Google'
printf '\n===== worker %s =====\n' "$(hostname -s 2>/dev/null || hostname)"
printf 'instance: '
curl -fsS -H "$header" "$metadata/instance/name"
printf '\nzone: '
curl -fsS -H "$header" "$metadata/instance/zone" | awk -F/ '{print $NF}'
printf 'service account: '
curl -fsS -H "$header" "$metadata/instance/service-accounts/default/email"
printf '\nuser identity: '
id
printf '\n\nBlock devices (read-only inventory):\n'
lsblk -e 7 -o NAME,TYPE,SIZE,FSTYPE,FSAVAIL,MOUNTPOINTS,MODEL
printf '\nMounted filesystems:\n'
findmnt -rn -o TARGET,SOURCE,FSTYPE,OPTIONS
printf '\nCapacity of existing local mount roots:\n'
df -hT / /home /mnt /dev/shm 2>/dev/null || true
printf '\nHost memory available to tmpfs-backed scratch:\n'
awk '/^MemAvailable:/ {print}' /proc/meminfo
printf '\nExisting directories immediately below /mnt and /media:\n'
find /mnt /media -mindepth 1 -maxdepth 2 -type d -printf '%p\n' 2>/dev/null | sort || true
REMOTE_BASH
)

  ensure_gcloud_ssh_key "$SSH_KEY_FILE"
  gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
    --project="$GCP_PROJECT" \
    --zone="$TPU_ZONE" \
    --worker=all \
    --ssh-key-file="$SSH_KEY_FILE" \
    --command="bash -c $(printf '%q' "$remote_command")" \
    --quiet
}

main "$@"
