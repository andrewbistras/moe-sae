#!/usr/bin/env bash
# Deploy one exact, clean Git commit and bootstrap every TPU VM worker.
# Safe to invoke from zsh; execution is pinned to Bash by this shebang.

set -euo pipefail
IFS=$'\n\t'
umask 077

readonly DEFAULT_UV_VERSION=0.11.14
DEPLOY_TEMP_DIR=''

usage() {
  cat <<'EOF'
Usage: scripts/tpu/deploy_workers.sh

Required environment variables (none are secrets):
  GCP_PROJECT                 GCP project containing the TPU VM
  TPU_NAME                    existing TPU VM resource name
  TPU_ZONE                    TPU VM zone
  TPU_SERVICE_ACCOUNT         expected worker metadata service-account email
  PROJECT_COMMIT              full 40-character commit currently checked out
  TPU_PROJECT_ROOT            absolute checkout path on every worker
  LOCAL_BUFFER_ROOT           explicit host-local directory on every worker
  BULK_GCS_ROOT               durable root, e.g. gs://bucket/moe-sae/gemma4_v1
  NFS_SERVER                  explicit Filestore/NFS hostname or IPv4 address
  NFS_EXPORT                  explicit export path, e.g. /moe_sae_control

Optional environment variables:
  UV_VERSION                  pinned uv version (default: 0.11.14)
  SSH_KEY_FILE                gcloud SSH key (default: ~/.ssh/google_compute_engine)
  TPU_ACCELERATOR_TYPE        exact expected type (default: v4-32)
  TPU_USE_INTERNAL_IP         1 to use private TPU VM addresses (default: 0);
                              required for v6e-16

Before setting LOCAL_BUFFER_ROOT, run scripts/tpu/inspect_workers.sh. This
script never creates a TPU, chooses a device, partitions a disk, or formats a
filesystem. It refuses a dirty checkout and deploys only PROJECT_COMMIT via a
self-contained Git bundle, so no GitHub credential reaches the workers.
EOF
}

log() {
  printf '[deploy] %s\n' "$*"
}

die() {
  printf '[deploy] ERROR: %s\n' "$*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command is unavailable: $1"
}

require_env() {
  local name=$1
  [[ -n "${!name:-}" ]] || die "required environment variable is unset: $name"
}

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

cleanup() {
  if [[ -n "${DEPLOY_TEMP_DIR:-}" && -d "$DEPLOY_TEMP_DIR" ]]; then
    rm -rf -- "$DEPLOY_TEMP_DIR"
  fi
}

validate_absolute_path() {
  local name=$1
  local value=$2
  [[ "$value" =~ ^/[A-Za-z0-9._/@+-]+$ ]] || \
    die "$name must be an absolute path containing only safe characters"
}

validate_inputs() {
  [[ "$GCP_PROJECT" =~ ^[a-z][a-z0-9-]{4,61}[a-z0-9]$ ]] || die "invalid GCP_PROJECT"
  [[ "$TPU_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9-]*$ ]] || die "invalid TPU_NAME"
  [[ "$TPU_ZONE" =~ ^[a-z0-9-]+$ ]] || die "invalid TPU_ZONE"
  [[ "$TPU_SERVICE_ACCOUNT" =~ ^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+$ ]] || \
    die "invalid TPU_SERVICE_ACCOUNT"
  [[ "$PROJECT_COMMIT" =~ ^[0-9a-f]{40}$ ]] || \
    die "PROJECT_COMMIT must be a lowercase, full 40-character Git SHA"
  validate_absolute_path TPU_PROJECT_ROOT "$TPU_PROJECT_ROOT"
  validate_absolute_path LOCAL_BUFFER_ROOT "$LOCAL_BUFFER_ROOT"
  [[ "$BULK_GCS_ROOT" =~ ^gs://[a-z0-9][a-z0-9._-]*/[A-Za-z0-9._/-]+$ ]] || \
    die "BULK_GCS_ROOT must be a non-root gs:// path"
  [[ "$BULK_GCS_ROOT" != */ ]] || die "BULK_GCS_ROOT must not end with '/'"
  [[ "$NFS_SERVER" =~ ^[A-Za-z0-9.-]+$ ]] || die "invalid NFS_SERVER"
  [[ "$NFS_EXPORT" =~ ^/[A-Za-z0-9._/-]+$ ]] || die "invalid NFS_EXPORT"
  [[ "$NFS_EXPORT" != */ ]] || die "NFS_EXPORT must not end with '/'"
  [[ "$UV_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || die "invalid UV_VERSION"
  [[ "$TPU_ACCELERATOR_TYPE" == "v4-32" || \
     "$TPU_ACCELERATOR_TYPE" == "v6e-16" ]] || \
    die "TPU_ACCELERATOR_TYPE must be exactly v4-32 or v6e-16"
  [[ "$TPU_USE_INTERNAL_IP" == "0" || "$TPU_USE_INTERNAL_IP" == "1" ]] || \
    die "TPU_USE_INTERNAL_IP must be exactly 0 or 1"
  if [[ "$TPU_ACCELERATOR_TYPE" == "v6e-16" ]]; then
    [[ "$TPU_USE_INTERNAL_IP" == "1" ]] || \
      die "v6e-16 deployment requires explicit TPU_USE_INTERNAL_IP=1"
  fi
}

ensure_gcloud_ssh_key() {
  local key_file=$1
  local -a ssh_command
  ssh_command=(
    gcloud compute tpus tpu-vm ssh "$TPU_NAME"
    --project="$GCP_PROJECT"
    --zone="$TPU_ZONE"
    --worker=0
    --ssh-key-file="$key_file"
    --command=true
    --quiet
  )
  if [[ "$TPU_USE_INTERNAL_IP" == "1" ]]; then
    ssh_command+=(--internal-ip)
  fi

  log "opening worker 0 once so gcloud can create/install its SSH key"
  "${ssh_command[@]}" >/dev/null
  [[ -f "$key_file" ]] || die "gcloud did not create the expected SSH key: $key_file"
  ssh-add "$key_file" >/dev/null 2>&1 || die \
    "could not add $key_file to an SSH agent; run 'eval \"\$(ssh-agent -s)\"' and retry"
}

write_config() {
  local output=$1
  local bundle_sha=$2
  local bootstrap_sha=$3
  local maxtext_commit=$4

  {
    printf 'DEPLOYMENT_SCHEMA=1\n'
    printf 'PROJECT_COMMIT=%s\n' "$PROJECT_COMMIT"
    printf 'BUNDLE_SHA256=%s\n' "$bundle_sha"
    printf 'BOOTSTRAP_SHA256=%s\n' "$bootstrap_sha"
    printf 'EXPECTED_MAXTEXT_COMMIT=%s\n' "$maxtext_commit"
    printf 'GCP_PROJECT=%s\n' "$GCP_PROJECT"
    printf 'EXPECTED_SERVICE_ACCOUNT=%s\n' "$TPU_SERVICE_ACCOUNT"
    printf 'PROJECT_ROOT=%s\n' "$TPU_PROJECT_ROOT"
    printf 'LOCAL_BUFFER_ROOT=%s\n' "$LOCAL_BUFFER_ROOT"
    printf 'BULK_GCS_ROOT=%s\n' "$BULK_GCS_ROOT"
    printf 'NFS_SERVER=%s\n' "$NFS_SERVER"
    printf 'NFS_EXPORT=%s\n' "$NFS_EXPORT"
    printf 'UV_VERSION=%s\n' "$UV_VERSION"
  } > "$output"
}

main() {
  if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
  fi
  (( $# == 0 )) || die "this command accepts no arguments; use environment variables"

  local name
  for name in \
    GCP_PROJECT TPU_NAME TPU_ZONE TPU_SERVICE_ACCOUNT PROJECT_COMMIT \
    TPU_PROJECT_ROOT LOCAL_BUFFER_ROOT BULK_GCS_ROOT NFS_SERVER NFS_EXPORT; do
    require_env "$name"
  done
  UV_VERSION=${UV_VERSION:-$DEFAULT_UV_VERSION}
  SSH_KEY_FILE=${SSH_KEY_FILE:-$HOME/.ssh/google_compute_engine}
  TPU_ACCELERATOR_TYPE=${TPU_ACCELERATOR_TYPE:-v4-32}
  TPU_USE_INTERNAL_IP=${TPU_USE_INTERNAL_IP:-0}
  validate_inputs

  require_command git
  require_command gcloud
  require_command ssh-add
  require_command awk
  if ! command -v sha256sum >/dev/null 2>&1; then
    require_command shasum
  fi

  local script_dir
  local repo_root
  local head_commit
  local checkout_status
  local maxtext_commit
  local temp_dir
  local bundle_file
  local bootstrap_file
  local config_file
  local bundle_sha
  local bootstrap_sha
  local deployment_id
  local remote_dir
  local remote_command
  local tpu_description
  local tpu_state
  local accelerator_type
  local -a gcloud_common

  script_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
  repo_root=$(git -C "$script_dir/../.." rev-parse --show-toplevel)
  head_commit=$(git -C "$repo_root" rev-parse HEAD)
  [[ "$head_commit" == "$PROJECT_COMMIT" ]] || \
    die "checkout HEAD is $head_commit, not PROJECT_COMMIT=$PROJECT_COMMIT"
  git -C "$repo_root" cat-file -e "$PROJECT_COMMIT^{commit}" || \
    die "PROJECT_COMMIT is not available in the local repository"
  checkout_status=$(git -C "$repo_root" status --porcelain --untracked-files=all)
  [[ -z "$checkout_status" ]] || \
    die "local checkout is dirty; commit intentional changes before production deployment"

  maxtext_commit=$(
    awk '$1 == "maxtext_fork_commit:" {print $2}' \
      "$repo_root/configs/gemma4_contribution_v1.yaml"
  )
  [[ "$maxtext_commit" =~ ^[0-9a-f]{40}$ ]] || \
    die "could not read the pinned Experiment-2 MaxText fork commit"

  temp_dir=$(mktemp -d "${TMPDIR:-/tmp}/moe-sae-deploy.XXXXXX")
  DEPLOY_TEMP_DIR=$temp_dir
  trap cleanup EXIT
  bundle_file="$temp_dir/repository.bundle"
  bootstrap_file="$temp_dir/bootstrap_worker.sh"
  config_file="$temp_dir/worker.conf"

  log "building a self-contained bundle whose advertised HEAD is $PROJECT_COMMIT"
  git -C "$repo_root" bundle create "$bundle_file" HEAD
  git -C "$repo_root" bundle verify "$bundle_file" >/dev/null
  git -C "$repo_root" show \
    "$PROJECT_COMMIT:scripts/tpu/bootstrap_worker.sh" > "$bootstrap_file"
  chmod 0700 "$bootstrap_file"
  cmp -s "$bootstrap_file" "$script_dir/bootstrap_worker.sh" || \
    die "working bootstrap helper does not match PROJECT_COMMIT"

  bundle_sha=$(sha256_file "$bundle_file")
  bootstrap_sha=$(sha256_file "$bootstrap_file")
  write_config "$config_file" "$bundle_sha" "$bootstrap_sha" "$maxtext_commit"
  deployment_id="${PROJECT_COMMIT:0:12}-${bundle_sha:0:12}"
  remote_dir="\$HOME/.cache/moe-sae-deploy/$deployment_id"

  tpu_description=$(gcloud compute tpus tpu-vm describe "$TPU_NAME" \
    --project="$GCP_PROJECT" \
    --zone="$TPU_ZONE" \
    --format='csv[no-heading](state,acceleratorType)') || \
    die "cannot describe TPU VM $TPU_NAME"
  IFS=',' read -r tpu_state accelerator_type <<< "$tpu_description"
  [[ "$tpu_state" == "READY" ]] || \
    die "TPU VM state is '$tpu_state'; wait for READY before deployment"
  [[ "$accelerator_type" == "$TPU_ACCELERATOR_TYPE" ]] || \
    die "TPU accelerator type is '$accelerator_type', not '$TPU_ACCELERATOR_TYPE'"
  log "target TPU is READY with accelerator type $TPU_ACCELERATOR_TYPE"
  ensure_gcloud_ssh_key "$SSH_KEY_FILE"

  gcloud_common=(
    --project="$GCP_PROJECT"
    --zone="$TPU_ZONE"
    --ssh-key-file="$SSH_KEY_FILE"
    --quiet
  )
  if [[ "$TPU_USE_INTERNAL_IP" == "1" ]]; then
    gcloud_common+=(--internal-ip)
  fi
  log "creating deployment directory on every worker"
  gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
    "${gcloud_common[@]}" \
    --worker=all \
    --command="install -d -m 0700 $remote_dir"

  log "copying bundle, config, and bootstrap helper to every worker"
  gcloud compute tpus tpu-vm scp \
    "$bundle_file" "$bootstrap_file" "$config_file" \
    "$TPU_NAME:~/.cache/moe-sae-deploy/$deployment_id/" \
    "${gcloud_common[@]}" \
    --worker=all

  remote_command="bash $remote_dir/bootstrap_worker.sh --config $remote_dir/worker.conf --bundle $remote_dir/repository.bundle"
  log "bootstrapping every worker concurrently; any worker failure fails this deployment"
  gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
    "${gcloud_common[@]}" \
    --worker=all \
    --command="$remote_command"

  log "all workers are bootstrapped at exact commit $PROJECT_COMMIT"
  log "next gate: run the bounded Experiment-2 contribution qualifier"
}

main "$@"
