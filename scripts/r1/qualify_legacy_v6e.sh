#!/usr/bin/env bash
#
# Bounded v6e-64 qualification through the legacy Cloud TPU API.
#
# The persistent controller runs this script after an operator has created one
# Spot queued-resource.  It never creates compute: it validates the exact
# 16-host slice, installs one generation-pinned region-local runtime on every
# fresh worker, starts one fixed benchmark under systemd on all hosts, retains
# bounded evidence only on the controller disk, and forcibly deletes that exact
# queued-resource on every terminal path.

set -Eeuo pipefail
IFS=$'\n\t'
umask 077

readonly PROJECT="${PROJECT:-neural-dragnet}"
readonly DEPLOYMENT_PROFILE="${DEPLOYMENT_PROFILE:-euw4a}"
case "$DEPLOYMENT_PROFILE" in
  euw4a)
    readonly ZONE="europe-west4-a"
    readonly REGION_CODE="euw4a"
    readonly EXPECTED_SUBNETWORK="projects/neural-dragnet/regions/europe-west4/subnetworks/default"
    ;;
  use1d)
    readonly ZONE="us-east1-d"
    readonly REGION_CODE="use1d"
    readonly EXPECTED_SUBNETWORK="projects/neural-dragnet/regions/us-east1/subnetworks/default"
    ;;
  *)
    printf 'unsupported DEPLOYMENT_PROFILE=%s\n' "$DEPLOYMENT_PROFILE" >&2
    exit 64
    ;;
esac
readonly ATTEMPT_SEQUENCE="${ATTEMPT_SEQUENCE:-3}"
readonly QUEUED_RESOURCE="r1-v6e64-spot-${REGION_CODE}-a${ATTEMPT_SEQUENCE}"
readonly TPU_NAME="${QUEUED_RESOURCE}-node"
readonly EXPECTED_ACCELERATOR="v6e-64"
readonly EXPECTED_RUNTIME_VERSION="v6e-ubuntu-2404"
readonly EXPECTED_SERVICE_ACCOUNT="655343590563-compute@developer.gserviceaccount.com"
readonly EXPECTED_NETWORK="projects/neural-dragnet/global/networks/default"
readonly EXPECTED_LAUNCH_REF="qualification-legacy-native-ep8-fsdp8-${REGION_CODE}-a${ATTEMPT_SEQUENCE}"
readonly EXPECTED_RUNTIME_COMMIT="0b01bc2b76e375271249eb6f7e0499d5b016a1da"
readonly EXPECTED_BOOTSTRAP_SHA256="482d5c2e00740e6638986b439bedf9cbe693e7f942ce3afedf73441c40b5ed5e"
readonly EXPECTED_VALIDATOR_SHA256="0f30113726e878cfaf544c86e98d8b81edd80d4a08dea7c05462d63e187b5c07"
readonly ATTEMPT="${REGION_CODE}-legacy-a${ATTEMPT_SEQUENCE}-ep8-fsdp8"
readonly MESH="EP8/FSDP8"
readonly FORMATION_TIMEOUT_SECONDS="${FORMATION_TIMEOUT_SECONDS:-28800}"
readonly WORKER_BOOTSTRAP_TIMEOUT_SECONDS="${WORKER_BOOTSTRAP_TIMEOUT_SECONDS:-2400}"
readonly BENCHMARK_TIMEOUT_SECONDS="${BENCHMARK_TIMEOUT_SECONDS:-3600}"
readonly SLICE_HARD_DEADLINE_SECONDS="7200"
readonly GCLOUD_READ_TIMEOUT_SECONDS="60"
readonly GCLOUD_READ_ATTEMPTS="3"
readonly POLL_SECONDS="${POLL_SECONDS:-30}"
readonly STATE_ROOT="${STATE_ROOT:-/var/lib/moe-sae-r1-controller/legacy-qualification/${ATTEMPT}}"
readonly RESULT_ROOT="/var/lib/moe-sae-r1/qualification/${ATTEMPT}"
readonly RESULT_PATH="${RESULT_ROOT}/result.json"
readonly SUCCESS_PATH="${RESULT_ROOT}/worker.success"
readonly UNIT="moe-sae-r1-legacy-qual-${ATTEMPT}"
readonly BOOTSTRAP_UNIT="moe-sae-r1-legacy-bootstrap-${ATTEMPT}"
readonly REAPER_UNIT="moe-sae-r1-legacy-reaper-${ATTEMPT}"
readonly REAPER_CONTROLLER="${STATE_ROOT}/reaper-controller.sh"
readonly FORMATION_DEADLINE_PATH="${STATE_ROOT}/formation/formation-deadline.epoch"
readonly FIRST_ACTIVE_PATH="${STATE_ROOT}/formation/first-active.epoch"
readonly RESOURCE_CREATE_TIME_PATH="${STATE_ROOT}/formation/resource-create-time"
readonly REAPER_DEADLINE_PATH="${STATE_ROOT}/cleanup/reaper-deadline.epoch"
readonly REAPER_RELEASED_PATH="${STATE_ROOT}/cleanup/resource-released"
readonly REAPER_SERVICE_PATH="/etc/systemd/system/${REAPER_UNIT}.service"
readonly SSH_USER="abistras"
readonly SSH_KEY_FILE="/home/abistras/.ssh/moe_sae_r1_legacy"
readonly REMOTE_BOOTSTRAP_DIR="/home/${SSH_USER}/.cache/moe-sae-r1/${ATTEMPT}"
readonly REMOTE_BOOTSTRAP="${REMOTE_BOOTSTRAP_DIR}/bootstrap_legacy_v6e_worker.sh"
readonly SCRIPT_DIR="$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)"
readonly BOOTSTRAP_SCRIPT="${BOOTSTRAP_SCRIPT:-${SCRIPT_DIR}/bootstrap_legacy_v6e_worker.sh}"
readonly RESULT_VALIDATOR="${RESULT_VALIDATOR:-${SCRIPT_DIR}/validate_legacy_v6e_result.py}"
readonly SSH_KEY_EXPIRATION="$(
  date -u --date='+12 hours' '+%Y-%m-%dT%H:%M:%SZ'
)"

log() {
  printf '%s [legacy-v6e-controller] %s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

die() {
  log "ERROR: $*" >&2
  exit 1
}

require_name() {
  local value="$1"
  local label="$2"
  [[ "$value" =~ ^[a-z][a-z0-9-]{0,62}$ ]] ||
    die "$label is malformed"
}

require_positive_integer() {
  local value="$1"
  local label="$2"
  [[ "$value" =~ ^[1-9][0-9]*$ ]] ||
    die "$label must be a positive integer"
}

sha256_file() {
  sha256sum "$1" | awk '{print $1}'
}

snapshot_queued_resource() {
  local snapshot
  local error_path
  snapshot="$(queued_resource_snapshot_path)"
  error_path="$(queued_resource_stderr_path)"
  local temporary="${snapshot}.tmp"
  local error_temporary="${error_path}.tmp"
  if ! timeout \
    --signal=TERM \
    --kill-after=5s \
    "${GCLOUD_READ_TIMEOUT_SECONDS}s" \
    gcloud compute tpus queued-resources describe "$QUEUED_RESOURCE" \
      --project="$PROJECT" \
      --zone="$ZONE" \
      --format=json >"$temporary" 2>"$error_temporary"; then
    rm -f "$temporary"
    mv "$error_temporary" "$error_path"
    return 1
  fi
  rm -f "$error_temporary" "$error_path"
  mv "$temporary" "$snapshot"
}

queued_resource_snapshot_is_exact() {
  local snapshot
  snapshot="$(queued_resource_snapshot_path)"
  jq -e \
    --arg resource "$QUEUED_RESOURCE" \
    --arg node "$TPU_NAME" \
    --arg accelerator "$EXPECTED_ACCELERATOR" \
    --arg runtime "$EXPECTED_RUNTIME_VERSION" \
    --arg service_account "$EXPECTED_SERVICE_ACCOUNT" \
    --arg network "$EXPECTED_NETWORK" \
    --arg subnetwork "$EXPECTED_SUBNETWORK" \
    --arg launch_ref "$EXPECTED_LAUNCH_REF" \
    '
      (.name | split("/") | last) == $resource and
      (.spot | type) == "object" and
      (.tpu.nodeSpec | length) == 1 and
      .tpu.nodeSpec[0].nodeId == $node and
      .tpu.nodeSpec[0].node.acceleratorType == $accelerator and
      .tpu.nodeSpec[0].node.runtimeVersion == $runtime and
      .tpu.nodeSpec[0].node.serviceAccount.email == $service_account and
      .tpu.nodeSpec[0].node.serviceAccount.scope ==
        ["https://www.googleapis.com/auth/cloud-platform"] and
      (
        .tpu.nodeSpec[0].node.networkConfig.network == "default" or
        .tpu.nodeSpec[0].node.networkConfig.network == $network
      ) and
      .tpu.nodeSpec[0].node.networkConfig.subnetwork == $subnetwork and
      .tpu.nodeSpec[0].node.metadata["moe-sae-r1-launch-ref"] == $launch_ref
    ' "$snapshot" >/dev/null
}

bind_or_validate_resource_incarnation() {
  local snapshot
  snapshot="$(queued_resource_snapshot_path)"
  local create_time
  create_time="$(
    jq -er '
      .createTime |
      select(
        type == "string" and
        test(
          "^[0-9]{4}-[0-9]{2}-[0-9]{2}T" +
          "[0-9]{2}:[0-9]{2}:[0-9]{2}(\\.[0-9]+)?Z$"
        )
      )
    ' "$snapshot"
  )" || return 1
  if [[ ! -e "$RESOURCE_CREATE_TIME_PATH" ]]; then
    local temporary="${RESOURCE_CREATE_TIME_PATH}.tmp.$$"
    printf '%s\n' "$create_time" >"$temporary"
    chmod 0600 "$temporary"
    sync -f "$temporary"
    mv -n "$temporary" "$RESOURCE_CREATE_TIME_PATH"
    rm -f "$temporary"
    sync -f "$(dirname "$RESOURCE_CREATE_TIME_PATH")"
  fi
  [[ -f "$RESOURCE_CREATE_TIME_PATH" && ! -L "$RESOURCE_CREATE_TIME_PATH" ]] ||
    return 1
  [[ "$(<"$RESOURCE_CREATE_TIME_PATH")" == "$create_time" ]]
}

queued_resource_identity_is_exact() {
  queued_resource_snapshot_is_exact &&
    bind_or_validate_resource_incarnation
}

validate_queued_resource() {
  queued_resource_identity_is_exact ||
    die "queued-resource identity changed"
}

snapshot_node() {
  local temporary="$STATE_ROOT/formation/node.json.tmp"
  if ! timeout \
    --signal=TERM \
    --kill-after=5s \
    "${GCLOUD_READ_TIMEOUT_SECONDS}s" \
    gcloud compute tpus tpu-vm describe "$TPU_NAME" \
      --project="$PROJECT" \
      --zone="$ZONE" \
      --format=json >"$temporary" \
      2>"$STATE_ROOT/formation/node.stderr.tmp"; then
    rm -f "$temporary"
    mv \
      "$STATE_ROOT/formation/node.stderr.tmp" \
      "$STATE_ROOT/formation/node.stderr"
    return 1
  fi
  rm -f "$STATE_ROOT/formation/node.stderr.tmp" "$STATE_ROOT/formation/node.stderr"
  mv "$temporary" "$STATE_ROOT/formation/node.json"
}

retry_snapshot_queued_resource() {
  local attempt
  for attempt in $(seq 1 "$GCLOUD_READ_ATTEMPTS"); do
    snapshot_queued_resource && return 0
    ((attempt == GCLOUD_READ_ATTEMPTS)) || sleep 5
  done
  return 1
}

retry_snapshot_node() {
  local attempt
  for attempt in $(seq 1 "$GCLOUD_READ_ATTEMPTS"); do
    snapshot_node && return 0
    ((attempt == GCLOUD_READ_ATTEMPTS)) || sleep 5
  done
  return 1
}

queued_resource_error_is_not_found() {
  local path="$1"
  [[ -f "$path" ]] && grep -Eq '(^|[[:space:]])NOT_FOUND:' "$path"
}

validate_ready_node() {
  local snapshot="$STATE_ROOT/formation/node.json"
  jq -e \
    --arg queued_resource "$QUEUED_RESOURCE" \
    --arg accelerator "$EXPECTED_ACCELERATOR" \
    --arg runtime "$EXPECTED_RUNTIME_VERSION" \
    --arg service_account "$EXPECTED_SERVICE_ACCOUNT" \
    --arg network "$EXPECTED_NETWORK" \
    --arg subnetwork "$EXPECTED_SUBNETWORK" \
    '
      .state == "READY" and
      .acceleratorType == $accelerator and
      .runtimeVersion == $runtime and
      .serviceAccount.email == $service_account and
      .serviceAccount.scope ==
        ["https://www.googleapis.com/auth/cloud-platform"] and
      (.networkConfig.enableExternalIps // false) == false and
      .networkConfig.network == $network and
      .networkConfig.subnetwork == $subnetwork and
      (.queuedResource | split("/") | last) == $queued_resource and
      (.networkEndpoints | length) == 16 and
      ([.networkEndpoints[].ipAddress] | unique | length) == 16 and
      all(.networkEndpoints[]; ((.accessConfig? // {}) == {})) and
      ((.health // "HEALTHY") == "HEALTHY")
    ' "$snapshot" >/dev/null ||
    die "READY TPU node identity/topology changed"
}

tpu_ssh() {
  local worker="$1"
  local command_timeout_seconds="$2"
  local remote_command="$3"
  require_positive_integer "$command_timeout_seconds" command_timeout_seconds
  timeout \
    --signal=TERM \
    --kill-after=5s \
    "${command_timeout_seconds}s" \
    gcloud compute tpus tpu-vm ssh "${SSH_USER}@${TPU_NAME}" \
      --project="$PROJECT" \
      --zone="$ZONE" \
      --worker="$worker" \
      --internal-ip \
      --quiet \
      --ssh-key-file="$SSH_KEY_FILE" \
      --ssh-key-expiration="$SSH_KEY_EXPIRATION" \
      --ssh-flag="-oBatchMode=yes" \
      --ssh-flag="-oConnectTimeout=20" \
      --ssh-flag="-oConnectionAttempts=1" \
      --ssh-flag="-oServerAliveInterval=15" \
      --ssh-flag="-oServerAliveCountMax=2" \
      --command="$remote_command"
}

tpu_scp_bootstrap() {
  local worker="$1"
  timeout --signal=TERM --kill-after=5s 120s \
    gcloud compute tpus tpu-vm scp \
      "$BOOTSTRAP_SCRIPT" \
      "${SSH_USER}@${TPU_NAME}:${REMOTE_BOOTSTRAP}.tmp" \
      --project="$PROJECT" \
      --zone="$ZONE" \
      --worker="$worker" \
      --internal-ip \
      --quiet \
      --ssh-key-file="$SSH_KEY_FILE" \
      --ssh-key-expiration="$SSH_KEY_EXPIRATION" \
      --scp-flag="-oBatchMode=yes" \
      --scp-flag="-oConnectTimeout=20" \
      --scp-flag="-oConnectionAttempts=1" \
      --scp-flag="-oServerAliveInterval=15" \
      --scp-flag="-oServerAliveCountMax=2"
}

run_all_workers() {
  local label="$1"
  local remote_command="$2"
  local output_root="$3"
  local command_timeout_seconds="${4:-95}"
  local worker
  local failed=0
  local -a pids=()
  mkdir -p "$output_root"
  for worker in $(seq 0 15); do
    (
      tpu_ssh "$worker" "$command_timeout_seconds" "$remote_command"
    ) >"$output_root/${worker}.tmp" 2>&1 &
    pids+=("$!")
  done
  for worker in $(seq 0 15); do
    if wait "${pids[$worker]}"; then
      mv "$output_root/${worker}.tmp" "$output_root/${worker}.ok"
    else
      mv "$output_root/${worker}.tmp" "$output_root/${worker}.failed"
      failed=1
    fi
  done
  [[ "$failed" == 0 ]] || die "$label failed on one or more workers"
}

probe_all_worker_identities() {
  local output_root="$1"
  local worker
  local failed=0
  local -a pids=()
  mkdir -p "$output_root"
  rm -f "$output_root"/*.ok "$output_root"/*.failed "$output_root"/*.tmp
  for worker in $(seq 0 15); do
    (
      tpu_ssh "$worker" 95 true
    ) >"$output_root/${worker}.tmp" 2>&1 &
    pids+=("$!")
  done
  for worker in $(seq 0 15); do
    if wait "${pids[$worker]}"; then
      mv "$output_root/${worker}.tmp" "$output_root/${worker}.ok"
    else
      mv "$output_root/${worker}.tmp" "$output_root/${worker}.failed"
      failed=1
    fi
  done
  [[ "$failed" == 0 ]]
}

copy_bootstrap_to_all_workers() {
  local worker
  local failed=0
  local -a pids=()
  for worker in $(seq 0 15); do
    (
      tpu_scp_bootstrap "$worker"
    ) >"$STATE_ROOT/bootstrap/scp/${worker}.tmp" 2>&1 &
    pids+=("$!")
  done
  for worker in $(seq 0 15); do
    if wait "${pids[$worker]}"; then
      mv \
        "$STATE_ROOT/bootstrap/scp/${worker}.tmp" \
        "$STATE_ROOT/bootstrap/scp/${worker}.ok"
    else
      mv \
        "$STATE_ROOT/bootstrap/scp/${worker}.tmp" \
        "$STATE_ROOT/bootstrap/scp/${worker}.failed"
      failed=1
    fi
  done
  [[ "$failed" == 0 ]] || die "bootstrap copy failed on one or more workers"
}

seal_evidence_best_effort() {
  (
    cd "$STATE_ROOT"
    local temporary="SHA256SUMS.$$.tmp"
    find . \
      -type f \
      ! -name SHA256SUMS \
      ! -name controller.log \
      ! -name controller.lock \
      ! -name '*.tmp' \
      -print0 |
      sort -z |
      xargs -0 -r sha256sum >"$temporary"
    sync -f "$temporary"
    mv "$temporary" SHA256SUMS
    sync -f .
  )
}

benchmark_started=0
qualification_complete=0
resource_validated=0
resource_released=0
reaper_armed=0
GCLOUD_BIN=''
snapshot_suffix=''

queued_resource_snapshot_path() {
  printf '%s/formation/queued-resource%s.json\n' \
    "$STATE_ROOT" "$snapshot_suffix"
}

queued_resource_stderr_path() {
  printf '%s/formation/queued-resource%s.stderr\n' \
    "$STATE_ROOT" "$snapshot_suffix"
}

stop_units_best_effort() {
  local worker
  local remote_stop
  local -a pids=()
  read -r -d '' remote_stop <<EOF || true
set +e
sudo install -d -o moe_sae -g moe_sae -m 0700 /var/lib/moe-sae-r1/qualification
sudo -u moe_sae touch /var/lib/moe-sae-r1/qualification/${ATTEMPT}.abort
sudo systemctl stop --no-block ${UNIT}.service >/dev/null 2>&1
exit 0
EOF
  for worker in $(seq 0 15); do
    (
      tpu_ssh "$worker" 95 "$remote_stop"
    ) >"$STATE_ROOT/cleanup/${worker}.log" 2>&1 &
    pids+=("$!")
  done
  for worker in $(seq 0 15); do
    wait "${pids[$worker]}" || true
  done
}

collect_evidence_best_effort() {
  local worker
  local remote_collect
  local -a pids=()
  read -r -d '' remote_collect <<EOF || true
set +e
printf '%s\n' '--- systemd ---'
sudo systemctl show ${UNIT}.service --no-pager \
  --property=LoadState,ActiveState,SubState,Result,ExecMainStatus
printf '%s\n' '--- journal ---'
sudo journalctl -u ${UNIT}.service \
  --no-pager --output=short-iso-precise -n 5000
printf '%s\n' '--- result ---'
if sudo test -f ${RESULT_PATH}; then
  sudo cat ${RESULT_PATH}
fi
EOF
  for worker in $(seq 0 15); do
    (
      tpu_ssh "$worker" 95 "$remote_collect"
    ) >"$STATE_ROOT/evidence/${worker}.tmp" 2>&1 &
    pids+=("$!")
  done
  for worker in $(seq 0 15); do
    if wait "${pids[$worker]}"; then
      mv "$STATE_ROOT/evidence/${worker}.tmp" "$STATE_ROOT/evidence/${worker}.log"
    else
      mv \
        "$STATE_ROOT/evidence/${worker}.tmp" \
        "$STATE_ROOT/evidence/${worker}.fetch-failed"
    fi
  done
}

collect_results_best_effort() {
  local worker
  local remote_result="if sudo test -f ${RESULT_PATH}; then sudo cat ${RESULT_PATH}; fi"
  local -a pids=()
  rm -f \
    "$STATE_ROOT"/result/*.raw \
    "$STATE_ROOT"/result/*.stderr \
    "$STATE_ROOT"/result/*.fetch-failed \
    "$STATE_ROOT"/result/*.tmp
  for worker in $(seq 0 15); do
    (
      tpu_ssh "$worker" 95 "$remote_result"
    ) >"$STATE_ROOT/result/${worker}.tmp" \
      2>"$STATE_ROOT/result/${worker}.stderr.tmp" &
    pids+=("$!")
  done
  for worker in $(seq 0 15); do
    if wait "${pids[$worker]}"; then
      mv "$STATE_ROOT/result/${worker}.tmp" "$STATE_ROOT/result/${worker}.raw"
      mv \
        "$STATE_ROOT/result/${worker}.stderr.tmp" \
        "$STATE_ROOT/result/${worker}.stderr"
    else
      {
        cat "$STATE_ROOT/result/${worker}.tmp"
        cat "$STATE_ROOT/result/${worker}.stderr.tmp"
      } >"$STATE_ROOT/result/${worker}.fetch-failed"
      rm -f \
        "$STATE_ROOT/result/${worker}.tmp" \
        "$STATE_ROOT/result/${worker}.stderr.tmp"
    fi
  done
}

write_reaper_deadline_if_earlier() {
  local candidate="$1"
  require_positive_integer "$candidate" reaper_deadline
  local current=''
  if [[ -f "$REAPER_DEADLINE_PATH" && ! -L "$REAPER_DEADLINE_PATH" ]]; then
    current="$(<"$REAPER_DEADLINE_PATH")"
    require_positive_integer "$current" existing_reaper_deadline
    if ((current <= candidate)); then
      return
    fi
  elif [[ -e "$REAPER_DEADLINE_PATH" ]]; then
    die "reaper deadline path is not a regular file"
  fi
  local temporary="${REAPER_DEADLINE_PATH}.tmp.$$"
  printf '%s\n' "$candidate" >"$temporary"
  chmod 0600 "$temporary"
  sync -f "$temporary"
  mv "$temporary" "$REAPER_DEADLINE_PATH"
  sync -f "$(dirname "$REAPER_DEADLINE_PATH")"
}

publish_release_marker() {
  local temporary="${REAPER_RELEASED_PATH}.tmp.$$"
  printf '%s\n' "$(date -u +%s)" >"$temporary"
  chmod 0600 "$temporary"
  sync -f "$temporary"
  mv "$temporary" "$REAPER_RELEASED_PATH"
  sync -f "$(dirname "$REAPER_RELEASED_PATH")"
  if ! seal_evidence_best_effort; then
    log "release marker is durable but evidence checksum sealing failed"
  fi
}

initialize_formation_deadline() {
  if [[ -f "$FORMATION_DEADLINE_PATH" && ! -L "$FORMATION_DEADLINE_PATH" ]]; then
    local existing
    existing="$(<"$FORMATION_DEADLINE_PATH")"
    require_positive_integer "$existing" existing_formation_deadline
    return
  fi
  [[ ! -e "$FORMATION_DEADLINE_PATH" ]] ||
    die "formation deadline path is not a regular file"
  local temporary="${FORMATION_DEADLINE_PATH}.tmp.$$"
  printf '%s\n' "$(( $(date -u +%s) + FORMATION_TIMEOUT_SECONDS ))" \
    >"$temporary"
  chmod 0600 "$temporary"
  sync -f "$temporary"
  mv "$temporary" "$FORMATION_DEADLINE_PATH"
  sync -f "$(dirname "$FORMATION_DEADLINE_PATH")"
}

install_persistent_reaper() {
  [[ ! -e "$REAPER_RELEASED_PATH" ]] ||
    die "attempt identity is already sealed released; use a fresh attempt/name"
  local controller_sha
  controller_sha="$(sha256_file "$0")"
  if [[ -e "$REAPER_CONTROLLER" ]]; then
    [[ -f "$REAPER_CONTROLLER" && ! -L "$REAPER_CONTROLLER" ]] ||
      die "bound reaper controller is not a regular file"
    [[ "$(sha256_file "$REAPER_CONTROLLER")" == "$controller_sha" ]] ||
      die "bound reaper controller changed for this attempt"
  else
    local controller_temporary="${REAPER_CONTROLLER}.tmp.$$"
    install -o root -g root -m 0500 "$0" "$controller_temporary"
    sync -f "$controller_temporary"
    mv "$controller_temporary" "$REAPER_CONTROLLER"
    sync -f "$(dirname "$REAPER_CONTROLLER")"
  fi

  initialize_formation_deadline
  write_reaper_deadline_if_earlier \
    "$(( $(<"$FORMATION_DEADLINE_PATH") + SLICE_HARD_DEADLINE_SECONDS ))"

  local service_temporary="${REAPER_SERVICE_PATH}.tmp.$$"
  {
    printf '%s\n' \
      '[Unit]' \
      "Description=Persistent exact-resource reaper for ${QUEUED_RESOURCE}" \
      'Wants=network-online.target' \
      'After=network-online.target' \
      'StartLimitIntervalSec=0' \
      '' \
      '[Service]' \
      'Type=exec' \
      "Environment=PROJECT=${PROJECT}" \
      "Environment=DEPLOYMENT_PROFILE=${DEPLOYMENT_PROFILE}" \
      "Environment=ATTEMPT_SEQUENCE=${ATTEMPT_SEQUENCE}" \
      "Environment=STATE_ROOT=${STATE_ROOT}" \
      "ExecStart=${REAPER_CONTROLLER} --reap" \
      'Restart=on-failure' \
      'RestartSec=30s' \
      'TimeoutStopSec=20s' \
      'KillMode=mixed' \
      'UMask=0077' \
      '' \
      '[Install]' \
      'WantedBy=multi-user.target'
  } >"$service_temporary"
  chmod 0644 "$service_temporary"
  sync -f "$service_temporary"
  mv "$service_temporary" "$REAPER_SERVICE_PATH"
  sync -f /etc/systemd/system
  systemctl daemon-reload
  systemctl enable --now "${REAPER_UNIT}.service"
  systemctl is-active --quiet "${REAPER_UNIT}.service" ||
    die "persistent exact-resource reaper did not become active"
  reaper_armed=1
  log "persistent formation watchdog is active before resource discovery"
}

tighten_reaper_for_active_slice() {
  local observed_at
  observed_at="$(date -u +%s)"
  write_reaper_deadline_if_earlier \
    "$((observed_at + SLICE_HARD_DEADLINE_SECONDS))"
  if [[ ! -e "$FIRST_ACTIVE_PATH" ]]; then
    local temporary="${FIRST_ACTIVE_PATH}.tmp.$$"
    printf '%s\n' "$observed_at" >"$temporary"
    chmod 0600 "$temporary"
    sync -f "$temporary"
    mv -n "$temporary" "$FIRST_ACTIVE_PATH"
    rm -f "$temporary"
    sync -f "$(dirname "$FIRST_ACTIVE_PATH")"
  fi
  log "slice deadline monotonically bounded to $(<"$REAPER_DEADLINE_PATH")"
}

reap_exact_resource() {
  [[ "$#" == 0 ]] || die "reaper accepts no additional arguments"
  [[ "$EUID" == 0 ]] || die "reaper must run as root"
  for command in \
    chmod date dirname find gcloud grep jq mkdir mv rm seq sha256sum sleep sort \
    sync timeout xargs; do
    command -v "$command" >/dev/null 2>&1 ||
      die "reaper command is unavailable: $command"
  done
  require_positive_integer "$ATTEMPT_SEQUENCE" ATTEMPT_SEQUENCE
  require_name "$QUEUED_RESOURCE" QUEUED_RESOURCE
  require_name "$TPU_NAME" TPU_NAME
  require_name "$ATTEMPT" ATTEMPT
  snapshot_suffix='.reaper'
  mkdir -p "$STATE_ROOT"/{formation,cleanup}
  while true; do
    [[ ! -e "$REAPER_RELEASED_PATH" ]] || {
      log "release marker already confirms queued-resource absence"
      return 0
    }
    [[ -f "$REAPER_DEADLINE_PATH" && ! -L "$REAPER_DEADLINE_PATH" ]] ||
      die "persistent reaper deadline is absent or unsafe"
    local deadline
    deadline="$(<"$REAPER_DEADLINE_PATH")"
    require_positive_integer "$deadline" reaper_deadline
    local now
    now="$(date -u +%s)"
    if ((now < deadline)); then
      if [[ ! -e "$FIRST_ACTIVE_PATH" ]] && snapshot_queued_resource; then
        if ! queued_resource_identity_is_exact; then
          log "resource name now refers to a different identity; refusing deletion"
          printf '%s\n' "$now" \
            >"$STATE_ROOT/cleanup/resource-identity-mismatch"
          return 0
        fi
        if [[ "$(
          jq -r '.state.state // "STATE_UNSPECIFIED"' \
            "$(queued_resource_snapshot_path)"
        )" == ACTIVE ]]; then
          tighten_reaper_for_active_slice
          continue
        fi
      fi
      local remaining=$((deadline - now))
      ((remaining > 60)) && remaining=60
      sleep "$remaining"
      continue
    fi

    if snapshot_queued_resource; then
      if ! queued_resource_identity_is_exact; then
        log "resource name now refers to a different identity; refusing deletion"
        printf '%s\n' "$(date -u +%s)" \
          >"$STATE_ROOT/cleanup/resource-identity-mismatch"
        return 0
      fi
      log "hard deadline reached; requesting deletion of exact queued-resource"
      timeout \
        --signal=TERM \
        --kill-after=5s \
        540s \
        gcloud compute tpus queued-resources delete "$QUEUED_RESOURCE" \
          --project="$PROJECT" \
          --zone="$ZONE" \
          --force \
          --async \
          --quiet || true
    elif queued_resource_error_is_not_found \
      "$(queued_resource_stderr_path)"; then
      publish_release_marker
      log "queued-resource absence confirmed after hard deadline"
      return 0
    else
      log "control-plane read failed after hard deadline; retrying"
    fi
    sleep 30
  done
}

confirm_queued_resource_absent() {
  local attempt
  for attempt in $(seq 1 20); do
    if snapshot_queued_resource; then
      queued_resource_identity_is_exact ||
        return 1
    elif queued_resource_error_is_not_found \
      "$(queued_resource_stderr_path)"; then
      publish_release_marker
      return 0
    fi
    sleep 15
  done
  return 1
}

release_queued_resource_best_effort() {
  [[ "$resource_validated" == 1 && "$resource_released" == 0 ]] || return 0
  if ! snapshot_queued_resource; then
    if queued_resource_error_is_not_found \
      "$(queued_resource_stderr_path)"; then
      publish_release_marker
      resource_released=1
      systemctl disable --now "${REAPER_UNIT}.service" >/dev/null 2>&1 || true
      reaper_armed=0
      log "queued-resource was already absent at release"
      return 0
    fi
    log "fresh release identity read failed; refusing deletion"
    return 1
  fi
  if ! queued_resource_identity_is_exact; then
    log "fresh release identity changed; refusing deletion"
    return 1
  fi
  log "requesting forced deletion of exact queued-resource $QUEUED_RESOURCE"
  timeout \
    --signal=TERM \
    --kill-after=5s \
    540s \
    gcloud compute tpus queued-resources delete "$QUEUED_RESOURCE" \
      --project="$PROJECT" \
      --zone="$ZONE" \
      --force \
      --async \
      --quiet || true
  if confirm_queued_resource_absent; then
    resource_released=1
    systemctl disable --now "${REAPER_UNIT}.service" >/dev/null 2>&1 || true
    reaper_armed=0
    log "queued-resource absence independently confirmed"
    return 0
  fi
  log "deletion is not yet confirmed; persistent reaper remains armed"
  return 1
}

on_exit() {
  local rc="$?"
  trap - EXIT INT TERM
  if [[ "$benchmark_started" == 1 && "$qualification_complete" == 0 ]]; then
    set +e
    log "controller exited before the qualification seal; stopping reachable units"
    stop_units_best_effort
    collect_evidence_best_effort
    seal_evidence_best_effort
  fi
  set +e
  release_queued_resource_best_effort
  exit "$rc"
}

trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

main() {
  if [[ "${1:-}" == --reap ]]; then
    shift
    reap_exact_resource "$@"
    return
  fi
  [[ "$#" == 0 ]] || die "this fixed controller accepts no arguments"
  [[ "$EUID" == 0 ]] || die "controller must run as root"
  for command in \
    awk cat chmod date dirname find flock gcloud grep install jq mkdir mktemp mv \
    rm seq sha256sum sleep sort sync systemctl tee timeout tr wc xargs; do
    command -v "$command" >/dev/null 2>&1 ||
      die "required command is unavailable: $command"
  done
  require_name "$PROJECT" PROJECT
  require_name "$ZONE" ZONE
  require_positive_integer "$ATTEMPT_SEQUENCE" ATTEMPT_SEQUENCE
  require_name "$QUEUED_RESOURCE" QUEUED_RESOURCE
  require_name "$TPU_NAME" TPU_NAME
  require_name "$ATTEMPT" ATTEMPT
  [[ "$MESH" == EP8/FSDP8 ]] ||
    die "legacy bridge is identity-bound to EP8/FSDP8"
  require_positive_integer "$FORMATION_TIMEOUT_SECONDS" FORMATION_TIMEOUT_SECONDS
  require_positive_integer \
    "$WORKER_BOOTSTRAP_TIMEOUT_SECONDS" \
    WORKER_BOOTSTRAP_TIMEOUT_SECONDS
  require_positive_integer "$BENCHMARK_TIMEOUT_SECONDS" BENCHMARK_TIMEOUT_SECONDS
  require_positive_integer "$POLL_SECONDS" POLL_SECONDS
  [[ "$STATE_ROOT" == /var/lib/moe-sae-r1-controller/legacy-qualification/* ]] ||
    die "STATE_ROOT escapes the legacy qualification root"
  [[ -f "$BOOTSTRAP_SCRIPT" && ! -L "$BOOTSTRAP_SCRIPT" ]] ||
    die "worker bootstrap script is absent or symbolic"
  [[ "$(sha256_file "$BOOTSTRAP_SCRIPT")" == "$EXPECTED_BOOTSTRAP_SHA256" ]] ||
    die "worker bootstrap script SHA-256 changed"
  [[ -f "$RESULT_VALIDATOR" && ! -L "$RESULT_VALIDATOR" ]] ||
    die "result validator is absent or symbolic"
  [[ "$(sha256_file "$RESULT_VALIDATOR")" == "$EXPECTED_VALIDATOR_SHA256" ]] ||
    die "result validator SHA-256 changed"
  GCLOUD_BIN="$(command -v gcloud)"
  [[ "$GCLOUD_BIN" == /* ]] || die "gcloud path is not absolute"

  mkdir -p \
    "$STATE_ROOT"/{formation,identity,bootstrap/scp,bootstrap/run,readiness,launch,status,evidence,result,cleanup}
  exec 9>"$STATE_ROOT/controller.lock"
  flock -n 9 || die "another controller owns attempt $ATTEMPT"
  exec > >(tee -a "$STATE_ROOT/controller.log") 2>&1

  install_persistent_reaper
  local formation_deadline
  formation_deadline="$(<"$FORMATION_DEADLINE_PATH")"
  while true; do
    if ! snapshot_queued_resource; then
      if queued_resource_error_is_not_found \
        "$(queued_resource_stderr_path)"; then
        log "waiting for the exact queued-resource to be created"
      else
        log "queued-resource describe failed within its hard timeout"
      fi
      (($(date -u +%s) < formation_deadline)) ||
        die "queued-resource was not observable before the formation deadline"
      sleep "$POLL_SECONDS"
      continue
    fi
    validate_queued_resource
    resource_validated=1
    local queued_state
    queued_state="$(
      jq -r '.state.state // "STATE_UNSPECIFIED"' \
        "$(queued_resource_snapshot_path)"
    )"
    log "queued-resource state=$queued_state"
    case "$queued_state" in
      ACTIVE) break ;;
      FAILED | SUSPENDED | DELETING)
        die "queued-resource entered terminal state $queued_state"
        ;;
      WAITING_FOR_RESOURCES | ACCEPTED | PROVISIONING | CREATING)
        ;;
      *) die "queued-resource entered unknown state $queued_state" ;;
    esac
    (($(date -u +%s) < formation_deadline)) ||
      die "v6e-64 queued-resource did not activate before timeout"
    sleep "$POLL_SECONDS"
  done
  tighten_reaper_for_active_slice

  while true; do
    if snapshot_node; then
      local node_state
      node_state="$(jq -r '.state // "STATE_UNSPECIFIED"' "$STATE_ROOT/formation/node.json")"
      log "TPU node state=$node_state"
      if [[ "$node_state" == READY ]]; then
        validate_ready_node
        break
      fi
      case "$node_state" in
        CREATING | STARTING | REIMAGING | RESTARTING) ;;
        *) die "TPU node entered unexpected state $node_state" ;;
      esac
    else
      log "TPU node resource is not yet observable"
    fi
    (($(date -u +%s) < formation_deadline)) ||
      die "v6e-64 node did not become READY before timeout"
    sleep "$POLL_SECONDS"
  done

  local identity_deadline=$((SECONDS + WORKER_BOOTSTRAP_TIMEOUT_SECONDS))
  while ! tpu_ssh 0 95 true >"$STATE_ROOT/identity/0.tmp" 2>&1; do
    ((SECONDS < identity_deadline)) ||
      die "controller SSH identity did not reach worker 0"
    log "waiting for TPU SSH identity propagation"
    sleep "$POLL_SECONDS"
  done
  mv "$STATE_ROOT/identity/0.tmp" "$STATE_ROOT/identity/0.ok"
  local identity_round=0
  while true; do
    local round_root="$STATE_ROOT/identity/all-${identity_round}"
    if probe_all_worker_identities "$round_root"; then
      log "all 16 TPU SSH identities are ready"
      break
    fi
    local ready_count
    ready_count="$(
      find "$round_root" -maxdepth 1 -type f -name '*.ok' |
        wc -l |
        tr -d ' '
    )"
    log "waiting for all TPU SSH identities ready=$ready_count/16"
    ((SECONDS < identity_deadline)) ||
      die "all 16 TPU SSH identities did not become ready"
    ((identity_round += 1))
    sleep "$POLL_SECONDS"
  done

  local remote_mkdir="install -d -m 0700 ${REMOTE_BOOTSTRAP_DIR}"
  run_all_workers \
    "remote bootstrap-directory creation" \
    "$remote_mkdir" \
    "$STATE_ROOT/bootstrap/mkdir"
  copy_bootstrap_to_all_workers

  local remote_bootstrap
  read -r -d '' remote_bootstrap <<EOF || true
set -Eeuo pipefail
printf '%s  %s\n' \
  ${EXPECTED_BOOTSTRAP_SHA256} \
  ${REMOTE_BOOTSTRAP}.tmp |
  sha256sum --check --status
mv ${REMOTE_BOOTSTRAP}.tmp ${REMOTE_BOOTSTRAP}
chmod 0500 ${REMOTE_BOOTSTRAP}
sudo install -o root -g root -m 0755 \
  ${REMOTE_BOOTSTRAP} \
  /usr/local/sbin/bootstrap_legacy_v6e_worker.sh
if ! sudo systemctl show ${BOOTSTRAP_UNIT}.service >/dev/null 2>&1; then
  sudo systemd-run \
    --unit=${BOOTSTRAP_UNIT} \
    --property=Type=exec \
    --property=RemainAfterExit=yes \
    --property=Restart=no \
    --property=RuntimeMaxSec=${WORKER_BOOTSTRAP_TIMEOUT_SECONDS} \
    --property=TimeoutStopSec=60 \
    --property=KillMode=mixed \
    --property=UMask=0077 \
    /usr/local/sbin/bootstrap_legacy_v6e_worker.sh
fi
deadline=\$((SECONDS + ${WORKER_BOOTSTRAP_TIMEOUT_SECONDS}))
while true; do
  load_state="\$(sudo systemctl show ${BOOTSTRAP_UNIT}.service --property=LoadState --value)"
  active_state="\$(sudo systemctl show ${BOOTSTRAP_UNIT}.service --property=ActiveState --value)"
  sub_state="\$(sudo systemctl show ${BOOTSTRAP_UNIT}.service --property=SubState --value)"
  result="\$(sudo systemctl show ${BOOTSTRAP_UNIT}.service --property=Result --value)"
  exit_status="\$(sudo systemctl show ${BOOTSTRAP_UNIT}.service --property=ExecMainStatus --value)"
  if [[ "\$load_state" == loaded &&
    "\$active_state" == active &&
    "\$sub_state" == exited &&
    "\$result" == success &&
    "\$exit_status" == 0 ]]; then
    break
  fi
  if [[ "\$load_state" != loaded ||
    "\$active_state" == failed ||
    "\$result" == exit-code ||
    "\$result" == timeout ||
    "\$result" == signal ]]; then
    sudo journalctl -u ${BOOTSTRAP_UNIT}.service --no-pager -n 200
    exit 1
  fi
  ((SECONDS < deadline)) || exit 124
  sleep 5
done
EOF
  run_all_workers \
    "exact runtime bootstrap" \
    "$remote_bootstrap" \
    "$STATE_ROOT/bootstrap/run" \
    "$((WORKER_BOOTSTRAP_TIMEOUT_SECONDS + 180))"

  local remote_readiness
  read -r -d '' remote_readiness <<EOF || true
set -Eeuo pipefail
printf '%s  %s\n' \
  ${EXPECTED_BOOTSTRAP_SHA256} \
  /usr/local/sbin/bootstrap_legacy_v6e_worker.sh |
  sudo sha256sum --check --status
sudo /usr/local/sbin/bootstrap_legacy_v6e_worker.sh
test "\$(id -u moe_sae)" = 20001
numeric_vfio_nodes="\$(
  find /dev/vfio -maxdepth 1 -type c -printf '%f\n' 2>/dev/null |
    grep -E '^[0-9]+\$' |
    sort -n
)"
test "\$(printf '%s\n' "\$numeric_vfio_nodes" | wc -l)" = 4
for node in \$numeric_vfio_nodes; do
  sudo -u moe_sae test -r "/dev/vfio/\$node"
  sudo -u moe_sae test -w "/dev/vfio/\$node"
done
sudo -u moe_sae test -r /dev/vfio/vfio
sudo -u moe_sae test -w /dev/vfio/vfio
EOF
  run_all_workers \
    "all-host runtime/device readiness" \
    "$remote_readiness" \
    "$STATE_ROOT/readiness"

  retry_snapshot_queued_resource ||
    die "queued-resource control-plane reads failed after readiness"
  validate_queued_resource
  [[ "$(
    jq -r '.state.state // "STATE_UNSPECIFIED"' \
      "$(queued_resource_snapshot_path)"
  )" == ACTIVE ]] || die "queued-resource changed after readiness"
  retry_snapshot_node || die "TPU node disappeared after readiness"
  validate_ready_node

  local remote_launch
  read -r -d '' remote_launch <<EOF || true
set -Eeuo pipefail
marker=/var/lib/moe-sae-r1/qualification/${ATTEMPT}.launched
abort_marker=/var/lib/moe-sae-r1/qualification/${ATTEMPT}.abort
sudo install -d -o moe_sae -g moe_sae -m 0700 \
  /var/lib/moe-sae-r1/qualification \
  ${RESULT_ROOT}
sudo test ! -e "\$abort_marker"
if sudo test -e ${SUCCESS_PATH}; then
  sudo test -e "\$marker"
  exit
fi
if sudo test -e "\$marker"; then
  sudo systemctl is-active --quiet ${UNIT}.service
  exit
fi
sudo -u moe_sae /bin/bash -c 'set -C; : > "\$1"' _ "\$marker"
if sudo test -e "\$abort_marker"; then
  sudo rm -f "\$marker"
  exit 1
fi
if ! sudo systemd-run \
  --unit=${UNIT} \
  --uid=moe_sae \
  --gid=moe_sae \
  --working-directory=/var/lib/moe-sae-r1 \
  --setenv=HOME=/var/lib/moe-sae-r1 \
  --setenv=PYTHONNOUSERSITE=1 \
  --setenv=PYTHONDONTWRITEBYTECODE=1 \
  --setenv=PYTHONUNBUFFERED=1 \
  --setenv=TMPDIR=/run/${UNIT} \
  --setenv=XDG_CACHE_HOME=/var/cache/moe-sae-r1 \
  --property=Type=exec \
  --property=RemainAfterExit=yes \
  --property=Restart=no \
  --property=RuntimeDirectory=${UNIT} \
  --property=RuntimeDirectoryMode=0700 \
  --property=RuntimeMaxSec=${BENCHMARK_TIMEOUT_SECONDS} \
  --property=TimeoutStopSec=110 \
  --property=KillMode=control-group \
  --property=LimitMEMLOCK=infinity \
  --property=LimitNOFILE=100000 \
  --property=UMask=0077 \
  --property=NoNewPrivileges=yes \
  --property=ProtectSystem=strict \
  --property=ProtectHome=yes \
  --property=PrivateTmp=yes \
  --property='ReadWritePaths=/var/lib/moe-sae-r1 /var/cache/moe-sae-r1 /run/${UNIT} /dev/shm' \
  --property='RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6' \
  /bin/bash -c 'set -Eeuo pipefail
  /opt/moe-sae-r1/runtime/bin/python -I \
      -m moe_sae.r1.prefix_benchmark_runner \
      --maxtext-base-yml /opt/moe-sae-r1/runtime/lib/python3.12/site-packages/maxtext/configs/base.yml \
      --output-directory ${RESULT_ROOT} \
      --mesh ${MESH} \
      --seed 0 \
      --result-json ${RESULT_PATH}
  : > ${SUCCESS_PATH}'; then
  sudo rm -f "\$marker"
  exit 1
fi
if [[ "\$(sudo systemctl show ${UNIT}.service --property=LimitMEMLOCK --value)" != infinity ||
  "\$(sudo systemctl show ${UNIT}.service --property=LimitNOFILE --value)" != 100000 ]]; then
  sudo systemctl stop ${UNIT}.service
  sudo rm -f "\$marker"
  exit 1
fi
EOF

  benchmark_started=1
  run_all_workers \
    "distributed benchmark admission" \
    "$remote_launch" \
    "$STATE_ROOT/launch"
  log "all 16 systemd-owned benchmark processes admitted"

  local remote_status="if sudo test -f ${SUCCESS_PATH}; then printf 'WorkerSuccess=yes\\n'; else printf 'WorkerSuccess=no\\n'; fi; sudo systemctl show ${UNIT}.service --no-pager --property=LoadState,ActiveState,SubState,Result,ExecMainStatus"
  local benchmark_deadline=$((SECONDS + BENCHMARK_TIMEOUT_SECONDS))
  local failed_count=0
  local succeeded_count=0
  while true; do
    retry_snapshot_queued_resource ||
      die "queued-resource control-plane reads failed during benchmark"
    validate_queued_resource
    [[ "$(
      jq -r '.state.state // "STATE_UNSPECIFIED"' \
        "$(queued_resource_snapshot_path)"
    )" == ACTIVE ]] || die "queued-resource left ACTIVE during benchmark"
    retry_snapshot_node || die "TPU node disappeared during benchmark"
    validate_ready_node

    rm -f "$STATE_ROOT"/status/*.ok "$STATE_ROOT"/status/*.failed
    run_all_workers \
      "distributed benchmark status" \
      "$remote_status" \
      "$STATE_ROOT/status"
    failed_count=0
    succeeded_count=0
    local worker
    for worker in $(seq 0 15); do
      local status_file="$STATE_ROOT/status/${worker}.ok"
      if grep -q '^WorkerSuccess=yes$' "$status_file"; then
        ((succeeded_count += 1))
      elif grep -q '^ActiveState=failed$' "$status_file" ||
        ! grep -q '^LoadState=loaded$' "$status_file" ||
        ! grep -q '^ActiveState=active$' "$status_file" ||
        grep -q '^SubState=exited$' "$status_file"; then
        ((failed_count += 1))
      fi
    done
    log "benchmark succeeded=$succeeded_count failed=$failed_count running=$((16 - succeeded_count - failed_count))"
    [[ "$failed_count" == 0 ]] || break
    [[ "$succeeded_count" == 16 ]] && break
    ((SECONDS < benchmark_deadline)) || break
    sleep "$POLL_SECONDS"
  done

  collect_evidence_best_effort
  collect_results_best_effort
  seal_evidence_best_effort
  [[ "$failed_count" == 0 ]] ||
    die "one or more benchmark processes failed"
  [[ "$succeeded_count" == 16 ]] ||
    die "benchmark exceeded its hard runtime limit"

  local result_count=0
  local result_path=''
  local worker
  for worker in $(seq 0 15); do
    [[ ! -e "$STATE_ROOT/result/${worker}.fetch-failed" ]] ||
      die "result fetch failed on worker $worker"
    if [[ -s "$STATE_ROOT/result/${worker}.raw" ]]; then
      ((result_count += 1))
      result_path="$STATE_ROOT/result/${worker}.raw"
    fi
  done
  [[ "$result_count" == 1 ]] ||
    die "expected exactly one process-zero result, found $result_count"
  "$RESULT_VALIDATOR" "$result_path" >"$STATE_ROOT/result/validation.json"
  qualification_complete=1
  seal_evidence_best_effort
  release_queued_resource_best_effort ||
    die "qualification passed but exact TPU resource release failed"
  log ">=51% qualification completed; evidence retained at $STATE_ROOT"
}

main "$@"
