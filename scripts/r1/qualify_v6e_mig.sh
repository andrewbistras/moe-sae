#!/usr/bin/env bash
#
# Qualification-only launcher for the baked DeepSeek-R1 v6e-64 prefix image.
#
# This script runs on the small persistent controller VM. It waits for one
# already-created bulk MIG to become a complete 16-VM/8x8 slice, starts the
# same fixed benchmark as a systemd-owned process on every private worker, and
# retains bounded logs locally. It never creates, repairs, resizes, or deletes
# TPU resources and it never uploads bulk evidence to GCS.

set -Eeuo pipefail
umask 077

readonly PROJECT="${PROJECT:-neural-dragnet}"
readonly ZONE="${ZONE:-us-east1-d}"
readonly MIG="${MIG:-r1-v6e-bench-27e25e5-ue1d-a1}"
readonly EXPECTED_TEMPLATE="${EXPECTED_TEMPLATE:-r1-v6e-spot-stop-27e25e5-us-east1-v1}"
readonly ATTEMPT="${ATTEMPT:-ue1d-a1-ep16-fsdp4}"
readonly MESH="${MESH:-EP16/FSDP4}"
readonly FORMATION_TIMEOUT_SECONDS="${FORMATION_TIMEOUT_SECONDS:-21600}"
readonly WORKER_READINESS_TIMEOUT_SECONDS="${WORKER_READINESS_TIMEOUT_SECONDS:-900}"
readonly BENCHMARK_TIMEOUT_SECONDS="${BENCHMARK_TIMEOUT_SECONDS:-3600}"
readonly POLL_SECONDS="${POLL_SECONDS:-30}"
readonly STATE_ROOT="${STATE_ROOT:-/var/lib/moe-sae-r1-controller/qualification/${ATTEMPT}}"
readonly UNIT="moe-sae-r1-qual-${ATTEMPT}"
readonly RESULT_ROOT="/var/lib/moe-sae-r1/qualification/${ATTEMPT}"
readonly RESULT_PATH="${RESULT_ROOT}/result.json"
readonly SUCCESS_PATH="${RESULT_ROOT}/worker.success"
readonly IMAGE_OBSERVATION_SHA256="629b8baf0e80f694be537df504fa7214eff283e00af97ced23b910a17d05c47f"
readonly SSH_KEY_FILE="/home/abistras/.ssh/moe_sae_r1_controller"
readonly SSH_KEY_EXPIRATION="$(
  date -u --date='+12 hours' '+%Y-%m-%dT%H:%M:%SZ'
)"

die() {
  printf '%s ERROR %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2
  exit 1
}

log() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

require_name() {
  local value="$1"
  local label="$2"
  [[ "$value" =~ ^[a-z][a-z0-9-]{0,62}$ ]] || die "$label is malformed"
}

require_positive_integer() {
  local value="$1"
  local label="$2"
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "$label must be a positive integer"
}

for command in date find flock gcloud grep jq mkdir mv sha256sum sleep sort tee timeout tr wc xargs; do
  command -v "$command" >/dev/null 2>&1 || die "missing command: $command"
done
require_name "$PROJECT" PROJECT
require_name "$ZONE" ZONE
require_name "$MIG" MIG
require_name "$EXPECTED_TEMPLATE" EXPECTED_TEMPLATE
require_name "$ATTEMPT" ATTEMPT
case "$MESH" in
  EP16/FSDP4 | EP32/FSDP2 | EP8/FSDP8) ;;
  *) die "MESH is unsupported" ;;
esac
require_positive_integer "$FORMATION_TIMEOUT_SECONDS" FORMATION_TIMEOUT_SECONDS
require_positive_integer \
  "$WORKER_READINESS_TIMEOUT_SECONDS" \
  WORKER_READINESS_TIMEOUT_SECONDS
require_positive_integer "$BENCHMARK_TIMEOUT_SECONDS" BENCHMARK_TIMEOUT_SECONDS
require_positive_integer "$POLL_SECONDS" POLL_SECONDS
[[ "$STATE_ROOT" == /var/lib/moe-sae-r1-controller/qualification/* ]] ||
  die "STATE_ROOT escapes the controller qualification root"

mkdir -p \
  "$STATE_ROOT"/{formation,readiness,launch,launch-cleanup,status,journal,result}
exec 9>"$STATE_ROOT/controller.lock"
flock -n 9 || die "another controller owns attempt $ATTEMPT"
exec > >(tee -a "$STATE_ROOT/controller.log") 2>&1

snapshot_mig() {
  local temporary="$STATE_ROOT/formation/mig.json.tmp"
  gcloud compute instance-groups managed describe "$MIG" \
    --project="$PROJECT" \
    --zone="$ZONE" \
    --format=json >"$temporary"
  mv "$temporary" "$STATE_ROOT/formation/mig.json"

  temporary="$STATE_ROOT/formation/instances.json.tmp"
  gcloud compute instance-groups managed list-instances "$MIG" \
    --project="$PROJECT" \
    --zone="$ZONE" \
    --format=json >"$temporary"
  mv "$temporary" "$STATE_ROOT/formation/instances.json"
}

compute_ssh() {
  local worker="$1"
  local remote_command="$2"
  timeout --signal=TERM --kill-after=5s 75s \
    gcloud compute ssh "$worker" \
    --project="$PROJECT" \
    --zone="$ZONE" \
    --internal-ip \
    --quiet \
    --ssh-key-file="$SSH_KEY_FILE" \
    --ssh-key-expiration="$SSH_KEY_EXPIRATION" \
    --ssh-flag="-oBatchMode=yes" \
    --ssh-flag="-oConnectTimeout=15" \
    --ssh-flag="-oConnectionAttempts=1" \
    --ssh-flag="-oServerAliveInterval=15" \
    --ssh-flag="-oServerAliveCountMax=2" \
    --command="$remote_command"
}

formation_deadline=$((SECONDS + FORMATION_TIMEOUT_SECONDS))
while true; do
  snapshot_mig
  template="$(
    jq -r '.instanceTemplate | split("/") | last' \
      "$STATE_ROOT/formation/mig.json"
  )"
  [[ "$template" == "$EXPECTED_TEMPLATE" ]] ||
    die "MIG template changed: $template"
  [[ "$(jq -r '.targetSize' "$STATE_ROOT/formation/mig.json")" == 16 ]] ||
    die "MIG target size is not 16"
  [[ "$(
    jq -r '.targetSizePolicy.mode' "$STATE_ROOT/formation/mig.json"
  )" == BULK ]] || die "MIG target-size policy is not BULK"

  member_count="$(jq 'length' "$STATE_ROOT/formation/instances.json")"
  running_count="$(
    jq '[.[] | select(.instanceStatus == "RUNNING")] | length' \
      "$STATE_ROOT/formation/instances.json"
  )"
  topology="$(
    jq -r '.status.appliedAcceleratorTopologies[0].state // "ABSENT"' \
      "$STATE_ROOT/formation/mig.json"
  )"
  log "formation members=$member_count running=$running_count topology=$topology"

  if [[ "$member_count" == 16 && "$running_count" == 16 &&
    "$topology" == ACTIVE ]]; then
    break
  fi
  ((SECONDS < formation_deadline)) ||
    die "complete v6e-64 slice did not form before timeout"
  sleep "$POLL_SECONDS"
done

mapfile -t workers < <(
  jq -r '.[].instance | split("/") | last' \
    "$STATE_ROOT/formation/instances.json" | sort
)
[[ "${#workers[@]}" == 16 ]] || die "expected exactly 16 workers"
[[ "$(
  jq '[.[].instance | split("/") | last] | unique | length' \
    "$STATE_ROOT/formation/instances.json"
)" == 16 ]] ||
  die "worker names are not unique"
printf '%s\n' "${workers[@]}" >"$STATE_ROOT/workers.txt"

# Compute can report RUNNING before sshd, project SSH metadata, the baked
# service user, or all four VFIO devices are ready. Establish the controller's
# SSH identity once, then require the complete 16-host image/device boundary
# before any process enters distributed JAX.
readiness_deadline=$((SECONDS + WORKER_READINESS_TIMEOUT_SECONDS))
while ! timeout --signal=TERM --kill-after=5s 75s \
  gcloud compute ssh "${workers[0]}" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --internal-ip \
  --quiet \
  --ssh-key-file="$SSH_KEY_FILE" \
  --ssh-key-expiration="$SSH_KEY_EXPIRATION" \
  --ssh-flag="-oBatchMode=yes" \
  --ssh-flag="-oConnectTimeout=15" \
  --ssh-flag="-oConnectionAttempts=1" \
  --ssh-flag="-oServerAliveInterval=15" \
  --ssh-flag="-oServerAliveCountMax=2" \
  --command=true \
  >"$STATE_ROOT/readiness/${workers[0]}.identity.log" 2>&1; do
  ((SECONDS < readiness_deadline)) ||
    die "controller SSH identity did not become usable before timeout"
  log "waiting for first worker SSH identity propagation"
  sleep "$POLL_SECONDS"
done

read -r -d '' remote_readiness <<EOF || true
set -Eeuo pipefail
test "\$(id -u moe_sae)" = 20001
sudo test -x /opt/moe-sae-r1/runtime/bin/python
sudo test -r /usr/local/lib/moe-sae-r1/image-observation.json
printf '%s  %s\n' \
  ${IMAGE_OBSERVATION_SHA256} \
  /usr/local/lib/moe-sae-r1/image-observation.json |
  sudo sha256sum --check --status
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

while true; do
  find "$STATE_ROOT/readiness" \
    -maxdepth 1 \
    -type f \
    \( -name '*.ready' -o -name '*.tmp' \) \
    -delete
  readiness_pids=()
  for worker in "${workers[@]}"; do
    (
      compute_ssh "$worker" "$remote_readiness" \
        >"$STATE_ROOT/readiness/${worker}.tmp" 2>&1 &&
      mv "$STATE_ROOT/readiness/${worker}.tmp" \
        "$STATE_ROOT/readiness/${worker}.ready"
    ) &
    readiness_pids+=("$!")
  done
  readiness_failed=0
  for pid in "${readiness_pids[@]}"; do
    wait "$pid" || ((readiness_failed += 1))
  done
  ready_count="$(
    find "$STATE_ROOT/readiness" -maxdepth 1 -type f -name '*.ready' |
      wc -l |
      tr -d ' '
  )"
  log "worker readiness ready=$ready_count failed=$readiness_failed"
  [[ "$readiness_failed" == 0 && "$ready_count" == 16 ]] && break
  ((SECONDS < readiness_deadline)) ||
    die "all 16 workers did not pass the image/VFIO boundary before timeout"
  sleep "$POLL_SECONDS"
done

# The readiness round and the MIG snapshot are distinct observations. Refuse
# to launch if any member or the 8x8 topology changed between them.
snapshot_mig
mapfile -t post_readiness_workers < <(
  jq -r '.[].instance | split("/") | last' \
    "$STATE_ROOT/formation/instances.json" | sort
)
[[ "${post_readiness_workers[*]}" == "${workers[*]}" ]] ||
  die "MIG membership changed after the all-host readiness barrier"
[[ "$(
  jq '[.[] | select(.instanceStatus == "RUNNING")] | length' \
    "$STATE_ROOT/formation/instances.json"
)" == 16 ]] || die "a worker stopped after the all-host readiness barrier"
[[ "$(
  jq -r '.status.appliedAcceleratorTopologies[0].state // "ABSENT"' \
    "$STATE_ROOT/formation/mig.json"
)" == ACTIVE ]] || die "8x8 topology stopped being active after readiness"

read -r -d '' remote_launch <<EOF || true
set -Eeuo pipefail
marker=/var/lib/moe-sae-r1/qualification/${ATTEMPT}.launched
abort_marker=/var/lib/moe-sae-r1/qualification/${ATTEMPT}.abort
sudo install -d -o moe_sae -g moe_sae -m 0700 /var/lib/moe-sae-r1/qualification
sudo install -d -o moe_sae -g moe_sae -m 0700 ${RESULT_ROOT}
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
EOF

benchmark_started=0
qualification_complete=0
launch_pids=()
status_pids=()

terminate_controller_fanout_best_effort() {
  local pid
  for pid in "${launch_pids[@]}" "${status_pids[@]}"; do
    [[ -n "$pid" ]] || continue
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  for pid in "${launch_pids[@]}" "${status_pids[@]}"; do
    [[ -n "$pid" ]] || continue
    wait "$pid" 2>/dev/null || true
  done
}

stop_units_best_effort() {
  local worker
  local -a cleanup_pids=()
  local remote_stop
  read -r -d '' remote_stop <<EOF || true
set +e
sudo install -d -o moe_sae -g moe_sae -m 0700 /var/lib/moe-sae-r1/qualification
sudo -u moe_sae touch /var/lib/moe-sae-r1/qualification/${ATTEMPT}.abort
sudo systemctl stop --no-block ${UNIT}.service 2>/dev/null
exit 0
EOF
  for worker in "${workers[@]}"; do
    (
      compute_ssh "$worker" "$remote_stop"
    ) >"$STATE_ROOT/launch-cleanup/${worker}.log" 2>&1 &
    cleanup_pids+=("$!")
  done
  for pid in "${cleanup_pids[@]}"; do
    wait "$pid" || true
  done
  # A launch command already executing on a worker can race the first
  # no-block stop. The abort marker prevents new admission; this second pass
  # closes the remaining narrow systemd-run/stop ordering window.
  sleep 2
  cleanup_pids=()
  for worker in "${workers[@]}"; do
    (
      compute_ssh "$worker" "$remote_stop"
    ) >>"$STATE_ROOT/launch-cleanup/${worker}.log" 2>&1 &
    cleanup_pids+=("$!")
  done
  for pid in "${cleanup_pids[@]}"; do
    wait "$pid" || true
  done
}

collect_evidence_best_effort() {
  local worker
  local -a collect_pids=()
  for worker in "${workers[@]}"; do
    (
      set +e
      if [[ ! -f "$STATE_ROOT/journal/${worker}.ok" ]]; then
        if compute_ssh \
          "$worker" \
          "sudo journalctl -u ${UNIT}.service --no-pager --output=short-iso-precise -n 4000" \
          >"$STATE_ROOT/journal/${worker}.tmp" 2>&1; then
          mv \
            "$STATE_ROOT/journal/${worker}.tmp" \
            "$STATE_ROOT/journal/${worker}.log"
          : >"$STATE_ROOT/journal/${worker}.ok"
        else
          mv \
            "$STATE_ROOT/journal/${worker}.tmp" \
            "$STATE_ROOT/journal/${worker}.fetch-error"
        fi
      fi
      if [[ ! -f "$STATE_ROOT/result/${worker}.ok" ]]; then
        if compute_ssh \
          "$worker" \
          "if sudo test -f ${RESULT_PATH}; then sudo cat ${RESULT_PATH}; fi" \
          >"$STATE_ROOT/result/${worker}.tmp" \
          2>"$STATE_ROOT/result/${worker}.stderr.tmp"; then
          mv \
            "$STATE_ROOT/result/${worker}.tmp" \
            "$STATE_ROOT/result/${worker}.json"
          mv \
            "$STATE_ROOT/result/${worker}.stderr.tmp" \
            "$STATE_ROOT/result/${worker}.stderr"
          : >"$STATE_ROOT/result/${worker}.ok"
        else
          mv \
            "$STATE_ROOT/result/${worker}.tmp" \
            "$STATE_ROOT/result/${worker}.fetch-error"
          mv \
            "$STATE_ROOT/result/${worker}.stderr.tmp" \
            "$STATE_ROOT/result/${worker}.fetch-error.stderr"
        fi
      fi
    ) &
    collect_pids+=("$!")
  done
  for pid in "${collect_pids[@]}"; do
    wait "$pid" || true
  done
}

evidence_fetch_complete() {
  local worker
  for worker in "${workers[@]}"; do
    [[ -f "$STATE_ROOT/journal/${worker}.ok" ]] || return 1
    [[ -f "$STATE_ROOT/result/${worker}.ok" ]] || return 1
  done
}

seal_evidence_best_effort() {
  (
    cd "$STATE_ROOT"
    find . \
      -type f \
      ! -name SHA256SUMS \
      ! -name controller.log \
      ! -name controller.lock \
      ! -name '*.tmp' \
      -print0 |
      sort -z |
      xargs -0 -r sha256sum >SHA256SUMS
  )
}

on_exit() {
  local rc="$?"
  trap - EXIT INT TERM
  if [[ "$benchmark_started" == 1 && "$qualification_complete" == 0 ]]; then
    set +e
    log "controller exit before qualification seal; stopping reachable units"
    terminate_controller_fanout_best_effort
    stop_units_best_effort
    collect_evidence_best_effort
    seal_evidence_best_effort
  fi
  exit "$rc"
}

trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

benchmark_started=1
for worker in "${workers[@]}"; do
  (
    compute_ssh "$worker" "$remote_launch"
  ) >"$STATE_ROOT/launch/${worker}.log" 2>&1 &
  launch_pids+=("$!")
done

launch_failed=0
for index in "${!launch_pids[@]}"; do
  if ! wait "${launch_pids[$index]}"; then
    launch_failed=1
    log "launch failed on ${workers[$index]}"
  fi
done
if [[ "$launch_failed" != 0 ]]; then
  die "distributed launch fanout was incomplete"
fi
log "all 16 systemd-owned benchmark processes admitted"

remote_status="if sudo test -f ${SUCCESS_PATH}; then printf 'WorkerSuccess=yes\\n'; else printf 'WorkerSuccess=no\\n'; fi; sudo systemctl show ${UNIT}.service --no-pager --property=LoadState,ActiveState,SubState,Result,ExecMainStatus"
benchmark_deadline=$((SECONDS + BENCHMARK_TIMEOUT_SECONDS))
while true; do
  find "$STATE_ROOT/status" \
    -maxdepth 1 \
    -type f \
    \( -name '*.txt' -o -name '*.tmp' \) \
    -delete
  status_pids=()
  for worker in "${workers[@]}"; do
    (
      compute_ssh "$worker" "$remote_status" \
        >"$STATE_ROOT/status/${worker}.txt.tmp" 2>&1 &&
      mv "$STATE_ROOT/status/${worker}.txt.tmp" \
        "$STATE_ROOT/status/${worker}.txt"
    ) &
    status_pids+=("$!")
  done
  status_failed=0
  for pid in "${status_pids[@]}"; do
    wait "$pid" || status_failed=1
  done
  [[ "$status_failed" == 0 ]] ||
    die "worker status fanout failed; slice may have been preempted"

  failed_count=0
  succeeded_count=0
  for worker in "${workers[@]}"; do
    status_file="$STATE_ROOT/status/${worker}.txt"
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
if ! evidence_fetch_complete; then
  log "evidence fetch incomplete; retrying missing worker artifacts once"
  sleep 5
  collect_evidence_best_effort
fi
evidence_fetch_complete ||
  die "normal completion did not fetch journals and result probes from all workers"
seal_evidence_best_effort

if [[ "$failed_count" != 0 ]]; then
  die "one or more benchmark processes failed"
fi
if [[ "$succeeded_count" != 16 ]]; then
  die "benchmark exceeded its hard runtime limit"
fi
result_count=0
for result in "$STATE_ROOT"/result/*.json; do
  if jq -e \
    '.schema == "moe-sae.r1.synthetic-prefix-qualification.v1"' \
    "$result" >/dev/null 2>&1; then
    ((result_count += 1))
  fi
done
[[ "$result_count" == 1 ]] ||
  die "expected one process-zero result, found $result_count"
qualification_complete=1
log "qualification completed; evidence retained at $STATE_ROOT"
