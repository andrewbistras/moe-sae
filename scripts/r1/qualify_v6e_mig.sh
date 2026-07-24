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
readonly BENCHMARK_TIMEOUT_SECONDS="${BENCHMARK_TIMEOUT_SECONDS:-3600}"
readonly POLL_SECONDS="${POLL_SECONDS:-30}"
readonly STATE_ROOT="${STATE_ROOT:-/var/lib/moe-sae-r1-controller/qualification/${ATTEMPT}}"
readonly UNIT="moe-sae-r1-qual-${ATTEMPT}"
readonly RESULT_ROOT="/var/lib/moe-sae-r1/qualification/${ATTEMPT}"
readonly RESULT_PATH="${RESULT_ROOT}/result.json"

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

for command in date find flock gcloud grep jq mkdir mv sha256sum sleep sort tee wc xargs; do
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
require_positive_integer "$BENCHMARK_TIMEOUT_SECONDS" BENCHMARK_TIMEOUT_SECONDS
require_positive_integer "$POLL_SECONDS" POLL_SECONDS
[[ "$STATE_ROOT" == /var/lib/moe-sae-r1-controller/qualification/* ]] ||
  die "STATE_ROOT escapes the controller qualification root"

mkdir -p "$STATE_ROOT"/{formation,launch,status,journal,result}
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

# Establish the controller's project SSH identity before the parallel fanout.
# Workers have no public IP; this connection stays on the default VPC.
gcloud compute ssh "${workers[0]}" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --internal-ip \
  --quiet \
  --command='sudo test -x /opt/moe-sae-r1/runtime/bin/python && test "$(find /dev/vfio -maxdepth 1 -type c 2>/dev/null | wc -l)" -ge 4'

read -r -d '' remote_launch <<EOF || true
set -Eeuo pipefail
marker=/var/lib/moe-sae-r1/qualification/${ATTEMPT}.launched
sudo install -d -o moe_sae -g moe_sae -m 0700 /var/lib/moe-sae-r1/qualification
sudo install -d -o moe_sae -g moe_sae -m 0700 ${RESULT_ROOT}
if sudo test -e "\$marker"; then
  sudo systemctl is-active --quiet ${UNIT}.service
  exit
fi
sudo -u moe_sae /bin/bash -c 'set -C; : > "\$1"' _ "\$marker"
if ! sudo systemd-run \
  --unit=${UNIT} \
  --uid=moe_sae \
  --gid=moe_sae \
  --working-directory=/var/lib/moe-sae-r1 \
  --setenv=HOME=/var/lib/moe-sae-r1 \
  --setenv=PYTHONNOUSERSITE=1 \
  --setenv=PYTHONDONTWRITEBYTECODE=1 \
  --setenv=PYTHONUNBUFFERED=1 \
  --setenv=TMPDIR=/var/tmp/moe-sae-r1 \
  --setenv=XDG_CACHE_HOME=/var/cache/moe-sae-r1 \
  --property=Type=exec \
  --property=RemainAfterExit=yes \
  --property=Restart=no \
  --property=TimeoutStopSec=110 \
  --property=KillMode=mixed \
  --property=UMask=0077 \
  --property=NoNewPrivileges=yes \
  --property=ProtectSystem=strict \
  --property=ProtectHome=yes \
  --property=PrivateTmp=yes \
  --property='ReadWritePaths=/var/lib/moe-sae-r1 /var/cache/moe-sae-r1 /var/tmp/moe-sae-r1 /dev/shm' \
  --property='RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6' \
  /opt/moe-sae-r1/runtime/bin/python -I \
    -m moe_sae.r1.prefix_benchmark_runner \
    --maxtext-base-yml /opt/moe-sae-r1/runtime/lib/python3.12/site-packages/maxtext/configs/base.yml \
    --output-directory ${RESULT_ROOT} \
    --mesh ${MESH} \
    --seed 0 \
    --result-json ${RESULT_PATH}; then
  sudo rm -f "\$marker"
  exit 1
fi
EOF

launch_pids=()
for worker in "${workers[@]}"; do
  (
    gcloud compute ssh "$worker" \
      --project="$PROJECT" \
      --zone="$ZONE" \
      --internal-ip \
      --quiet \
      --command="$remote_launch"
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
[[ "$launch_failed" == 0 ]] || die "distributed launch fanout was incomplete"
log "all 16 systemd-owned benchmark processes admitted"

remote_status="sudo systemctl show ${UNIT}.service --no-pager --property=LoadState,ActiveState,SubState,Result,ExecMainStatus"
benchmark_deadline=$((SECONDS + BENCHMARK_TIMEOUT_SECONDS))
while true; do
  status_pids=()
  for worker in "${workers[@]}"; do
    (
      gcloud compute ssh "$worker" \
        --project="$PROJECT" \
        --zone="$ZONE" \
        --internal-ip \
        --quiet \
        --command="$remote_status" \
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
  for status_file in "$STATE_ROOT"/status/*.txt; do
    if grep -q '^ActiveState=failed$' "$status_file"; then
      ((failed_count += 1))
    fi
    if grep -q '^SubState=exited$' "$status_file"; then
      ((succeeded_count += 1))
    fi
  done
  log "benchmark succeeded=$succeeded_count failed=$failed_count running=$((16 - succeeded_count - failed_count))"
  [[ "$failed_count" == 0 ]] || break
  [[ "$succeeded_count" == 16 ]] && break
  ((SECONDS < benchmark_deadline)) || break
  sleep "$POLL_SECONDS"
done

collect_pids=()
for worker in "${workers[@]}"; do
  (
    gcloud compute ssh "$worker" \
      --project="$PROJECT" \
      --zone="$ZONE" \
      --internal-ip \
      --quiet \
      --command="sudo journalctl -u ${UNIT}.service --no-pager --output=short-iso-precise -n 4000" \
      >"$STATE_ROOT/journal/${worker}.log" 2>&1
    gcloud compute ssh "$worker" \
      --project="$PROJECT" \
      --zone="$ZONE" \
      --internal-ip \
      --quiet \
      --command="sudo test -f ${RESULT_PATH} && sudo cat ${RESULT_PATH} || true" \
      >"$STATE_ROOT/result/${worker}.json" \
      2>"$STATE_ROOT/result/${worker}.stderr"
  ) &
  collect_pids+=("$!")
done
for pid in "${collect_pids[@]}"; do
  wait "$pid" || true
done

(
  cd "$STATE_ROOT"
  find . -type f ! -name SHA256SUMS ! -name '*.tmp' -print0 |
    sort -z |
    xargs -0 sha256sum >SHA256SUMS
)

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
log "qualification completed; evidence retained at $STATE_ROOT"
