#!/bin/bash
# Resilient pipeline-capture runner (spec §20 step 2 data collection).
#
# Survives: terminal/session close (run it under nohup), laptop sleep (the
# process pauses and resumes on wake; caffeinate -i prevents *idle* sleep so
# the machine won't doze off on its own — lid close still sleeps), and crashes
# (auto-restarts with --resume from captures/<name>/progress.json; the driver
# checkpoints every 100 examples, so at most ~5 min of work is ever redone).
#
# Rerun anytime: a completed capture is detected and the script exits 0.
#
#   nohup scripts/run_pipeline_capture.sh >> logs/capture_runner.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
mkdir -p logs

OUT="${OUT:-captures/pipeline_1m}"
EXAMPLES="${EXAMPLES:-3500}"
VISUAL_CAP="${VISUAL_CAP:-512}"
SEED="${SEED:-7}"
DEVICE="${DEVICE:-mps}"
LOG="logs/capture_pipeline.log"
MAX_ATTEMPTS=50

is_complete() {
  python3 - "$OUT" <<'EOF' 2>/dev/null
import json, sys
from pathlib import Path
p = Path(sys.argv[1]) / "progress.json"
sys.exit(0 if p.exists() and json.loads(p.read_text()).get("complete") else 1)
EOF
}

consumed() {
  python3 -c "import json;print(json.load(open('$OUT/progress.json'))['stream_consumed'])" \
    < /dev/null 2>/dev/null || echo 0
}

# Attempts count only CONSECUTIVE no-progress failures: any advance in
# progress.json resets the counter, so a long healthy run interrupted many
# times (sleep/wake cycles) never exhausts the budget, while a genuine crash
# storm stops after MAX_ATTEMPTS with exponential backoff (10s → 300s cap).
attempt=0
backoff=10
last_consumed=$(consumed)
while ! is_complete; do
  now_consumed=$(consumed)
  if [ "$now_consumed" -gt "$last_consumed" ]; then
    attempt=0; backoff=10
  fi
  last_consumed=$now_consumed
  attempt=$((attempt + 1))
  if [ "$attempt" -gt "$MAX_ATTEMPTS" ]; then
    echo "[runner] $MAX_ATTEMPTS consecutive no-progress attempts — giving up" | tee -a "$LOG"
    exit 1
  fi
  RESUME=""
  [ -f "$OUT/progress.json" ] && RESUME="--resume"
  # Sweep any orphaned worker for this capture root: killing the uv/caffeinate
  # wrapper can orphan the python child, and two workers on one root collide.
  pkill -9 -f "run_capture.py --out $OUT" 2>/dev/null && sleep 3
  echo "[runner] attempt $attempt $(date '+%F %T') $RESUME" >> "$LOG"
  # stdin MUST come from /dev/null: the runner may outlive the terminal that
  # launched it, and a worker inheriting a dead pty fd 0 dies at interpreter
  # startup (init_sys_streams / EBADF — observed 2026-07-07 00:13).
  PYTHONUNBUFFERED=1 caffeinate -is uv run --no-sync python scripts/run_capture.py \
    --out "$OUT" --examples "$EXAMPLES" --visual-cap "$VISUAL_CAP" \
    --seed "$SEED" --device "$DEVICE" $RESUME < /dev/null >> "$LOG" 2>&1
  rc=$?
  echo "[runner] attempt $attempt exited rc=$rc" >> "$LOG"
  [ $rc -eq 0 ] && continue  # loop re-checks completion
  sleep "$backoff"
  backoff=$((backoff * 2)); [ "$backoff" -gt 300 ] && backoff=300
done
echo "[runner] capture complete: $OUT $(date '+%F %T')" | tee -a "$LOG"
