#!/bin/bash
# Overnight pipeline orchestrator (spec §20 steps 2-3 + step-5 sanity gate).
#
# Waits for the capture runner to finish, then: stats pass → train pipeline
# conditions A, B, C (1M tokens, f=1024, k̄=32) → reconstruction eval + dev
# FeatureDumps → A-vs-C sanity gate. Every step is idempotent (checked by its
# final artifact) and retried once, so this script can be re-run at any time
# and continues from wherever it stopped. Detach with:
#
#   nohup scripts/run_pipeline_night.sh >> logs/night_runner.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
mkdir -p logs reports
LOG="logs/pipeline_night.log"
CAPTURE=captures/pipeline_1m
STATS=stats/pipeline_1m
DEVICE="${DEVICE:-mps}"

say() { echo "[night $(date '+%F %T')] $*" | tee -a "$LOG"; }

run_step() {  # run_step <name> <artifact-that-marks-done> <cmd...>
  local name="$1" artifact="$2"; shift 2
  if [ -e "$artifact" ]; then say "$name: done already ($artifact)"; return 0; fi
  for try in 1 2 3; do
    say "$name: attempt $try"
    # stdin from /dev/null: this chain outlives the launching terminal, and a
    # child inheriting a dead pty fd 0 dies at python startup (init_sys_streams).
    PYTHONUNBUFFERED=1 caffeinate -is "$@" < /dev/null >> "$LOG" 2>&1
    if [ -e "$artifact" ]; then say "$name: OK"; return 0; fi
    say "$name: attempt $try failed (rc=$?, artifact missing)"
    sleep 30
  done
  say "$name: FAILED twice — stopping the chain"; exit 1
}

# --- 0. wait for capture completion --------------------------------------- #
say "waiting for capture completion at $CAPTURE"
until python3 - "$CAPTURE" <<'EOF' 2>/dev/null
import json, sys
from pathlib import Path
p = Path(sys.argv[1]) / "progress.json"
sys.exit(0 if p.exists() and json.loads(p.read_text()).get("complete") else 1)
EOF
do sleep 60; done
say "capture complete"

# --- 1. statistics pass (§20 step 2) --------------------------------------- #
run_step "stats" "$STATS/stats.safetensors" \
  uv run --no-sync python scripts/run_stats.py --capture "$CAPTURE" --out "$STATS"

# --- 2. train pipeline conditions (§20 step 3) ----------------------------- #
for cond in A B C; do
  run="runs/pipeline_1m_${cond}"
  if [ ! -f "$run/final/theta_star.json" ] && [ -d "$run" ]; then
    say "train $cond: wiping incomplete $run"; rm -rf "$run"
  fi
  run_step "train $cond" "$run/final/theta_star.json" \
    uv run --no-sync python scripts/run_train.py \
      --config "configs/pipeline_1m_${cond}.yaml" --device "$DEVICE"
done

# --- 3. reconstruction eval + dev dumps ------------------------------------ #
for cond in A B C; do
  run="runs/pipeline_1m_${cond}"
  run_step "eval $cond" "$run/eval_dev/metrics.json" \
    uv run --no-sync python scripts/run_eval.py \
      --run "$run" --capture "$CAPTURE" --split dev \
      --dump "dumps/pipeline_1m_${cond}/dev" --device "$DEVICE"
done

# --- 4. A-vs-C implementation sanity gate (§20 step 5) --------------------- #
run_step "sanity-gate" "reports/sanity_gate_pipeline.json" \
  uv run --no-sync python -c "
from moe_sae.eval.sanity_gate import compare_sanity
v = compare_sanity('runs/pipeline_1m_A', 'runs/pipeline_1m_C',
                   'reports/sanity_gate_pipeline.json')
print('sanity gate:', 'PASS' if v['passed'] else 'FAIL', v['ratio_a_over_c'])
"
say "pipeline chain complete — see reports/sanity_gate_pipeline.json"
