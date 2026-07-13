#!/bin/bash
# Discovery phase on a Colab Pro A100-40 (§20 step 4), session-death-proof.
#
# Colab VMs are ephemeral and Pro sessions disconnect — so ALL state lives in
# a GCS bucket and every phase is idempotent: rerun this script in any fresh
# session and it syncs down, continues from wherever the last session died,
# and syncs up as it goes. Phases:
#   1. capture: WORKERS parallel processes, each a disjoint slice of the
#      deterministic stream (--start/--examples), checkpoint+resume per worker
#   2. merge workers → captures/discovery_50m
#   3. stats pass
#   4. the 14 discovery training runs (sequential, --device cuda, skip-if-done)
#   5. eval dumps (A/C seed1, dev) + §20.5 sanity gate
#
# Budget note (Colab Pro ≈ 100 compute units/mo; A100 ≈ 11.8 CU/h): capture
# ~3 h + suite ~2.5 h + eval ≈ ~70 CU. Sessions may die midway — that's fine.
#
# SUPERSEDED (2026-07-13): the training phases here predate the A12 backend
# map (v0.1.2 addendum), which freezes ALL 14 discovery runs on the TPU/JAX
# lane; and the capture phase overruns Colab's ~200 GB disk at full scale
# (~770 GB, no eviction). Use scripts/cloud/colab_capture.sh for capture and
# scripts/tpu/03_discovery_suite.sh for training. This script is retained
# ONLY as the A10 cross-backend audit instrument (torch/CUDA replication of
# the A-vs-C gate on a box with disk for the shards, e.g. GCE + gcsfuse) and
# requires explicit opt-in:
[ "${AUDIT:-0}" = 1 ] || {
  echo "superseded — capture: scripts/cloud/colab_capture.sh; training: TPU lane (A12)."
  echo "set AUDIT=1 only for the A10 cross-backend A-vs-C replication."
  exit 2
}
set -u
cd "$(dirname "$0")/../.."

BUCKET="${BUCKET:?set BUCKET=gs://<bucket>/discovery}"
WORKERS="${WORKERS:-3}"
EXAMPLES="${EXAMPLES:-96000}"
VISUAL_CAP="${VISUAL_CAP:-512}"
SEED="${SEED:-7}"
PER=$((EXAMPLES / WORKERS))
CAP_ROOT=captures/discovery_50m
STATS=stats/discovery_50m
RUNS=(discovery_A_seed1 discovery_A_seed2 discovery_A_seed3
      discovery_B_seed1 discovery_B_seed2 discovery_B_seed3
      discovery_C_seed1 discovery_C_seed2 discovery_C_seed3
      discovery_F_seed1 discovery_F_seed2 discovery_F_seed3
      discovery_D_seed1 discovery_E_seed1)

say() { echo "[colab $(date '+%F %T')] $*"; }
pull() { gcloud storage rsync -r -q "$BUCKET/$1" "$1" 2>/dev/null || true; }
push() { gcloud storage rsync -r -q "$1" "$BUCKET/$1"; }

worker_done() {
  python3 -c "import json,sys;p=json.load(open('${CAP_ROOT}_w$1/progress.json'));sys.exit(0 if p['complete'] else 1)" 2>/dev/null
}

# ---- phase 1: parallel capture --------------------------------------------- #
capture_phase() {
  local all_done=1
  for w in $(seq 0 $((WORKERS - 1))); do pull "${CAP_ROOT}_w$w"; worker_done "$w" || all_done=0; done
  [ "$all_done" = 1 ] && { say "capture already complete"; return 0; }

  say "launching $WORKERS capture workers ($PER examples each)"
  local pids=()
  for w in $(seq 0 $((WORKERS - 1))); do
    worker_done "$w" && continue
    RESUME=""; [ -f "${CAP_ROOT}_w$w/progress.json" ] && RESUME="--resume"
    PYTHONUNBUFFERED=1 python scripts/run_capture.py \
      --out "${CAP_ROOT}_w$w" --start $((w * PER)) --examples "$PER" \
      --visual-cap "$VISUAL_CAP" --seed "$SEED" --device cuda $RESUME \
      < /dev/null > "logs/capture_w$w.log" 2>&1 &
    pids+=($!)
  done
  # sync worker checkpoints up every 10 min while any worker lives
  while :; do
    local alive=0
    for pid in "${pids[@]}"; do kill -0 "$pid" 2>/dev/null && alive=1; done
    for w in $(seq 0 $((WORKERS - 1))); do push "${CAP_ROOT}_w$w"; done
    [ "$alive" = 0 ] && break
    sleep 600
  done
  for w in $(seq 0 $((WORKERS - 1))); do
    worker_done "$w" || { say "worker $w incomplete — rerun this script to resume"; exit 1; }
  done
  say "capture complete"
}

# ---- main chain ------------------------------------------------------------ #
mkdir -p logs
capture_phase

if [ ! -f "$CAP_ROOT/progress.json" ]; then
  pull "$CAP_ROOT"
fi
if ! python3 -c "import json,sys;p=json.load(open('$CAP_ROOT/progress.json'));sys.exit(0 if p['complete'] else 1)" 2>/dev/null; then
  say "merging workers"
  python scripts/merge_captures.py $(for w in $(seq 0 $((WORKERS - 1))); do echo "${CAP_ROOT}_w$w"; done) --out "$CAP_ROOT"
  push "$CAP_ROOT"
fi

pull "$STATS"
if [ ! -f "$STATS/stats.safetensors" ]; then
  say "stats pass"
  python scripts/run_stats.py --capture "$CAP_ROOT" --out "$STATS"
  push "$STATS"
fi

for run in "${RUNS[@]}"; do
  pull "runs/$run"
  if [ -f "runs/$run/final/theta_star.json" ]; then say "$run: done"; continue; fi
  [ -d "runs/$run" ] && { say "$run: wiping incomplete"; rm -rf "runs/$run"; }
  say "$run: training"
  python scripts/run_train.py --config "configs/$run.yaml" --device cuda \
    < /dev/null 2>&1 | tail -5
  [ -f "runs/$run/final/theta_star.json" ] || { say "$run FAILED"; exit 1; }
  push "runs/$run"
done

for run in discovery_A_seed1 discovery_C_seed1; do
  pull "dumps/$run"
  if [ ! -f "runs/$run/eval_dev/metrics.json" ]; then
    say "eval $run"
    python scripts/run_eval.py --run "runs/$run" --capture "$CAP_ROOT" \
      --split dev --dump "dumps/$run/dev" --device cuda
    push "runs/$run"; push "dumps/$run"
  fi
done

python - << 'EOF'
from moe_sae.eval.sanity_gate import compare_sanity
v = compare_sanity("runs/discovery_A_seed1", "runs/discovery_C_seed1",
                   "reports/sanity_gate_discovery.json")
print("DISCOVERY SANITY GATE (spec §20.5):", "PASS" if v["passed"] else "FAIL",
      "| A/C ratio:", round(v["ratio_a_over_c"], 4))
EOF
push reports
say "discovery chain complete — sync'd to $BUCKET"
