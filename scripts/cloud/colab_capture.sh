#!/bin/bash
# Discovery capture (§20 step 4 data) on a Colab A100, streaming to GCS —
# the disk-safe replacement for the capture phase of colab_discovery.sh.
#
# Why slices: Colab's ~200 GB disk cannot hold the ~770 GB capture, and the
# shard writer numbers new shards by counting files on disk (capture/writer.py
# ShardWriter), so evicting already-uploaded shards mid-run would corrupt
# numbering. The 96k-example stream is therefore cut into SLICES disjoint
# worker slices (--start/--examples, the driver's native parallelism); each
# slice is captured to completion, pushed, verified against the bucket, then
# wiped before its slot takes the next slice. CONCURRENCY slots run in
# parallel (two ~7 GB model copies fit a 40 GB A100 and hide data loading).
# Peak disk ≈ CONCURRENCY × ~64 GB at the defaults.
#
# All state lives in the bucket; rerun this script in any fresh session to
# resume: complete slices are skipped, an interrupted one pulls its checkpoint
# back down and continues via --resume (a background pusher syncs every
# 10 min, so a dead session redoes ≤10 min of work).
#
# Phases:
#   0. A7 gate on THIS substrate — FP32 hook validation with --device cuda,
#      plus the A3 split-contamination re-verification. HARD GATE: no shard
#      counts unless it passes here; a PASS report is cached in the bucket.
#   1. sliced capture.
# Afterwards (not this script): scripts/cloud/merge_in_bucket.py (server-side
# merge, no data transits), then the stats pass in-region over gcsfuse on a
# TPU host, then scripts/tpu/03_discovery_suite.sh. Training NEVER runs here
# (A12 backend map: all 14 runs on TPU/JAX).
set -euo pipefail
cd "$(dirname "$0")/../.."

BUCKET="${BUCKET:?set BUCKET=gs://<bucket> (no trailing slash)}"
EXAMPLES="${EXAMPLES:-96000}"
SLICES="${SLICES:-12}"
CONCURRENCY="${CONCURRENCY:-2}"
VISUAL_CAP="${VISUAL_CAP:-512}"
SEED="${SEED:-7}"
CAP="captures/discovery_50m"
PER=$((EXAMPLES / SLICES))
[ $((PER * SLICES)) -eq "$EXAMPLES" ] || { echo "EXAMPLES must divide evenly by SLICES"; exit 1; }

say() { echo "[capture $(date '+%F %T')] $*"; }
mkdir -p logs reports

# Parallel composite uploads break under concurrent rsyncs of the same dir
# (the pusher + a slice finalization): each rsync garbage-collects the other's
# temp components -> GcsNotFoundError storms. Plain uploads are plenty fast
# for ~180 MB shards and keep objects md5-verifiable.
gcloud config set storage/parallel_composite_upload_enabled False 2>/dev/null || true

# HF CDN connections drop mid-parquet-block under load (ChunkedEncodingError
# through fsspec); give hub reads a longer timeout on top of its own backoff.
export HF_HUB_DOWNLOAD_TIMEOUT=60

# A7 substrate guard: shards from an ungated GPU class are inadmissible.
nvidia-smi --query-gpu=name --format=csv,noheader | grep -q "A100" \
  || { say "not an A100 runtime (A7: gate + capture must share a substrate) — refusing"; exit 1; }

complete_json() { # $1 = progress.json path; exit 0 iff capture complete
  python3 -c "import json,sys;sys.exit(0 if json.load(open(sys.argv[1])).get('complete') else 1)" "$1" 2>/dev/null
}
bucket_slice_complete() { # $1 = slice index: progress complete AND all shards present.
  # progress.json can reach the bucket before the last shards do (rsync order
  # is arbitrary), so "complete" is only trusted with a matching object count.
  gcloud storage cp "$BUCKET/${CAP}_w$1/progress.json" "/tmp/prog_$1.json" >/dev/null 2>&1 \
    && complete_json "/tmp/prog_$1.json" || return 1
  local want got
  want=$(python3 -c "import json;print(sum(json.load(open('/tmp/prog_$1.json'))['shards'].values()))")
  got=$(gcloud storage ls "$BUCKET/${CAP}_w$1/*/shard_*.safetensors" 2>/dev/null | wc -l | tr -d ' ')
  [ "$want" -eq "$got" ]
}
verify_local_shards() { # $1 = worker root: files on disk match progress.json exactly
  python3 - "$1" <<'EOF'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
prog = json.load(open(root / "progress.json"))
for pool, n in prog.get("shards", {}).items():
    have = sorted(p.name for p in (root / pool).glob("shard_*.safetensors"))
    want = [f"shard_{i:05d}.safetensors" for i in range(n)]
    if have[: len(want)] != want:  # missing/misnamed prefix => unusable checkpoint
        sys.exit(1)
sys.exit(0)
EOF
}

# ---- phase 0: A7 gate + A3 re-verification (PASS cached in the bucket) ----- #
GATE=reports/hook_validation_cuda.json
if gcloud storage cp "$BUCKET/$GATE" "$GATE" >/dev/null 2>&1 \
   && python3 -c "import json,sys;sys.exit(0 if json.load(open('$GATE'))['gate']=='PASS' else 1)"; then
  say "A7 gate: PASS report already in bucket ($GATE)"
else
  say "A7 gate: running FP32 hook validation on this runtime (~40 min)"
  python scripts/run_capture.py --export-validation-inputs validation_inputs --seed "$SEED"
  # the streamed FP32 reference forward reads safetensors straight from the HF
  # cache (CheckpointStreamer) — prefetch weights on a fresh VM before gating
  say "prefetching model weights into the HF cache"
  python - <<'EOF'
from huggingface_hub import snapshot_download
from moe_sae.model_loading import MODEL_ID
snapshot_download(MODEL_ID, allow_patterns=["*.safetensors*", "*.json", "*.model", "*.txt"])
EOF
  # exits nonzero on FAIL -> set -e halts everything (halt, never skip)
  python scripts/run_validate_gate.py --inputs validation_inputs --out "$GATE" --device cuda
  say "A3: split-contamination re-verification (consequence freeze)"
  python -m pytest tests/test_data_splits.py -q
  gcloud storage cp "$GATE" "$BUCKET/$GATE"
  say "A7 gate PASS + A3 green — report cached in bucket; logbook this"
fi

# ---- background pusher: checkpoint freshness for session-death resume ------ #
pusher_loop() {
  while :; do
    sleep 600
    for d in ${CAP}_w*; do
      [ -d "$d" ] || continue
      # a completed slice belongs to its slot's verified finalization push —
      # never rsync it concurrently (temp-object races corrupt the upload)
      complete_json "$d/progress.json" && continue
      gcloud storage rsync -r -q "$d" "$BUCKET/$d" 2>/dev/null || true
    done
  done
}
# Kill-drill lesson (docs/RUNBOOK.md): an interrupted previous run can orphan
# capture workers and its pusher — Jupyter interrupts don't reliably reach
# background subshells, and two writers on one capture dir corrupt shard
# numbering. Sweep before launching anything.
pkill -f "scripts/run_capture.py --out captures/" 2>/dev/null || true
pkill -f "gcloud storage rsync .*discovery_50m" 2>/dev/null || true
sleep 2

pusher_loop & PUSHER=$!
trap 'kill "$PUSHER" 2>/dev/null || true' EXIT

# ---- phase 1: sliced capture ------------------------------------------------ #
run_slice() {
  local k=$1 out="${CAP}_w$k"
  if bucket_slice_complete "$k"; then
    rm -rf "$out"
    say "slice $k: already complete in bucket — skipped"
    return 0
  fi
  local RESUME=""
  if [ -f "$out/progress.json" ]; then
    RESUME="--resume"
  elif gcloud storage ls "$BUCKET/$out/progress.json" >/dev/null 2>&1; then
    say "slice $k: pulling checkpoint from bucket"
    gcloud storage rsync -r -q "$BUCKET/$out" "$out" || true
    if verify_local_shards "$out"; then
      RESUME="--resume"
    else
      # a push died between shards and progress.json: checkpoint unusable
      say "slice $k: bucket checkpoint inconsistent — recapturing slice from scratch"
      rm -rf "$out"
      gcloud storage rm -r "$BUCKET/$out" >/dev/null 2>&1 || true
    fi
  fi
  # Progress-aware retry (same doctrine as run_pipeline_capture.sh): HF-stream
  # failures land in the iterator, OUTSIDE per-example isolation, and kill the
  # worker — transient CDN weather, so retry with backoff as long as each
  # attempt advances the checkpoint; give up only when attempts stop paying.
  local attempts=0 last_captured=-1 captured
  while :; do
    say "slice $k: capturing $PER examples from --start $((k * PER)) $RESUME"
    PYTHONUNBUFFERED=1 python scripts/run_capture.py \
      --out "$out" --start $((k * PER)) --examples "$PER" \
      --visual-cap "$VISUAL_CAP" --seed "$SEED" --device cuda $RESUME \
      < /dev/null >> "logs/capture_w$k.log" 2>&1 || true
    complete_json "$out/progress.json" && break
    captured=$(python3 -c "import json;print(json.load(open('$out/progress.json')).get('captured',0))" 2>/dev/null || echo 0)
    if [ "$captured" -gt "$last_captured" ]; then attempts=0; last_captured=$captured; else attempts=$((attempts + 1)); fi
    if [ "$attempts" -ge 5 ]; then
      say "slice $k: no progress over 5 attempts (logs/capture_w$k.log) — systematic, giving up"
      return 1
    fi
    say "slice $k: worker died at $captured/$PER — retry $attempts/5 in $((30 * attempts + 15))s"
    sleep $((30 * attempts + 15))
    [ -f "$out/progress.json" ] && RESUME="--resume"
  done
  say "slice $k: pushing to bucket"
  # --delete-unmatched: drop bucket-side strays from interrupted pushes so the
  # bucket copy exactly mirrors the completed slice before we verify + wipe
  gcloud storage rsync -r -q --delete-unmatched-destination-objects "$out" "$BUCKET/$out"
  bucket_slice_complete "$k" || { say "slice $k: bucket progress.json not complete after push"; return 1; }
  local want got
  want=$(find "$out" -name 'shard_*.safetensors' | wc -l | tr -d ' ')
  got=$(gcloud storage ls "$BUCKET/$out/*/shard_*.safetensors" | wc -l | tr -d ' ')
  [ "$want" -eq "$got" ] || { say "slice $k: bucket has $got shards, local has $want — NOT wiping"; return 1; }
  rm -rf "$out"
  say "slice $k: complete — $want shards verified in bucket, local wiped"
}

slot_loop() { # slot s takes slices s, s+CONCURRENCY, s+2*CONCURRENCY, ...
  local s=$1 k
  for ((k = s; k < SLICES; k += CONCURRENCY)); do
    run_slice "$k" || exit 1
  done
}

say "phase 1: $SLICES slices x $PER examples, $CONCURRENCY concurrent slots"
pids=()
for ((s = 0; s < CONCURRENCY; s++)); do
  slot_loop "$s" &
  pids+=($!)
done
fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
[ "$fail" = 0 ] || { say "some slices incomplete — rerun this script (it resumes)"; exit 1; }

say "capture complete: ${SLICES} slices at $BUCKET/${CAP}_w*"
say "next: python scripts/cloud/merge_in_bucket.py --bucket $BUCKET --workers $SLICES --execute"
say "then (TPU host, in-region): gcsfuse mount -> run_stats.py -> 03_discovery_suite.sh"
