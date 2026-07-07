#!/bin/bash
# Discovery capture (§20 step 4 data) on a rented GPU box — the capture path
# stays on the FP32-gate-validated PyTorch implementation; only SAE training
# moves to TPU. Shards stream straight to the us-central2 bucket (no laptop
# involved; datasets stream from HF hub).
#
# Sizing: ~96k examples at visual-cap 512 ≈ 50M tokens; ~0.3 s/example on an
# A100 ≈ 8 h ≈ $10-25 spot. Disk: shards buffer locally then rsync (needs
# ~800 GB scratch, or rsync incrementally as below).
set -euo pipefail

BUCKET="${BUCKET:?set BUCKET=gs://...}"
EXAMPLES="${EXAMPLES:-96000}"

cd ~/moe-sae
# resilient runner, same semantics as the laptop version (checkpoint+resume):
OUT=captures/discovery_50m EXAMPLES=$EXAMPLES VISUAL_CAP=512 SEED=7 DEVICE=cuda \
  scripts/run_pipeline_capture.sh &
CAPTURE_PID=$!

# incremental sync every 10 min so the box can be preempted safely; progress
# checkpoint travels with the shards, so a fresh box resumes from the bucket:
while kill -0 $CAPTURE_PID 2>/dev/null; do
  sleep 600
  gcloud storage rsync -r captures/discovery_50m "$BUCKET/captures/discovery_50m" || true
done
gcloud storage rsync -r captures/discovery_50m "$BUCKET/captures/discovery_50m"

# stats pass (fast, same box), then the TPU side takes over:
uv run --no-sync python scripts/run_stats.py \
  --capture captures/discovery_50m --out stats/discovery_50m
gcloud storage rsync -r stats/discovery_50m "$BUCKET/stats/discovery_50m"
echo "capture + stats in $BUCKET — hand off to scripts/tpu/03_discovery_suite.sh"
