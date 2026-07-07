#!/bin/bash
# Per-host environment for the v4-32 slice (4 TPU-VM hosts, 4 chips each).
# Run on every worker:
#   gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all \
#     --command="bash -s" < scripts/tpu/01_setup_hosts.sh
set -euo pipefail

REPO_URL="${REPO_URL:?set REPO_URL=<git url or gcs tarball>}"
BUCKET="${BUCKET:?set BUCKET=gs://...}"

sudo apt-get install -y -q python3.11 python3.11-venv git >/dev/null || true
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

git clone "$REPO_URL" ~/moe-sae || (cd ~/moe-sae && git pull)
cd ~/moe-sae
uv sync --no-dev
# torch stays CPU-only on TPU hosts (data pipeline / BatchComposer only);
# jax gets the TPU runtime:
uv pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
uv run --no-sync python -c "import jax; print(jax.devices())"  # expect 4 TpuDevice per host

# Stage shards into host RAM-backed storage for the compute-bound fast path
# (each v4 host has ~400 GB RAM; the 770 GB discovery set shards 1/4 per host,
# pipeline-scale fits trivially):
mkdir -p /dev/shm/moe_sae
gcloud storage rsync -r "$BUCKET/captures/pipeline_1m" /dev/shm/moe_sae/pipeline_1m
