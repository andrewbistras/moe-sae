#!/bin/bash
# TPU v4-32 provisioning (GCP us-central2-b — the region where v4 pods live;
# do NOT place the bucket elsewhere: in-region GCS reads are free, cross-region
# or cross-cloud egress on ~770 GB of shards is the one large avoidable cost).
#
# NOTHING in this file runs automatically. Review, set PROJECT, then execute
# stanzas by hand. Works for both paid quota and TRC grants (TRC allocations
# are typically in us-central2 as well).
set -euo pipefail

PROJECT="${PROJECT:?set PROJECT=<gcp-project-id>}"
ZONE=us-central2-b
REGION=us-central2
BUCKET="gs://${PROJECT}-moe-sae"
TPU_NAME=moe-sae-v4-32
RUNTIME=tpu-ubuntu2204-base

# --- 1. bucket, colocated with the TPUs ------------------------------------ #
gcloud storage buckets create "$BUCKET" \
  --project="$PROJECT" --location="$REGION" \
  --uniform-bucket-level-access

# --- 2. the v4-32 slice (queued resource; add --spot for preemptible) ------- #
gcloud compute tpus queued-resources create "$TPU_NAME" \
  --project="$PROJECT" --zone="$ZONE" \
  --node-id="$TPU_NAME" \
  --accelerator-type=v4-32 \
  --runtime-version="$RUNTIME"
# poll: gcloud compute tpus queued-resources describe $TPU_NAME --zone=$ZONE

# --- 3. shard upload (choose ONE source) ------------------------------------ #
# (a) pipeline shards from the laptop (~16 GB — fine for smoke tests):
#   gcloud storage rsync -r captures/pipeline_1m "$BUCKET/captures/pipeline_1m"
#   gcloud storage rsync -r stats/pipeline_1m   "$BUCKET/stats/pipeline_1m"
#   gcloud storage rsync -r runs/pipeline_1m_A  "$BUCKET/runs/pipeline_1m_A"   # for parity vs MPS
# (b) discovery shards (~770 GB): do NOT upload from a laptop — capture them
#     in-cloud on a GPU box writing directly to $BUCKET (see scripts/cloud/).
