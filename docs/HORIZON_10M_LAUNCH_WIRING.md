# Gemma 10M controller launch wiring

This is the concrete controller-side handoff for the 10M SAE/transcoder
learning-horizon study. It deliberately contains no qualification command.
Every accelerator training invocation is the direct, checkpoint-resumable
path with a bound of at most 3,600 seconds.

The currently safe fallback is one already-READY private `v6e-16`:

```text
gemma4-horizon-sae-f
europe-west4-a
10.253.33.106:/moe_sae_control
gs://moe-sae-horizon-ew4-neural-dragnet
```

It can train all SAE and transcoder A/R/B/F trajectories sequentially. Extra
Europe slices may execute independent trajectories concurrently because they
share the same Europe Filestore and mirror. Do not split the four conditions
of one scientific model across Europe and US-East unless their small run
control trees are explicitly reconciled before the model's joint evaluator.

## Seal-time inputs

Exactly these values remain unknown until capture finishes:

```zsh
export PROJECT_COMMIT='<full clean 40-character experiment commit>'
export TARGET_MANIFEST_SHA256='<sealed 10M contribution manifest SHA-256>'
export OVERLAY_CAPTURE_CONTRACT_SHA256='<target-bound overlay capture config hash>'
export OVERLAY_MANIFEST_SHA256='<sealed 10M routed-input overlay manifest SHA-256>'
export TRANSCODER_HARVEST_RECEIPT_SHA256='<sealed overlay harvest receipt SHA-256>'
export TRANSCODER_HARVEST_CHIP_SECONDS='<nonnegative measured shared capture cost>'
```

All six are scientific inputs. The TPU name, create time, controller/JAX
permutation, cache capacity, and regional mirror generations are operational
identities discovered and sealed after allocation.

Run the controller as the dedicated `tpuops` account. It is UID/GID 2001,
which owns the central NFS control tree and matches the systemd unit. The
clean controller checkout and venv belong under that account:

```zsh
sudo -iu tpuops
```

The expected source paths and prefixes are:

```zsh
export REPO='/home/tpuops/src/moe-sae'
export CENTRAL_PROD='/mnt/moe-sae-control'
export TPU_REPO='/home/abistras/moe-sae'
export TPU_PROD="$TPU_REPO/artifacts/gemma4_v1/production"
export REMOTE_UV='/home/abistras/.local/bin/uv'

export TARGET_TRAIN_REL='experiment2/contributions/direct-b9c54aa7dce5/phase3_full_width_pilot/train'
export TARGET_EVAL_REL='experiment2/contributions/direct-b9c54aa7dce5/phase2_mechanism_pilot/eval'
export OVERLAY_TRAIN_REL='horizon-10m/captures/transcoder-overlay/train'
export OVERLAY_EVAL_REL='experiment3/routed-input-overlay/eval'
export SIDECAR_REL='experiment2/qualification/20260725T085844Z-e5d2b6250ec1/b128-ep8-fsdp2/recomposition-sidecar.json'

export TARGET_TRAIN_PREFIX='gs://moe-sae-bucket/moe-sae/gemma4_v1/experiment2/contributions/train'
export TARGET_EVAL_PREFIX='gs://moe-sae-bucket/moe-sae/gemma4_v1/experiment2/contributions/eval'
export OVERLAY_TRAIN_PREFIX='gs://moe-sae-bucket/moe-sae/gemma4_v1/horizon-10m/transcoder-overlay/train'
export OVERLAY_EVAL_PREFIX='gs://moe-sae-bucket/moe-sae/gemma4_v1/experiment3/routed-input-overlay/eval'
```

Before proceeding, compare the three supplied SHA-256s to the manifests:

```zsh
test "$(jq -r .manifest_sha256 "$CENTRAL_PROD/$TARGET_TRAIN_REL/manifest.json")" = "$TARGET_MANIFEST_SHA256"
test "$(jq -r .manifest_sha256 "$CENTRAL_PROD/$OVERLAY_TRAIN_REL/manifest.json")" = "$OVERLAY_MANIFEST_SHA256"
test "$(jq -r .manifest_sha256 "$CENTRAL_PROD/$TARGET_EVAL_REL/manifest.json")" = '31454e5ce8693521c41a2a95f2ef88d40c5d26c726a0efac5c26d882ed341f5e'
test "$(jq -r .manifest_sha256 "$CENTRAL_PROD/$OVERLAY_EVAL_REL/manifest.json")" = 'f68e2fe6e4cf568e694925697dd5d65fe10160a52d357c641892de85ee5c3da7'
```

## Finalize the executable configs

The target-bound overlay capture config is finalized before overlay harvest.
Its printed `capture_contract_sha256` becomes
`OVERLAY_CAPTURE_CONTRACT_SHA256`:

```zsh
cd "$REPO"
uv run python -m moe_sae.contribution_transcoder_capture_config finalize \
  --template configs/gemma4_transcoder_horizon_v1.yaml \
  --output "$CENTRAL_PROD/horizon-10m/contracts/transcoder-overlay-capture.yaml" \
  --train-target-manifest-sha256 "$TARGET_MANIFEST_SHA256"
```

After the overlay seals, finalize both training configs:

```zsh
uv run python -m moe_sae.learning_horizon_config finalize \
  --template configs/gemma4_contribution_horizon_v1.yaml \
  --output "$CENTRAL_PROD/horizon-10m/contracts/sae.yaml" \
  --train-target-manifest-sha256 "$TARGET_MANIFEST_SHA256"

uv run python -m moe_sae.learning_horizon_config finalize \
  --template configs/gemma4_transcoder_horizon_v1.yaml \
  --output "$CENTRAL_PROD/horizon-10m/contracts/transcoder.yaml" \
  --train-target-manifest-sha256 "$TARGET_MANIFEST_SHA256" \
  --train-source-overlay-manifest-sha256 "$OVERLAY_MANIFEST_SHA256" \
  --source-overlay-capture-contract-sha256 "$OVERLAY_CAPTURE_CONTRACT_SHA256"
```

## Mirror once per region

Both regional plans contain all four datasets. This means the Europe fallback
can execute either scientific model without another inter-region transfer.
US-East remains ready for an opportunistic transcoder slice.

```zsh
for REGION BUCKET in \
  europe-west4 gs://moe-sae-horizon-ew4-neural-dragnet \
  us-east1     gs://moe-sae-horizon-use1-neural-dragnet
do
  export REGIONAL_ROOT="$CENTRAL_PROD/horizon-10m/regional/$REGION"
  install -d -m 0700 "$REGIONAL_ROOT"
  uv run python scripts/tpu/prepare_regional_mirror.py plan \
    --run-id gemma4-horizon-10m \
    --region "$REGION" \
    --mirror-prefix "$BUCKET/horizon-10m" \
    --target-control-root "$CENTRAL_PROD/$TARGET_TRAIN_REL" \
    --target-object-prefix "$TARGET_TRAIN_PREFIX" \
    --eval-target-control-root "$CENTRAL_PROD/$TARGET_EVAL_REL" \
    --eval-target-object-prefix "$TARGET_EVAL_PREFIX" \
    --recomposition-sidecar "$CENTRAL_PROD/$SIDECAR_REL" \
    --overlay-control-root "$CENTRAL_PROD/$OVERLAY_TRAIN_REL" \
    --overlay-object-prefix "$OVERLAY_TRAIN_PREFIX" \
    --eval-overlay-control-root "$CENTRAL_PROD/$OVERLAY_EVAL_REL" \
    --eval-overlay-object-prefix "$OVERLAY_EVAL_PREFIX" \
    --output "$REGIONAL_ROOT/mirror-plan.json"
  uv run python scripts/tpu/prepare_regional_mirror.py copy-and-seal \
    --project neural-dragnet \
    --region "$REGION" \
    --plan "$REGIONAL_ROOT/mirror-plan.json" \
    --output "$REGIONAL_ROOT/mirror-seal.json" \
    --workers 16 \
    --apply
done
```

The copy is generation-pinned, destination-create-only, and replay-safe. It
copies each unique shard once per region, not once per TPU.

## Deploy and stage the small POSIX plane

The regional GCS mirror does not include manifests or configs. Deploy first,
which mounts the Europe Filestore on all four controllers:

```zsh
export GCP_PROJECT='neural-dragnet'
export TPU_NAME='gemma4-horizon-sae-f'
export TPU_ZONE='europe-west4-a'
export TPU_SERVICE_ACCOUNT='655343590563-compute@developer.gserviceaccount.com'
export TPU_PROJECT_ROOT="$TPU_REPO"
export LOCAL_BUFFER_ROOT='/dev/shm/moe-sae-local'
export BULK_GCS_ROOT='gs://moe-sae-horizon-ew4-neural-dragnet/horizon-10m'
export NFS_SERVER='10.253.33.106'
export NFS_EXPORT='/moe_sae_control'
export TPU_ACCELERATOR_TYPE='v6e-16'
export TPU_USE_INTERNAL_IP='1'

bash scripts/tpu/deploy_workers.sh
```

Build a create-only bundle containing only required control files:

```zsh
export CONTROL_BUNDLE="/tmp/gemma4-horizon-control-$PROJECT_COMMIT.tar"
uv run python scripts/tpu/stage_regional_control.py build \
  --source-root "$CENTRAL_PROD" \
  --relative-file "$TARGET_TRAIN_REL/manifest.json" \
  --relative-file "$TARGET_EVAL_REL/manifest.json" \
  --relative-file "$OVERLAY_TRAIN_REL/manifest.json" \
  --relative-file "$OVERLAY_EVAL_REL/manifest.json" \
  --relative-file "$SIDECAR_REL" \
  --relative-file 'horizon-10m/contracts/sae.yaml' \
  --relative-file 'horizon-10m/contracts/transcoder.yaml' \
  --relative-file 'horizon-10m/regional/europe-west4/mirror-seal.json' \
  --output "$CONTROL_BUNDLE"

gcloud compute tpus tpu-vm scp "$CONTROL_BUNDLE" \
  "$TPU_NAME:/tmp/horizon-control.tar" \
  --project="$GCP_PROJECT" \
  --zone="$TPU_ZONE" \
  --worker=0 \
  --internal-ip \
  --quiet

gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
  --project="$GCP_PROJECT" \
  --zone="$TPU_ZONE" \
  --worker=0 \
  --internal-ip \
  --quiet \
  --command="$REMOTE_UV run --frozen --extra tpu python $TPU_REPO/scripts/tpu/stage_regional_control.py apply --bundle /tmp/horizon-control.tar --destination-root $TPU_PROD"
```

Reapplying identical bytes succeeds; any different existing path fails. There
is no `rsync --delete`, overwrite, schema autodetection, or bulk shard copy in
this step.

## Attest and probe the ready slice

Run the attester on worker zero so its outputs land directly on the mounted
regional Filestore:

```zsh
export TOPOLOGY_REL="horizon-10m/topology/$TPU_NAME"
export TOPOLOGY_REMOTE="$TPU_PROD/$TOPOLOGY_REL"
export PROBE_ID="horizon10m-${PROJECT_COMMIT[1,12]}-$(date -u +%Y%m%dT%H%M%SZ)"

gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
  --project="$GCP_PROJECT" \
  --zone="$TPU_ZONE" \
  --worker=0 \
  --internal-ip \
  --quiet \
  --command="$REMOTE_UV run --frozen --extra tpu python $TPU_REPO/scripts/tpu/attest_accelerator.py --run-id gemma4-horizon-10m --scientific-model sae --condition F --project $GCP_PROJECT --tpu-name $TPU_NAME --zone $TPU_ZONE --accelerator-type v6e-16 --output-root $TOPOLOGY_REMOTE"

uv run python scripts/tpu/probe_controller_identity.py \
  --project "$GCP_PROJECT" \
  --tpu-name "$TPU_NAME" \
  --zone "$TPU_ZONE" \
  --project-commit "$PROJECT_COMMIT" \
  --project-root "$TPU_REPO" \
  --remote-uv "$REMOTE_UV" \
  --contract "$TOPOLOGY_REMOTE/accelerator-contract.json" \
  --probe-id "$PROBE_ID" \
  --probe-root "$TOPOLOGY_REMOTE/probes/$PROBE_ID" \
  --cache-root '/dev/shm/moe-sae-local/horizon-10m/cache' \
  --required-lane-bytes 159000000000 \
  --required-free-bytes 206158430208 \
  --internal-ip
```

This is an identity/capacity probe, not a qualification or performance run.
The two paths consumed by every direct attempt are:

```zsh
export CONTROLLER_MAP="$TOPOLOGY_REMOTE/probes/$PROBE_ID/controller-mapping.json"
export CACHE_INVENTORY="$TOPOLOGY_REMOTE/probes/$PROBE_ID/cache-inventory.json"
export MIRROR_SEAL="$TPU_PROD/horizon-10m/regional/europe-west4/mirror-seal.json"
```

## Direct segmented training

Every segment uses the same scientific `run-id` and a new monotonic
`attempt-id`. It resumes only through
`runs/<run-id>/checkpoints/latest.json`. A controller should accept a segment
boundary only if that pointer changed. `training-result.json` is the sole
completion.

The common environment is:

```zsh
typeset -a COMMON_ENV
COMMON_ENV=(
  "GCP_PROJECT=$GCP_PROJECT"
  "TPU_NAME=$TPU_NAME"
  "TPU_ZONE=$TPU_ZONE"
  "PROJECT_COMMIT=$PROJECT_COMMIT"
  "TPU_PROJECT_ROOT=$TPU_REPO"
  "REMOTE_UV=$REMOTE_UV"
  'TPU_ACCELERATOR_TYPE=v6e-16'
  'TPU_USE_INTERNAL_IP=1'
  "ARTIFACT_ROOT=$TPU_PROD"
  "BULK_GCS_ROOT=$BULK_GCS_ROOT"
  "CONTRIBUTION_TRAIN_CAPTURE_ROOT=$TPU_PROD/$TARGET_TRAIN_REL"
  "CONTRIBUTION_EVAL_CAPTURE_ROOT=$TPU_PROD/$TARGET_EVAL_REL"
  "CONTRIBUTION_OBJECT_PREFIX=$TARGET_TRAIN_PREFIX"
  "CONTRIBUTION_EVAL_OBJECT_PREFIX=$TARGET_EVAL_PREFIX"
  "CONTRIBUTION_RECOMPOSITION_SIDECAR=$TPU_PROD/$SIDECAR_REL"
  'CONTRIBUTION_SAE_CACHE_ROOT=/dev/shm/moe-sae-local/horizon-10m/cache'
  'CONTRIBUTION_SAE_CACHE_MAX_BYTES=171798691840'
  "CONTRIBUTION_REGIONAL_MIRROR_SEAL=$MIRROR_SEAL"
  "CONTRIBUTION_FRESH_CONTROLLER_MAPPING=$CONTROLLER_MAP"
  "CONTRIBUTION_LOCAL_CACHE_INVENTORY=$CACHE_INVENTORY"
)
```

The exact per-segment SAE invocation is:

```zsh
export MODEL='sae'
export CONDITION='A'  # then R, B, F
export RUN_ID="horizon10m-$MODEL-${(L)CONDITION}-seed0"
export ATTEMPT_ID="segment-$(date -u +%Y%m%dT%H%M%SZ)-$RANDOM"
env "${COMMON_ENV[@]}" \
  "CONTRIBUTION_SAE_CONTROL_ROOT=$TPU_PROD/horizon-10m/runs/$MODEL" \
  'CONTRIBUTION_SAE_CHECKPOINT_ROOT=gs://moe-sae-horizon-ew4-neural-dragnet/horizon-10m/checkpoints/sae' \
  uv run python scripts/tpu/train_contribution.py \
    --action train \
    --scientific-model sae \
    --config "$TPU_PROD/horizon-10m/contracts/sae.yaml" \
    --phase learning_horizon_10m \
    --condition "$CONDITION" \
    --seed 0 \
    --run-id "$RUN_ID" \
    --attempt-id "$ATTEMPT_ID" \
    --maximum-runtime-seconds 3500
```

The exact transcoder invocation adds the sealed overlay:

```zsh
export MODEL='transcoder'
export CONDITION='A'  # then R, B, F
export RUN_ID="horizon10m-$MODEL-${(L)CONDITION}-seed0"
export ATTEMPT_ID="segment-$(date -u +%Y%m%dT%H%M%SZ)-$RANDOM"
env "${COMMON_ENV[@]}" \
  "CONTRIBUTION_SOURCE_OVERLAY_ROOT=$TPU_PROD/$OVERLAY_TRAIN_REL" \
  "CONTRIBUTION_SOURCE_OVERLAY_PREFIX=$OVERLAY_TRAIN_PREFIX" \
  "CONTRIBUTION_SAE_CONTROL_ROOT=$TPU_PROD/horizon-10m/runs/$MODEL" \
  'CONTRIBUTION_SAE_CHECKPOINT_ROOT=gs://moe-sae-horizon-ew4-neural-dragnet/horizon-10m/checkpoints/transcoder' \
  uv run python scripts/tpu/train_contribution.py \
    --action train \
    --scientific-model transcoder \
    --config "$TPU_PROD/horizon-10m/contracts/transcoder.yaml" \
    --phase learning_horizon_10m \
    --condition "$CONDITION" \
    --seed 0 \
    --run-id "$RUN_ID" \
    --attempt-id "$ATTEMPT_ID" \
    --maximum-runtime-seconds 3500
```

There is intentionally no `qualify` action and no qualification bundle. A
cloud-resident supervisor repeats a segment only for a witnessed checkpoint
boundary, Spot preemption/resource loss, or SSH transport loss. Numerical,
manifest, identity, schema, token-drop, and overflow failures are terminal.

## Evaluate 2M, 5M, and 10M

Each evaluator invocation jointly evaluates A/R/B/F. Therefore there are
three accelerator invocations per scientific model, not twelve redundant
invocations. The resulting `scientific-result.json` is the shared immutable
result for all four condition roles at that milestone.

SAE:

```zsh
for TOKENS in 2000000 5000000 10000000
do
  env "${COMMON_ENV[@]}" \
    "CONTRIBUTION_SAE_CONTROL_ROOT=$TPU_PROD/horizon-10m/runs/sae" \
    'CONTRIBUTION_SAE_CHECKPOINT_ROOT=gs://moe-sae-horizon-ew4-neural-dragnet/horizon-10m/checkpoints/sae' \
    uv run python scripts/tpu/evaluate_contribution.py \
      --config "$TPU_PROD/horizon-10m/contracts/sae.yaml" \
      --phase learning_horizon_10m \
      --seed 0 \
      --checkpoint-tokens "$TOKENS" \
      --result-root "$TPU_PROD/horizon-10m/evaluation/sae/$TOKENS" \
      --run-root-a "$TPU_PROD/horizon-10m/runs/sae/runs/horizon10m-sae-a-seed0" \
      --run-root-r "$TPU_PROD/horizon-10m/runs/sae/runs/horizon10m-sae-r-seed0" \
      --run-root-b "$TPU_PROD/horizon-10m/runs/sae/runs/horizon10m-sae-b-seed0" \
      --run-root-f "$TPU_PROD/horizon-10m/runs/sae/runs/horizon10m-sae-f-seed0"
done
```

Transcoder:

```zsh
for TOKENS in 2000000 5000000 10000000
do
  env "${COMMON_ENV[@]}" \
    "CONTRIBUTION_SOURCE_OVERLAY_ROOT=$TPU_PROD/$OVERLAY_TRAIN_REL" \
    "CONTRIBUTION_EVAL_SOURCE_OVERLAY_ROOT=$TPU_PROD/$OVERLAY_EVAL_REL" \
    "CONTRIBUTION_SOURCE_OVERLAY_PREFIX=$OVERLAY_TRAIN_PREFIX" \
    "CONTRIBUTION_EVAL_SOURCE_OVERLAY_PREFIX=$OVERLAY_EVAL_PREFIX" \
    "CONTRIBUTION_SAE_CONTROL_ROOT=$TPU_PROD/horizon-10m/runs/transcoder" \
    'CONTRIBUTION_SAE_CHECKPOINT_ROOT=gs://moe-sae-horizon-ew4-neural-dragnet/horizon-10m/checkpoints/transcoder' \
    "TRAINING_PROJECT_COMMIT=$PROJECT_COMMIT" \
    'TRANSCODER_PEAK_FLOPS_PER_SECOND=14688000000000000' \
    "TRANSCODER_HARVEST_RECEIPT_SHA256=$TRANSCODER_HARVEST_RECEIPT_SHA256" \
    "TRANSCODER_HARVEST_CHIP_SECONDS=$TRANSCODER_HARVEST_CHIP_SECONDS" \
    uv run python scripts/tpu/evaluate_transcoder.py \
      --config "$TPU_PROD/horizon-10m/contracts/transcoder.yaml" \
      --phase learning_horizon_10m \
      --seed 0 \
      --checkpoint-tokens "$TOKENS" \
      --result-root "$TPU_PROD/horizon-10m/evaluation/transcoder/$TOKENS" \
      --run-root-a "$TPU_PROD/horizon-10m/runs/transcoder/runs/horizon10m-transcoder-a-seed0" \
      --run-root-r "$TPU_PROD/horizon-10m/runs/transcoder/runs/horizon10m-transcoder-r-seed0" \
      --run-root-b "$TPU_PROD/horizon-10m/runs/transcoder/runs/horizon10m-transcoder-b-seed0" \
      --run-root-f "$TPU_PROD/horizon-10m/runs/transcoder/runs/horizon10m-transcoder-f-seed0"
done
```

The six output directories each contain the immutable scientific JSON; the
transcoder evaluator also renders `REPORT.md`. Preserve them before deleting
the TPU or regional Filestore.

## Parallel capacity policy

The fallback above is scientifically complete with one slice. If additional
Europe `v6e-16` slices become READY, deploy/stage/attest/probe each incarnation
and assign different trajectories concurrently. Their `run-id` and
checkpoint namespace stay unchanged. Never allow two slices to own the same
trajectory simultaneously.

If all four US-East slices become READY, the whole transcoder model
(A/R/B/F plus its three joint evaluations) may move to US-East using:

```text
10.59.245.2:/moe_sae_control
gs://moe-sae-horizon-use1-neural-dragnet
us-east1-d
```

This preserves one region per joint evaluator and avoids unnecessary
cross-region checkpoint reads. The GCS object mirror remains one copy per
region regardless of how many slices are added.
