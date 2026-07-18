# Production runbook

Every stage is fail closed. Do not continue past a failed parity, storage,
integrity, MFU, goodput, health, or exact-count gate. Production commands that
initialize JAX must be launched collectively with the same arguments on all
four v4-32 controllers; the snippets below show that per-controller command.

## Stage 1 — prepare the immutable data and model inputs

Stage 1 happens before the TPU run. It is complete when the packed train and
validation data are sealed in GCS, Gemma has been converted from one immutable
Hugging Face commit, logit parity has passed, and you have recorded the six
locations/digests needed by the TPU stages.

Your v4-32 is covered by the TRC allocation. That removes TPU cost anxiety, but
it does not change this preparation or any acceptance gate. Use the project,
zone, service account, and provisioning process in the TRC award. Do not create
a personal billed TPU. GCS and a temporary high-memory conversion VM may still
count against separate TRC quotas.

### 1.1 Check the repository and local tools

On the machine where you will stage C4:

```zsh
cd /Users/abistras/Desktop/moe-sae

# In zsh, make a failing command on the left of `| tee` fail the pipeline.
setopt PIPE_FAIL

git status --short
export REPOSITORY_URL="$(git remote get-url origin)"
export PROJECT_COMMIT="$(git rev-parse HEAD)"
print -r -- "$REPOSITORY_URL"
print -r -- "$PROJECT_COMMIT"
uv --version
gcloud --version
jq --version
```

For a production run, `git status --short` should print nothing. Commit the
implementation first so every later fingerprint refers to a stable tree. If
`jq` is missing on the Mac, install it with `brew install jq`.

Keep the large optional dependency sets separate. This prevents CPU Torch and
data libraries from leaking into the TPU runtime:

```zsh
export DATA_ENV="$HOME/.virtualenvs/moe-sae-data"
export CONVERSION_ENV="$HOME/.virtualenvs/moe-sae-conversion"

UV_PROJECT_ENVIRONMENT="$DATA_ENV" \
  uv sync --frozen --extra data
```

Create the conversion environment later on the Linux conversion host, not on
the Mac. Every data command below explicitly selects `DATA_ENV`.

### 1.2 Select the TRC project and authenticate both clients

Set the project ID from the TRC award, then authenticate the `gcloud` CLI and
Python Application Default Credentials. They are separate credential paths:

```zsh
export GCP_PROJECT="<TRC project ID>"

gcloud auth login
gcloud config set project "$GCP_PROJECT"
gcloud auth application-default login
gcloud auth application-default set-quota-project "$GCP_PROJECT"

gcloud services enable \
  compute.googleapis.com \
  storage.googleapis.com \
  tpu.googleapis.com
```

If enabling an API is denied, ask the TRC project administrator. Do not switch
silently to a personal project.

### 1.3 Select or create the GCS bucket

If TRC supplied a bucket, use it and skip creation. Otherwise choose a globally
unique name and ask the TRC administrator which location is permitted. A
v4-32 is commonly provisioned in `us-central2-b`, but ordinary Cloud Storage
does not offer `us-central2` as a regional bucket location. `US` or
`us-central1` are reasonable examples only when the administrator approves
them:

```zsh
export BUCKET="<globally-unique-bucket-name>"
export BUCKET_LOCATION="<TRC-approved location, for example US>"

gcloud storage buckets create "gs://$BUCKET" \
  --project="$GCP_PROJECT" \
  --location="$BUCKET_LOCATION" \
  --default-storage-class=STANDARD \
  --uniform-bucket-level-access \
  --public-access-prevention \
  --soft-delete-duration=0
```

This is a scratch/experiment bucket, so the command explicitly disables the
otherwise-default seven-day soft-delete window. Deleted qualification and
benchmark objects would remain billable during that window. If TRC requires
recovery retention, omit this flag and accept the separate GCS cost. Do not
change an existing shared bucket's policy without its owner. See Google's
[bucket-location list](https://cloud.google.com/storage/docs/locations) and
[soft-delete documentation](https://cloud.google.com/storage/docs/soft-delete)
before choosing differently.

Define the Stage 1 paths. The two source/model revisions are already pinned in
`configs/gemma4_26b_a4b.yaml`; operators do not supply revision flags.

```zsh
export ARTIFACT_ROOT="$PWD/artifacts/gemma4_v1/production"
export C4_SOURCE_CACHE="$HOME/.cache/moe-sae/c4-source"
export LOCAL_C4_STAGE="$HOME/moe-sae-work/c4-stage"
# One 319 MB file is simpler and more reliable over ordinary HTTP than Xet.
export HF_HUB_DISABLE_XET=1

export BULK_GCS_ROOT="gs://$BUCKET/moe-sae/gemma4_v1"
export MODEL_ROOT="$BULK_GCS_ROOT/model"
export SAE_CHECKPOINT_ROOT="$BULK_GCS_ROOT/sae-checkpoints"

mkdir -p \
  "$ARTIFACT_ROOT/operator-results" \
  "$C4_SOURCE_CACHE" \
  "$LOCAL_C4_STAGE"
```

Check both GCS authentication paths:

```zsh
gcloud storage ls "gs://$BUCKET"

UV_PROJECT_ENVIRONMENT="$DATA_ENV" \
  uv run --frozen --extra data python -c \
  'import os; from google.cloud import storage; b=storage.Client(project=os.environ["GCP_PROJECT"]).bucket(os.environ["BUCKET"]); print("bucket exists:", b.exists())'
```

The second command must print `bucket exists: True`.

### 1.4 Understand the bounded C4 input

The pipeline does not build the 806+ GiB TFDS corpus. It downloads and accepts
only these two objects from the immutable `allenai/c4` revision
`1588ec454efa1a09f29cd18ddd04fe05fc8653a2`:

| Split | Exact source object | Bytes | SHA-256 |
|---|---|---:|---|
| train | `en/c4-train.00000-of-01024.json.gz` | 319,308,785 | `8ef8d75b0e045dec4aa5123a671b4564466b0707086a7ed1ba8721626dfffbc9` |
| validation | `en/c4-validation.00000-of-00008.json.gz` | 40,471,190 | `1f25b6af12da84115301d4ee93ea5246c8fea5bb4a2008472794d95b917cc97f` |

Those bounded shards contain more than the required 50,000,000 train and
2,000,000 held-out Gemma tokens. The stager downloads them itself, verifies
their byte size and SHA-256 before parsing, preserves line order, and fails if
the inventory is exhausted. Do not download, rename, or edit them manually.

The repositories are public at the pinned commits. First try without an HF
token. If Hugging Face returns an authentication or rate-limit error, enter a
read-only token without echoing it:

```zsh
read -s "HF_TOKEN?Hugging Face read token: "
export HF_TOKEN
echo
```

Never put the token in a command argument, file, notebook output, or Git.
`HF_HUB_DISABLE_XET=1` changes only the C4 download transport; the verified
source revision, byte size, and SHA-256 remain identical. It avoids a known
failure mode where Xet establishes connections but does not assemble this
single large gzip object on some laptops.

### 1.5 Run a small staging smoke test

This first command downloads and verifies the train source, then downloads only
Gemma's `tokenizer.json` and `tokenizer_config.json` at the pinned model commit.
Their exact 32 MB payload/file hashes are checked before Transformers loads a
two-file, local-only view. It then writes a few ArrayRecords, seals a manifest,
and exercises the same code used for production:

```zsh
UV_PROJECT_ENVIRONMENT="$DATA_ENV" \
  uv run --frozen --extra data moe-sae stage-c4 \
    --artifact-root "$ARTIFACT_ROOT" \
    --split train \
    --source-cache-dir "$C4_SOURCE_CACHE" \
    --output-directory "$LOCAL_C4_STAGE/smoke/train" \
    --target-tokens 65536 \
    --records-per-shard 128 \
    --local-smoke
```

The JSON result contains `manifest_location`, `grain_files`,
`scientific_tokens`, `record_count`, `source_examples`, and `resumed`. The
first run should normally say `"resumed": false`. Run the exact command again;
it should return the identical inventory with `"resumed": true`.

If this fails, stop here. The exception identifies whether source download,
hashing, tokenizer loading, JSON decoding, packing, ArrayRecord reopening, or
resume validation failed.

### 1.6 Stage and publish the production splits

Run train first:

```zsh
UV_PROJECT_ENVIRONMENT="$DATA_ENV" \
  uv run --frozen --extra data moe-sae stage-c4 \
    --artifact-root "$ARTIFACT_ROOT" \
    --split train \
    --source-cache-dir "$C4_SOURCE_CACHE" \
    --output-directory "$LOCAL_C4_STAGE/train" \
    --records-per-shard 4096 \
    --publish-gcs "$BULK_GCS_ROOT/datasets/c4/train" \
  | tee "$ARTIFACT_ROOT/operator-results/stage-train.json"
```

Then validation:

```zsh
UV_PROJECT_ENVIRONMENT="$DATA_ENV" \
  uv run --frozen --extra data moe-sae stage-c4 \
    --artifact-root "$ARTIFACT_ROOT" \
    --split validation \
    --source-cache-dir "$C4_SOURCE_CACHE" \
    --output-directory "$LOCAL_C4_STAGE/validation" \
    --records-per-shard 4096 \
    --publish-gcs "$BULK_GCS_ROOT/datasets/c4/validation" \
  | tee "$ARTIFACT_ROOT/operator-results/stage-validation.json"
```

These are safe to re-run after interruption. Resume is by sealed shard state,
not by counting files. A completed staging inventory can be slightly larger
than 50M/2M because records and four host lanes remain statically aligned;
capture masks the tail to the exact scientific token totals.

This is laptop CPU and disk work, not TPU work. Expect train to dominate and
leave the terminal awake; validation is much smaller. The bounded C4 downloads
total about 360 MB and the tokenizer adds about 32 MB once to the shared cache;
the packed output is measured in GB rather than hundreds of GB. An interruption
is inconvenient, not catastrophic: rerun the exact split command and let the
immutable state decide what is complete.

The manifest schema is `gemma4-c4-packed-arrayrecord-v2`. It binds the exact
source repository, commit, file path, compressed size and SHA-256; the exact
Gemma tokenizer commit plus the byte/Git identities of its two local files;
packed record schema; scientific token counts; host lane counts; and the
ordered content-addressed ArrayRecord inventory.

Extract the four values needed by preflight and harvesting:

```zsh
export TRAIN_DATASET_MANIFEST="$(
  jq -r '.manifest_location' \
    "$ARTIFACT_ROOT/operator-results/stage-train.json"
)"
export TRAIN_GRAIN_FILES="$(
  jq -r '.grain_files' \
    "$ARTIFACT_ROOT/operator-results/stage-train.json"
)"
export EVAL_DATASET_MANIFEST="$(
  jq -r '.manifest_location' \
    "$ARTIFACT_ROOT/operator-results/stage-validation.json"
)"
export EVAL_GRAIN_FILES="$(
  jq -r '.grain_files' \
    "$ARTIFACT_ROOT/operator-results/stage-validation.json"
)"

print -r -- "$TRAIN_DATASET_MANIFEST"
print -r -- "$TRAIN_GRAIN_FILES"
print -r -- "$EVAL_DATASET_MANIFEST"
print -r -- "$EVAL_GRAIN_FILES"
```

All four values must begin with `gs://`. Preserve them in your private operator
notes; they are identities, not secrets. The staging CLI calls the held-out
split `validation`; capture and SAE commands call that same data `eval`.

Do not try to revoke write access on just this apparent "directory" unless the
bucket was deliberately created with managed folders and a tested folder-level
IAM policy. Ordinary bucket IAM is not prefix-scoped, and removing bucket-wide
writes would also break later capture and checkpoint publication. The sealed
manifest's object generations and hashes make accidental replacement fail
closed. A dedicated read-only dataset bucket can be added later if the TRC
administrator wants stronger separation.

### 1.7 Prepare the high-memory Linux conversion host

Do not attempt Gemma 26B conversion on the MacBook. The official parity check
uses float32 and holds the Hugging Face and MaxText models concurrently. Require
at least 384 GiB RAM; a 512 GiB host is the safer default. Also require at least
200 GiB free on a fast local filesystem. The model snapshot alone is roughly
52 GB. TPU quota does not automatically make this CPU VM or its disk free.

If TRC gives you a suitable host, use it. If the administrator tells you to
create one, the following is a concrete default. Run it on your Mac after
replacing all three placeholders. `n2-highmem-64` supplies 512 GB of RAM; if
that type or quota is unavailable, stop and ask the administrator rather than
choosing a smaller machine:

```zsh
export CONVERSION_VM="gemma4-conversion"
export CONVERSION_ZONE="<zone with approved N2 quota, for example us-central1-a>"
export CONVERSION_SERVICE_ACCOUNT="<TRC service-account email>"

gcloud compute instances create "$CONVERSION_VM" \
  --project="$GCP_PROJECT" \
  --zone="$CONVERSION_ZONE" \
  --machine-type=n2-highmem-64 \
  --boot-disk-size=300GB \
  --boot-disk-type=pd-ssd \
  --image-family=ubuntu-2404-lts-amd64 \
  --image-project=ubuntu-os-cloud \
  --service-account="$CONVERSION_SERVICE_ACCOUNT" \
  --scopes=https://www.googleapis.com/auth/cloud-platform \
  --labels=purpose=gemma4-conversion

gcloud compute ssh "$CONVERSION_VM" \
  --project="$GCP_PROJECT" \
  --zone="$CONVERSION_ZONE"
```

That service account needs permission to list/read the bucket and create
objects below the model and handoff prefixes. If VM creation or bucket access
is denied, send the exact error to the TRC administrator; do not add a service
account key file or switch projects.

The remaining commands in this section run on the Linux host. Install the small
operator toolset and enter zsh:

```zsh
sudo apt-get update
sudo apt-get install -y apt-transport-https ca-certificates curl git gnupg jq zsh

if ! command -v gcloud >/dev/null; then
  curl -fsSLo /tmp/google-cloud-apt-key.gpg \
    https://packages.cloud.google.com/apt/doc/apt-key.gpg
  sudo gpg --batch --yes --dearmor \
    --output /usr/share/keyrings/cloud.google.gpg \
    /tmp/google-cloud-apt-key.gpg
  echo 'deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main' \
    | sudo tee /etc/apt/sources.list.d/google-cloud-sdk.list >/dev/null
  sudo apt-get update
  sudo apt-get install -y google-cloud-cli
fi

if ! command -v uv >/dev/null; then
  curl -LsSf -o /tmp/uv-install.sh https://astral.sh/uv/install.sh
  sh /tmp/uv-install.sh
fi
export PATH="$HOME/.local/bin:$PATH"
exec zsh
```

The `exec zsh` line starts a fresh shell. Re-establish the safety option and all
cross-host values; paste the exact URL and 40-character commit printed in 1.1:

```zsh
setopt PIPE_FAIL

export GCP_PROJECT="<same TRC project ID>"
export BUCKET="<same bucket name>"
export REPOSITORY_URL="<exact git remote URL from 1.1>"
export PROJECT_COMMIT="<exact 40-character commit from 1.1>"

gcloud config set project "$GCP_PROJECT"
gcloud auth list
curl -fsS \
  -H 'Metadata-Flavor: Google' \
  'http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/email'
print
gcloud storage ls "gs://$BUCKET"
```

The metadata command must print the intended service-account email and the
storage command must succeed. A Compute Engine attached service account is
automatically available to Python ADC; do not run `gcloud auth
application-default login` on that VM. On a non-GCE TRC host, authenticate
`gcloud` and ADC exactly as in 1.2 instead.

Clone and verify this repository. If it is private, authenticate Git first;
never place a GitHub token in `REPOSITORY_URL`:

```zsh
export PROJECT_ROOT="$HOME/src/moe-sae"
mkdir -p "$HOME/src"

git clone "$REPOSITORY_URL" "$PROJECT_ROOT"
git -C "$PROJECT_ROOT" checkout --detach "$PROJECT_COMMIT"
test "$(git -C "$PROJECT_ROOT" rev-parse HEAD)" = "$PROJECT_COMMIT"
test -z "$(git -C "$PROJECT_ROOT" status --porcelain=v1 --untracked-files=all)"
cd "$PROJECT_ROOT"
```

If `$PROJECT_ROOT` already exists, do not clone over it. Verify its remote,
fetch, detach at `$PROJECT_COMMIT`, and require an empty status before
continuing.

Create the isolated conversion environment, then prove the host really has the
required memory, disk, tools, Python ADC, and bucket permissions:

```zsh
export CONVERSION_ENV="$HOME/.virtualenvs/moe-sae-conversion"
UV_PROJECT_ENVIRONMENT="$CONVERSION_ENV" \
  uv sync --frozen --extra conversion

export CHECKPOINT_WORK="$HOME/checkpoint-work"
export HF_CACHE_DIR="$CHECKPOINT_WORK/huggingface"
mkdir -p "$HF_CACHE_DIR"

free -h
df -h "$CHECKPOINT_WORK"
export HOST_MEMORY_KIB="$(awk '/MemTotal/ {print $2}' /proc/meminfo)"
export WORK_FREE_BYTES="$(df --output=avail -B1 "$CHECKPOINT_WORK" | tail -1)"
(( HOST_MEMORY_KIB >= 384 * 1024 * 1024 ))
(( WORK_FREE_BYTES >= 200 * 1024 * 1024 * 1024 ))

git --version
uv --version
gcloud --version
jq --version

UV_PROJECT_ENVIRONMENT="$CONVERSION_ENV" \
  uv run --frozen --extra conversion python -c \
  'import os; from google.cloud import storage; c=storage.Client(project=os.environ["GCP_PROJECT"]); next(c.list_blobs(os.environ["BUCKET"], max_results=1), None); print("bucket object access: ok")'
```

Both arithmetic checks must return silently with exit status zero, and the last
command must print `bucket object access: ok`. A failed check means stop; do not
hope that swap or an assumed `/mnt/disks/...` mount will rescue parity.

Prepare the exact MaxText checkout:

```zsh
export MAXTEXT_COMMIT="48c7da3e28e8f57fe36ddf09d00eec53072a2ea0"
export MAXTEXT_ROOT="$HOME/src/maxtext-gemma4"

git clone https://github.com/AI-Hypercomputer/maxtext.git "$MAXTEXT_ROOT"
git -C "$MAXTEXT_ROOT" checkout --detach "$MAXTEXT_COMMIT"
test "$(git -C "$MAXTEXT_ROOT" rev-parse HEAD)" = "$MAXTEXT_COMMIT"
test -z "$(git -C "$MAXTEXT_ROOT" status --porcelain=v1 --untracked-files=all)"
```

If the checkout already exists, do not clone over it; fetch the commit, check
it out detached, and run the same two tests.

Finally define a unique, never-reused model output for this one conversion
attempt:

```zsh
export ARTIFACT_ROOT="$PROJECT_ROOT/artifacts/gemma4_v1/production"
export BULK_GCS_ROOT="gs://$BUCKET/moe-sae/gemma4_v1"
export MODEL_ROOT="$BULK_GCS_ROOT/model"
export MODEL_ATTEMPT_ID="$(date -u +%Y%m%dT%H%M%SZ)-$(print -r -- "$PROJECT_COMMIT" | cut -c1-12)"
export MODEL_OUTPUT="$MODEL_ROOT/$MODEL_ATTEMPT_ID"

mkdir -p "$ARTIFACT_ROOT/operator-results"
print -r -- "$MODEL_OUTPUT"
```

The conversion CLI proves this path contains no GCS objects before downloading
the model. If an attempt fails after it starts writing, generate a new
`MODEL_ATTEMPT_ID`; never delete or reuse the partial prefix. It also refuses
to run when `$ARTIFACT_ROOT/checkpoint/conversion.json` is already sealed. That
means Stage 1 already has an authoritative conversion; do not overwrite it.

### 1.8 Convert the exact Gemma snapshot and prove parity

Make sure the `google/gemma-4-26B-A4B` license has been accepted by the account
behind your Hugging Face token. Enter the read-only token in this new host's
shell if it is not already present:

```zsh
if [[ -z "${HF_TOKEN:-}" ]]; then
  read -s "HF_TOKEN?Hugging Face read token: "
  export HF_TOKEN
  echo
fi
```

First validate the plan without downloading 52 GB:

```zsh
UV_PROJECT_ENVIRONMENT="$CONVERSION_ENV" \
  uv run --frozen --extra conversion moe-sae convert-checkpoint \
    --artifact-root "$ARTIFACT_ROOT" \
    --maxtext-root "$MAXTEXT_ROOT" \
    --hf-cache-dir "$HF_CACHE_DIR" \
    --output "$MODEL_OUTPUT" \
    --dry-run
```

The printed plan must name Gemma revision
`24548b62aa021d562695c04aaf7758a1ea47990b`, print 64-character repository and
snapshot inventory SHA-256 values, and show `max_kl_div_threshold: 0.03`. Then
run it for real:

```zsh
UV_PROJECT_ENVIRONMENT="$CONVERSION_ENV" \
  uv run --frozen --extra conversion moe-sae convert-checkpoint \
    --artifact-root "$ARTIFACT_ROOT" \
    --maxtext-root "$MAXTEXT_ROOT" \
    --hf-cache-dir "$HF_CACHE_DIR" \
    --output "$MODEL_OUTPUT"
```

The CLI downloads the exact commit into a commit-addressed local snapshot. It
requires the exact ten-file repository tree pinned in code, verifies all three
LFS payload SHA-256/size pairs, and verifies every ordinary file's Git blob
identity and SHA-256. Missing, extra, corrupt, or locally modified cache files
fail before conversion. Both MaxText conversion and the Hugging Face parity
model/tokenizer consume that same local path. The Hub token is removed from
both subprocess environments. The CLI re-hashes the complete 51.6 GB snapshot
after parity and refuses to attest if anything changed.

The CPU parity subprocess explicitly disables MaxText's Pallas-backed
MegaBlox GMM and retains sparse top-8 routing through the portable
`jax.lax.ragged_dot` path. Do not remove the `megablox=false` override or
replace it with a TPU-only kernel.

Recovery is permitted only when the converter committed the canonical
`$MODEL_OUTPUT/0/items` checkpoint but the same command then failed during
parity, before `conversion.json` was sealed. In that case, do not delete the
checkpoint, choose a new output, or rerun the converter. Keep `MODEL_OUTPUT`
set to the failed attempt's exact prefix and run:

```zsh
UV_PROJECT_ENVIRONMENT="$CONVERSION_ENV" \
  uv run --frozen --extra conversion moe-sae convert-checkpoint \
    --artifact-root "$ARTIFACT_ROOT" \
    --maxtext-root "$MAXTEXT_ROOT" \
    --hf-cache-dir "$HF_CACHE_DIR" \
    --output "$MODEL_OUTPUT" \
    --resume-validation
```

Recovery uses the already-cached pinned Hugging Face snapshot without network
access, never reruns conversion, requires the canonical checkpoint path, and
hashes the checkpoint inventory both before and after parity. A missing,
partial, or changing checkpoint fails without sealing an attestation. The
result records that the original conversion duration is unavailable. Never use
this mode for an output whose provenance is unknown or for a failure that
occurred before Orbax wrote `commit_success.txt`.

Success means conversion exits zero, parity exits zero at
`max_kl_div <= 0.03`, `$MODEL_OUTPUT/0/items` exists, and
`$ARTIFACT_ROOT/checkpoint/conversion.json` exists. Quiet periods while large
weights load or hash are normal; do not interrupt solely because no new line
has appeared. Reserve a several-hour window: download speed, two full snapshot
hash passes, Orbax publication, and CPU parity all contribute. This Stage 1
timing is not part of either harvester MFU or marginal SAE chip-seconds.

The CLI removes the token from its child processes; remove it from the
interactive shell as soon as the command finishes:

```zsh
unset HF_TOKEN
```

The attestation records the model ID and revision, expected repository-tree and
byte-level snapshot inventory hashes, clean MaxText commit, checkpoint
inventory hash, timings, parity threshold and result, and its own self-hash.
Extract the two authoritative values:

```zsh
export MODEL_CHECKPOINT="$(
  jq -r '.checkpoint' "$ARTIFACT_ROOT/checkpoint/conversion.json"
)"
export CHECKPOINT_SHA256="$(
  jq -r '.checkpoint_sha256' "$ARTIFACT_ROOT/checkpoint/conversion.json"
)"

print -r -- "$MODEL_CHECKPOINT"
print -r -- "$CHECKPOINT_SHA256"
```

`CHECKPOINT_SHA256` must be exactly 64 lowercase hexadecimal characters. Never
shorten it or compute a replacement manually.

If the conversion host does not mount the later shared POSIX control plane,
preserve the exact self-hashed file at an attestation-addressed GCS object:

```zsh
export CONVERSION_ATTESTATION_SHA256="$(
  jq -r '.attestation_sha256' "$ARTIFACT_ROOT/checkpoint/conversion.json"
)"
export CONVERSION_HANDOFF="$BULK_GCS_ROOT/control-handoff/conversion/$CONVERSION_ATTESTATION_SHA256/conversion.json"
gcloud storage cp --if-generation-match=0 \
  "$ARTIFACT_ROOT/checkpoint/conversion.json" \
  "$CONVERSION_HANDOFF"
```

Before production preflight, copy those exact bytes to
`$ARTIFACT_ROOT/checkpoint/conversion.json` on the shared control plane. Exact
byte copying is useful audit discipline. The loader canonicalizes parsed JSON,
so whitespace or key ordering alone does not change the self-hash, but changing
any attested field does. Do not edit or regenerate the file manually.

The TPU runtime does not install or import Torch. Every production model stage
recomputes the Orbax inventory on every controller, requires controller
agreement, and compares it with `CHECKPOINT_SHA256`.

Once the handoff object and all extracted values are safely recorded, stop the
temporary conversion VM from the Mac so it cannot accrue idle compute usage:

```zsh
gcloud compute instances stop "$CONVERSION_VM" \
  --project="$GCP_PROJECT" \
  --zone="$CONVERSION_ZONE"
```

After the shared-control-plane copy passes production preflight, delete the VM
and its auto-delete boot disk if TRC policy permits. Stopped persistent disks
can still be billed.

### Stage 1 completion checklist

- [ ] The implementation is committed and the production checkout is clean.
- [ ] `gcloud` and Python ADC both access the TRC bucket.
- [ ] The smoke stage succeeded twice, including deterministic resume.
- [ ] Train and validation ArrayRecords and manifests are sealed in GCS.
- [ ] `TRAIN_DATASET_MANIFEST`, `TRAIN_GRAIN_FILES`,
  `EVAL_DATASET_MANIFEST`, and `EVAL_GRAIN_FILES` are saved.
- [ ] MaxText is at the exact clean pinned commit.
- [ ] Gemma conversion used the immutable pinned local snapshot.
- [ ] Hugging Face/MaxText parity passed at `max_kl_div <= 0.03`.
- [ ] `MODEL_CHECKPOINT` and the 64-character `CHECKPOINT_SHA256` are saved.
- [ ] The exact `conversion.json` is preserved for the shared control plane.
- [ ] No TFDS build, Dataflow job, full-C4 download, or TPU allocation was
  needed for Stage 1.

## Deployment storage roles after Stage 1

On the v4-32, use one stable layout. `ARTIFACT_ROOT` must resolve to the same
strongly consistent POSIX mount on all four controllers:

```zsh
export ARTIFACT_ROOT="$PWD/artifacts/gemma4_v1/production"
export LOCAL_BUFFER_ROOT="/mnt/localssd/moe-sae"
export BULK_GCS_ROOT="gs://$BUCKET/moe-sae/gemma4_v1"
export SAE_CHECKPOINT_ROOT="$BULK_GCS_ROOT/sae-checkpoints"
```

The paths have deliberately different jobs:

- `ARTIFACT_ROOT` is the small shared POSIX control plane for locks,
  attestations, identities, receipts, ledgers, sealed manifests, timing records,
  and checkpoint pointers. It must not hold activation or Orbax payloads.
- `LOCAL_BUFFER_ROOT` is private scratch on each controller. It buffers capture
  writes and hosts that controller's bounded, verified `capture-cache/`.
- `BULK_GCS_ROOT` is the durable immutable data plane for staged inputs and
  activation shards.
- `SAE_CHECKPOINT_ROOT` is the immutable Orbax payload plane below the same GCS
  root.

Install only the TPU environment on every controller:

```zsh
uv sync --frozen --extra tpu
```

## 4. Qualify all three storage planes

Storage qualification must complete on the allocated v4-32 before production
preflight. Launch this collectively on all four controllers:

```zsh
uv run --extra tpu moe-sae qualify-storage \
  --artifact-root "$ARTIFACT_ROOT" \
  --local-buffer-root "$LOCAL_BUFFER_ROOT" \
  --bulk-gcs-root "$BULK_GCS_ROOT" \
  --sae-checkpoint-root "$SAE_CHECKPOINT_ROOT"
```

Production thresholds are pinned by the configuration and cannot be
overridden: at least 10 GB free on the shared control volume, 128 GiB on each
local buffer, an incompressible 8 GiB GCS probe per process in 8 MiB blocks,
and at least 4 GB/s aggregate for both upload and generation-pinned download.
The qualifier also proves cross-process control-plane visibility and `flock`,
atomic rename, visibility of disjoint per-host control records, process-private
local-buffer mounts using same-name/different-payload probes, conditional GCS
creation, CRC32C and SHA-256 round trips, exact-generation reads, and exact-
generation cleanup.

It then performs an actual collective Orbax checkpoint probe below
`$SAE_CHECKPOINT_ROOT/qualification/`: save a small globally sharded tree to a
unique immutable URI, wait for Orbax finalization, round-trip its custom
metadata, restore it on every process, and compare each process's restored
shard digest. This finalized probe is retained as qualification evidence.

Only after all of those checks pass does the coordinator atomically write the
self-hashed `$ARTIFACT_ROOT/preflight/storage.json`. Benchmark, harvest,
training, evaluation, reporting, and `run` reject a changed config,
environment, topology, control path, local-buffer layout, GCS root, cache
limit, checkpoint root, or Orbax proof.

`qualify-storage --local` is a reduced single-process object-store smoke test;
it does not produce the production attestation and cannot authorize a TPU run.

## 5. Run production preflight

After storage qualification, launch preflight collectively:

```zsh
uv run --extra tpu moe-sae preflight \
  --artifact-root "$ARTIFACT_ROOT" \
  --checkpoint "$MODEL_CHECKPOINT" \
  --grain-train "$TRAIN_GRAIN_FILES" \
  --grain-eval "$EVAL_GRAIN_FILES" \
  --dataset-train-manifest "$TRAIN_DATASET_MANIFEST" \
  --dataset-eval-manifest "$EVAL_DATASET_MANIFEST"
```

Preflight requires Python 3.12, the exact MaxText provenance, a 16-chip v4
runtime, a clean secret scan, the qualified storage attestation, the explicit
checkpoint, and exact train/eval staged-data inventories. A merely host-local
artifact root fails the cross-process control-plane checks.

Do not proceed unless every line is `PASS`. `--json` emits the same result as
machine-readable JSON. `--local` skips TPU/MaxText and production-storage
requirements and is not an authorization for production.

## 6. Benchmark capture

Run the required nine-candidate sweep:

```zsh
uv run --extra tpu moe-sae benchmark-harvest \
  --artifact-root "$ARTIFACT_ROOT" \
  --checkpoint "$MODEL_CHECKPOINT" \
  --checkpoint-sha256 "$CHECKPOINT_SHA256" \
  --steps 200
```

The order is EP16/FSDP1/TP1, EP8/FSDP2/TP1, then EP8/FSDP1/TP2, each with
per-device sequence batches 1, 2, and 4. Each candidate compiles once, runs
five warmups, and executes at least 200 synchronized steps in a fresh process.

Selection maximizes persisted tokens/s among candidates with useful-prefix MFU
at least 50%, no OOM, recompile, or dropped route, writer backlog below two
batches, and identical instrumented output. Useful FLOPs count only blocks
1--14 plus target attention/router/routed FFN; the denominator is 4.4 PFLOP/s
times wall time. Benchmark persistence uses the same local-buffer-to-GCS path
as production capture, and its temporary immutable objects are deleted by
exact generation after the timed candidate completes.

The selected, provenance-bound result is
`$ARTIFACT_ROOT/benchmark/selected.json`. A failed or timed-out candidate keeps
its supervisor telemetry but cannot poison a later candidate.

## 7. Run the real-data canary and seal captures

The train canary captures the first 2,000,000 production tokens. It uses the
selected mesh and batch by default and resumes by immutable input-range ID:

```zsh
uv run --extra tpu moe-sae harvest \
  --artifact-root "$ARTIFACT_ROOT" \
  --split train \
  --canary \
  --checkpoint "$MODEL_CHECKPOINT" \
  --checkpoint-sha256 "$CHECKPOINT_SHA256" \
  --dataset-manifest "$TRAIN_DATASET_MANIFEST" \
  --grain-files "$TRAIN_GRAIN_FILES"
```

Continue only when real-data goodput is at least 90% of synthetic, input wait
is below 5%, persisted goodput is at least 90% of device-output goodput, and
writer backlog stays below two batches. The canary's objects are already
durable production objects; the subsequent train harvest adopts their receipts
and continues rather than recomputing those ranges.

Complete and seal train, then capture and seal eval:

```zsh
uv run --extra tpu moe-sae harvest \
  --artifact-root "$ARTIFACT_ROOT" \
  --split train \
  --checkpoint "$MODEL_CHECKPOINT" \
  --checkpoint-sha256 "$CHECKPOINT_SHA256" \
  --dataset-manifest "$TRAIN_DATASET_MANIFEST" \
  --grain-files "$TRAIN_GRAIN_FILES"

uv run --extra tpu moe-sae harvest \
  --artifact-root "$ARTIFACT_ROOT" \
  --split eval \
  --checkpoint "$MODEL_CHECKPOINT" \
  --checkpoint-sha256 "$CHECKPOINT_SHA256" \
  --dataset-manifest "$EVAL_DATASET_MANIFEST" \
  --grain-files "$EVAL_GRAIN_FILES"
```

There is no post-seal bulk publication step and no `--publish-gcs` harvest
option. For every capture range, the owning host:

1. serializes one safetensors shard in its local buffer;
2. conditionally uploads it directly to its immutable, content-addressed GCS
   URI;
3. verifies URI, generation, size, CRC32C, and SHA-256 identity;
4. atomically writes a self-hashed upload receipt on the POSIX control plane;
5. commits the identical entry to its host ledger.

If a host dies after the receipt but before the ledger update, resume verifies
the remote object and adopts the receipt. It never infers progress by counting
files. After the global harvest barrier, the coordinator validates every
receipt and remote object identity, proves contiguous ranges with no gaps or
duplicates, checks exact token counts, merges the shard FP64 statistics, and
atomically seals the shared POSIX manifest. Training and evaluation consume
only that sealed manifest. The required totals are exactly 50,000,000 train
tokens and 2,000,000 eval tokens.

Capture hashing, manifest verification, and sealing are reported separately
from device execution, while their reserved-cluster time remains part of the
shared harvest cost used by end-to-end ratios. Campaign telemetry is bound to
the sealed manifest SHA, config, checkpoint, conversion attestation, dataset
fingerprint, environment, storage attestation, and selected benchmark.

## 8. Train without quality peeking

Train seed 1 for A, B, and F to the 5M health boundary without comparing
relative quality:

```zsh
for condition in A B F; do
  uv run --extra tpu moe-sae train \
    --artifact-root "$ARTIFACT_ROOT" \
    --condition "$condition" \
    --seed 1 \
    --stop-after-tokens 5000000
done
```

Each run must have finite state, exact L0=64, a final 64-update loss EMA no more
than 80% of the initial EMA, and at least 95% live features for A/F. Stop and
diagnose a failure; do not add auxiliary losses. On pass, resume seed 1 to 50M
and then run seeds 2 and 3:

```zsh
for condition in A B F; do
  uv run --extra tpu moe-sae train \
    --artifact-root "$ARTIFACT_ROOT" \
    --condition "$condition" \
    --seed 1
done

for seed in 2 3; do
  for condition in A B F; do
    uv run --extra tpu moe-sae train \
      --artifact-root "$ARTIFACT_ROOT" \
      --condition "$condition" \
      --seed "$seed"
  done
done
```

Before measured optimization, each controller materializes its assigned
sealed-capture partition from GCS into its bounded local `capture-cache`.
Cold downloads are generation-pinned and SHA-256 verified; every cache hit is
rehashed before use. Cache prefetch time and bytes are reported but excluded
from marginal SAE training chip-seconds. Timed training includes synchronized
optimizer steps and input stalls; compilation, checkpoint serialization, and
evaluation are separate.
Condition A's fixed-order capacity prepass remains included in A's measured
marginal cost.

At each checkpoint boundary, Orbax writes to a new attempt URI below
`$SAE_CHECKPOINT_ROOT/runs/<run-identity-sha256>/`; payloads are never
overwritten. Only after `wait_until_finished` and finalized-marker validation
does the coordinator atomically publish the self-hashed
`$ARTIFACT_ROOT/runs/.../latest.json` pointer. Completion publishes
`final.json` to the same finalized final payload. Resume follows only
`latest.json`, and evaluation follows only `final.json`. Once a replacement
`latest.json` is durable, the controllers synchronously remove the superseded
payload; a cleanup failure stops the run with the new pointer still safely
resumable. No payload is deleted before its replacement is finalized and
published, and the final/latest pointers share the one retained final payload.

## 9. Evaluate and report

Evaluate all nine final checkpoints against the same sealed held-out capture
and common SST, then build the report:

```zsh
uv run --extra tpu moe-sae eval \
  --artifact-root "$ARTIFACT_ROOT" \
  --all

uv run --extra tpu moe-sae report \
  --artifact-root "$ARTIFACT_ROOT" \
  --output "$ARTIFACT_ROOT/report.json"
```

Evaluation prefetches its assigned GCS capture partition into the same bounded
local cache before evaluation timing, validates each `final.json` pointer, and
restores the referenced finalized Orbax payload. `report.json` is the
authoritative machine-readable result. It includes storage qualification and
capture-prefetch diagnostics as well as per-seed EV/NMSE, exact L0, dead rate,
candidate exposure, chip-seconds, harvester MFU, persisted tokens/s, and wall
time.

If F-minus-B is non-positive, G is undefined. If its paired interval includes
zero, the report is marked unstable/exploratory rather than headline-ready.

## 10. Gated orchestration

After C4 publication, checkpoint conversion, storage qualification, and the
explicit production preflight above, `run` can execute or resume the remaining
gated state machine:

```zsh
uv run --extra tpu moe-sae run \
  --artifact-root "$ARTIFACT_ROOT" \
  --checkpoint "$MODEL_CHECKPOINT" \
  --checkpoint-sha256 "$CHECKPOINT_SHA256" \
  --grain-train "$TRAIN_GRAIN_FILES" \
  --grain-eval "$EVAL_GRAIN_FILES" \
  --dataset-train-manifest "$TRAIN_DATASET_MANIFEST" \
  --dataset-eval-manifest "$EVAL_DATASET_MANIFEST"
```

`run` does not qualify storage or stage/convert inputs. It requires their
attestations, creates the benchmark if absent, runs the train canary, seals
both captures, enforces the seed-1 health boundary before relative evaluation,
finishes all nine runs, evaluates them, and writes the report. It reuses only
provenance-matching sealed stages and stops at the first unmet gate.
