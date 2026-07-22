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

Stage 1 also freezes one target-runtime lifecycle requirement: the TPU OS image
and bootstrap must support systemd user lingering, and the production user must
later prove `Linger=yes` on every controller. This is a hard prerequisite for
host-RAM caches and detached supervisors; it is not an optional convenience.
Do not select an image or access policy that prevents the bootstrap from
enabling and verifying it.

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

If TRC supplied a bucket, use it and skip creation. The current campaign's
existing `gs://moe-sae-bucket` is confirmed as regional `US-CENTRAL2`, matching
the v4-32 region. Otherwise choose a globally unique name and ask the TRC
administrator which currently available location is permitted; never infer a
location from an old region list:

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
- [ ] The target TPU image/bootstrap plan can enable and verify systemd
  `Linger=yes` for the production user on every controller.
- [ ] No TFDS build, Dataflow job, full-C4 download, or TPU allocation was
  needed for Stage 1.

## Stage 2 — provision the v4-32 and its small POSIX control plane

Run Stage 2 from the Mac/control terminal, not from the stopped conversion VM.
The model and dataset identities below are confirmed for this campaign. The
TPU allocation mode, TPU name, and TPU service account are intentionally *not*
filled in: those must come from the TRC award or a TRC administrator.

### 2.1 Load confirmed identities and fill the operator choices

The following GCS values are copyable as written:

```zsh
setopt PIPE_FAIL

export GCP_PROJECT="neural-dragnet"
export TPU_ZONE="us-central2-b"
export BUCKET="moe-sae-bucket"
export TPU_NETWORK="default"

export BULK_GCS_ROOT="gs://moe-sae-bucket/moe-sae/gemma4_v1"
export SAE_CHECKPOINT_ROOT="$BULK_GCS_ROOT/sae-checkpoints"
export MODEL_CHECKPOINT="gs://moe-sae-bucket/moe-sae/gemma4_v1/model/20260718T101406Z-0600f5d98d32/0/items"
export CHECKPOINT_SHA256="fbb5e2fe394d8ca03a30e15247b7a4171461b913406b63005a29dc47c3ed8281"
export CONVERSION_HANDOFF="gs://moe-sae-bucket/moe-sae/gemma4_v1/control-handoff/conversion/fddcde27083510fad4c3a32f51e760bf040a08bfe6b816202bd1dfd1542f9c9e/conversion.json"

export TRAIN_DATASET_MANIFEST="gs://moe-sae-bucket/moe-sae/gemma4_v1/datasets/c4/train/manifests/8737195093e083fe1baf982e22e8c99d0b7e8be1e201e14d91e3c3cb37b5e22a.json"
export TRAIN_GRAIN_FILES='gs://moe-sae-bucket/moe-sae/gemma4_v1/datasets/c4/train/shards/*.array_record'
export EVAL_DATASET_MANIFEST="gs://moe-sae-bucket/moe-sae/gemma4_v1/datasets/c4/validation/manifests/a3c6da01b87ecac4076b5f123f62f8998905c4b12889ff1a1432645f59b4f630.json"
export EVAL_GRAIN_FILES='gs://moe-sae-bucket/moe-sae/gemma4_v1/datasets/c4/validation/shards/*.array_record'

export FILESTORE_NAME="gemma4-control"
export FILESTORE_SHARE="moe_sae_control"
```

Now fill these values. Keep the angle-bracket placeholders quoted, and do not
run a create command until every placeholder has been replaced:

```zsh
# Choose a new resource name; this is not a pre-existing confirmed name.
export TPU_NAME="<new unique TPU node name, for example gemma4-harvest-01>"

# One of: direct-on-demand, direct-reserved, queued-on-demand, queued-reserved.
export TPU_ALLOCATION_MODE="<mode confirmed by the TRC award/admin>"

# Required only for a queued mode; otherwise leave it empty.
export QUEUED_RESOURCE_NAME="<new queued-resource name, or empty>"

# The account TRC authorizes you to attach to the TPU workers.
export TPU_SERVICE_ACCOUNT="<TRC-approved service-account email>"
```

The project default account observed during conversion was
`655343590563-compute@developer.gserviceaccount.com`, and it could read the
bucket. That observation does **not** prove that TRC wants it attached to this
TPU. Use it only if the award or administrator confirms it.

Finally, commit the exact operator implementation before deployment. The
commit shown in an earlier Stage 1 handoff is not a substitute for the commit
that contains the current activation supervisor and worker scripts:

```zsh
cd /Users/abistras/Desktop/moe-sae
git status --short
export PROJECT_COMMIT="$(git rev-parse HEAD)"
test ${#PROJECT_COMMIT} -eq 40
print -r -- "$PROJECT_COMMIT"
```

`git status --short` must print nothing. The worker deployer refuses a dirty
checkout.

### 2.2 Determine the TRC allocation mode before creating anything

Do not infer this choice from the fact that the TPU is gratis to you. Look for
the following exact ideas in the award email/console, or ask the administrator:

| TRC wording | `TPU_ALLOCATION_MODE` | Creation behavior |
|---|---|---|
| On-demand quota; direct node creation approved | `direct-on-demand` | CreateNode, no special allocation flag |
| A reservation is named; direct consumption approved | `direct-reserved` | CreateNode with `--reserved` |
| On-demand quota; queued request requested/recommended | `queued-on-demand` | Non-preemptible queued resource |
| A reservation is named and queued consumption requested | `queued-reserved` | Queued resource with `--reserved` |

`--reserved` does not mean "use my free quota"; it means "consume a Cloud TPU
reservation." Never add it unless TRC confirms a reservation. Never add
`--spot` or `--preemptible` to this overnight production run. Google documents
the distinction in its [TPU VM creation](https://docs.cloud.google.com/tpu/docs/create-tpu-vm)
and [queued-resource](https://docs.cloud.google.com/tpu/docs/queued-resources)
guides.

The commands below use the default external worker addresses. If TRC requires
internal-only workers, stop and obtain the approved bastion/IAP and Private
Google Access procedure first; adding `--internal-ips` alone would make the
Mac-side SSH/deployment steps unreachable.

### 2.3 Run the non-destructive cloud inventory

These API reads catch the common wrong-project, wrong-zone, wrong-runtime,
wrong-network, and missing-input errors before allocation:

```zsh
gcloud config set project "$GCP_PROJECT"
gcloud auth list

gcloud compute tpus tpu-vm accelerator-types describe v4-32 \
  --project="$GCP_PROJECT" \
  --zone="$TPU_ZONE"

gcloud compute tpus tpu-vm versions describe tpu-ubuntu2204-base \
  --project="$GCP_PROJECT" \
  --zone="$TPU_ZONE"

gcloud compute networks describe "$TPU_NETWORK" \
  --project="$GCP_PROJECT"

gcloud iam service-accounts describe "$TPU_SERVICE_ACCOUNT" \
  --project="$GCP_PROJECT"

gcloud storage buckets describe "gs://$BUCKET" \
  --project="$GCP_PROJECT"

gcloud storage ls "$MODEL_CHECKPOINT/commit_success.txt"
gcloud storage ls "$CONVERSION_HANDOFF"
gcloud storage ls "$TRAIN_DATASET_MANIFEST"
gcloud storage ls "$TRAIN_GRAIN_FILES"
gcloud storage ls "$EVAL_DATASET_MANIFEST"
gcloud storage ls "$EVAL_GRAIN_FILES"

gcloud compute tpus tpu-vm list \
  --project="$GCP_PROJECT" \
  --zone="$TPU_ZONE"
gcloud compute tpus queued-resources list \
  --project="$GCP_PROJECT" \
  --zone="$TPU_ZONE"
gcloud filestore instances list \
  --project="$GCP_PROJECT" \
  --zone="$TPU_ZONE"
```

The two TPU capability descriptions must succeed, every GCS command must show
at least one object, and your proposed names must not collide with an existing
resource. These commands do not create cloud resources. Later `gcloud ... ssh`
inspection is different: its remote inventory payload is non-mutating, but
Google may create or update SSH authorization metadata as part of connecting.

### 2.4 Create exactly the allocation type TRC approved

First reject unfilled or inconsistent choices:

```zsh
[[ "$TPU_NAME" != *'<'* && -n "$TPU_NAME" ]]
[[ "$TPU_SERVICE_ACCOUNT" != *'<'* && -n "$TPU_SERVICE_ACCOUNT" ]]

case "$TPU_ALLOCATION_MODE" in
  direct-on-demand|direct-reserved)
    export QUEUED_RESOURCE_NAME=""
    ;;
  queued-on-demand|queued-reserved)
    [[ "$QUEUED_RESOURCE_NAME" != *'<'* && -n "$QUEUED_RESOURCE_NAME" ]]
    ;;
  *)
    print -u2 -- "Unrecognized or unfilled TPU_ALLOCATION_MODE"
    false
    ;;
esac
```

Then run this one `case`. It selects one—and only one—Google creation command:

```zsh
case "$TPU_ALLOCATION_MODE" in
  direct-on-demand)
    gcloud compute tpus tpu-vm create "$TPU_NAME" \
      --project="$GCP_PROJECT" \
      --zone="$TPU_ZONE" \
      --accelerator-type=v4-32 \
      --version=tpu-ubuntu2204-base \
      --network="$TPU_NETWORK" \
      --service-account="$TPU_SERVICE_ACCOUNT" \
      --scopes=https://www.googleapis.com/auth/cloud-platform \
      --labels=purpose=gemma4-harvest
    ;;
  direct-reserved)
    gcloud compute tpus tpu-vm create "$TPU_NAME" \
      --project="$GCP_PROJECT" \
      --zone="$TPU_ZONE" \
      --accelerator-type=v4-32 \
      --version=tpu-ubuntu2204-base \
      --network="$TPU_NETWORK" \
      --service-account="$TPU_SERVICE_ACCOUNT" \
      --scopes=https://www.googleapis.com/auth/cloud-platform \
      --labels=purpose=gemma4-harvest \
      --reserved
    ;;
  queued-on-demand)
    gcloud compute tpus queued-resources create "$QUEUED_RESOURCE_NAME" \
      --project="$GCP_PROJECT" \
      --zone="$TPU_ZONE" \
      --node-id="$TPU_NAME" \
      --accelerator-type=v4-32 \
      --runtime-version=tpu-ubuntu2204-base \
      --network="$TPU_NETWORK" \
      --service-account="$TPU_SERVICE_ACCOUNT" \
      --scopes=https://www.googleapis.com/auth/cloud-platform \
      --labels=purpose=gemma4-harvest
    ;;
  queued-reserved)
    gcloud compute tpus queued-resources create "$QUEUED_RESOURCE_NAME" \
      --project="$GCP_PROJECT" \
      --zone="$TPU_ZONE" \
      --node-id="$TPU_NAME" \
      --accelerator-type=v4-32 \
      --runtime-version=tpu-ubuntu2204-base \
      --network="$TPU_NETWORK" \
      --service-account="$TPU_SERVICE_ACCOUNT" \
      --scopes=https://www.googleapis.com/auth/cloud-platform \
      --labels=purpose=gemma4-harvest \
      --reserved
    ;;
esac
```

For a queued allocation, inspect rather than repeatedly recreating it:

```zsh
if [[ -n "$QUEUED_RESOURCE_NAME" ]]; then
  gcloud compute tpus queued-resources describe "$QUEUED_RESOURCE_NAME" \
    --project="$GCP_PROJECT" \
    --zone="$TPU_ZONE"
fi
```

Re-run that description after a few minutes until the resource is active and
the node exists. Capacity delay is normal. Do not submit duplicate queued
requests.

For either allocation path, this is the authoritative ready check:

```zsh
gcloud compute tpus tpu-vm describe "$TPU_NAME" \
  --project="$GCP_PROJECT" \
  --zone="$TPU_ZONE" \
  --format='yaml(name,state,acceleratorType,runtimeVersion,networkConfig,serviceAccount)'

test "$(gcloud compute tpus tpu-vm describe "$TPU_NAME" \
  --project="$GCP_PROJECT" --zone="$TPU_ZONE" \
  --format='value(state)')" = "READY"
test "$(gcloud compute tpus tpu-vm describe "$TPU_NAME" \
  --project="$GCP_PROJECT" --zone="$TPU_ZONE" \
  --format='value(acceleratorType)')" = "v4-32"
test "$(gcloud compute tpus tpu-vm describe "$TPU_NAME" \
  --project="$GCP_PROJECT" --zone="$TPU_ZONE" \
  --format='value(runtimeVersion)')" = "tpu-ubuntu2204-base"
```

All three `test` commands must exit silently with status zero.

### 2.5 Create the 1 TiB Filestore control plane

Create Filestore only after the TPU is allocated, so a delayed TPU request does
not leave an idle filesystem accruing cost. This volume stores small control
records, not activations. Basic HDD's minimum is 1 TiB; see Google's
[Filestore tiers](https://docs.cloud.google.com/filestore/docs/service-tiers).

```zsh
gcloud services enable file.googleapis.com \
  --project="$GCP_PROJECT"

gcloud filestore instances create "$FILESTORE_NAME" \
  --project="$GCP_PROJECT" \
  --zone="$TPU_ZONE" \
  --tier=BASIC_HDD \
  --file-share=name="$FILESTORE_SHARE",capacity=1TB \
  --network=name="$TPU_NETWORK" \
  --labels=purpose=gemma4-control

gcloud filestore instances describe "$FILESTORE_NAME" \
  --project="$GCP_PROJECT" \
  --zone="$TPU_ZONE" \
  --format='yaml(name,state,tier,fileShares,networks)'

test "$(gcloud filestore instances describe "$FILESTORE_NAME" \
  --project="$GCP_PROJECT" --zone="$TPU_ZONE" \
  --format='value(state)')" = "READY"
test "$(gcloud filestore instances describe "$FILESTORE_NAME" \
  --project="$GCP_PROJECT" --zone="$TPU_ZONE" \
  --format='value(tier)')" = "BASIC_HDD"

export NFS_SERVER="$(gcloud filestore instances describe "$FILESTORE_NAME" \
  --project="$GCP_PROJECT" --zone="$TPU_ZONE" \
  --format='value(networks[0].ipAddresses[0])')"
export NFS_EXPORT="/$FILESTORE_SHARE"
print -r -- "$NFS_SERVER:$NFS_EXPORT"
```

Both Filestore checks must be silent and `NFS_SERVER` must print an RFC1918 address.
The TPU and Filestore must use the same VPC. Do not expose NFS publicly.

### Stage 2 completion checklist

- [ ] TRC explicitly confirmed the allocation mode and attached service account.
- [ ] No Spot or preemptible flag was used.
- [ ] The node is `READY`, `v4-32`, and uses `tpu-ubuntu2204-base`.
- [ ] The node can be described as `$TPU_NAME` in `us-central2-b`.
- [ ] Basic HDD Filestore is `READY` on the same VPC.
- [ ] `NFS_SERVER` and `NFS_EXPORT` are recorded.

## Stage 3 — inspect, deploy, and bootstrap all four controllers

Stage 3 deliberately separates inspection from mutation. The inspection
script never chooses, partitions, formats, or mounts a scratch device. You read
its output, select an already-mounted local filesystem with enough space, and
give that exact choice to the fail-closed deployer.

### 3.1 Inspect every worker before choosing scratch

From the clean repository on the Mac:

```zsh
cd /Users/abistras/Desktop/moe-sae

export GCP_PROJECT="neural-dragnet"
export TPU_ZONE="us-central2-b"
# Reuse the exact name chosen in Stage 2.
export TPU_NAME="<the created TPU node name>"

mkdir -p artifacts/gemma4_v1/production/operator-results
scripts/tpu/inspect_workers.sh | tee \
  "artifacts/gemma4_v1/production/operator-results/worker-inventory.txt"
```

If this first connection says no SSH agent is available, run the following and
retry the inspection:

```zsh
eval "$(ssh-agent -s)"
```

The helper creates the gcloud key if necessary and adds it to that agent.

Print the home path explicitly on all four workers:

```zsh
gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
  --project="$GCP_PROJECT" \
  --zone="$TPU_ZONE" \
  --worker=all \
  --command='printf "HOME=%s HOST=%s\n" "$HOME" "$(hostname -s)"'
```

The script runs `lsblk`, `findmnt`, `df`, metadata identity reads, and `id` on
all four workers. Its *remote inventory payload* does not write disks or
filesystems. The SSH connection itself is not literally write-free: `gcloud`
may create a local SSH key and may update authorized-key or OS Login metadata.

Read all four sections. They must agree on the service-account email and
numeric user ID/group ID. Select either an existing local `ext4`/`xfs`
filesystem or the direct `/dev/shm` `tmpfs` mount. Pick one absolute directory
path beneath that private host-local filesystem which has at least
`25769803776` free bytes (24 GiB) on **every** worker. This is the
activation-capture requirement: two simultaneous 8 GiB qualification files
plus an 8 GiB reserve for the rolling safetensor buffer and runtime overhead.
Do not assume `/mnt/localssd` exists merely because that name appears in an
example.

For example, use `$HOME/moe-sae-local` only if `/home` is on a local `ext4` or
`xfs` device and every worker shows at least 24 GiB free there. The fixed
100 GiB TPU boot disk is acceptable when it passes this post-install check.
For this v4-32, `/dev/shm/moe-sae-local` is preferable when every inventory
section shows `/dev/shm tmpfs tmpfs rw,nosuid,nodev`, at least 24 GiB free, and
at least 24 GiB `MemAvailable`. The bootstrap accepts `tmpfs` only strictly
below that direct mount and rechecks its capacity. It rejects `/run`, arbitrary
tmpfs mounts, missing `nosuid`/`nodev`, and every shared filesystem. `/dev/shm`
is erased on reboot; that is safe here because it holds only bounded,
reconstructible staging files, while receipts and payloads live on NFS/GCS.
If no existing private filesystem passes, stop. The helpers intentionally will not improvise
by formatting a device; provision suitable per-worker storage with the TRC
administrator and inspect again.

Record the identical worker home and your explicit choice:

```zsh
export TPU_HOME="<absolute home common to all four workers, for example /home/abistras>"
export TPU_PROJECT_ROOT="$TPU_HOME/src/moe-sae"
export LOCAL_BUFFER_ROOT="/dev/shm/moe-sae-local"

[[ "$TPU_HOME" == /* && "$TPU_HOME" != *'<'* ]]
[[ "$LOCAL_BUFFER_ROOT" == /* && "$LOCAL_BUFFER_ROOT" != *'<'* ]]
```

`LOCAL_BUFFER_ROOT` must be outside the repository and outside the NFS mount.
It may have the same pathname on all four hosts; the later storage qualifier
proves that the backing filesystems are actually private.

### 3.2 Deploy one exact commit without a GitHub credential

Set the remaining values on the Mac, then run the deployer:

```zsh
cd /Users/abistras/Desktop/moe-sae

export PROJECT_COMMIT="$(git rev-parse HEAD)"
export BULK_GCS_ROOT="gs://moe-sae-bucket/moe-sae/gemma4_v1"
export TPU_SERVICE_ACCOUNT="<same approved email used in Stage 2>"
export NFS_SERVER="$(gcloud filestore instances describe "$FILESTORE_NAME" \
  --project="$GCP_PROJECT" --zone="$TPU_ZONE" \
  --format='value(networks[0].ipAddresses[0])')"
export NFS_EXPORT="/$FILESTORE_SHARE"

git status --short
scripts/tpu/deploy_workers.sh
```

If the script asks for an SSH agent, run this once and retry:

```zsh
eval "$(ssh-agent -s)"
ssh-add "$HOME/.ssh/google_compute_engine"
```

The deployer refuses a dirty checkout, creates a self-contained Git bundle,
hashes it, copies the bundle and a secret-free config to every worker, and
checks out only `PROJECT_COMMIT`. No GitHub username, password, PAT, or HF token
is sent to the TPU. On every worker, the bootstrap then:

1. verifies the metadata service account and GCS access;
2. enables systemd lingering for the production user and refuses to continue
   unless `loginctl` reports `Linger=yes`;
3. installs only the small OS toolset, pinned `uv`, and Python 3.12;
4. installs the frozen TPU dependency set and verifies the MaxText commit;
5. mounts `$NFS_SERVER:$NFS_EXPORT` **directly** at
   `$TPU_PROJECT_ROOT/artifacts/gemma4_v1/production`;
6. validates, creates, and writes only the explicit `LOCAL_BUFFER_ROOT`.

The linger gate is required even when `nohup` appears to preserve a process.
On the observed TPU image, `systemd-logind` had `RemoveIPC=yes`; when the last
non-lingering SSH session ended, it removed user-owned `/dev/shm` state. A
detached parent and its later fresh numerical children therefore cannot rely
on a qualified resident cache unless the user remains lingered.

The artifact path must be the NFS mountpoint itself, not a symlink to some
other mount. Production path validation resolves symlinks and rejects a mount
outside the repository's `artifacts/gemma4_v1` namespace.

### 3.3 Copy the exact conversion handoff onto the shared mount

Connect to worker 0:

```zsh
gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
  --project="$GCP_PROJECT" \
  --zone="$TPU_ZONE" \
  --worker=0
```

After the prompt changes to the TPU worker, run this block there. It copies to
a temporary file, compares an existing destination if present, and renames
only after the download succeeds:

```zsh
set -euo pipefail

export PROJECT_ROOT="$HOME/src/moe-sae"
export ARTIFACT_ROOT="$PROJECT_ROOT/artifacts/gemma4_v1/production"
export CONVERSION_HANDOFF="gs://moe-sae-bucket/moe-sae/gemma4_v1/control-handoff/conversion/fddcde27083510fad4c3a32f51e760bf040a08bfe6b816202bd1dfd1542f9c9e/conversion.json"
export CONVERSION_DESTINATION="$ARTIFACT_ROOT/checkpoint/conversion.json"

mkdir -p "$ARTIFACT_ROOT/checkpoint"
temporary="$(mktemp "$ARTIFACT_ROOT/checkpoint/.conversion.json.XXXXXX")"
gcloud storage cp "$CONVERSION_HANDOFF" "$temporary"

test -s "$temporary"
test "$(gcloud storage cat "$CONVERSION_HANDOFF" | sha256sum | awk '{print $1}')" = \
  "$(sha256sum "$temporary" | awk '{print $1}')"

if [[ -e "$CONVERSION_DESTINATION" ]]; then
  cmp --silent "$temporary" "$CONVERSION_DESTINATION"
  rm -- "$temporary"
else
  mv -- "$temporary" "$CONVERSION_DESTINATION"
fi
sync "$CONVERSION_DESTINATION"
ls -l "$CONVERSION_DESTINATION"
exit
```

Because `ARTIFACT_ROOT` is the shared NFS mount, one successful worker-0 copy
is visible to all four controllers. Do not manually edit this self-hashed JSON.

### 3.4 Define the collective-launch helpers on the Mac

Every command that initializes JAX must begin on all four controllers with
identical arguments. Define this zsh helper in the Mac terminal that will
operate the run:

```zsh
tpu_all() {
  local remote_command="$1"
  gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
    --project="$GCP_PROJECT" \
    --zone="$TPU_ZONE" \
    --worker=all \
    --command="$remote_command"
}

export ARTIFACT_ROOT="$TPU_PROJECT_ROOT/artifacts/gemma4_v1/production"
export REMOTE_UV="$TPU_HOME/.local/bin/uv"
export SAE_CHECKPOINT_ROOT="$BULK_GCS_ROOT/sae-checkpoints"
```

These locations have separate jobs:

- `ARTIFACT_ROOT` is the shared NFS control plane for locks, attestations,
  identities, receipts, ledgers, sealed manifests, and timing records. It must
  not contain activation or Orbax payloads.
- `LOCAL_BUFFER_ROOT` is private scratch on each controller for bounded capture
  writes. Stage 8 qualifies a separate consumer cache after both captures seal.
- `BULK_GCS_ROOT` is the immutable bulk plane for staged data and activations.
- `SAE_CHECKPOINT_ROOT` is the immutable Orbax payload plane for later SAE
  training.

Prove the deployment is identical before starting JAX:

```zsh
tpu_all "set -eu
cd '$TPU_PROJECT_ROOT'
test \"\$(git rev-parse HEAD)\" = '$PROJECT_COMMIT'
test -z \"\$(git status --porcelain=v1 --untracked-files=all)\"
test -x '$REMOTE_UV'
test \"\$(loginctl show-user \"\$(id -un)\" -p Linger --value)\" = yes
findmnt -rn -o OPTIONS --mountpoint '$ARTIFACT_ROOT' | tr ',' '\n' | grep -qx 'hard'
findmnt -rn -o OPTIONS --mountpoint '$ARTIFACT_ROOT' | tr ',' '\n' | grep -qx 'noac'
findmnt -rn -o OPTIONS --mountpoint '$ARTIFACT_ROOT' | tr ',' '\n' | grep -qx 'lookupcache=none'
! findmnt -rn -o OPTIONS --mountpoint '$ARTIFACT_ROOT' | tr ',' '\n' | grep -Eq '^(soft|softerr|nolock|local_lock=(all|flock))$'
test \"\$(findmnt -rn -o SOURCE --mountpoint '$ARTIFACT_ROOT')\" = '$NFS_SERVER:$NFS_EXPORT'
test -d '$LOCAL_BUFFER_ROOT'
printf 'READY %s %s\\n' \"\$(hostname -s)\" \"\$(git rev-parse HEAD)\""
```

You need four `READY` lines with the same commit. Never launch a JAX stage on
worker 0 alone, and never resume just the worker that failed. Re-run the exact
same collective command on all four.

### Stage 3 completion checklist

- [ ] All four inventory sections were inspected before scratch was selected.
- [ ] The same service account and numeric user identity appear on all workers.
- [ ] `loginctl` reports `Linger=yes` for the production user on all four
  controllers.
- [ ] Every selected host-local filesystem has at least 24 GiB free after bootstrap.
- [ ] The deployer finished successfully on all four controllers.
- [ ] Every checkout is clean and at the same full `PROJECT_COMMIT`.
- [ ] NFS is mounted directly at the production artifact root on all four with
  `hard,noac,lookupcache=none` and remote locking (`nolock`, `local_lock=all`,
  and `local_lock=flock` are forbidden). This is a correctness-first control
  mount; activation shards and checkpoints remain in GCS.
- [ ] The exact conversion handoff is present on the shared mount.

## Stage 4 — qualify all three storage planes

Run this from the Mac using the `tpu_all` helper. This is a foreground gate:
leave the terminal connected and wait for all four remote commands to exit.

```zsh
tpu_all "cd '$TPU_PROJECT_ROOT' && \
  '$REMOTE_UV' run --frozen --extra tpu moe-sae qualify-storage \
    --artifact-root '$ARTIFACT_ROOT' \
    --local-buffer-root '$LOCAL_BUFFER_ROOT' \
    --bulk-gcs-root '$BULK_GCS_ROOT' \
    --sae-checkpoint-root '$SAE_CHECKPOINT_ROOT'"
```

This is equivalent to an explicit `gcloud compute tpus tpu-vm ssh ...
--worker=all --command=...`; the helper does not run locally or only on worker
0. Production thresholds are pinned: at least 10 GB free on the shared control
volume, 24 GiB on each activation buffer, an incompressible 8 GiB GCS probe per
process in 8 MiB blocks, at least 4 GB/s aggregate local-buffer write
throughput, and at least 4 GB/s aggregate upload. The local-write timing uses
pre-generated incompressible blocks and includes `fsync`, so it measures the
backing store rather than pseudorandom generation; its file is deleted before
the simultaneous upload/download probes. The
generation-pinned download must pass CRC32C and SHA-256 integrity checks, but
its throughput is diagnostic for activation capture: capture consumes the
tiny packed-C4 input and does not read activation shards back. The configured
4 GB/s download requirement remains sealed in the attestation for the later,
separate SAE-cache qualification.

The probe and `ArtifactWriter` use the same large-object transport. Each
producer computes SHA-256 and whole-object CRC32C in its existing generation or
serialization pass, binds both digests to the file's device/inode/size/times,
and the store checks that state before and after transfer. It does not rescan
each large file on the timed upload path. The TPU process then execs a JAX-cold
helper, which uploads 16 MiB parts with 32 worker processes, and the parent
conditionally copies the unique temporary object into its content-addressed
final name. Generation-pinned probe downloads use 32 MiB ranges with 16
workers. The helper boundary is deliberate: only the fresh JAX-cold helper
creates a process pool. Transfers get a 30-minute helper deadline plus a
60-second parent grace period for cancellation and cleanup.

The qualifier proves cross-process control-plane visibility and `flock`, atomic
rename, absent-to-present visibility after every client has primed a negative
lookup, disjoint per-host control records, genuinely private local-buffer mounts
using same-name/different-payload probes, conditional GCS creation, CRC32C/SHA-256
round trips, and exact-generation reads and cleanup. It also performs a real
collective Orbax save/restore below
`$SAE_CHECKPOINT_ROOT/qualification/`, including finalized metadata and
cross-process restored-shard digest checks.

Only after every check passes does the coordinator atomically write the
self-hashed `$ARTIFACT_ROOT/preflight/storage.json`. Every later production
command rejects a changed config, environment, topology, control path, local
layout, GCS root, checkpoint root, or Orbax proof. This record is explicitly
scoped to `activation-capture`; it does not claim that the 100 GiB SAE cache is
available. The negative-lookup visibility proof is part of storage-attestation
schema v7; older attestations are intentionally incompatible and must be
regenerated by this collective command after deploying the new commit.

`qualify-storage --local` is only a single-process smoke test. It cannot
authorize production.

## Stage 5 — run production preflight

Launch preflight collectively from the Mac:

```zsh
tpu_all "cd '$TPU_PROJECT_ROOT' && \
  '$REMOTE_UV' run --frozen --extra tpu moe-sae preflight \
    --artifact-root '$ARTIFACT_ROOT' \
    --checkpoint '$MODEL_CHECKPOINT' \
    --grain-train '$TRAIN_GRAIN_FILES' \
    --grain-eval '$EVAL_GRAIN_FILES' \
    --dataset-train-manifest '$TRAIN_DATASET_MANIFEST' \
    --dataset-eval-manifest '$EVAL_DATASET_MANIFEST'"
```

Preflight requires Python 3.12, the exact MaxText provenance, a 16-chip v4
runtime, a clean secret scan, the qualified storage attestation, the explicit
checkpoint, and exact train/eval staged-data inventories. A merely host-local
artifact root fails the cross-process control-plane checks.

Do not proceed unless every check is `PASS`. `--local` skips TPU/MaxText and
production-storage requirements and is not production authorization.

The `activation-farm` supervisor reruns this preflight in a fresh subprocess at
the start of every invocation. This explicit Stage 5 run is still valuable: it
isolates deployment errors before the long benchmark begins.

## Stage 6 — launch the one supervised activation campaign

Production benchmark and harvesting must run under one lifetime lease and one
four-controller supervisor. Do not run standalone `benchmark-harvest` or
`harvest` commands for this campaign: they do not own the outer all-phase
supervisor that detects a missing controller between child processes.

`activation-farm` always walks the fixed sequence preflight, benchmark, canary,
train, eval. There are intentionally no phase-selection flags. On resume it
validates and skips any provenance-matching sealed phase, then continues at the
first incomplete one.

Build the exact all-phase command and shell-quote it once in local zsh:

```zsh
activation_command() {
  print -r -- "cd '$TPU_PROJECT_ROOT' && \
'$REMOTE_UV' run --frozen --extra tpu moe-sae activation-farm \
  --artifact-root '$ARTIFACT_ROOT' \
  --checkpoint '$MODEL_CHECKPOINT' \
  --checkpoint-sha256 '$CHECKPOINT_SHA256' \
  --grain-train '$TRAIN_GRAIN_FILES' \
  --grain-eval '$EVAL_GRAIN_FILES' \
  --dataset-train-manifest '$TRAIN_DATASET_MANIFEST' \
  --dataset-eval-manifest '$EVAL_DATASET_MANIFEST'"
}

export REMOTE_FARM_COMMAND="$(activation_command)"
export REMOTE_FARM_QUOTED="$(printf '%q' "$REMOTE_FARM_COMMAND")"
export REMOTE_FARM_LOG="$TPU_HOME/activation-farm.log"
export REMOTE_FARM_PID="$TPU_HOME/activation-farm.pid"

export REMOTE_FARM_LAUNCH="set -eu
if test -s '$REMOTE_FARM_PID' && kill -0 \$(cat '$REMOTE_FARM_PID') 2>/dev/null; then
  echo 'activation-farm is already running on this controller' >&2
  exit 1
fi
nohup bash -lc $REMOTE_FARM_QUOTED >> '$REMOTE_FARM_LOG' 2>&1 < /dev/null &
echo \$! > '$REMOTE_FARM_PID'
echo \"started activation-farm pid=\$(cat '$REMOTE_FARM_PID') host=\$(hostname -s)\""

tpu_all "$REMOTE_FARM_LAUNCH"
```

For an operator-authorized code-only retry after a complete production
preflight has already passed against the same checkpoint and immutable dataset
manifests, append both of these flags to `REMOTE_FARM_COMMAND`:

```text
--skip-preflight --reuse-verified-dataset
```

`--skip-preflight` bypasses only the fresh preflight subprocess.
`--reuse-verified-dataset` omits the slow scalar decode/recount of every packed
ArrayRecord in later capture subprocesses; manifest schema, ordered inventory,
GCS object generations, sizes, and content hashes are still verified.  The
bypass values are part of the four-controller operation identity and are
written to the activation log.  Never use this path for a first launch or
after changing a checkpoint, dataset manifest, staging implementation, or
packed-record validator.  A code change still requires deployment and a new
environment-bound storage attestation before this retry.

This is the **one numerical launch** for Stages 6 and 7. You must see four
`started activation-farm` lines. If fewer than four workers start, the
supervisor rendezvous fails closed; do not manually start only the missing
worker. Wait for the other processes to return, diagnose deployment, then
relaunch the same command on all four.

`nohup` and complete stream redirection keep each supervisor alive if the Mac
sleeps or the SSH session closes. The shared control state and per-harvest
heartbeats—not the PID files—are authoritative.

The first numerical phase is the nine-candidate benchmark. Its sweep order is
EP16/FSDP1/TP1, EP8/FSDP2/TP1, then EP8/FSDP1/TP2, each with per-device sequence
batches 1, 2, and 4. Each candidate compiles once, runs five warmups, and runs
at least 200 synchronized steps in a fresh process.

Selection requires useful-prefix MFU at least 50%, no OOM, recompilation, or
dropped route, writer backlog below two batches, and identical instrumented
output. Among passing candidates it maximizes persisted tokens/s. A failed or
timed-out candidate keeps telemetry but cannot poison a later candidate. If no
candidate passes, all four activation supervisors exit and canary never starts.

Benchmark timing separates the synchronized device path from durable wall
time. `device_elapsed_seconds` covers each compiled prefix invocation through
`block_until_ready`; useful-prefix MFU and device tokens/s use that denominator.
`elapsed_seconds` additionally covers input handling, production-format
persistence, and the final writer drain, so persisted tokens/s continues to be
an end-to-end rate. `persistence_seconds` reports foreground persistence and
drain time. Benchmark records from the earlier combined-timing schema are
rejected rather than silently reinterpreted.

## Stage 7 — pass the canary, then farm train and eval activations overnight

### 7.1 Let the supervised 2M-token canary enforce its hard gate

There is no second launch here. After benchmark selection passes, the same four
supervisors collectively validate its record and run the 2,000,000-token train
canary. The next phase cannot start unless all of these are true:

- `canary_passed` is true and captured tokens equal 2,000,000;
- real-data goodput is at least 90% of synthetic;
- input wait is below 5%;
- persisted goodput is at least 90% of device-output goodput;
- writer backlog remains below two batches.

The canary objects are already immutable production objects. If the gate
passes, train capture adopts their receipts instead of recomputing them. If it
fails, all four supervisors exit and train capture never starts. Do not bypass
the gate with a standalone `harvest` invocation.

### 7.2 Let the same supervisors seal train, then eval

After the canary passes, the already-running campaign continues train capture
to exactly 50,000,000 tokens, seals it, then captures and seals exactly
2,000,000 held-out eval tokens. There is no second command to paste.

**`activation-farm` stops after held-out capture. It never starts SAE training,
SAE evaluation, or reporting.** Do not substitute `moe-sae run`; `run`
continues into all nine A/B/F training runs.

If the campaign stops safely or fails, resolve the cause and rerun the Stage 6
detached launch on all four controllers. The supervisor revalidates and skips
each complete benchmark/canary/capture record and resumes only the first
incomplete phase. It never trusts filenames without provenance checks.

### 7.3 Check status without disturbing the collective

The shared status command is safe to run from worker 0 while capture is live:

```zsh
gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
  --project="$GCP_PROJECT" \
  --zone="$TPU_ZONE" \
  --worker=0 \
  --command="cd '$TPU_PROJECT_ROOT' && \
    '$REMOTE_UV' run --frozen --extra tpu moe-sae activation-status \
      --artifact-root '$ARTIFACT_ROOT'"
```

Inspect each controller's process and recent log without sending a signal:

```zsh
tpu_all "printf '\n===== %s =====\n' \"\$(hostname -s)\"
if test -s '$REMOTE_FARM_PID' && kill -0 \$(cat '$REMOTE_FARM_PID') 2>/dev/null; then
  echo \"supervisor running pid=\$(cat '$REMOTE_FARM_PID')\"
else
  echo 'supervisor not running'
fi
tail -n 40 '$REMOTE_FARM_LOG' 2>/dev/null || true"
```

Healthy status means one coherent phase is `running` (or has just become
`passed`) and no phase is `failed`. During canary/train/eval, harvest-control
heartbeats remain fresh, shard/receipt counts advance, and writer backlog stays
bounded. During benchmark, use the four log tails and process checks. A quiet
compile is not itself a failure; stale heartbeats or a dead supervisor are.

Never `kill`, `pkill`, close only one worker, or restart one controller while a
collective phase is active.

### 7.4 Request a cooperative stop if needed

From worker 0, write one durable abort request:

```zsh
gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
  --project="$GCP_PROJECT" \
  --zone="$TPU_ZONE" \
  --worker=0 \
  --command="cd '$TPU_PROJECT_ROOT' && \
    '$REMOTE_UV' run --frozen --extra tpu moe-sae activation-abort \
      --artifact-root '$ARTIFACT_ROOT' \
      --reason 'operator requested a safe stop'"
```

The request is cooperative. During harvesting it is mirrored into the active
campaign, observed collectively at a safe batch boundary, and left latched so
all four supervisors see it. Wait until the process/log check above says
`supervisor not running` on all four. Only then clear the request:

```zsh
gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
  --project="$GCP_PROJECT" \
  --zone="$TPU_ZONE" \
  --worker=0 \
  --command="cd '$TPU_PROJECT_ROOT' && \
    '$REMOTE_UV' run --frozen --extra tpu moe-sae activation-abort \
      --artifact-root '$ARTIFACT_ROOT' \
      --clear"
```

This is not a force-kill mechanism. A process hard-hung inside an XLA compile,
driver call, or collective that never returns to Python cannot observe the
marker. If heartbeats go stale and none of the four supervisors exits, preserve
`activation-status`, all four log tails, and the TPU description first. Do not
kill one worker. The last-resort recovery is to stop/delete the **entire** TPU
slice (delete the entire queued resource for a queued allocation), then resume
later by launching the identical command on all four replacement controllers.
Already verified immutable shards remain resumable. Do not clear the abort
while any old supervisor or inherited child is still alive; the CLI also
refuses that unsafe clear.

Resume by rebuilding `REMOTE_FARM_COMMAND`, `REMOTE_FARM_QUOTED`, and
`REMOTE_FARM_LAUNCH` from Stage 6 and launching it on all four again. Existing
benchmark, canary, shards, receipts, and sealed outputs are reused only after
their identities validate.

### 7.5 Morning acceptance

Do not release the TPU until all four supervisor processes have exited and
`activation-status` reports all four phases `passed`. On worker 0, inspect:

```zsh
gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
  --project="$GCP_PROJECT" \
  --zone="$TPU_ZONE" \
  --worker=0 \
  --command="jq '{phase,captured_tokens,total_tokens,capture_manifest_sha256,manifest_path}' \
    '$ARTIFACT_ROOT/harvest/train.json' \
    '$ARTIFACT_ROOT/harvest/eval.json'"
```

Train must report exactly 50,000,000 captured tokens and eval exactly 2,000,000,
each with a nonempty sealed-manifest SHA/path. There is no later bulk upload:
every safetensors shard was conditionally written directly to its immutable,
content-addressed GCS URI, verified by generation, size, CRC32C, and SHA-256,
then represented by a self-hashed POSIX receipt and ledger entry.

If a worker dies after a receipt but before its ledger update, a collective
resume verifies and adopts the remote object. Progress is never inferred by
counting files. Final sealing proves contiguous ranges, no gaps/duplicates,
exact counts, remote identities, and merged FP64 statistics.

Capture hashing, manifest verification, and sealing are reported separately
from device execution, while reserved-cluster time remains part of shared
harvest cost. Campaign telemetry is bound to the sealed manifest SHA, config,
checkpoint, conversion attestation, dataset fingerprint, environment, storage
attestation, and selected benchmark.

For a direct TPU node, stop or delete it according to TRC policy. A queued
resource cannot be stopped; delete the queued resource when the allocation is
no longer needed. Do not release either resource after capture alone when the
SAE run will continue on this deployment. Filestore remains authoritative
until Stage 10 exports and independently verifies the complete control-plane
evidence bundle.

## Stage 8 — prove and seal one SAE operation

Stages 8–10 use one immutable SAE operation identity. Do not launch standalone
`train`, `eval`, `report`, or `run` commands alongside this workflow. The
four-controller `sae-farm` owns phase order and starts each numerical phase in
a fresh, fenced subprocess.

Keep the activation-scoped `preflight/storage.json` from Stage 4 exactly as
written. Its SHA is already embedded in every capture shard and campaign
record. The SAE smoke suite and resident-cache qualification add new immutable
proofs; neither replaces the producer proof.

Before creating either cache, re-prove the login lifecycle on the exact
deployment. This gate exists because one observed qualification successfully
verified 73.66 GB in `/dev/shm` at 4.946 GB/s, yet the cache disappeared about
20 seconds after the qualifier's SSH sessions exited. The image used
`RemoveIPC=yes` and the production user had no linger. In a controlled
comparison, a worker with `Linger=yes` retained its `/dev/shm` sentinel while
the three otherwise matching non-lingering workers lost theirs.

Run a real disconnect boundary, not just a configuration read:

```zsh
export SAE_LINGER_SENTINEL="/dev/shm/moe-sae-linger-$PROJECT_COMMIT"

tpu_all "set -eu
test \"\$(loginctl show-user \"\$(id -un)\" -p Linger --value)\" = yes
test ! -e '$SAE_LINGER_SENTINEL'
printf '%s\n' '$PROJECT_COMMIT' > '$SAE_LINGER_SENTINEL'"

# The first collective SSH command has now exited. Wait beyond the observed
# approximately 20-second logind cleanup window, then reconnect independently.
sleep 30

tpu_all "set -eu
test \"\$(loginctl show-user \"\$(id -un)\" -p Linger --value)\" = yes
test \"\$(cat '$SAE_LINGER_SENTINEL')\" = '$PROJECT_COMMIT'
rm -- '$SAE_LINGER_SENTINEL'"
```

All four second connections must pass. If any sentinel vanishes, stop and fix
the image/bootstrap lifecycle before smoke or cache qualification. `nohup`, a
fast reconnect, or a previously successful storage attestation is not a
substitute for this test.

### 8.1 Run full-shape smokes in the exact order F, A, B

Define two different host-private paths on the Mac. They may have identical
pathnames on all four controllers because the underlying `/dev/shm` mounts are
private:

```zsh
export SAE_SMOKE_CACHE_ROOT="/dev/shm/moe-sae-sae-smoke"
export SAE_CACHE_ROOT="/dev/shm/moe-sae-sae-cache"

[[ "$SAE_SMOKE_CACHE_ROOT" != "$SAE_CACHE_ROOT" ]]
tpu_all "set -eu
test ! -e '$SAE_SMOKE_CACHE_ROOT'
test ! -e '$SAE_CACHE_ROOT'
df -B1 /dev/shm"
```

Both roots must be fresh. Run exactly three foreground collective commands:

```zsh
for condition in F A B; do
  tpu_all "cd '$TPU_PROJECT_ROOT' && \
    '$REMOTE_UV' run --frozen --extra tpu moe-sae sae-smoke \
      --artifact-root '$ARTIFACT_ROOT' \
      --condition '$condition' \
      --cache-root '$SAE_SMOKE_CACHE_ROOT'"
done
```

Each `tpu_all` call starts one new process on every controller and must return
successfully on all four before the next condition starts. Do not add
`--attempt-id`; process 0 creates it and broadcasts the same value to the other
three processes. The B result seals and prints the exact F/A/B suite SHA.

Each smoke constructs the production-sized condition, performs one complete
16,384-token distributed update, verifies exact TopK-64 behavior, saves a real
Orbax checkpoint, inspects its finalized metadata, restores it with a fresh
checkpointer, and compares the restored state. Condition A also runs its exact
fixed-order capacity prepass. The input cache is isolated below that attempt's
`SAE_SMOKE_CACHE_ROOT` path. After a collective pass, the command removes only
that attempt cache; its self-hashed evidence remains on the shared control
plane and its immutable checkpoint remains in GCS. This cleanup is what leaves
enough `/dev/shm` headroom for the resident cache.

Do not manually delete, rename, copy, or synthesize a smoke attempt. On failure,
preserve the four command outputs and the immutable failure record, diagnose
the cause, and do not continue to cache qualification. One successful attempt
per condition is the production contract.

### 8.2 Qualify the resident train-plus-eval cache

Only after all three smokes pass, cold-fill the distinct resident cache:

```zsh
tpu_all "cd '$TPU_PROJECT_ROOT' && \
  '$REMOTE_UV' run --frozen --extra tpu moe-sae qualify-sae-storage \
    --artifact-root '$ARTIFACT_ROOT' \
    --cache-root '$SAE_CACHE_ROOT'"
```

Each worker receives the exact contiguous train-plus-eval partition used by
distributed training. The gate downloads every assigned immutable object once
with four concurrent objects, 16 inner GCS workers, and 16 MiB chunks; verifies
generation, size, and SHA-256; fsyncs each file; and commits by atomic rename.
The approximately 68.6 GiB assignment must remain below the sealed 100 GiB
bound. Before filling, free space must cover that assignment, four concurrent
temporary objects, and 128 GiB still free afterward. Aggregate verified
throughput must be at least 4 GB/s.

Success publishes the immutable
`preflight/sae-storage/<consumer-environment-sha256>.json`, scoped to
`sae-training`. It binds the producer proof, clean consumer environment,
4-process/4-local-device/16-global-device topology, data4×feature4 mesh, both
capture manifests, exact cache inventory, and producer/consumer compatibility
proof. A failed qualification leaves diagnostic files but publishes no
attestation; diagnose it and use a new empty cache path.

From this point through the final report, do not clean `$SAE_CACHE_ROOT`, reboot
a controller, redeploy, or let another job consume its `/dev/shm` headroom. The
runtime is resident-only and never silently refills a missing object. Loss of
one qualified file invalidates this deployment; `Linger=yes` does not soften
that rule. Stop rather than treating the old attestation as permission to
continue or refilling the old path under the old proof. The v1 attestation is
keyed by consumer environment, so same-environment requalification is not a
supported recovery path. Preserve the old proof and deploy an audited new
consumer environment before qualifying a fresh empty cache and sealing a new
operation identity. Replace this limitation with explicitly selected,
content-addressed cache attestations before the Spot R1 campaign.

### 8.3 Seal the operation and record its exact SHA

`sae-begin` validates the F/A/B suite and the new resident-cache attestation,
then derives one operation SHA from all scientific, implementation, storage,
waiver, and deployment identities. Invoke it collectively and retain process
0's JSON:

```zsh
export REMOTE_SAE_BEGIN_RESULT="$TPU_HOME/sae-begin.json"

tpu_all "set -o pipefail
cd '$TPU_PROJECT_ROOT'
'$REMOTE_UV' run --frozen --extra tpu moe-sae sae-begin \
  --artifact-root '$ARTIFACT_ROOT' \
  --cache-root '$SAE_SMOKE_CACHE_ROOT' \
  | tee '$REMOTE_SAE_BEGIN_RESULT'"

export OPERATION_SHA256="$(
  gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
    --project="$GCP_PROJECT" \
    --zone="$TPU_ZONE" \
    --worker=0 \
    --command="jq -er '.operation_sha256' '$REMOTE_SAE_BEGIN_RESULT'"
)"

print -r -- "$OPERATION_SHA256" | grep -Eq '^[0-9a-f]{64}$'
mkdir -p "$PWD/artifacts/gemma4_v1/operator-results"
print -r -- "$OPERATION_SHA256" > \
  "$PWD/artifacts/gemma4_v1/operator-results/sae-operation-sha256.txt"
print -r -- "SAE operation: $OPERATION_SHA256"
```

The same `OPERATION_SHA256` is mandatory for farm, status, abort, every
authoritative run/evaluation, and the final report. Never infer a "latest"
operation by directory order. A safe retry of `sae-begin` can only return the
same identity; a changed input creates a different operation.

The durable 2M canary remains truthfully failed when its performance-only
waiver is present. The operation binds the exact self-hashed waiver and sealed
50M recovery chain. If either the harvester MFU or real-data canary performance
gate was waived, the final report remains exploratory and headline-ineligible
even when every SAE phase passes.

## Stage 9 — launch and supervise the SAE farm

### 9.1 Start one detached supervisor on every controller

Build the exact command once in local zsh, quote it once, and launch it on all
four controllers:

```zsh
sae_farm_command() {
  print -r -- "cd '$TPU_PROJECT_ROOT' && \
'$REMOTE_UV' run --frozen --extra tpu moe-sae sae-farm \
  --artifact-root '$ARTIFACT_ROOT' \
  --operation-sha256 '$OPERATION_SHA256'"
}

export REMOTE_SAE_FARM_COMMAND="$(sae_farm_command)"
export REMOTE_SAE_FARM_QUOTED="$(printf '%q' "$REMOTE_SAE_FARM_COMMAND")"
export REMOTE_SAE_FARM_LOG="$TPU_HOME/sae-farm-$OPERATION_SHA256.log"
export REMOTE_SAE_FARM_PID="$TPU_HOME/sae-farm-$OPERATION_SHA256.pid"

export REMOTE_SAE_FARM_LAUNCH="set -eu
if test -s '$REMOTE_SAE_FARM_PID' && \
   kill -0 \$(cat '$REMOTE_SAE_FARM_PID') 2>/dev/null; then
  echo 'sae-farm is already running on this controller' >&2
  exit 1
fi
nohup bash -lc $REMOTE_SAE_FARM_QUOTED \
  >> '$REMOTE_SAE_FARM_LOG' 2>&1 < /dev/null &
echo \$! > '$REMOTE_SAE_FARM_PID'
echo \"started sae-farm pid=\$(cat '$REMOTE_SAE_FARM_PID') host=\$(hostname -s)\""

tpu_all "$REMOTE_SAE_FARM_LAUNCH"
```

You must see four `started sae-farm` lines. The long-lived parent imports no
JAX. It holds one kernel-backed lane lease for its lifetime and starts every
numerical phase in a fresh process while passing that lease only as a fencing
witness. Four matching lanes must rendezvous before a phase can run; a child
exit code alone cannot pass a phase because the parent independently validates
its immutable result envelope. The farm polls NFS control/lock state every five
seconds and publishes an unchanged heartbeat at most every 30 seconds. Do not
lower `--poll-seconds` on the shared NFS mount: a non-blocking remote `flock`
still invokes the NFSv3 Network Lock Manager, and aggressive polling can stall
all four clients.

The fixed order is:

1. validate the exact operation prerequisites;
2. bind the already-passed smoke evidence for F, A, then B;
3. train seed-1 A, B, and F to the 5M health boundary, without relative
   evaluation;
4. finish A/B/F for seeds 1, 2, and 3 in seed-major order;
5. evaluate those nine final checkpoints in the same seed-major order; and
6. publish the operation-scoped efficiency report.

The seed-1 gate requires finite state, exact L0=64, a final 64-update loss EMA
at most 80% of the initial EMA, and at least 95% live features for A/F. A gate
failure stops the operation; do not add an auxiliary loss or change the
contract in place. Training time includes synchronized optimizer steps and
input stalls. Compilation, checkpoint serialization, evaluation, and cache
qualification are recorded separately; A's accepted capacity prepass remains
part of A's measured marginal chip-seconds.

Checkpoints use unique attempt URIs below the operation-bound GCS checkpoint
root. The coordinator publishes a self-hashed `latest.json` only after Orbax
finalization; evaluation follows only `final.json`. The prior payload is not
removed until its replacement is finalized, published, and synchronously
cleaned according to the retention rule.

`sae-phase` is an internal hidden command. Never invoke it manually, put it in
an operator script, or use it to repair a failed phase. It is accepted only in
a fresh farm child with the inherited live lane descriptor and the exact active
phase/attempt tuple. Likewise, do not run direct `train`, `eval`, `report`, or
`run` commands while any farm lane is held.

### 9.2 Inspect status without importing JAX

`sae-status` is read-only and safe to run on worker 0 while the farm is live:

```zsh
gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
  --project="$GCP_PROJECT" \
  --zone="$TPU_ZONE" \
  --worker=0 \
  --command="cd '$TPU_PROJECT_ROOT' && \
    '$REMOTE_UV' run --frozen --extra tpu moe-sae sae-status \
      --artifact-root '$ARTIFACT_ROOT' \
      --operation-sha256 '$OPERATION_SHA256'"
```

For routine monitoring, run this command no more than once every 30 seconds.
The supervisors already perform their own five-second health checks; a tighter
external status loop adds NFS/NLM load without improving the fail-closed
decision. Heartbeats are declared stale only after 130 seconds (four missed
30-second publications plus two control-poll observations), and phase
rendezvous/completion waits remain finitely bounded at 520 seconds. A fresh
in-progress status without its lock still fails immediately; released terminal
records are durable history. These margins cover control-plane stalls, not dead
or split supervisors.

Also inspect all four parent processes and their recent outer logs:

```zsh
tpu_all "printf '\n===== %s =====\n' \"\$(hostname -s)\"
if test -s '$REMOTE_SAE_FARM_PID' && \
   kill -0 \$(cat '$REMOTE_SAE_FARM_PID') 2>/dev/null; then
  echo \"supervisor running pid=\$(cat '$REMOTE_SAE_FARM_PID')\"
else
  echo 'supervisor not running'
fi
tail -n 40 '$REMOTE_SAE_FARM_LOG' 2>/dev/null || true"
```

Healthy live status has `campaign_status: running`, exactly four held and
non-stale lanes on one phase/attempt, no orphaned or vanished lanes, and a
passed prefix followed by one running phase. At completion it has
`campaign_status: complete`, `next_phase: null`, and every phase is `passed`.
Per-phase stdout/stderr logs are already durable below
`$ARTIFACT_ROOT/operator/sae/operations/$OPERATION_SHA256/logs/sae-farm/`.

A quiet compile or checkpoint save is not itself a failure. A failed phase,
orphaned lane, vanished lane, split attempt, stale held lease, or four-parent
process disagreement is a stop condition. Preserve status and logs before
changing infrastructure. Never kill, restart, or relaunch only one controller.

### 9.3 Request a cooperative boundary stop

From worker 0, latch one operation-specific abort request:

```zsh
gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
  --project="$GCP_PROJECT" \
  --zone="$TPU_ZONE" \
  --worker=0 \
  --command="cd '$TPU_PROJECT_ROOT' && \
    '$REMOTE_UV' run --frozen --extra tpu moe-sae sae-abort \
      --artifact-root '$ARTIFACT_ROOT' \
      --operation-sha256 '$OPERATION_SHA256' \
      --reason 'operator requested a safe boundary stop'"
```

This is not a signal or force-kill. It prevents the next phase from starting;
an active numerical child must first return through its collective boundary.
Wait until the all-host process check reports `supervisor not running` on all
four. Only then clear the request:

```zsh
gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
  --project="$GCP_PROJECT" \
  --zone="$TPU_ZONE" \
  --worker=0 \
  --command="cd '$TPU_PROJECT_ROOT' && \
    '$REMOTE_UV' run --frozen --extra tpu moe-sae sae-abort \
      --artifact-root '$ARTIFACT_ROOT' \
      --operation-sha256 '$OPERATION_SHA256' \
      --clear"
```

The CLI refuses an unsafe clear while a phase runs or any lane lease is held.
After diagnosis, reuse the same `OPERATION_SHA256` only when the committed
consumer environment is unchanged and the failure is a transient,
restart-safe interruption. Passed phases are skipped only after their
immutable evidence revalidates. A source fix changes the consumer environment
identity: retain the old operation as failed evidence, deploy the audited fix,
requalify the resident cache for that environment, repeat the required F/A/B
smokes, and create a new operation with `sae-begin`. Do not edit or delete the
old state, `run.json`, checkpoint pointers, or phase records. Do not transplant
an old-operation checkpoint into the replacement operation: checkpoint and
run identities deliberately bind the old operation SHA, so even a numerically
healthy checkpoint is diagnostic evidence rather than an authorized resume
seed.

Training and evaluation run records are shared multi-process publications.
Every field in them must therefore be either replicated by construction or
made canonical by a collective. Capture-prefetch telemetry is an ordered
all-gather of each process's `objects` and `bytes`, with top-level totals that
represent aggregate per-process materialization work. A capture shard crossing
two process token partitions is intentionally counted twice; these counters
are not unique capture inventory. A payload disagreement is a failed phase,
not permission to discard the differing field or accept only process zero's
view.

If a child is hard-hung inside XLA or a collective, preserve status, all logs,
and the TPU description; recovery acts on the entire slice, never one worker.
Because the qualified cache is volatile, a TPU reboot or replacement also
ends this exact consumer deployment rather than authorizing an implicit cache
refill.

## Stage 10 — accept, export evidence, and clean up

### 10.1 Accept the operation and inspect the report

Do not release any resource until all four parent processes have exited and
`sae-status` says `campaign_status: complete`. Inspect the operation-scoped
report from worker 0:

```zsh
export SAE_OPERATION_ROOT="$ARTIFACT_ROOT/operator/sae/operations/$OPERATION_SHA256"
export SAE_REPORT="$SAE_OPERATION_ROOT/artifacts/report.json"

gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
  --project="$GCP_PROJECT" \
  --zone="$TPU_ZONE" \
  --worker=0 \
  --command="jq '{status,
    headline_eligible: (.headline_eligible // .primary.quality.headline_eligible),
    gain_recovery_percent: .primary.quality.gain_recovery_percent,
    cost_ratio_percent: .primary.cost.cost_ratio_percent,
    headline}' '$SAE_REPORT'"
```

The report contains per-seed EV/NMSE, exact L0, dead rate, candidate exposure,
chip-seconds, harvester MFU, persisted capture throughput, cache diagnostics,
and wall time. If F-minus-B is non-positive, G is undefined. If its paired
interval includes zero, the ratio is unstable/exploratory. A benchmark or
canary performance waiver makes `headline_eligible` false regardless of G and
C; preserve that status rather than editing the report.

### 10.2 Export the complete control plane before deleting Filestore

GCS already contains the immutable capture shards and Orbax payloads, but it
does not by itself contain every authoritative POSIX control record. Filestore
must remain mounted until a complete archive of `$ARTIFACT_ROOT` and all four
outer supervisor logs has been uploaded to an operation-specific GCS prefix
and verified from a second copy.

First upload each stopped supervisor's outer log under its content hash:

```zsh
export EVIDENCE_PREFIX="$BULK_GCS_ROOT/evidence/operations/$OPERATION_SHA256"

tpu_all "set -eu
if test -s '$REMOTE_SAE_FARM_PID' && \
   kill -0 \$(cat '$REMOTE_SAE_FARM_PID') 2>/dev/null; then
  echo 'refusing evidence export while sae-farm is live' >&2
  exit 1
fi
test -s '$REMOTE_SAE_FARM_LOG'
log_sha=\$(sha256sum '$REMOTE_SAE_FARM_LOG' | awk '{print \$1}')
log_uri='$EVIDENCE_PREFIX/supervisors/'\$(hostname -s)-\$log_sha.log
if gcloud storage ls \"\$log_uri\" >/dev/null 2>&1; then
  test \"\$(gcloud storage cat \"\$log_uri\" | sha256sum | awk '{print \$1}')\" = \"\$log_sha\"
else
  gcloud storage cp --if-generation-match=0 \
    '$REMOTE_SAE_FARM_LOG' \"\$log_uri\"
fi
test \"\$(gcloud storage cat \"\$log_uri\" | sha256sum | awk '{print \$1}')\" = \"\$log_sha\"
printf 'VERIFIED %s %s\n' \"\$(hostname -s)\" \"\$log_uri\""
```

Then create the quiescent control-plane archive on worker 0, upload it with an
object-creation precondition, and verify the uploaded bytes:

```zsh
export LOCAL_EVIDENCE_RECORD="$PWD/artifacts/gemma4_v1/operator-results/sae-evidence-export.txt"
mkdir -p "${LOCAL_EVIDENCE_RECORD:h}"

gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
  --project="$GCP_PROJECT" \
  --zone="$TPU_ZONE" \
  --worker=0 \
  --command="set -eu
archive='/dev/shm/gemma4-evidence-$OPERATION_SHA256.tar.gz'
rm -f \"\$archive\"
tar --sort=name --mtime='UTC 1970-01-01' \
  --owner=0 --group=0 --numeric-owner \
  -czf \"\$archive\" -C '$ARTIFACT_ROOT' .
evidence_sha=\$(sha256sum \"\$archive\" | awk '{print \$1}')
evidence_uri='$EVIDENCE_PREFIX/control/'\$evidence_sha.tar.gz
if gcloud storage ls \"\$evidence_uri\" >/dev/null 2>&1; then
  test \"\$(gcloud storage cat \"\$evidence_uri\" | sha256sum | awk '{print \$1}')\" = \"\$evidence_sha\"
else
  gcloud storage cp --if-generation-match=0 \"\$archive\" \"\$evidence_uri\"
fi
test \"\$(gcloud storage cat \"\$evidence_uri\" | sha256sum | awk '{print \$1}')\" = \"\$evidence_sha\"
printf 'EVIDENCE_SHA256=%s\nEVIDENCE_URI=%s\n' \
  \"\$evidence_sha\" \"\$evidence_uri\"" \
  | tee "$LOCAL_EVIDENCE_RECORD"
```

The archive includes configuration, conversion/storage/compatibility proofs,
benchmark and waiver records, both capture manifests, receipts and host
ledgers, smoke suite, operation state, phase-result envelopes, per-phase logs,
metrics journals, checkpoint pointers, evaluations, timings, and report. It
does not duplicate the large GCS capture or Orbax payload bytes.

Download the archive to a second filesystem, verify its SHA, extract it, and
check the minimum reconstruction identities:

```zsh
export EVIDENCE_SHA256="$(grep '^EVIDENCE_SHA256=' "$LOCAL_EVIDENCE_RECORD" | cut -d= -f2-)"
export EVIDENCE_URI="$(grep '^EVIDENCE_URI=' "$LOCAL_EVIDENCE_RECORD" | cut -d= -f2-)"
export EVIDENCE_RECOVERY_ROOT="$HOME/moe-sae-evidence/$OPERATION_SHA256"

mkdir -p "$EVIDENCE_RECOVERY_ROOT"
gcloud storage cp "$EVIDENCE_URI" "$EVIDENCE_RECOVERY_ROOT/control.tar.gz"
test "$(shasum -a 256 "$EVIDENCE_RECOVERY_ROOT/control.tar.gz" | awk '{print $1}')" = \
  "$EVIDENCE_SHA256"
tar -xzf "$EVIDENCE_RECOVERY_ROOT/control.tar.gz" \
  -C "$EVIDENCE_RECOVERY_ROOT"

jq -e --arg operation "$OPERATION_SHA256" \
  '.operation_sha256 == $operation and
   ([.phases[].status] | all(. == "passed"))' \
  "$EVIDENCE_RECOVERY_ROOT/operator/sae/operations/$OPERATION_SHA256/state.json"
jq -e '.total_tokens == 50000000' \
  "$EVIDENCE_RECOVERY_ROOT/captures/train/manifest.json"
jq -e '.total_tokens == 2000000' \
  "$EVIDENCE_RECOVERY_ROOT/captures/eval/manifest.json"
jq -e --arg operation "$OPERATION_SHA256" \
  '.experiment.operation_sha256 == $operation' \
  "$EVIDENCE_RECOVERY_ROOT/operator/sae/operations/$OPERATION_SHA256/artifacts/report.json"
```

These checks must run against the downloaded copy, not the still-mounted NFS
tree. Also retain the four `VERIFIED` supervisor-log URIs and confirm the bulk
capture and checkpoint prefixes still exist. If any archive hash, extracted
identity, token count, phase status, or report binding fails, Filestore remains
authoritative: do not delete it.

### 10.3 Release resources only after verified export

After the downloaded evidence copy passes, it is safe for `/dev/shm` to vanish
with the TPU. Retain the GCS capture objects, final Orbax payloads, smoke
checkpoints, evidence archive, and supervisor logs for analysis and audit. Do
not delete them as cache cleanup.

Delete the whole TPU allocation according to how it was created, then delete
Filestore:

```zsh
if [[ -n "$QUEUED_RESOURCE_NAME" ]]; then
  gcloud compute tpus queued-resources delete "$QUEUED_RESOURCE_NAME" \
    --project="$GCP_PROJECT" \
    --zone="$TPU_ZONE"
else
  gcloud compute tpus tpu-vm delete "$TPU_NAME" \
    --project="$GCP_PROJECT" \
    --zone="$TPU_ZONE"
fi

gcloud filestore instances delete "$FILESTORE_NAME" \
  --project="$GCP_PROJECT" \
  --zone="$TPU_ZONE"
```

Never delete one TPU worker, delete Filestore before the evidence check, or
treat an outer `nohup` process ending as proof that the scientific operation
completed. The operation status, immutable report, content-hashed archive, and
downloaded reconstruction checks are the teardown authority.
