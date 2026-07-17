# Production runbook

Every stage is fail closed. Do not continue past a failed parity, storage,
integrity, MFU, goodput, health, or exact-count gate. Production commands that
initialize JAX must be launched collectively with the same arguments on all
four v4-32 controllers; the snippets below show that per-controller command.

## 1. Security and deployment variables

Revoke and replace any Hugging Face credential that appeared in notebook or
terminal output. Authenticate through `HF_TOKEN` in the process environment;
never pass it as a CLI argument. Preflight scans repository file contents for
common token shapes without printing token values.

Use one stable production layout. The artifact root must be below
`artifacts/gemma4_v1/` and resolve to the same strongly consistent POSIX mount
on every controller:

```zsh
export ARTIFACT_ROOT="$PWD/artifacts/gemma4_v1/production"
export TFDS_DATA_DIR="/path/to/prepared/tfds"
export LOCAL_C4_STAGE="/path/to/cpu-local/c4-stage"
export BULK_GCS_ROOT="gs://BUCKET/moe-sae/gemma4_v1"
export SAE_CHECKPOINT_ROOT="$BULK_GCS_ROOT/sae-checkpoints"
export LOCAL_BUFFER_ROOT="/mnt/localssd/moe-sae"
export TOKENIZER_REVISION="<40-character-gemma-tokenizer-commit>"
export MAXTEXT_ROOT="/opt/maxtext"
export MODEL_CHECKPOINT="gs://BUCKET/gemma4-26b-pt/0/items"
```

The paths have deliberately different jobs:

- `ARTIFACT_ROOT` is the small shared POSIX control plane. It contains locks,
  attestations, identities, receipts, host ledgers, sealed manifests, timing
  records, and checkpoint pointers. It must not contain activation or Orbax
  payloads.
- `LOCAL_BUFFER_ROOT` is per-controller local scratch. Capture shards are
  serialized there before upload, and its `capture-cache/` holds verified GCS
  shards for training and evaluation. The configured 100 GiB cache limit is
  enforced independently on each controller; the cache is not shared POSIX
  state.
- `BULK_GCS_ROOT` is the durable bulk-data plane for immutable staged data and
  activation shards.
- `SAE_CHECKPOINT_ROOT` is the immutable Orbax payload plane. It must be in the
  same bucket and lexically below `BULK_GCS_ROOT`.

Install only the dependencies needed by each environment:

```zsh
# CPU staging host
uv sync --extra data

# Isolated CPU checkpoint-conversion environment
uv sync --extra conversion

# Every TPU controller
uv sync --extra tpu
```

## 2. Stage and publish deterministic C4

Do this before reserving the v4-32. The stager accepts only an already-prepared
TFDS `c4/en:3.1.0`; it never downloads C4 or silently relabels another version.
The tokenizer is pinned to the supplied 40-character revision. Packing is
ordered, offline, and independent of TPU batch size.

Stage and publish both TFDS splits from CPU-local scratch:

```zsh
for split in train validation; do
  uv run --extra data moe-sae stage-c4 \
    --artifact-root "$ARTIFACT_ROOT" \
    --split "$split" \
    --tfds-data-dir "$TFDS_DATA_DIR" \
    --tokenizer-revision "$TOKENIZER_REVISION" \
    --output-directory "$LOCAL_C4_STAGE/$split" \
    --records-per-shard 4096 \
    --publish-gcs "$BULK_GCS_ROOT/datasets/c4/$split"
done
```

Production targets come from the configuration: 50,000,000 scientific train
tokens and 2,000,000 scientific validation tokens. `--target-tokens` is only
valid with `--local-smoke` when testing a reduced local input. Resume occurs
only at a completely sealed shard boundary.

The exact manifest schema is `gemma4-c4-packed-arrayrecord-v1`. Its top-level
fields are:

```text
schema_version, dataset, source, tokenizer, sequence_length,
record_format, packing, consumer, target_scientific_tokens,
source_examples, scientific_tokens, record_count,
lane_scientific_tokens, files
```

Each fixed record uses `moe-sae-packed-sequence-v1` and contains
`token_ids:int32[1024]`, `positions:uint16[1024]`, and
`segment_ids:uint16[1024]`. Padding, BOS, and EOS (IDs 0, 2, and 1) are excluded
from scientific counts. Records are assigned by
`record_index % 4`, every controller lane meets its token quota, and only the
dataset tail is padded to the static 64-record multiple. A file entry contains
`uri`, `sha256`, `size_bytes`, `first_record_index`, `record_count`, and
`scientific_tokens`. A published GCS entry additionally records the exact
object generation, CRC32C, size, and SHA-256 identity.

Publication conditionally uploads full-SHA-256-named ArrayRecord objects, then
publishes the content-addressed dataset manifest. Save each command's printed
`manifest_location` and `grain_files`; the remote values look like:

```zsh
export TRAIN_DATASET_MANIFEST="gs://BUCKET/.../train/manifests/<sha256>.json"
export TRAIN_GRAIN_FILES='gs://BUCKET/.../train/shards/*.array_record'
export EVAL_DATASET_MANIFEST="gs://BUCKET/.../validation/manifests/<sha256>.json"
export EVAL_GRAIN_FILES='gs://BUCKET/.../validation/shards/*.array_record'
```

The CLI calls the TFDS held-out split `validation`; capture and SAE commands
call the same split `eval`. Preflight hashes the ordered objects, reopens their
ArrayRecords, recomputes counts, verifies that the pinned generations stayed
unchanged across the scan, and uses the canonical manifest hash as the dataset
fingerprint.

After publication, make the dataset prefix read-only to every campaign service
account (no object update/delete permission) for the lifetime of the run.
Generation pins detect drift; the IAM boundary prevents an authorized external
writer from replacing an object between preflight and Grain consumption.

## 3. Convert and validate the model checkpoint

Run conversion in the isolated CPU environment. `--maxtext-root` must point at
the clean checkout of commit
`48c7da3e28e8f57fe36ddf09d00eec53072a2ea0`. The command converts unscanned,
BF16, text-only base/PT weights and runs the MaxText/Hugging Face logit checker
with `max_kl_div=0.03`:

```zsh
uv run --extra conversion moe-sae convert-checkpoint \
  --artifact-root "$ARTIFACT_ROOT" \
  --maxtext-root "$MAXTEXT_ROOT" \
  --output gs://BUCKET/gemma4-26b-pt
```

The resulting items path is `OUTPUT/0/items`. The command inventories it and
writes the self-hashed control record
`$ARTIFACT_ROOT/checkpoint/conversion.json`. Set the exact 64-character
`checkpoint_sha256` from that file; never invent or truncate it:

```zsh
export CHECKPOINT_SHA256="<64 lowercase hex characters from conversion.json>"
```

The TPU runtime does not install or import Torch. Every production model stage
recomputes the checkpoint inventory on every controller, requires agreement,
and compares it with `CHECKPOINT_SHA256`.

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
