# Production runbook

Every stage is fail closed. Do not continue past a failed parity, integrity,
MFU, goodput, health, or exact-count gate.

## 1. Security and environment

Revoke and replace any Hugging Face credential that appeared in notebook or
terminal output. Authenticate through `HF_TOKEN` in the process environment;
never pass it as a CLI argument. The repository preflight scans file contents
for common token shapes without printing token values.

On every TPU host:

```sh
uv sync --extra tpu
uv run --extra tpu moe-sae preflight \
  --checkpoint gs://BUCKET/gemma4-26b-pt/0/items \
  --grain-train 'gs://BUCKET/c4-3.1.0/train/*.array_record*' \
  --grain-eval 'gs://BUCKET/c4-3.1.0/validation/*.array_record*' \
  --dataset-train-manifest gs://BUCKET/c4-3.1.0/train.manifest.json \
  --dataset-eval-manifest gs://BUCKET/c4-3.1.0/validation.manifest.json
```

Preflight requires Python 3.12, the exact MaxText provenance, a 16-chip TPU
runtime, staged train/eval patterns, an explicit checkpoint, a clean secret
scan, and an artifact path below `artifacts/gemma4_v1/`.

`artifacts/gemma4_v1/` must be one shared, strongly consistent POSIX mount
visible at the same path on every worker. A merely host-local directory is not
valid: preflight writes a nonce per process and proves that all workers can see
all nonces before allowing production work.

## 2. Convert and validate the checkpoint

Use an isolated CPU conversion environment with CPU Torch installed. Point
`--maxtext-root` at the exact pinned checkout; the command verifies Git HEAD,
requires a clean worktree, forces both subprocesses to import from that
checkout's source tree, converts unscanned BF16 text-only weights, and then
runs the MaxText/Hugging Face parity checker with `max_kl_div=0.03`.

```sh
uv run --extra conversion moe-sae convert-checkpoint \
  --maxtext-root /opt/maxtext \
  --output gs://BUCKET/gemma4-26b-pt
```

The resulting model items path is `OUTPUT/0/items`. After the logit check
passes, the command inventories the checkpoint and writes
`artifacts/gemma4_v1/checkpoint/conversion.json`. Copy its
`checkpoint_sha256` value into every benchmark/harvest/run invocation; never
invent or truncate this digest. The file is self-hashed and records a passing
parity result; benchmark, capture, campaign, training, evaluation, and report
stages reject a missing, modified, or mismatched attestation. Compilation/training environments do not
install or import Torch. Before loading MaxText, each production invocation
recomputes the staged local/GCS inventory on every controller, requires the
controllers to agree, and rejects a mismatch with the supplied digest.

## 3. Pre-stage deterministic C4

Convert C4 English 3.1.0 train and validation to deterministic Grain
ArrayRecord shards before reserving the v4-32. Live Hugging Face streaming is
not an allowed production input. Each split must have a
`gemma4-c4-arrayrecord-v1` JSON manifest with these exact top-level fields:
`schema_version`, `dataset`, `tokenizer_id`, `sequence_length`, `packing`, and
`files`. The ordered `files` entries contain `uri`, `sha256`, `size_bytes`,
`record_count`, and `scientific_tokens`. `dataset` fixes C4/en 3.1.0 and the
actual split; `packing` fixes 1,024-token packing and exclusion of padding,
BOS, and EOS (IDs 0, 2, and 1). Before model execution, every listed object is
hashed and compared with the exact `--grain-files` expansion, and Grain
reopens the ArrayRecord source to recompute its record count. The capture loop
then derives scientific-token counts from the actual packed segmentation and
special-token mask. The canonical manifest hash becomes the dataset
fingerprint; it is never entered manually.

## 4. Benchmark capture

Run all nine candidates, in this order:

1. EP16/FSDP1/TP1;
2. EP8/FSDP2/TP1;
3. EP8/FSDP1/TP2;

with per-device sequence batches 1, 2, and 4. Each candidate compiles once,
runs five warmups, then at least 200 synchronized steps.

```sh
CHECKPOINT_SHA256='64 lowercase hex characters from conversion.json'

uv run --extra tpu moe-sae benchmark-harvest \
  --checkpoint gs://BUCKET/gemma4-26b-pt/0/items \
  --checkpoint-sha256 "$CHECKPOINT_SHA256" \
  --output artifacts/gemma4_v1/benchmark
```

Selection maximizes persisted tokens/s among candidates with useful-prefix MFU
at least 50%, no OOM/recompile/dropped route, writer backlog below two batches,
and identical instrumented output. Useful FLOPs count only blocks 1–14 plus
target attention/router/routed FFN; the denominator is 4.4 PFLOP/s times wall
time. The public command remains JAX-cold: one short bootstrap verifies the
slice/checkpoint/environment, then every mesh/batch runs in a fresh coordinated
worker process. A shared one-hour deadline kills the entire local process group,
collects every controller's termination status, and publishes a failed attempt
before the next candidate starts. A candidate exception, OOM, or hang therefore
cannot poison the rest of the sweep; its logs and status remain under the
benchmark supervisor directory.

## 5. Canary and harvest

The real-data canary writes the first 2M production training tokens and is
resumable by immutable input-range ID.

```sh
uv run --extra tpu moe-sae harvest --split train --canary \
  --checkpoint gs://BUCKET/gemma4-26b-pt/0/items \
  --checkpoint-sha256 "$CHECKPOINT_SHA256" \
  --dataset-manifest gs://BUCKET/c4-3.1.0/train.manifest.json \
  --grain-files 'gs://BUCKET/c4-3.1.0/train/*.array_record*'

uv run --extra tpu moe-sae harvest --split train \
  --checkpoint gs://BUCKET/gemma4-26b-pt/0/items \
  --checkpoint-sha256 "$CHECKPOINT_SHA256" \
  --dataset-manifest gs://BUCKET/c4-3.1.0/train.manifest.json \
  --grain-files 'gs://BUCKET/c4-3.1.0/train/*.array_record*'

uv run --extra tpu moe-sae harvest --split eval \
  --checkpoint gs://BUCKET/gemma4-26b-pt/0/items \
  --checkpoint-sha256 "$CHECKPOINT_SHA256" \
  --dataset-manifest gs://BUCKET/c4-3.1.0/validation.manifest.json \
  --grain-files 'gs://BUCKET/c4-3.1.0/validation/*.array_record*'
```

Continue only when real goodput is at least 90% of synthetic, input wait is
below 5%, persisted goodput is at least 90% of device-output goodput, and the
writer backlog remains below two batches. Resume the same store to exactly 50M
training tokens, then capture exactly 2M validation tokens. Host manifests are
merged only after every hash/count validates. Publish to GCS only after the
sealed merged manifest exists.

Manifest hashing/statistics verification and atomic publication are timed
separately from device capture, but their reserved-cluster time is included in
the shared harvest chip-seconds used by end-to-end cost ratios. Campaign v2
telemetry binds that cost to the sealed manifest SHA, config, base checkpoint,
conversion-parity attestation, dataset order, environment, and selected
benchmark; stale cost files cannot be reused with a replacement capture. If a
crash leaves publication time indeterminate, authoritative cost recovery stops
instead of charging offline downtime to the TPU campaign.

## 6. Train without quality peeking

Train seed 1 for A, B, and F to the 5M health boundary without comparing their
relative quality:

```sh
for condition in A B F; do
  uv run --extra tpu moe-sae train \
    --condition "$condition" --seed 1 --stop-after-tokens 5000000
done
```

Each run must have finite state, exact L0=64, a final 64-update loss EMA no more
than 80% of the initial EMA, and at least 95% live features for A/F. Stop and
diagnose a failure; do not add auxiliary losses. On pass, resume seed 1 to 50M,
then run seeds 2 and 3 with the same activation order.

Training checkpoints near 10M-token boundaries retain only `latest` and
`final`. Timing brackets synchronized optimizer steps and input stalls, while
compilation/checkpoint/evaluation times are logged separately.
Condition A's measured fixed-order capacity scan is also included in its
marginal wall/chip-seconds; a manual production capacity override is rejected.
Every run also has an opaque instance ID embedded in its Orbax metadata. Final
evaluation requires that ID plus exactly 50M restored tokens and 3,052 updates.
Condition A derives a separate fixed held-out routing capacity before
evaluation; that time is logged as excluded evaluation cost and cannot inflate
the marginal training-cost result.

## 7. Evaluate and report

Evaluate all nine final checkpoints on the same sealed held-out capture and its
common SST. Then build the report:

```sh
uv run --extra tpu moe-sae eval --all
uv run --extra tpu moe-sae report
```

`report.json` is the authoritative machine-readable result. If F-minus-B is
non-positive or its paired interval includes zero, the renderer refuses a
headline-ready status and reports the prescribed undefined/exploratory form.

`moe-sae run` executes the entire gated sequence, stopping at the first unmet
gate and reusing only provenance-matching sealed stages:

```sh
uv run --extra tpu moe-sae run \
  --checkpoint gs://BUCKET/gemma4-26b-pt/0/items \
  --checkpoint-sha256 "$CHECKPOINT_SHA256" \
  --grain-train 'gs://BUCKET/c4-3.1.0/train/*.array_record*' \
  --grain-eval 'gs://BUCKET/c4-3.1.0/validation/*.array_record*' \
  --dataset-train-manifest gs://BUCKET/c4-3.1.0/train.manifest.json \
  --dataset-eval-manifest gs://BUCKET/c4-3.1.0/validation.manifest.json
```
