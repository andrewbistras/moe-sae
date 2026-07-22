# moe-sae

This repository runs one focused Gemma 4 experiment:

> Router-conditioned A recovers **G%** of the full-dictionary reconstruction
> gain at **C%** of its measured marginal SAE training cost.

The active model is `google/gemma-4-26B-A4B`, text only. Activations are the
aggregate routed-MoE branch at one-indexed block 15, immediately after
`post_feedforward_layernorm_2`. Conditions A, B, and F reconstruct exactly the
same globally centered target with exact per-token TopK-64.

The previous DeepSeek-VL2/spec-era implementation is intentionally absent from
the active tree. It is preserved at Git tag
`deepseek-vl2-final-a291707`; ignored historical `runs/`, `captures/`, and
`stats/` directories are never read or migrated.

## Active surface

- [`configs/gemma4_26b_a4b.yaml`](configs/gemma4_26b_a4b.yaml) — the single
  production configuration.
- `moe-sae` — the single CLI, with `stage-c4`, `qualify-storage`,
  `qualify-sae-storage`, `preflight`,
  `convert-checkpoint`, `benchmark-harvest`, `harvest`, `activation-farm`,
  `activation-status`, `activation-abort`, `sae-smoke`, `sae-begin`,
  `sae-farm`, `sae-status`, `sae-abort`, `train`, `eval`, `report`, and `run`.
  Production SAE operation goes through the `sae-*` commands; `sae-phase` is
  a hidden, lease-fenced farm child and is never an operator command.
- [`src/moe_sae/data.py`](src/moe_sae/data.py) — deterministic C4 packing and
  the fixed-record Grain/ArrayRecord reader.
- [`src/moe_sae/model.py`](src/moe_sae/model.py) — pinned MaxText prefix adapter.
- [`src/moe_sae/artifacts.py`](src/moe_sae/artifacts.py) — content-addressed
  capture shards, atomic manifests, and range-identity resume.
- [`src/moe_sae/activation_operator.py`](src/moe_sae/activation_operator.py)
  and [`src/moe_sae/harvest_control.py`](src/moe_sae/harvest_control.py) — the
  four-controller activation-only supervisor, lifetime leases, durable status,
  and cooperative safe-stop records.
- [`src/moe_sae/storage.py`](src/moe_sae/storage.py) and
  [`src/moe_sae/object_store.py`](src/moe_sae/object_store.py) — storage-plane
  qualification, immutable GCS publication, and the bounded local cache.
- [`src/moe_sae/checkpointing.py`](src/moe_sae/checkpointing.py) — immutable
  Orbax payload namespaces and self-hashed POSIX control pointers.
- [`src/moe_sae/sae.py`](src/moe_sae/sae.py) — exact A/B/F scoring, distributed
  TopK, sparse decode, and optimizer math.
- [`src/moe_sae/sae_operator.py`](src/moe_sae/sae_operator.py) and
  [`src/moe_sae/sae_farm.py`](src/moe_sae/sae_farm.py) — the four-controller
  SAE operation identity, lifetime lane leases, fixed phase plan, durable
  status, and cooperative boundary aborts.
- [`src/moe_sae/metrics.py`](src/moe_sae/metrics.py) — reconstruction and cost
  report contract.

## Local development

Python 3.12 and [uv](https://docs.astral.sh/uv/) are required.

```sh
uv sync
uv run pytest -q
uv run moe-sae --help
```

TPU hosts additionally install the pinned MaxText runtime:

```sh
uv sync --extra tpu
```

Checkpoint conversion uses a separate CPU-only Torch environment:

```sh
uv sync --extra conversion
```

MaxText resolves from commit
`48c7da3e28e8f57fe36ddf09d00eec53072a2ea0`. Torch is not a training-runtime
dependency; install CPU Torch only in the isolated checkpoint-conversion
environment.

## Stage C4 before reserving the TPU

The stager accepts one explicit train object and one explicit validation object
from `allenai/c4` revision
`1588ec454efa1a09f29cd18ddd04fe05fc8653a2`. Their paths, compressed sizes,
and SHA-256 hashes are pinned in code and verified before parsing. It does not
build or download the full TFDS corpus. The Gemma tokenizer is a verified
two-file local view from model revision
`24548b62aa021d562695c04aaf7758a1ea47990b`; both files are hash/size checked
before and after loading.

Documents remain in source order, are packed into fixed 1,024-token records,
and exclude padding, BOS, and EOS from scientific counts. The sealed train and
validation inventories target 50M and 2M tokens, balance the four controller
lanes, and pad only the record tail to a supported static batch multiple.

Run both splits on a CPU staging machine with local scratch. Resume is allowed
only from a completely sealed shard boundary. `--publish-gcs` conditionally
uploads SHA-256-named ArrayRecord objects and publishes the content-addressed
manifest last:

```zsh
export ARTIFACT_ROOT="$PWD/artifacts/gemma4_v1/production"
export C4_SOURCE_CACHE="$HOME/.cache/moe-sae/c4-source"
export LOCAL_C4_STAGE="/path/to/local/c4-stage"
export BULK_GCS_ROOT="gs://BUCKET/moe-sae/gemma4_v1"
export HF_HUB_DISABLE_XET=1

for split in train validation; do
  uv run --frozen --extra data moe-sae stage-c4 \
    --artifact-root "$ARTIFACT_ROOT" \
    --split "$split" \
    --source-cache-dir "$C4_SOURCE_CACHE" \
    --output-directory "$LOCAL_C4_STAGE/$split" \
    --publish-gcs "$BULK_GCS_ROOT/datasets/c4/$split"
done
```

Keep the printed `manifest_location` and `grain_files` values for `preflight`,
`harvest`, or `run`. Staging calls the held-out split `validation`; harvesting calls
the same held-out input `eval`. Remote file entries pin GCS generation, size,
CRC32C, and SHA-256; preflight verifies those identities before and after its
full byte/count scan.

## Split storage architecture

Production deliberately separates coordination from bulk I/O:

- The control plane is the small `--artifact-root` below
  `artifacts/gemma4_v1/`. It must be a strongly consistent shared POSIX mount
  visible at the same path on all controllers. It holds attestations, locks,
  upload receipts, host ledgers, sealed manifests, timing records, run
  identities, and checkpoint pointers—not activation or checkpoint payloads.
- Each controller's `--local-buffer-root` is host-local scratch. Capture shards
  are serialized there before upload, and training/evaluation use its bounded,
  SHA-256-verifying `capture-cache`. The mount-point string may be identical on
  every controller, but the underlying storage must be process-private.
- `--bulk-gcs-root` is the durable data plane. Harvesters upload immutable,
  content-addressed safetensors directly from their local buffers. Each ledger
  entry binds the object's URI, generation, size, CRC32C, and SHA-256. After all
  hosts finalize, the coordinator verifies exact object identities and
  contiguous ranges/counts, merges FP64 statistics, and atomically seals the
  POSIX manifest. Readers see only that sealed manifest.
- Orbax SAE payloads use unique attempt paths below
  `--sae-checkpoint-root`. That GCS root must be in the same bucket and below
  the qualified bulk root. The corresponding run directory on POSIX contains
  self-hashed `latest.json` and `final.json` pointers; a payload is never
  overwritten or trusted before Orbax reports it finalized. After a new latest
  pointer is durable, the superseded payload is removed synchronously; the
  final and latest pointers share the retained final payload.

Qualify this layout on the allocated v4-32, on all four controllers, before
production preflight:

```zsh
# On the current v4-32 image, use this only after inspect_workers.sh confirms
# that /dev/shm is the direct, private tmpfs mount with sufficient capacity.
export LOCAL_BUFFER_ROOT="/dev/shm/moe-sae-local"
export SAE_CHECKPOINT_ROOT="$BULK_GCS_ROOT/sae-checkpoints"

uv run --extra tpu moe-sae qualify-storage \
  --artifact-root "$ARTIFACT_ROOT" \
  --local-buffer-root "$LOCAL_BUFFER_ROOT" \
  --bulk-gcs-root "$BULK_GCS_ROOT" \
  --sae-checkpoint-root "$SAE_CHECKPOINT_ROOT"
```

The activation-capture thresholds are pinned and cannot be overridden: at
least 10GB free on the control volume, 24GiB per local rolling buffer,
an 8GiB local-write probe per process at least 4GB/s aggregate, an
incompressible 8GiB GCS probe per process, and at least 4GB/s aggregate
generation-pinned upload. Download integrity is mandatory and its throughput
is recorded diagnostically; the configured 4GB/s download requirement is
sealed for the later SAE-cache qualification. `/dev/shm` is eligible only as
the direct `tmpfs` mount and is deliberately ephemeral; durable state remains
in GCS and the shared control plane. Qualification also proves
that equal local-buffer paths resolve to process-private storage, cross-process
`flock`, atomic rename and visibility, disjoint host files, durable directory
fsync, end-to-end object hashes, and exact-generation cleanup. It then performs
a tiny collective Orbax save, finalization check, and
fresh-checkpointer restore at the exact checkpoint prefix; that immutable probe
and its timings are sealed into the self-hashed, `activation-capture`-scoped
`preflight/storage.json`. Benchmark and harvest accept that scope. After both
captures seal, `qualify-sae-storage` cold-fills each host's exact contiguous
partition into a 100 GiB bounded cache (outer-4 objects, inner-16 workers,
16 MiB chunks), requires 128 GiB free afterward and 4 GB/s aggregate verified
throughput, then publishes the separate immutable, `sae-training`-scoped
`preflight/sae-storage/<consumer-environment-sha256>.json`. Train, eval,
report, and post-capture `run` select the proof for their exact environment
and require both producer and consumer records. All commands reject a
different config, environment, topology, path layout, Orbax proof, or storage
attestation. The qualified `/dev/shm` files are part of that proof: if they are
lost, the current deployment stops instead of silently refilling from GCS.

The first-run order is therefore: stage and publish both C4 splits and convert
the model checkpoint off-cluster; qualify the three storage planes on the
v4-32; run `preflight`; then use `activation-farm` for the supervised
benchmark, canary, train capture, and held-out capture. It stops before SAE
training. After both captures seal, run fresh full-shape `sae-smoke` processes
in the exact order F, A, B; qualify the distinct resident SAE cache; seal those
proofs with `sae-begin`; then launch one detached `sae-farm` supervisor on each
of the four controllers. The farm alone advances the seed-1 gates, all nine
training runs, all nine evaluations, and the final report under one immutable
operation SHA. The converter's
self-hashed parity attestation, checkpoint inventory digest, storage
attestation, and C4 manifest fingerprints are bound through the resulting
benchmark, capture, campaign, checkpoint, and report metadata.

Read [`docs/EXPERIMENT.md`](docs/EXPERIMENT.md) for the scientific contract and
[`docs/RUNBOOK.md`](docs/RUNBOOK.md) for the fail-closed execution sequence.
