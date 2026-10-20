# Gemma mechanism Gate 1

This is a diagnostic audit of already-trained checkpoints. It is not a new
confirmatory experiment and cannot by itself establish that randomized-route
training is worse than true-route training.

The protocol is frozen by `configs/gemma4_gate1_audit.yaml`. The runner hashes
the canonical parsed protocol before reading checkpoint payloads or activation
shards. Every output binds that protocol hash, the exact checkpoint payload
URI and metadata, the sealed capture manifests, the route transformation, and
the evaluated token ranges.

## Questions

Gate 1 answers three bounded questions:

1. Does A trail B on both sampled training activations and held-out
   activations, or only on held-out data?
2. Does a trained A checkpoint reconstruct materially better with the true
   route IDs than with a deterministic per-sequence permutation?
3. Does exposing all 128 learned A banks recover quality hidden by the
   router's eight-bank visibility rule?

The retained 5M A/B/F checkpoints and the retained 40M A checkpoint provide
secondary learning-curve evidence. They do not replace a periodic held-out
curve because B/F 40M payloads were not retained.

## Frozen transformations

`true` uses the captured expert IDs unchanged.

`shuffled` constructs a deterministic uniform permutation of expert labels for
each `sequence_id`, keyed by the protocol's domain and key, and applies that
same permutation to every token in the sequence. It preserves eight unique
in-range IDs per token and the within-sequence stability of the historical
condition-C control. Its capacity is measured from the transformed fixed
order; overflow or token dropping is fatal.

`all_banks` restores A's learned 131,072-feature parameters but scores them
with global-F visibility. It is diagnostic only: banks encounter activations
outside their training routes and their scores were not calibrated for global
competition.

## Statistics and interpretation

The runner retains additive SSE, SST, and token counts by `sequence_id`.
Comparisons use identical tokens and paired sequence-clustered bootstrap
resamples. The primary report includes all per-seed estimates; no across-seed
claim may be reduced to a token-only interval.

The precommitted material-effect scale is 10% of the completed experiment's
F-over-B gain:

```text
0.10 * 0.033926403809130146 = 0.0033926403809130147 EV
```

- A below B on both train and held-out data argues against ordinary overfit.
- A above B on train but below B on held-out data supports an A-specific
  generalization failure.
- True-route A above shuffled-route A shows that the trained banks use router
  identity. It does not prove a training-time advantage over a randomized
  control.
- All-bank A above true-route A indicates complementary learned features that
  the router does not expose. Equality is consistent with duplicated banks or
  a dictionary whose useful directions are already available in eight banks.

Whatever Gate 1 shows, a claim about the original expert-decomposed
architecture requires Gate 2: fresh weighted per-expert contributions and
matched true/random local-decomposition training.

## Operational boundary

Gate 1 may read the sealed capture manifests, immutable GCS activation shards,
and the exact immutable checkpoint payloads named in the protocol. It writes
only to a new diagnostic output root. It must not modify the completed
operation, its report, its final checkpoint pointers, or any captured shard.

The normal production evaluator deliberately accepts only a final control
pointer. Gate 1 uses a separate fail-closed direct-payload restore path which
checks Orbax finalization and exact custom metadata before and after restore.
It does not relax the production path.

## Exact operator run

The canonical protocol SHA-256 is:

```text
142102c565fe024e476d537f92e845dd149294c168b42324c16b619733bf094d
```

Deploy a clean commit to one four-controller `v4-32` using Stage 3 of
`docs/RUNBOOK.md`. Reuse the existing `gemma4-control` Filestore; do not
recreate it and do not rerun the retired activation or SAE farms. The
bootstrap must mount that share at the production artifact root and report
`Linger=yes` on every controller. Gate 1 reads activation objects and Orbax
payloads from GCS, uses a bounded host-private refillable cache, and writes
only below `diagnostics/gate1` on the NFS control plane. It does not require a
new storage attestation.

On the Mac, retain the Stage-3 `tpu_all` helper and define:

```zsh
export GATE1_PROTOCOL="$TPU_PROJECT_ROOT/configs/gemma4_gate1_audit.yaml"
export GATE1_OUTPUT_ROOT="$ARTIFACT_ROOT/diagnostics/gate1"
export GATE1_CACHE_ROOT="/dev/shm/moe-sae-gate1-cache"
export GATE1_PROTOCOL_SHA256="142102c565fe024e476d537f92e845dd149294c168b42324c16b619733bf094d"

tpu_all "set -eu
cd '$TPU_PROJECT_ROOT'
test \"\$(git rev-parse HEAD)\" = '$PROJECT_COMMIT'
test \"\$(loginctl show-user \"\$(id -un)\" -p Linger --value)\" = yes
test -f '$GATE1_PROTOCOL'
test -f '$ARTIFACT_ROOT/captures/train/manifest.json'
test -f '$ARTIFACT_ROOT/captures/eval/manifest.json'
mkdir -p '$GATE1_CACHE_ROOT'
test \"\$(stat -f -c %T '$GATE1_CACHE_ROOT')\" = tmpfs
test \"\$(df --output=avail -B1 '$GATE1_CACHE_ROOT' | tail -1)\" -ge 25769803776
printf 'GATE1_READY %s %s\\n' \"\$(hostname -s)\" \"\$(git rev-parse HEAD)\""
```

Require four `GATE1_READY` lines. Start all four JAX processes from detached
user-systemd services at one common future epoch:

```zsh
export GATE1_ATTEMPT="$(date -u +%Y%m%dT%H%M%SZ)"
export GATE1_UNIT="gemma4-gate1-$GATE1_ATTEMPT"
export GATE1_START_EPOCH="$(($(date +%s) + 90))"

export GATE1_REMOTE_COMMAND="set -eu
while test \"\$(date +%s)\" -lt '$GATE1_START_EPOCH'; do sleep 1; done
cd '$TPU_PROJECT_ROOT'
exec '$REMOTE_UV' run --frozen --extra tpu moe-sae gate1-audit \
  --config '$TPU_PROJECT_ROOT/configs/gemma4_26b_a4b.yaml' \
  --artifact-root '$ARTIFACT_ROOT' \
  --protocol '$GATE1_PROTOCOL' \
  --cache-root '$GATE1_CACHE_ROOT' \
  --cache-max-bytes 25769803776 \
  --output-root '$GATE1_OUTPUT_ROOT' \
  --all"
export GATE1_REMOTE_QUOTED="$(printf '%q' "$GATE1_REMOTE_COMMAND")"

tpu_all "systemd-run --user \
  --unit='$GATE1_UNIT' \
  --property=RuntimeMaxSec=4h \
  /bin/bash -lc $GATE1_REMOTE_QUOTED"
```

The service launch returns immediately. Connectivity to the Mac is no longer
part of the numerical job. Inspect all controllers without modifying it:

```zsh
tpu_all "systemctl --user show '$GATE1_UNIT' \
  -p ActiveState -p SubState -p Result -p ExecMainStatus
journalctl --user-unit '$GATE1_UNIT' -n 80 --no-pager"

gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
  --project="$GCP_PROJECT" \
  --zone="$TPU_ZONE" \
  --worker=0 \
  --command="find '$GATE1_OUTPUT_ROOT/gemma4-mechanism-gate1-20260724/$GATE1_PROTOCOL_SHA256' \
    -mindepth 2 -maxdepth 2 -name cell.json -type f | wc -l"
```

There are 22 unique cells. The runner evaluates the decisive final-checkpoint
comparisons first and then the secondary learning endpoints. A safe retry uses
a new `GATE1_ATTEMPT` and repeats the exact four-host launch. Completed cells
are validated and reused; partial process fragments are never counted as
completed evidence.

After all four services show `Result=success` and the count is exactly 22,
build the report on worker 0 only:

```zsh
gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
  --project="$GCP_PROJECT" \
  --zone="$TPU_ZONE" \
  --worker=0 \
  --command="cd '$TPU_PROJECT_ROOT' && \
    '$REMOTE_UV' run --frozen --extra tpu moe-sae gate1-report \
      --config '$TPU_PROJECT_ROOT/configs/gemma4_26b_a4b.yaml' \
      --artifact-root '$ARTIFACT_ROOT' \
      --protocol '$GATE1_PROTOCOL' \
      --cell-root '$GATE1_OUTPUT_ROOT'"
```

The immutable result is
`$GATE1_OUTPUT_ROOT/gemma4-mechanism-gate1-20260724/$GATE1_PROTOCOL_SHA256/report.json`.
Copy it off the TPU before deleting the node; the NFS copy remains durable.
Provisioning and bootstrap are expected to dominate setup time. Once JAX is
ready, the 22-cell audit is budgeted for roughly 30–75 minutes and reporting
for less than five minutes. These are planning estimates, not acceptance
criteria.
