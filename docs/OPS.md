# Operations manual — running a pre-registered experiment like a production job

Companion to `docs/RUNBOOK.md` (which maps spec §20 to commands). This file
adapts the industry large-training-run playbook to a **frozen, pre-registered
spec** (`spec/frozen_spec_v0.1.1.md` + `spec/frozen_spec_v0.1.2_addendum.md`).
The organizing rule, from which every entry below follows:

> **Ops may observe, checkpoint, halt, resume, and parallelize. Ops may never
> change a training-step number, reorder a batch, or touch anything §9–§12
> froze.** Industry playbooks optimize for "the model ships"; this project
> optimizes for "the measurement is the one we registered." Any recovery
> action that would be invisible in a capabilities run (skipping a batch,
> nudging a schedule) is a protocol violation here, because the training
> trajectory *is* the instrument.

Cross-referenced ops modules (written in parallel; bound by contract, i.e. by
the guarantees stated here, not by their current source):

- `src/moe_sae/util/metrics.py` — mirroring sinks (TensorBoard/CSV) fed from
  the same records that produce `metrics.jsonl`. Contract: `metrics.jsonl`
  stays the canonical artifact with its frozen line format
  (`docs/INTERFACES.md`, run-artifact layout); sinks are derived, write-only,
  and never read back by training or analysis.
- `scripts/preflight.py` — the preflight battery (§ "Preflight" below).
  Contract: read-only checks; exits nonzero on any failure; never mutates a
  run directory.
- `src/moe_sae/util/manifest.py` — run manifests + LOGBOOK append. Contract:
  one manifest per run attempt, written before step 0; LOGBOOK entries are
  append-only.
- `scripts/tpu/spot_controller.py` — preemption-safe fleet controller.
  Contract: launches/relaunches work from a queue; every relaunch resumes
  from the last durable checkpoint and must satisfy the bitwise-resume rule.

---

## 1. Adopt / adapt / reject / defer

### Adopt (pure observers or pure resilience — cannot touch the trajectory)

| Practice | One-line reason |
|---|---|
| Dashboards / TensorBoard sinks alongside canonical `metrics.jsonl` (`src/moe_sae/util/metrics.py`) | Humans read dashboards; the registration reads `metrics.jsonl` — mirrors are derived and write-only, so zero risk to the frozen line format. |
| MFU + goodput meters | Timing/throughput measurement reads clocks, not tensors; the RUNBOOK's ≥50% MFU target on A-shape steps needs a meter to be a target. |
| Preflight battery (`scripts/preflight.py`) | Every discovery failure so far (ENOSPC, missing weights, nested clones) was detectable before spending; read-only checks are free insurance. |
| Run manifests + logbook (`src/moe_sae/util/manifest.py`, `LOGBOOK.md`) | A7 requires backend substitutions "recorded with trigger"; a manifest per attempt + append-only logbook is the recording mechanism. |
| Resume-capable loops | Halting is only prereg-safe because resuming is bitwise (§4); resumability converts every anomaly response into "checkpoint + halt" instead of "improvise." |
| Preemption-safe fleet controller (`scripts/tpu/spot_controller.py`) | Spot capacity is ~3× cheaper and preemption is indistinguishable from a deliberate halt once resume is bitwise. |
| Determinism drills | The whole ops posture (halt-anytime, spot fleets, cross-backend lanes) rests on bitwise resume and the A7 parity suite; a guarantee that is never drilled is a hope. |

### Reject — with the pre-registration reasoning

| Industry practice | Why it is forbidden here |
|---|---|
| Skip-bad-batch (drop the batch that spiked, continue) | Breaks spec §10 "same training-data order across matched conditions": all 14 runs consume the identical `data_seed = 1000 + seed` batch sequence; skipping in one run desynchronizes every A/B/C/F comparison and silently edits the frozen data stream. Candidate dropping is *never permitted* (A7). |
| Rewind + reshuffle (restore an earlier checkpoint, reshuffle data, retrain past the spike) | Same violation by another route: §9's shuffling is fixed by the composer RNG; a reshuffle is an unregistered data order, and the "retrained-past" segment is no longer the registered trajectory. |
| Mid-run data-mixture changes (curriculum tweaks, domain re-weighting) | §9 freezes composition at exactly 2048 visual + 2048 text per batch, four domains sampled evenly; RUNBOOK "Known scale realities" explicitly forbids relaxing it even when text positions repeat. |
| WSD branch-and-anneal (branch a checkpoint, anneal LR to peek at quality) | §11 freezes the schedule (3e-4, 1k-step linear warmup, cosine to 3e-5) and A9 closed the one permitted LR option as *waived*; an annealed branch is an unregistered second experiment whose results would contaminate frozen choices. |

The pattern: each rejected practice trades trajectory integrity for
convergence robustness. Industry makes that trade because only the endpoint
matters. Here the comparisons *between* trajectories are the result, so the
trade is not available at any price.

### Defer to v0.2+ — with trigger conditions

| Practice | Trigger condition |
|---|---|
| muP transfer | Only if §20.11 authorizes a width/model-scale sweep in a v0.2 spec bump. muP changes parameterization and effective LRs — training math — so it is unadoptable under any frozen spec; it becomes relevant the day a new spec version *wants* to transfer hyperparameters across widths. |
| Scaling-law ladders | Only after the confirmatory run completes and a v0.2 bump pre-registers a budget sweep. §10 fixes exactly three budgets (1M / 50M / 200M); intermediate rungs today would be unregistered experiments. |
| Orbax / Grain adoption | Only when the fleet stage (§5) makes sync safetensors checkpointing or the single-host torch `BatchComposer` feed the measured bottleneck — and only after demonstrating byte-identical checkpoint round-trip into the registered torch eval stack and bit-identical batch sequences vs the composer (the A7 parity bar). |
| Pallas kernels | Only if the MFU meter shows A-shape steps <50% MFU attributable to grouped-GEMM capacity padding (RUNBOOK: Cap=768 vs mean load 384, ~2× on A's encode), and only with selection/loss semantics exact per A7 (fp32 comparison math, no candidate dropped) plus parity-suite extension before first use. |

---

## 2. Spike / anomaly playbook — pre-registration edition

**Trip conditions** (observer-side, evaluated from the same records that feed
`metrics.jsonl`; monitoring thresholds are ops choices because they change
nothing about training):

- any NaN/inf in loss, θ_b, or parameters;
- pre-clip global grad norm > 100× its trailing-1000-step median (the frozen
  clip `min(1, 1/(‖g‖+1e-6))` (A5) bounds the *step*, so the pre-clip norm is
  the honest anomaly signal);
- θ_b EMA collapsing to ~0 or diverging while loss is finite;
- dead-latent count jumping by more than a full bank between windows.

**Response — always the same four steps, in order:**

1. **Checkpoint** at the current step boundary (never mid-step).
2. **HALT** the run. Do not let it continue "to see if it recovers."
3. **LOGBOOK entry** via `src/moe_sae/util/manifest.py`: run name, step,
   trip condition, checkpoint path, manifest id.
4. **Human audit** before any resume decision. Never silent continuation;
   never batch skipping (§1, Reject table).

**Why this differs from the industry default.** The industry response
(skip/rewind/reshuffle) exists because a spike in a capabilities run is a
nuisance obstructing the product. Here a spike is *data*: either (a) the
frozen math produced it, in which case the run continues through it unaltered
and the event is reported, or (b) an implementation/hardware fault produced
it, in which case the study halts for an audit under the same discipline as
the §20.5 gate. HALT is prereg-safe precisely because resume is bitwise
(§4) — a halted-and-resumed run is the *same experiment*; a skipped batch is
a different one.

**Evidence to collect before resuming** (the audit checklist):

- **Replay test (decisive):** re-run the offending step(s) from the prior
  checkpoint on the same backend. Determinism means a real spike replays
  bitwise. Reproduces → the frozen math did it (see below). Doesn't
  reproduce → infra fault (bad DIMM, kernel nondeterminism, corrupted shard
  read); fix infra, verify with the §4 drill, resume from the checkpoint.
- Offending batch identity: step index → batch is fully determined by
  `data_seed = 1000 + seed`; record the `example_id`s and shard(s).
- Shard integrity: re-hash the capture shards the batch read against capture
  `meta.json`; a corrupt shard is a capture problem and pauses **all 14
  runs** (shared stream), not just the tripped one.
- Context series: θ_b trajectory (`theta_trajectory.npy`), per-bank loads,
  dead counts, pre-clip grad norms for ±500 steps.
- Cross-condition check: did matched conditions trip on the same batch? Same
  batch across conditions → data-side cause; one condition only →
  condition-code suspicion → implementation audit before anything resumes.

**Resolution rules.** Genuine spike under frozen math: resume from the halt
checkpoint, run through it, disclose in the logbook (it will appear in
`metrics.jsonl` regardless — the record is the record). Implementation bug:
spec §0 discipline — written reason, version bump if any frozen semantics
were touched, and invalidation of affected runs; A must beat C with the bug
fixed, not around it.

---

## 3. Monitoring without an on-call rotation

One researcher, no pager. The repo's pattern is **detached runners +
artifact watchers + a morning-after checklist** — designed so nothing needs a
human awake, because nothing a watcher could do automatically is permitted
anyway (§2: the only automatic actions are checkpoint + halt + log).

**Detached runners** (RUNBOOK "Resilient local execution"): `nohup`'d
wrappers (`scripts/run_pipeline_capture.sh`, `scripts/run_pipeline_night.sh`)
that restart workers with `--resume`, sweep orphans, and abort loudly above a
5% per-example failure rate. On TPU, `scripts/tpu/spot_controller.py` plays
the same role fleet-wide: relaunch preempted slices from the queue, resume
from the last durable checkpoint, append every preemption to the LOGBOOK.

**Artifact watchers** poll for completion evidence rather than process
liveness: capture `progress.json` advancing; `runs/<run>/metrics.jsonl`
growing; `final/theta_star.json` existing (the "training finished" witness);
gate reports written. A process that is alive but not producing artifacts is
treated as dead. Watchers notify + log; they never remediate beyond a
restart-with-resume.

**Morning-after checklist** (in order, ~5 minutes per run):

1. `LOGBOOK.md` — any watcher/controller entries overnight? Preemptions are
   normal; §2 trips are not — those gate everything below.
2. `runs/<run>/metrics.jsonl` tail — last step vs expected (12,207 at
   discovery scale); loss finite and trending; `lr` on the frozen §11
   schedule (a wrong lr at a given step means a resume bug — halt).
3. `theta_trajectory.npy` — θ_b EMA smooth, no collapse/divergence; for D/E
   remember θ_b is log-only (A4) but still a health signal.
4. Dead counts — substantial dead rates are *expected by design* at
   discovery scale (§10 power note: ~12k fires/feature); the anomaly is a
   discontinuity, not a level. Check no bank is 100% dead (starved-bank
   signal for the §16.3 load floor).
5. Per-bank loads — burstiness is known and frozen policy (A7: grow-and-
   retry capacity); check capacity-grow events were logged, not silent.
6. Disk/RAM headroom on the box before the day's next stage (preflight
   re-run is cheaper than a 3 a.m. ENOSPC — the Phase-2 sizing bug in the
   git history is the cautionary tale).

---

## 4. Determinism policy

Determinism is not hygiene here; it is what makes HALT, spot preemption, and
cross-backend lanes compatible with pre-registration at all.

**Numerics ground rules** (already frozen — `docs/INTERFACES.md` Numerics,
A5, A7): BatchTopK scores compared in FP32 on every backend; loss
reductions/norms/EMA in FP32; selection semantics exact across backends with
no candidate dropped; the JAX lane is held to torch by the 21-test parity
suite.

**XLA / backend flags.** Every run manifest (`src/moe_sae/util/manifest.py`)
records the exact backend fingerprint: framework versions, `XLA_FLAGS`,
matmul-precision setting, device topology, and driver/runtime versions. On
XLA backends, deterministic-ops flags are ON and matmul precision is pinned
(highest available for the fp32 comparison paths); autotuning modes that can
pick different kernels per launch are disabled for training runs. The policy
is: *whatever flag set passes the drill below is frozen in the manifest and
reused verbatim for every run in the suite* — flags may not vary across the
14 runs of one experiment.

**Seed registry** (all frozen; contract in `docs/INTERFACES.md` and A2):

| Seed | Value | Consumer |
|---|---|---|
| Training seeds | 1, 2, 3 (A/B/C/F); 1 (D/E) | SAE init only |
| `data_seed` | **1000 + training seed** (convention, `RunConfig.data_seed`) | `BatchComposer` — matched conditions at the same seed see bit-identical batch sequences |
| Capture stream seed | 1 | pipeline + discovery capture order (source of stable `example_id`s) |
| Validation-input export seed | 7 | §20.1 gate inputs |
| Condition-C salt | `0xC0FFEEC`, PCG64(SALT ^ example_id) | per-example permutation — deliberately framework- and seed-independent (A2) |

No other RNG may exist in the training path. New ops code draws no random
numbers, or draws them from an explicitly registered, logged seed.

**The resume-must-be-bitwise rule.** A run killed at any step and resumed
from its last checkpoint must produce (a) `metrics.jsonl` lines from the
resume point onward byte-identical to an uninterrupted run, and (b) a final
checkpoint with identical tensor contents. This requires checkpoints to carry
*everything*: SAE weights, FP32 Adam moments and step count, θ EMA state,
DeadTracker clocks/stamps, composer stream position, and μ_e. Anything
excluded from the checkpoint is a resume bug by definition.

**The drill** (mandatory before any paid run on a new backend, topology, or
flag set; it is a smoke-ladder rung, run at pipeline dims):

1. Run a reference training job N steps uninterrupted; keep `metrics.jsonl`
   and final checkpoint hashes.
2. Re-run; SIGKILL at an arbitrary step m (not a checkpoint boundary you
   chose — let the killer pick); resume; run to N.
3. Diff: `metrics.jsonl` byte-equal from step m'+1 (first post-resume step)
   onward; final checkpoint tensor-equal. Any mismatch fails the drill;
   the backend/flag set may not carry real runs until it passes.
4. Log the drill outcome (pass hash, flag set, backend fingerprint) in the
   LOGBOOK; the spot controller refuses backends without a passing drill on
   record.

---

## 5. Scale-up path

### Today: TPU v4-32 (numbers from `docs/RUNBOOK.md`)

- Whole 14-run discovery suite ≈ **30 min** of slice time; SAE compute is
  tiny (~19 PFLOP·s per A-run; F is 10.7× A and the wall-clock straggler by
  construction).
- Binding resource is **activation streaming**: ~90 GB/s for one run at 50%
  MFU; solved by (1) one-stream-many-consumers — §10's matched data order
  means all 14 runs share one fetch (~6 GB/s per stream), and (2) a
  host-RAM cache — 770 GB of shards < ~1.6 TB across the 4 TPU-VM hosts, so
  after one free in-region GCS pass every read is memory-speed.
- Capture stays PyTorch on a rented GPU (§3-gate-validated path); ~8 h ≈
  $10–25 spot on A100; bucket must be in `us-central2` (in-region reads
  free; cross-region on 770 GB is the one large avoidable cost).
- MFU target ≥50% on A-shape steps; single controller
  (`scripts/tpu/03_discovery_suite.sh` lineage) drives the suite.

### Next: spot v6e-64 fleet (~13× v4-32 per slice)

Per-slice bf16 throughput ≈13× today's slice, on preemptible capacity, with
multiple slices for the confirmatory run's 3-seed × all-conditions matrix
(200M tokens = 4× the stream, 48,828 steps/run). What breaks, in the order
it breaks:

1. **Single-host RAM cache → sharded host caches.** The confirmatory capture
   is ~3.1 TB of shards (4× discovery) — no longer under one slice's
   aggregate host RAM with headroom, and spot slices lose their cache on
   every preemption. Fix: shard the cache across hosts keyed by batch range
   (the batch sequence is known ahead of time from `data_seed` — prefetch is
   trivially precomputable), backed by free in-region GCS re-reads on
   preemption.
2. **Sync checkpoints → async multi-tier.** At v4-32, checkpointing 14 small
   runs synchronously is noise; a preemptible fleet needs frequent durable
   checkpoints, and synchronous GCS writes of FP32 moments + tracker state
   stall the step loop. Fix: async snapshot to host RAM, tiered flush
   local-SSD → GCS. The bitwise-resume drill (§4) re-runs against the async
   path before it carries a real run — an async checkpoint that races the
   step loop is exactly the class of bug the drill exists to catch. (This is
   the Orbax trigger from §1's defer table.)
3. **Single controller → queue-coordinated fleet.** One shell script driving
   one slice cannot survive preemption of itself or coordinate N slices.
   `scripts/tpu/spot_controller.py`'s contract scales here: a durable work
   queue of (run, resume-step) items; idempotent claim/heartbeat/complete;
   relaunched slices claim the next item and resume from the last durable
   checkpoint; every preemption/relaunch appended to the LOGBOOK. The §10
   matched-data-order requirement is unaffected — each run's stream is a
   pure function of `data_seed`, so runs may float freely across slices.
4. **Capture-side reality — host models beyond ~10B.** deepseek-vl2-tiny
   (~3.4B) captures on one A100. A future host model past ~10B no longer
   fits a single-GPU PyTorch forward at capture batch sizes; the options are
   (a) sharded PyTorch capture (tensor/pipeline parallel forward feeding the
   same `ShardWriter` format) or (b) a JAX capture path on the TPUs
   themselves. Either way, A7 is explicit: **every capture backend must pass
   the §3 FP32 gate (<1e-5 per token, 100k tokens) plus the BF16 report
   before its shards are used** — a re-run of the §20.1 hard gate, not a
   waiver. Training-side parity does not transfer to capture; the gate is
   per-backend.

Everything in this section is ops architecture. The spec quantities riding
on top — batch composition, data order, budgets, schedules, metrics — do not
change at any scale (§10: confirmatory = "identical architecture; no metric
or hypothesis changes").

---

## Decisions (this document)

- **Monitoring thresholds are ops-only and therefore choosable** (§2): they
  trigger checkpoint+halt, which is trajectory-preserving under the bitwise-
  resume rule; nothing frozen depends on them. Chosen values (100× trailing
  median pre-clip grad norm, etc.) are starting points and may be tuned in
  the LOGBOOK without spec implications.
- **HALT is the only automated anomaly response** (§2): permitted because
  resume is bitwise (§4); every trajectory-altering response is rejected
  with its spec citation (§1).
- **Mirroring sinks never feed back** (§1, `src/moe_sae/util/metrics.py`
  contract): `metrics.jsonl` remains the single canonical record; dashboards
  are projections.
- **Flag sets are frozen per experiment** (§4): determinism flags may differ
  between backends but not between runs of one suite; the manifest is the
  record.
- **Deferred-practice triggers are written down now** (§1) so that adopting
  muP/ladders/Orbax/Pallas later is a pre-declared engineering event, not a
  quiet degree of freedom.
