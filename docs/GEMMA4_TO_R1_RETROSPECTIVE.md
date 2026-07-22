# From Gemma 4 to DeepSeek-R1: evidence, debt, and the next system

**Snapshot:** 2026-07-22, after the Gemma 4 train and held-out activation
captures were sealed, but before the final nine SAE runs and exploratory
efficiency result. This is a candid engineering retrospective and a design
input for the R1 replication. It is not a claim that the Gemma experiment is
scientifically complete. Because the hard harvester-MFU gate and the real-data
canary gates were waived, this campaign can compute the planned G/C statistic
but cannot become preregistered or headline-ready retroactively, regardless of
how favorable that statistic is.

The words below have deliberately different meanings:

- **Observed** means a production record, immutable object, or direct runtime
  inspection exists.
- **Committed** means the behavior is represented in Git at or before
  `45a15f028751932f00b521e35fd894e68a38b165`.
- **In progress** means a repair exists only in the current worktree or
  operator artifacts and must not be treated as a released capability.
- **Inference** means the explanation is consistent with the evidence but has
  not itself been isolated experimentally.
- **Requirement** means the R1 system must satisfy it before scientific data
  count.

This distinction matters. The project repeatedly benefited from refusing to
turn a successful recovery into a retroactive claim that the original gate had
passed.

## Executive conclusion

The Gemma reset got the most important scientific abstraction right: one
aggregate routed-MoE target, one A/B/F comparison, exact per-token TopK, and a
measured cost denominator. It also got the durable bulk-data abstraction right:
immutable content-addressed GCS objects, exact input ranges, receipts, and a
manifest published last. Those choices are why a 50M-token capture survived
multiple code fixes, a failed canary, an NFS visibility race, and disconnected
operator sessions without being recaptured. This durability statement applies
to shard payloads. The authoritative manifests, waivers, host ledgers, and
other control evidence still live on zonal Filestore and are not yet an
independently durable experiment bundle.

The reset did **not** become the compact system originally imagined. From the
DeepSeek archive tag to the capture commit, the tree changed by roughly 43,000
insertions and 31,000 deletions. The committed CLI alone reached nearly 5,000
lines, and activation operations added several large state machines. Much of
that code was necessary hardening, but much of it was also a response to three
avoidable couplings: runtime installation on live TPU VMs, mutable POSIX/NFS
control records, and environment identity that treated every code change as a
scientific incompatibility.

For DeepSeek-R1, the unit of execution must be **one self-contained zonal
`v6e-64` slice**, not one host and not a cross-region JAX world. Multiple
regional slices can farm disjoint capture ranges or run independent
condition/seed jobs, coordinated by an always-on control service. They must
share immutable logical identities through object storage, not a shared NFS
mount. Spot loss is a normal scheduler event, not an exceptional code path.

## What happened

### 1. The DeepSeek-VL2/spec era

The repository originally tested a richer visual-cortex hypothesis on
DeepSeek-VL2-Tiny: 64 routed experts, top-6 routing, one SAE bank per expert,
gated per-expert contributions at one-indexed block 6, multimodal balancing,
BatchTopK, conditions A through F, probe analyses, and causal interventions.
The active tree from that period is preserved at tag
`deepseek-vl2-final-a291707`.

The scientific governance became elaborate but internally coherent. Spec
v0.1.2 added keyed splits, per-bank inference thresholds, auxiliary-loss
diagnostics, load floors, audit-first gates, and a mechanical confirmatory
trigger. That machinery answered the multimodal interpretation question; it
was not necessary for the narrower reconstruction-versus-cost question.

The operational lessons arrived before the science. Colab capture accumulated
local disk pressure, upload races, partial shards, orphan processes, and many
restart surfaces. Moving durable bulk state to GCS and execution to TPU was the
right conclusion. The completed 1M A/B/C runs were plumbing evidence only:
244 optimizer steps, all inside a 1,000-step warmup; no F condition; inference
sparsity different from training sparsity; and no usable TPU MFU. They cannot
support an efficiency headline.

### 2. The Gemma reset

The first proposed reset named Gemma 3 4B, but that would not test router
conditioning because the model is not the required routed MoE. Catching that
architecture contradiction before launch was important. The experiment moved
to the text-only base/PT `google/gemma-4-26B-A4B`: 30 blocks, width 2,816,
128 routed experts, top-8 routing, and one shared expert.

Commit `85ed074` replaced the governed multimodal surface with A/B/F and the
planned G/C metric. Commit `1e84111` introduced split storage. Commits from
`0600f5d` through `45a15f0` then hardened staging, conversion, supervision,
capture sharding, transfers, topology handling, timing, profiling, and the
auditable MFU exception. The intended experiment is documented in
[`EXPERIMENT.md`](EXPERIMENT.md), while [`ARCHIVE.md`](ARCHIVE.md) records the
historical boundary.

The final Gemma scientific contract is materially simpler:

- Capture the aggregate routed branch after block 15's
  `post_feedforward_layernorm_2`, before the shared expert and residual.
- A is 128 banks of width 1,024 and sees only the router's eight banks; B is a
  global 8,192-feature candidate-matched dictionary; F is a global
  131,072-feature parameter-matched dictionary.
- Every condition reconstructs the same globally centered activation using
  exact, strictly positive per-token TopK-64.
- The primary quality statistic is F-over-B gain recovered by A; the primary
  cost statistic is median paired A/F marginal SAE chip-seconds, including A's
  required accepted capacity prepass.

The scientific interpretation is narrower than “the router discovered causal
expert concepts.” A receives router IDs that also participated in producing
the routed-aggregate target, so the experiment measures the reconstruction
efficiency of conditional dictionary access. It does not by itself establish
causal expert specialization. The shared expert is excluded, and the result is
specific to one layer, corpus sample, token budget, and model.

Capturing the aggregate rather than eight slot vectors reduced the nominal
record to 5,666 bytes per scientific token and the 50M capture from roughly
2.25 TB to about 283 GB decimal. It was also the only mathematically sound
common target because Gemma applies a nonlinear branch RMSNorm after combining
routed experts.

### 3. Data and conversion

The implementation does not contain or stream all of “C4 English 3.1.0.” It
uses an exact bounded prefix from one pinned `allenai/c4` train object and one
validation object, at revision
`1588ec454efa1a09f29cd18ddd04fe05fc8653a2`. That is a reproducible corpus
sample, not a corpus-wide scaling-law dataset.

Observed staging results were:

| split | packed records | scientific tokens available | production target |
|---|---:|---:|---:|
| train | 60,928 | 50,035,528 | 50,000,000 |
| validation/eval | 2,496 | 2,016,660 | 2,000,000 |

The tokenizer and source bytes were independently pinned, packing was
deterministic, and BOS/EOS/padding did not enter scientific counts. This was a
substantial improvement over a live streaming dependency in the MFU-critical
path.

Checkpoint conversion ran on a custom N2 VM with 32 vCPUs and 512 GiB RAM in
`us-south1-a` after `n2-highmem-64` capacity was unavailable and the
project-wide CPU quota was initially 32. The bucket was in `us-central2`, so
this cross-region placement was a capacity-driven expedient, not a layout to
repeat for a 0.7--1.3 TB R1 checkpoint.

The 51.6 GB converted checkpoint completed, but the final CPU parity process
then selected the TPU-only MegaBlox/Pallas path and crashed. Commit `965b109`
added a CPU-safe validation path, reused the existing converted checkpoint,
and did **not** rerun conversion. The resumed comparison passed with maximum
single-token KL divergence `1.81e-05`, far below the `0.03` bound. The
attestation therefore records `converter_rerun=false` and no trustworthy
conversion duration. This is strong evidence that checkpoint bytes and the
full-model implementation agreed; it is not evidence of a clean atomic
convert-and-validate workflow, nor did it by itself prove the later prefix
adapter.

An early local notebook output contained a Hugging Face token-shaped secret.
The current workspace scan is clean, but repository evidence cannot prove that
the external credential was revoked or rotated. Source checkout also initially
used GitHub's removed password-authentication flow. The R1 workers must receive
an image or signed source bundle and attached machine identity, never a human
token.

### 4. TPU bring-up and preflight

The first queued v4 request failed with an internal code-13 error and was
recreated as the `-r2` attempt. The resulting v4-32 came up as four TPU VMs and
used the existing default Compute service account under an explicitly
expedient, broadly privileged operator policy. The live image performed
automatic package upgrades and required a coordinated reboot. Observed setup
times were 30m39.630s and 30m58.168s on workers 1 and 2, 30m44.584s on worker
3, and 41m41.829s on worker 0 because its initramfs tail ran longer. The timers
were then disabled and all four workers rebooted together. The operator also
had to learn GCP quota, VM creation, GitHub authentication, SSH,
service-account identity, and teardown while trying to start a production
experiment.

The first production preflight took roughly 35–37 minutes because every one of
the four processes independently decoded and recounted all 60,928 packed train
records. That was fail-closed, but it scaled work with `hosts × corpus` instead
of validating the corpus once and checking a short immutable proof on every
consumer. An audited reuse path was added in `5b0353b`; on the final retry the
operator explicitly chose to bypass a redundant full rerun while retaining
the previously verified data identities.

Target-runtime qualification exposed several assumptions that local tests had
not:

- MaxText layer metadata did not provide a reliable increasing `layer_idx`;
  the adapter had to validate structural order instead (`f416f6e`).
- A process can own several interleaved global device slices. Global JAX
  device order, process index, and TPU worker suffix are not interchangeable
  identities (`e127925`).
- Equal local row counts do not establish that two meshes selected the same
  input rows (`9d5dbe6`).
- Benchmark capture needed coalescing, valid-prefix sorting, and a prebound
  inference-pure graph to stop storage layout, padding, and tracing behavior
  from contaminating the measurement (`449c6b8`, `6fc0824`, `b40ce12`).
- Device time and durable wall time had to be separated before MFU and
  persisted goodput could be interpreted (`9809fb6`).

These were real correctness fixes, not cosmetic tuning.

#### Qualification attempt ledger

The commit list alone hides the cost and sequence of failed attempts. The
operator record shows this progression:

| attempt or fix | observed result | consequence |
|---|---|---|
| first production preflight | 37m23s | valid but redundantly scanned the full corpus on every worker |
| first benchmark supervisor | 24m29s; all nine candidates failed before compile because `layer_idx` was absent | structural layer-order validation replaced the metadata assumption; cleanup was also fixed so it could not mask the primary exception |
| first real mesh ownership checks | interleaved addressable slices violated the assumed rectangle; later TP2 had equal local counts but different rows | ownership became a proof over exact global row IDs, not counts or process suffixes |
| first benchmark writer | roughly one 23 MiB object per host per step | output was coalesced into bounded production-style shards |
| earlier complete EP/FSDP/TP sweep | maximum 2.445% MFU at EP8/FSDP2/TP1, batch 2, but timing still included host processing | useful for diagnosis, not admissible as the final exact-environment selection |
| device-only EP16/FSDP1/TP1 baseline | 154.4 of 166.1 timed seconds in prefix compute, 1.60% MFU; persistence was 12.8s, about 8% | established that writing was not the dominant MFU loss |
| kernel/graph corrections | valid-prefix `use_ragged_sort` avoided processing a 16x padded dropless-MoE buffer; typed PRNG profiler metadata was repaired; `nnx.jit_partial(graph=True, graph_updates=False)` stopped threading unchanged suffix state | produced a genuinely prefix-only, inference-pure measured path |
| optimized 10-step diagnostic | 2.319% MFU and 30.75k device-output tokens/s | improvement, but too short for selection |
| final exact-environment 200-step qualification | EP16/FSDP1/TP1 batch 1 only: 2.124329898% MFU, 28.16k device-output and 25,628.09 persisted tokens/s | became the sole observation in the final `selected.json`; the waiver explicitly disallowed cross-environment evidence |

The final qualification was therefore an expedited one-candidate rerun after
the broader sweep, not a second nine-candidate sweep. Both facts belong in the
record: the broad campaign explored nine configurations, while the exact
capture environment qualified and waived one.

#### Storage qualification ledger

Storage performance also improved through several concrete fixes rather than
one clean pass:

| observation | measured aggregate throughput | interpretation or fix |
|---|---:|---|
| initial qualifier | 1.22 GB/s upload; 0.503 GB/s download | failed the 4 GB/s target; in-process threaded I/O was not a trustworthy JAX-cold transfer test |
| isolated cold-transfer helper | 2.724 GB/s upload | exposed a redundant second hash pass over the 8 GiB payload |
| producer SHA-256/CRC32C reuse | 4.558748 GB/s upload | first clean pass without rereading the payload solely to hash it |
| later requalifications | 2.54 GB/s fail, 4.94 GB/s pass, then above 6 GB/s | demonstrated material transient GCS variance; one pass is not a capacity guarantee |
| sealed-shard read probe | 0.823--1.298 GB/s with one object; 5.148927 GB/s with four-object concurrency | justified a separate concurrent SAE-cache read gate |

Activation capture was write-heavy, so download became diagnostic for that
stage rather than a hard launch blocker. That decision was valid only because
Stage 8 later introduced a distinct hard consumer-cache qualifier. A bucket
lifecycle rule was also added to delete abandoned multipart uploads under the
experiment prefix after one day.

### 5. Benchmark, canary, and capture

No benchmark evidence met the planned 50% useful-prefix MFU gate. The earlier
nine-candidate sweep peaked at 2.445%, while the only observation under the
final exact capture code and environment was 2.124329898%. The bounded profiler
and timing split indicated that prefix model execution, not GCS writing,
dominated the device path. The exact causal decomposition was not completed,
so “the native MaxText prefix path was a poor fit for this small static
inference workload” is an inference, not a proven kernel-level diagnosis.

Commit `45a15f0` added a narrow, self-hashed exception that could waive only
the MFU minimum. It could not waive OOM, recompilation, token drops, output
parity, writer backlog, or invalid timing. The chosen synthetic run delivered
about 25,628 persisted scientific tokens/s and passed those integrity checks.
The exception did not make the 50% gate pass and permanently removes this
campaign from the original headline-ready category.

The first 2M real-data canary then failed three operational comparisons:

| canary observation | value |
|---|---:|
| real/persisted goodput | 16,227.24 tokens/s |
| device-output goodput | 23,758.43 tokens/s |
| synthetic goodput | 25,628.09 tokens/s |
| input-wait fraction | 19.18% |
| maximum writer backlog | 1 batch |
| recompilations / dropped routes | 0 / 0 |

The comparison was partly ill-posed: the synthetic path and real packed data
did not have identical valid-token density. The operator authorized completion
because the measured canary projected the remaining suffix under one hour and
all correctness/durability checks were clean. A separate canary waiver
preserved `canary_passed=false`, bound the exact evidence, and explicitly
recorded that the operator email was used only as a local audit identity; no
email was sent.

Caching the packed C4 inputs locally reduced input wait from about 19.2% to
about 0.7% in the production suffix. The sealed production records report
roughly 20.3k persisted train tokens/s, with aggregate input wait below 2%.
This improvement confirmed a host-input bottleneck in the original canary, but
it did not repair the very low prefix MFU.

The durable result was:

- exactly 50,000,000 train tokens in 384 immutable GCS shards;
- train manifest SHA-256
  `ed2ba84582a489f83209c2ce0756139f93f30e3370e54960fcfc56c82c331357`;
- exactly 2,000,000 held-out tokens in 16 immutable GCS shards;
- eval manifest SHA-256
  `d407cfc541cb249cf71156c91ec82545a480d12edde10abbf18be30d96d8302b`.

All 400 object identities were subsequently checked against GCS generation,
size, CRC32C, and SHA-256. Ranges were contiguous with no gaps or duplicates;
merged FP64 statistics were finite. The held-out SST was computed against the
training mean, as the report contract requires.

### 6. The seal/recovery incident

The final seal was not the first warning. Earlier clients exceeded the
15-second publication retry while waiting for newly created benchmark waiver
and start-release records. Those incidents were worked around, but they showed
that absent-to-present visibility on the shared mount was not bounded by the
protocol. After train capture finished, worker 0 successfully published the
final POSIX manifest and three NFS clients again retained a negative directory
entry and continued to see `ENOENT`. The collective split after the JAX
barrier, so the summary and held-out phase did not run even though the 50M data
plane was already clean. Branching on `exists()` before a collective barrier
made a cache artifact into a divergent distributed schedule.

The safe recovery required a **full unmount and mount** with
`lookupcache=positive` on every host. A remount was not sufficient because it
could preserve the stale client cache. A second issue then appeared: the
original canary waiver had bound mutable “latest host ledger” files, which
production capture necessarily replaced. An operator-only recovery wrapper
therefore bound the new exact ledger hashes, the sealed manifest, all original
immutable evidence, and the narrow authorization to recover the summary and
run eval. It did not recapture train and it retained the failed canary truth.

This was a competent emergency recovery. It was also evidence that both the
NFS publication protocol and the waiver binding model were wrong for a system
that must scale across regions. It was a repeated failure class, not an
isolated final-seal anomaly.

### 7. The Stage-8 boundary

The first storage design had required 128 GiB of scratch even though the TPU
boot disks were fixed at 100 GiB. Before activation launch, that overbroad gate
was correctly narrowed to a truthful 24 GiB rolling-buffer requirement scoped
to `activation-capture`. Training then discovered the other side of the scope
error: its separate 100 GiB consumer cache had never been qualified. The hosts
did have roughly 200 GiB of `/dev/shm`, and each train partition was only about
66 GiB, but the activation attestation could not truthfully authorize this new
use. The progression was therefore 128 GiB overqualification, 24 GiB
stage-correct qualification, then a missed downstream 100 GiB requirement.

The first dedicated SAE-cache qualification then passed its own contract:
73.66 GB per controller (294.64 GB aggregate) of `/dev/shm` data were downloaded
and verified at 4.946 GB/s aggregate. About
20 seconds after the foreground qualifier's SSH sessions exited, however, the
cache vanished. Runtime inspection found `systemd-logind` configured with
`RemoveIPC=yes` and the production user with `Linger=no`. A controlled
four-worker comparison enabled linger only on worker 0: its owned `/dev/shm`
sentinel survived the disconnect interval, while the otherwise matching
sentinels on workers 1--3 vanished. Enabling `Linger=yes` on all four workers
therefore became a prerequisite to requalification. The passed throughput and
hash proof had been valid; the target runtime invalidated its resident bytes
after the process boundary.

The Stage-8 hardening implementation preserves the immutable activation
attestation, adds a separate SAE-storage attestation for bounded `/dev/shm` and
Orbax paths, and adds a producer/consumer compatibility proof. That
compatibility proof is a transitional allowlist of selected full-file and AST
symbol hashes; despite its broad module description, it principally binds the
capture producer/schema semantics and is not a substitute for golden numerical
compatibility tests. It was committed and its full-shape F/A/B smoke suite
passed under consumer commit `67bb7f7`; the later cache-lifecycle repair
correctly requires a new identity-bound smoke suite and storage qualification.

The metrics journal repair is more concrete: rows are bound to one
`(run_instance_id, condition, seed)`, require strictly increasing optimizer
step and scientific-token counts, are flushed and `fsync`ed, and on resume may
atomically remove only the suffix ahead of the exact checkpoint boundary. The
checkpoint remains authoritative. This removes the earlier blind-append
rollback risk. Full-shape A/B/F smoke and exact Orbax restore passed once on
the production TPU; interrupted full-run resume and the SAE farm remain in
qualification. Farm stress testing has
already exposed abort-classification and peer-release races. The repair itself
has added several thousand lines, reinforcing the need to split orchestration
before R1.

The final SAE metric is still pending. When all nine runs, evaluation, and
reporting complete, this document may record the exploratory G/C result, but
the two performance waivers must remain attached and headline eligibility must
remain false. No production SAE comparison existed at this snapshot, so no
relative A/B/F quality had yet been inspected.

## What worked

1. **The question became answerable.** A/B are candidate matched; A/F are
   parameter matched; all three reconstruct the same aggregate target. The
   result no longer depends on visual-token analyses, threshold calibration,
   local losses, or auxiliary dead-feature objectives.

2. **The activation site respected Gemma's math.** Capturing one routed
   aggregate avoided inventing post-nonlinearity per-expert contributions. The
   router weights remained diagnostics while IDs alone controlled visibility.

3. **Conversion had a real numerical oracle.** The `1.81e-05` maximum KL check
   caught checkpoint/implementation mistakes before expensive capture.

4. **Immutable ranges beat file counting.** Resume was keyed by deterministic
   input-range identity. A receipt could be adopted after a crash only after
   the remote object's exact identity was reverified. This is why the 50M
   capture did not need to be repeated.

5. **The bulk shard plane was genuinely durable.** Shards were serialized to
   private local scratch, conditionally uploaded under their content hash, and
   represented by a final manifest only after exact range and object
   validation. FP64 sufficient statistics removed a separate statistics pass.
   This claim does not extend to the Filestore-resident evidence needed to
   interpret and operate those objects.

6. **Failures generally failed closed.** Layer order, row ownership, route
   uniqueness, dropped tokens, recompilation, static shapes, hashes, counts,
   and checkpoint finalization were checked. Several failures that looked like
   “infrastructure weirdness” were correctly treated as possible scientific
   corruption until disproven.

7. **Detached execution broke the laptop's numerical-liveness dependency.**
   Once all four supervisors were launched under `nohup` with durable
   heartbeats, a sleeping Mac or broken SSH session no longer killed the job.
   Diagnosis, remediation, and operator notification still depended on a Codex
   session running from the laptop, so this was liveness rather than autonomous
   cloud operations.

8. **Performance evidence was preserved rather than rewritten.** Both the MFU
   and canary exceptions live alongside their failed source observations. The
   eventual paper must say that the launch SLOs were waived; it must not say
   they passed.

9. **The cache bottleneck was measured and fixed.** Later GCS probes on real
   immutable capture shards showed that four-object concurrency raised
   aggregate download throughput from 0.82–1.30 GB/s to about 5.15 GB/s. That
   is enough to stage a per-host partition quickly; it also showed why a single
   sequential shard loop was the wrong qualifier.

## What failed

1. **The 50% harvester MFU objective failed by more than an order of
   magnitude.** This is the largest unresolved performance result. Fast GCS
   does not compensate for an underutilized model prefix, and R1 cannot inherit
   this path blindly.

2. **The canary compared unlike denominators.** Synthetic output tokens and
   scientific valid tokens from real packed sequences were not density
   matched. Real/synthetic goodput therefore mixed model execution, masking,
   and input transport.

3. **Preflight scaled redundantly.** Four full corpus scans turned a useful
   proof into a 37-minute barrier and encouraged bypassing it under time
   pressure.

4. **The “strong POSIX control plane” abstraction leaked NFS client-cache
   semantics.** Atomic rename on the server did not imply immediate
   absent-to-present visibility on every client. A barrier cannot repair a
   stale negative dentry.

5. **Waivers bound mutable names.** Binding `hosts/host-p*.json` by path made
   the canary waiver incompatible with the expected production update of those
   files. A waiver should bind immutable evidence objects or a sealed snapshot,
   never a mutable latest pointer.

6. **Storage qualification mixed stages.** Capture upload, later capture
   download, local capacity, and Orbax were bundled in one conceptual gate,
   while its durable attestation was scoped only to capture. The mismatch was
   found only at the handoff to training.

7. **Runtime provisioning was artisanal.** Live package upgrades, reboots,
   Git cloning, per-host setup variation, and manual shell quoting consumed
   scarce accelerator time and created host drift.

8. **The operator interface was too expert-dependent.** The runbook was
   comprehensive but expected fluency in quota classes, zones, SSH, service
   accounts, NFS, JAX collectives, and process supervision. Status required
   several commands and interpretation. The system did not notify or remediate
   through a phone-reachable control plane.

9. **The rewrite did not remain compact.** Large modules and a multi-thousand-
   line CLI made emergency reasoning harder. The code has many good local
   invariants, but too many of them are coupled through one orchestration
   surface.

10. **Local tests were necessary but insufficient.** Fake meshes did not
    expose real device ordering, MaxText layer metadata, NFS lookup behavior,
    TPU compile behavior, or multi-host ownership.

11. **The target login lifecycle destroyed qualified state.** `nohup` and a
    successful `/dev/shm` qualifier did not protect user-owned IPC state after
    the last SSH session exited. `RemoveIPC=yes` plus `Linger=no` turned a
    clean 73.66 GB cache into missing input between fresh processes. This is
    target-runtime lifecycle debt, not a GCS-throughput or cache-integrity
    failure.

12. **The experiment evidence was split across durability domains.** A 1 TiB
    minimum Basic HDD Filestore volume carried a small amount of authoritative
    control data while GCS carried the bulk objects. Filestore remained zonal,
    separately billed (quoted during operation at about $0.27/hour), and
    required for convenient reconstruction of campaign state. It must not be
    deleted until a complete immutable evidence export is copied to GCS and
    independently verified.

## Bodges and waivers that must not become architecture

These actions were reasonable under a ten-day deadline. They are listed so
that “it worked once” does not silently turn them into policy.

| action | why it was used | why it must not scale |
|---|---|---|
| Attach the default Compute service account | fastest authorized path to GCS and TPU operations | excessive blast radius; no per-role audit boundary |
| Convert in a distant custom N2 VM | local capacity and quota made it available quickly | R1 checkpoint transfer, egress, and startup must be co-located with regional storage |
| Upgrade the live TPU image and reboot | made the existing allocation usable | slow, nondeterministic, and hostile to ephemeral Spot replacement |
| Skip the repeated full preflight | prior exact corpus proof was still valid and time was short | bypass flags are easy to misuse unless proofs are stage-scoped and machine checked |
| Qualify only one candidate in the final exact environment | the broader sweep and repair history had consumed the launch window | exploratory cross-environment tuning evidence must not be presented as a complete final sweep |
| Waive the 50% MFU gate | all non-MFU benchmark integrity checks were clean | performance debt remains real; a 671B model magnifies it |
| Waive canary performance checks | projected capture fit the authorized hour and durable integrity was clean | the comparison was not calibrated and the source state remains failed |
| Use an external self-hashed canary wrapper | production code had no narrow canary exception | ignored/unreleased code is hard to review, reproduce, and discover |
| Recover sealing with a second wrapper | NFS split after the data were already sealed | recovery policy belongs in the state machine and must use immutable bindings |
| Keep Filestore as the coordination root | familiar locks, rename, and shared paths | it is zonal, client-cached, and unsuitable for globally distributed executors |
| Use `/dev/shm` as an emergency SAE cache | sufficient fast capacity already existed | it is volatile and competes with process memory; capacity must be qualified per image/topology |
| Enable user linger live during diagnosis | stopped `RemoveIPC=yes` from clearing worker 0's sentinel at SSH teardown | lifecycle policy belongs in the baked image and its acceptance suite, not an emergency mutation under allocation |
| Use the operator's email in a waiver | established a human audit identity | identity is not notification; it sent no message and grants no control |
| Run the repair agent from the laptop | enabled rapid interactive diagnosis | detached jobs survive disconnection, but nobody diagnoses, alerts, or repairs when the laptop session disappears |

## Latent scalability and correctness risks

### Risks still present in the Gemma implementation

- **Environment identity is too coarse.** Capture records bind one repository
  environment hash. A later storage-only fix can therefore reject perfectly
  valid activation bytes. Scientific identity, implementation identity, and
  deployment identity need separate schemas and explicit compatibility edges.

- **Process-major layouts are topology assumptions.** JAX process indices are
  useful execution identities, not stable corpus owners. A replacement slice
  or another API can enumerate devices differently.

- **The SAE mesh is host-shaped.** Parameters are feature-sharded across the
  four chips local to a process and replicated across the data axis. That is a
  deliberate latency tradeoff, not a universal v6e-64 layout. Wider R1
  dictionaries may require a different feature/data/possibly pipeline mesh.

- **Static A capacity is topology and order dependent.** The current full-order
  prepass is safer than IID estimates, but its result depends on process
  partitioning. A resumed run must either keep the same global assignment or
  recompute and recompile a topology-specific capacity.

- **The committed JSONL journal could duplicate logical steps after rollback.**
  The in-progress identity-bound, `fsync`ed, checkpoint-reconciled journal
  addresses this specific risk, pending real interrupted-TPU validation. R1
  should retain immutable `(run, global_step, attempt)` identity and
  deterministic reduction rather than regress to blind append semantics.

- **`eval --all` can retain several compiled JAX programs.** Nine evaluations
  should run in isolated processes or explicitly evict caches.

- **Checkpoint and final-pointer publication can encounter the same POSIX
  visibility class as capture manifests.** GCS payload durability does not
  make an NFS `latest.json` globally visible.

- **The complete evidence bundle is not yet in GCS.** Content-addressed
  activation shards alone do not preserve the manifests, waivers, receipts,
  timing records, and control history needed to interpret them. Export and
  verify those records before Filestore teardown.

- **Hard-coded byte gates age badly.** A fixed 100 GiB cache and 128 GiB free-
  space gate did not match 100 GiB boot disks, even though volatile RAM could
  safely hold the actual partition.

- **Cache qualification did not bind the user-session lifecycle.** The
  qualifier proved capacity, hashes, inventory, and speed, but not survival
  after the last SSH session. A fresh-process farm makes that boundary part of
  data availability. Until the runtime image proves `Linger=yes` and a
  disconnect-survival probe, a resident `/dev/shm` attestation is incomplete.

- **The prefix adapter depends on private MaxText/NNX structure.** It
  duck-types unscanned layer and MoE fields and relies on pinned Flax NNX
  `jit_partial(..., graph_updates=False)` behavior to keep unchanged blocks
  16--30 out of the compiled step. MaxText's dropless EP path also allocated a
  worst-case buffer 16 times the balanced EP16 assignment count until
  `use_ragged_sort` skipped padded rows. These are pinned implementation facts,
  not portable model-adapter contracts. Constructing and tracing an unscanned
  61-layer, 671B R1 graph this way may itself be prohibitive; build a real
  prefix module/checkpoint rather than relying on compiler pruning of an
  enormous full graph.

- **Router capture recomputes the gate.** The native routed kernel produces
  the aggregate and a second small gate computation emits IDs and weights.
  Parity tests make that defensible for Gemma, but it duplicates work and
  creates a second router-semantics surface. R1 should prefer a native kernel
  auxiliary output or instrumentation hook.

- **The final scientific path is not yet production-proven.** Until an
  interrupted real TPU SAE run restores, passes the 5M gate, checkpoints, and
  evaluates, the training claims remain test evidence rather than production
  evidence.

### Risks amplified by R1 and Spot

- DeepSeek-R1-0528 uses the DeepSeek-V3 architecture: 61 layers, hidden width
  7,168, three initial dense layers, 256 routed experts, top-8 routing, one
  shared expert, grouped sigmoid routing, a learned route-selection bias, and
  route scale 2.5. Reusing Gemma's softmax/router adapter would be wrong.
  These values are in the official
  [R1 config](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528/raw/main/config.json)
  and [MaxText DeepSeek 671B config](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/configs/models/deepseek3-671b.yml).

- The upstream model has 671B total parameters; approximately 37B active
  parameters per token is a commonly reported architecture-level estimate and
  must not be treated as a measured FLOP count for this prefix. A hypothetical
  full BF16 weight image is about 1.342 TB decimal before runtime buffers. The
  released source is FP8-quantized, so source precision, conversion semantics,
  and the intended runtime weight format must be frozen explicitly.

- A DeepSeek aggregate activation record with BF16 `[N,7168]`, eight uint8
  IDs, eight BF16 weights, int64 sequence ID, and uint16 position is 14,370
  bytes/token. Fifty million tokens are about 718.5 GB decimal (669 GiB).
  This still favors aggregate capture strongly, but regional replication and
  cache sizing must be explicit.

- With 256 banks, A exposes only `8/256 = 3.125%` of F's dictionary. If bank
  width remains 1,024, total width doubles to 262,144 and 50M tokens provide
  only half as many tokens per feature as Gemma. Either use width 512 to retain
  131,072 total features, or scale the corpus/training budget. Both choices,
  50M tokens, and TopK-64 are proposals. Compute route-frequency coverage and
  statistical power first: a blanket 95% live-bank/feature gate may be
  structurally unreachable for rare router banks even when training is sound.

- A `v6e-64` is one 8×8, 64-chip topology represented by 16 half-host VMs with
  four chips each. It has 2 TiB aggregate HBM, but each chip has only 32 GB.
  MaxText specifically warns that DeepSeek layers require expert or tensor
  parallelism rather than pure FSDP on Trillium. See the official
  [v6e topology](https://docs.cloud.google.com/tpu/docs/v6e) and
  [MaxText model-customization guide](https://maxtext.readthedocs.io/en/maxtext-v0.2.2/guides/optimization/custom_model.html).
  A full hypothetical BF16 image would leave only about 11 GB per chip on
  average for runtime buffers, so one v6e-64 is **not yet proven** to fit the
  official full model and parity workload. Inventory the true prefix and full
  runtime states; qualify v6e-128/v6e-256, host-offload, or another supported
  layout if the full oracle does not fit.

- TPU Spot VMs can be preempted at any time, cannot be restarted, and do not
  support the optional 120-second preemption notice. Recovery must assume zero
  warning. See [Manage TPU Spot VMs](https://docs.cloud.google.com/tpu/docs/spot).

- Cross-region execution cannot be one low-latency JAX collective. A zonal
  slice is the failure and synchronization domain; regional slices can only
  cooperate through durable work units.

## Hard lessons

1. **Freeze scientific semantics, not incidental deployment bytes.** Model,
   activation site, dataset order, objective, seed, width, and TopK belong in a
   scientific identity. Image digest, MaxText commit, XLA/libtpu build, zone,
   and mesh belong in implementation/deployment identities. A tested
   compatibility record may connect them; neither should impersonate the
   other.

2. **The output commit is the unit of progress.** A process, host, filename,
   or “number of shards present” is not progress. An immutable range plus a
   validated content object plus a fenced commit is progress.

3. **Performance gates and scientific gates are different.** A route drop,
   parity failure, duplicate token, or non-finite optimizer state invalidates
   science. Low MFU makes a run uneconomic. Both matter, but an authorized
   economic deviation should not require pretending the science gate passed.

4. **Benchmark the exact denominator.** Synthetic and real paths must expose
   the same sequence shape, valid-token density, instrumentation, and
   persistence contract. Useful FLOPs must describe exactly the executed
   prefix, while tokens/s must say whether it counts padded, model-output, or
   scientific tokens.

5. **Do not make the operator's laptop part of liveness.** The control service,
   credentials, logs, alerts, and remediation loop must stay in cloud. A phone
   should be an observation/approval surface, not a terminal that keeps the job
   alive. Distinguish three promises: detached workers provide numerical
   liveness, a cloud watcher plus bounded repair agent provide operational
   autonomy, and push notification plus one-time actions provide human
   reachability. Gemma achieved only the first.

6. **Bake environments; do not repair them under allocation.** Image build and
   vulnerability/package updates happen before accelerator acquisition.
   Workers prove an image digest and start the job. No production startup
   should require an OS upgrade or reboot.

7. **A shared filesystem is not a distributed database.** POSIX rename and
   `flock` are convenient within one kernel view. They do not provide the
   fencing, immutable history, or cross-region semantics required here.

8. **Target-runtime tests are part of implementation.** Device ordering,
   collectives, checkpoint restore, router parity, preemption, and storage
   publication need real-slice acceptance tests. So do login/session teardown,
   detached-parent survival, and host-cache survival across fresh-process
   handoff. Unit parity remains necessary, but it is not deployment
   qualification.

9. **Every emergency path must be promoted or retired.** A successful wrapper
   becomes a reviewed state-machine transition with tests, or it is explicitly
   forbidden next time. Operator-only scripts must not become invisible
   dependencies.

10. **Operational simplicity is scientific risk reduction.** Fewer mutable
    states, fewer manual commands, and fewer identity concepts exposed to the
    operator directly reduce the chance of comparing the wrong bytes or
    silently changing a denominator.

## Target architecture for DeepSeek-R1 on multi-region v6e-64

### Current external facts

As of this snapshot:

- MaxText lists DeepSeek V3 0324 and DeepSeek R1-0528 671B as supported JAX
  models. Pin a new exact MaxText commit; “current main” is not a production
  identity. See the [MaxText model list](https://github.com/AI-Hypercomputer/maxtext).
- `v6e-64` is an 8×8 topology with 64 chips, eight physical hosts, and 16
  four-chip VMs. Each chip provides 32 GB HBM and 918 BF16 TFLOP/s peak, so the
  slice peak is 58.752 PFLOP/s by multiplication.
- Currently advertised general v6e zones are `us-central1-b`, `us-east1-d`,
  `us-east5-a`, `us-east5-b`, `us-south1-ai1b`, `europe-west4-a`,
  `asia-northeast1-b`, and `southamerica-west1-a`. Availability is not a
  capacity guarantee; query the API immediately before scheduling. See
  [TPU regions and zones](https://docs.cloud.google.com/tpu/docs/regions-zones).
- Flex-start can queue a requested duration up to seven days. The current
  Compute Engine page lists v6e Flex-start in `asia-northeast1-b`,
  `us-east5-a`, and `us-south1-ai1b`. It is a different availability product,
  not a synonym for Spot. See
  [Create TPU Flex-start VMs](https://docs.cloud.google.com/tpu/docs/create-flex-start-compute).

These are time-sensitive cloud facts and must be rechecked at launch.

### Scientific plane

The model adapter and SAE geometry must be declarative:

```text
model revision
  -> exact architecture assertion
  -> exact target layer/site
  -> router semantics
  -> activation schema
  -> A/B/F geometry
  -> data/order/seed
  -> report contract
```

Proposed, not yet frozen:

- Pin an exact revision of `deepseek-ai/DeepSeek-R1-0528` and declare whether
  the scientific source is the released FP8 checkpoint or a precisely specified
  dequantized conversion. Use one fixed MoE block near relative depth 0.5;
  one-indexed block 31 is the obvious first candidate, but it must be checked
  against the exact MaxText layer structure.
- Capture the weighted routed aggregate immediately before addition of the
  shared expert and residual. Validate this against an unmodified full block.
  Prefer an upstream-native auxiliary output or hook over a second router pass
  and duck-typed access to private layer fields.
- Reproduce DeepSeek routing exactly: group restriction and top-8 IDs are
  selected using bias-adjusted sigmoid scores, while emitted weights come from
  the unadjusted scores, are normalized according to the model config, and are
  multiplied by the routed scaling factor. IDs control A visibility; weights
  remain diagnostic unless the frozen target says otherwise.
- Preserve A/B/F algebra. For bank width `w`, A and F have `256w` features, B
  has `8w`, and nominal exposure is 3.125%. Parameter count is
  `features × (2×7168 + 1)` for encoder, decoder, and bias. At `w=512`, A/F
  contain 131,072 features and about 1.879B parameters. At `w=1024`, they
  contain 262,144 features and about 3.758B parameters.
- Keep exact per-token TopK-64 initially so the algorithmic comparison stays
  homologous to Gemma. Reconsider `k` only before data results and with a
  written power/feature-utilization calculation.
- Run a same-distribution replication first if the goal is architecture
  transfer: pin the bounded C4 source bytes again, then restage them with the
  R1 tokenizer and explicitly record the resulting packing and scientific-token
  identity. Treat a reasoning-trace corpus as a second experiment: generated
  chain-of-thought tokens, prompt tokens, and ordinary web text are different
  measures and must not be silently mixed.

The full source checkpoint should pass official logit parity first, on a slice
large enough for the actual full-model runtime rather than assuming v6e-64.
Then build a content-derived prefix checkpoint containing only embeddings and
layers needed through the target. That prefix must have its own tensor inventory
and must reproduce the full checkpoint's target activation exactly. Loading all
post-target layers, unembedding, and MTP weights on every Spot attempt would
waste both HBM and startup time.

### Control plane

Run one small, non-Spot controller independent of the researcher's laptop. It
owns the experiment state machine, queued-resource requests, attempt fencing,
alerts, and policy. It does not participate in JAX collectives.

```text
always-on controller
  |-- global immutable experiment registry
  |-- regional capacity queues
  |-- fenced work leases and retry policy
  |-- status/alert/approval API
  `-- final manifest and cost reducer
          |
          +-- region A: one zonal v6e-64 collective
          +-- region B: one zonal v6e-64 collective
          `-- region C: one zonal v6e-64 collective
```

Controller state must be durable, strongly consistent for transitions, and
conditionally updated. Before launch, select and chaos-test the actual store:
for example, a managed transactional database, or a deliberately restricted
GCS generation-match state machine with proven lease/fencing semantics. “Use
object storage” is not by itself a concurrency protocol. A lease contains
`campaign_id`, `work_range_id`, `attempt_id`, an increasing fence token,
expiry, and expected scientific identity. A stale worker may finish an upload,
but the coordinator accepts its receipt only if the fence is still current or
the bytes are an exact content duplicate of the committed result.

A deterministic cloud watcher should emit a machine-readable incident bundle.
A bounded cloud-resident repair agent may inspect logs, retry an unchanged
attempt, or apply preapproved operational repairs; a separate verifier checks
its proposed transition. Every action, authority used, diff, and result enters
the attempt ledger. Scientific constants, data identity, and exception policy
remain outside the agent's grant.

Phone notifications should carry status and a bounded action such as “retry
same immutable work,” “switch Spot to Flex-start,” or “halt campaign.” Actions
must be one-time, expiring, and tied to the incident digest. The phone is not a
shell and the agent is not granted permission to change scientific constants.

### Execution plane

Each `v6e-64` is one atomic executor: 16 VMs, conventionally 16 JAX processes
with four local devices each (64 chips total), one topology, one image digest,
and one collective lifetime. Record the actual runtime enumeration rather than
deriving ownership from those counts. Never form a collective across regions.

- **Harvest:** allocate globally unique immutable token ranges to any healthy
  slice. Different regions can capture different ranges concurrently. Publish
  shards regionally, then seal one global logical manifest after every content
  identity and range is present. Range identity and token order must be derived
  from the corpus/scientific identity, not JAX process numbering or the
  topology that happened to execute it.
- **SAE training:** one condition/seed run stays within one slice at a time.
  It may resume on an equivalent slice in another region only after a
  cross-region checkpoint-restore parity test has passed. Conditions do not
  share a live optimizer collective. Matched per-seed activation order must be
  invariant across region and topology; A capacity must be recomputed and
  recompiled if ownership changes.
- **Evaluation:** run each condition/seed in an isolated process. It may be
  scheduled near the checkpoint/capture replica and reduced centrally.

Start from MaxText's native DeepSeek sharding and benchmark EP/TP/FSDP/pipeline
variants on the real prefix. Do not transplant Gemma's EP16 shortlist or its
host-local feature axis by name. The rough “10B parameters per MoE layer”
estimate must be replaced by the converted tensor inventory; in either case,
pure FSDP is unlikely to fit 32 GB HBM and expert or tensor sharding is a
first-class qualification candidate.

### Image and identity plane

Choose the execution substrate before choosing the image artifact. For
Compute Engine/Flex-start/Spot, bake and qualify a TPU-compatible custom **OS
image** derived from a supported accelerator Ubuntu image; Google documents
the required TPU drivers, agents, VFIO configuration, and the operator's
compatibility responsibility in
[Use a custom OS image](https://docs.cloud.google.com/tpu/docs/custom-os-image).
An OCI workload image may be layered on top, and is the natural unit on GKE,
but a container alone does not remove mutable host bootstrap. Pin the OS image,
Python, JAX, libtpu, MaxText and repository commits, locked dependencies, and
any OCI digest in one software inventory. Replicate regional image artifacts
if pull time or egress warrants it. Startup may mount scratch, fetch identities,
and launch; it may not apt-upgrade or reboot.

Image acceptance must also prove the runtime user's lifecycle. Bake
`loginctl enable-linger` (or an equivalently explicit service-owned lifetime)
into provisioning and require `loginctl show-user <runtime-user> -p Linger`
to return `yes` on every VM. Create an owned `/dev/shm` sentinel, disconnect
the final administrative session, wait beyond the image's logind cleanup
interval, reconnect, and prove both the sentinel and detached controller
survived before accepting the image. This is required even if the image keeps
`RemoveIPC=yes` as a sensible default for non-lingering users. An OCI workload
image alone does not prove the host-side lifecycle.

Use three identities:

1. **Scientific identity:** model/checkpoint tensor digest, tokenizer, corpus
   manifest, target layer/site, activation schema, condition geometry,
   optimizer, token order, and seed.
2. **Implementation identity:** repository commit, MaxText commit, image
   digest, JAX/libtpu/XLA versions, converter, and adapter implementation.
3. **Deployment identity:** cloud project, zone, consumption model, slice and
   topology, VM/process/device coordinates, service account, and attempt.

A compatibility attestation links a producer implementation to a consumer
implementation only after schema validation and numerical parity. A storage or
orchestrator patch must not require rewriting scientific provenance; an
adapter math change must.

### Data and artifact plane

Remove shared NFS from correctness. Use object storage for immutable payloads
and conditional publication. A logical content object can have several
regional replicas; record each replica's URI, generation, size, CRC32C, and
SHA-256. GCS generations are replica-specific, while SHA-256 is the portable
logical identity.

Before the Gemma Filestore is deleted, export manifests, receipts, waivers,
host ledgers, timing records, configuration, logs, and final checksums to an
immutable GCS evidence prefix and independently reconstruct the capture from
that export. R1 should create this bundle continuously, not as teardown work.

Use ephemeral host RAM/local SSD only as a verified cache. Compute cache limits
from the actual assigned partition and measured free memory. For a 50M R1
train capture, even distribution over 16 VMs is about 42 GiB per VM before
overhead; eval adds about 1.7 GiB per VM. Those figures exclude staging
temporaries, model/runtime pressure, and skew. They are only sizing inputs,
even though a four-chip v6e VM is documented with 720 GB host RAM; the
qualifier must measure actual free memory and enforce a safety margin.

Checkpoint payloads use unique attempt prefixes. A `latest` record is an
object-store CAS update published only after Orbax finalization and a
cross-process restore check. Never overwrite a payload. Never delete the prior
checkpoint until its successor is finalized, published, and independently
readable. Keep enough generations to survive a corrupt newest checkpoint and
make retention an explicit cost policy.

### Spot and Flex-start policy

Spot is appropriate for range-addressed harvest and, after chaos testing,
resumable SAE runs. Because there is no usable preemption warning, checkpoint
intervals must be chosen from measured save/restore time, workload progress,
and observed or conservatively assumed interruption hazard, not from a shutdown
hook. Detect slice loss externally, fence the attempt, preserve its logs,
recreate the entire 16-VM slice, and resume from the last committed state.

Flex-start is preferable for first full-model qualification, long F runs, or
any phase whose compile/load/checkpoint overhead makes Spot waste excessive.
Its queue wait is time-to-result, not accelerator chip-seconds; report it
separately. Requests are limited to seven days and supported zones differ from
general v6e availability.

Service accounts must be role-specific: controller, harvester, trainer, and
read-only verifier. Use Workload Identity or attached identities, not human
tokens and not the default broadly privileged Compute account.

## Migration sequence

1. **Finish and freeze Gemma evidence.** Complete all nine runs, held-out
   evaluation, report, exception disclosure, complete Filestore-to-GCS evidence
   export, and a final artifact inventory. Tag that exact state. Add the actual
   G/C result to the top of this document as exploratory and keep
   `headline_eligible=false` permanently.

2. **Extract protocols before adding R1.** Split the current CLI into model
   adapter, immutable artifact protocol, SAE math, run worker, and controller.
   Convert emergency canary/seal recovery into typed transitions or remove
   them. Replace JSONL append semantics and POSIX pointers.

3. **Build the object-store controller.** Implement fenced leases, immutable
   attempt logs, CAS state transitions, orphan adoption, stale-attempt
   rejection, regional quota/capacity discovery, budget stops, abandoned-
   attempt garbage collection, phone alerts, and whole-slice restart. Prove it
   with fake workers and forced races before using TPU.

4. **Build and publish the immutable TPU environment.** For Compute Engine,
   bake the custom TPU OS image and optionally its OCI workload layer; replicate
   artifacts regionally. Test it on a small v6e slice with a synthetic/tiny
   model. Prove `Linger=yes`, detached-controller survival, and `/dev/shm`
   sentinel survival after the last SSH session exits. Target cold worker
   readiness in minutes, with no package mutation or reboot.

5. **Pin and convert R1.** Select an exact current MaxText commit that contains
   R1-0528/DeepSeek 671B support. Freeze the exact source revision and FP8-to-
   runtime conversion policy, inventory every tensor, convert using the
   official path, pass logits parity on adequate hardware, derive the prefix
   checkpoint, and pass prefix-versus-full activation parity.

6. **Freeze the R1 scientific design.** Choose layer, target site, corpus,
   bank width, token budget, TopK, and seeds using only architecture, memory,
   route-frequency coverage, and power calculations. Forbid quality peeking
   during systems tuning.

7. **Qualify one Flex-start or on-demand slice.** Start with v6e-64 only if the
   tensor/runtime inventory predicts adequate full-oracle and prefix headroom;
   otherwise use a larger supported topology. Prove whole-slice readiness,
   image and checkpoint identities, model memory headroom, native router
   invariants, compile stability, exact row ownership, regional object
   throughput, and useful-prefix FLOP accounting.

8. **Benchmark the real prefix.** Use identical real and synthetic valid-token
   density. Sweep only predeclared MaxText mesh/batch candidates, retain every
   failure, and explain the gap to a roofline model. Do not authorize a global
   campaign with an unexplained Gemma-like 2% MFU.

9. **Run a 1M capture and preemption drill.** Kill the whole slice at random
   points: before upload, after upload before receipt, after receipt, and
   during manifest sealing. Resume in another zone and prove one exact
   contiguous manifest.

10. **Scale harvest regionally.** Give each slice disjoint immutable ranges,
    keep payloads near their producer, replicate sealed objects needed by SAE
    regions, and seal one global manifest. Measure duplicate and abandoned
    work explicitly.

11. **Qualify SAE training on real v6e-64.** Compile A/B/F, measure memory and
    TopK communication, perform interrupted-versus-uninterrupted restore, and
    run the blind 5M health gate. Width or token-budget changes after this
    point are a new experiment.

12. **Run one full paired seed, then three.** Schedule condition order to avoid
    systematic zone/time confounding, and preserve matched token order across
    topology changes. Pair or randomize A/F placement so region, time, and
    hardware do not masquerade as an architectural cost difference. Prefer
    Flex-start for first F if Spot checkpoint economics are unknown. Use seed 1
    only for operational/health evidence; do not inspect relative held-out
    quality until all predeclared production runs have completed. Then evaluate
    all seeds together.

13. **Publish two cost views.** Preserve the algorithmic G/C metric for
    comparison with Gemma, and add realized Spot/Flex-start cost and waste. A
    cheaper dollar total caused by regional price or random preemption is not
    an architectural speedup.

## Acceptance gates

### Identity and security

- Exact source model revision, tensor inventory, tokenizer, MaxText commit,
  image digest, and repository commit are present and self-consistent.
- No human credential exists on workers. Every service account has a reviewed
  minimal role set and can access only its stage's prefixes/actions.
- Scientific, implementation, and deployment identities are distinct and
  included in every receipt, checkpoint pointer, metric event, and report.

### Model and capture correctness

- Official HF/MaxText logits parity passes a frozen KL/error threshold.
- Prefix checkpoint and adapter match the full unmodified model at the target
  activation on a fixed validation set.
- Router IDs exactly match DeepSeek's grouped, bias-adjusted selection; eight
  IDs are unique and in `[0,255]`; diagnostic weights match unadjusted-score
  normalization and route scaling.
- Instrumentation does not change target output. No routed token is dropped.
- Every scientific row maps to one `(example, sequence, position)` and is
  invariant under each accepted mesh.

### Distributed runtime and performance

- All 16 VMs join or the attempt does no scientific work. A partial slice
  cannot publish accepted output.
- The baked runtime identity reports `Linger=yes` for the service user on every
  VM, and a disconnect test proves that both a detached controller and owned
  host-cache sentinel survive beyond the image's logind cleanup interval.
- One static compile serves all measured steps; no hidden recompilation occurs.
- Peak per-chip HBM retains at least a measured safety margin (default 15%) and
  no host approaches tmpfs/DRAM exhaustion.
- A roofline-derived useful-prefix MFU floor is frozen before production. A
  provisional planning ambition is at least 30% on v6e and at least 80% of a
  calibrated roofline, but neither number is current evidence or a valid gate
  until real prefix measurements establish the attainable model. Any lower
  launch is an explicit operational exception, never a silent gate change.
- Real-data scientific tokens/s is at least 90% of the density-matched
  synthetic path; input wait is below 5%; durable output is at least 90% of
  device output; writer queues remain bounded.

### Artifact and preemption safety

- Concurrent uploads are conditional and content-addressed. Seal proves exact
  total, contiguous ranges, no duplicates, and every regional replica's
  identity.
- Forced preemption at each publication boundary loses at most the documented
  uncommitted interval and cannot create an accepted duplicate or gap.
- Stale fence tokens cannot update campaign state after lease takeover.
- Checkpoint restore is proven after interruption on the same slice and on an
  equivalent slice in another accepted zone. Exact equality is preferred; if
  only numerical equivalence is achievable, its tolerance and scientific
  consequence are frozen before production.
- Corrupt or incomplete newest checkpoints fall back to a retained validated
  predecessor without mutating the source record.

### SAE and report correctness

- Tiny NumPy/JAX parity covers A/B/F scoring, deterministic distributed TopK,
  sparse decode, gradients, tangent projection, and renormalization at R1
  dimensions/topology.
- The blind 5M gate requires finite state, exact L0=64, the frozen loss-EMA
  decrease, no capacity overflow, and a live-feature threshold justified by
  the frozen router-frequency/power analysis. Retain 95% only if that analysis
  shows it is attainable for rare A banks and F under the chosen budget.
- All conditions use one held-out manifest, one training mean, one SST, and
  matched seed order. Evaluation runs in isolated processes.
- G remains unclipped; non-positive or interval-unstable F-over-B gain is
  labeled undefined/exploratory. Every operational exception appears in the
  headline eligibility field.

## Cost and waste accounting

The current Gemma primary metric should remain available:

```text
marginal_chipseconds(run)
  = (committed optimizer wall seconds
     + accepted A capacity-prepass wall seconds) × physical chips

C_algorithmic
  = 100 × median_seed(marginal_chipseconds(A) /
                      marginal_chipseconds(F))
```

Optimizer wall time includes step synchronization and input stalls. The second
term is zero for B/F and includes the one accepted A capacity prepass. The
metric excludes compile, checkpoint serialization, evaluation, and shared
harvest. This measures the marginal algorithm under one successful execution
path. Repeated capacity prepasses after failed or preempted attempts are not
silently added to the algorithmic denominator; they appear in realized attempt
cost below.

Spot requires a second ledger. For every attempt, record provisioned start/end,
zone, consumption model, chip count, phase, last committed checkpoint/range,
bytes transferred, and termination cause. Derive:

```text
attempt_chipseconds
  = all provisioned slice wall seconds × chips

wasted_chipseconds
  = failed/replayed work + abandoned compile/load/cache/checkpoint time

useful_fraction
  = marginal_chipseconds / attempt_chipseconds

C_realized
  = 100 × median_seed(total attempt chipseconds for A /
                      total attempt chipseconds for F)
```

Report at least these disjoint buckets:

- committed optimizer chip-seconds;
- replayed optimizer chip-seconds lost after the last durable checkpoint;
- model/checkpoint load and cache-fill chip-seconds;
- compilation chip-seconds, including failed compiles;
- checkpoint write/validation chip-seconds;
- evaluation chip-seconds;
- successful and abandoned harvest chip-seconds;
- idle provisioned chip-seconds and controller VM cost;
- regional object storage, operations, replication, and network egress;
- queue wait and wall-clock time to result (not billed accelerator time unless
  the provider actually allocates resources during it);
- realized currency cost using the price in that region and consumption model.

Do not compare raw dollars across conditions without preserving region and
price. Random Spot interruption is a nuisance variable: pair seeds, randomize
condition order, report attempt histories, and retain both `C_algorithmic` and
`C_realized`. The former supports the architecture claim; the latter tells an
operator what the experiment actually cost.

Shared harvesting remains separate and should be amortized over one and three
seeds, plus any additional follow-up reuse. Regional replication is not free
harvesting and belongs in end-to-end realized cost. Duplicate content produced
by a raced lease may be scientifically harmless, but its chip-seconds and bytes
are still waste.

## Evidence ledger at this snapshot

| claim | status | evidence |
|---|---|---|
| DeepSeek-era implementation preserved | committed | tag `deepseek-vl2-final-a291707` |
| Gemma checkpoint numerical parity | observed | max KL `1.81e-05`; conversion attestation bound into capture |
| bounded C4 staging identities | observed | train/eval stage results and immutable manifests |
| 50M train capture | observed and independently revalidated | 384 GCS objects; manifest `ed2ba…3157` |
| 2M held-out capture | observed and independently revalidated | 16 GCS objects; manifest `d407cf…302b` |
| earlier nine-candidate MFU sweep | observed, non-final environment | maximum 2.445%; useful diagnostic evidence only |
| final exact-environment MFU qualification | **failed and waived** | one EP16/FSDP1/TP1 batch-1 observation; 2.124329898% MFU and 25,628.09 persisted tokens/s |
| original real-data canary performance | **failed** | 16.2k vs 25.6k tokens/s and 19.18% input wait; narrow waiver used |
| cached production input path | observed | roughly 20.3k tokens/s; input wait below 2% |
| NFS publication protocol | **failed repeatedly** | waiver/start-release visibility incidents and negative-dentry split after successful 50M seal |
| immutable capture recovery | observed | no train recapture; exact wrapper/waiver bindings retained |
| `/dev/shm` lifecycle across SSH exit | **failed and isolated; remediation required** | 73.66 GB passed at 4.946 GB/s then vanished after about 20s; worker-0 `Linger=yes` sentinel survived while workers 1--3 `Linger=no` sentinels vanished |
| complete experiment evidence independent of Filestore | **not yet established** | immutable GCS payloads exist; control/evidence export and reconstruction remain required |
| SAE storage and producer/consumer compatibility | in progress | separate attestation and selected-code compatibility proof exist only in worktree |
| checkpoint-reconciled metrics journal | in progress | identity-bound, monotonic, `fsync`ed implementation and tests exist in worktree; interrupted TPU validation pending |
| nine final SAE runs and exploratory G/C | **not yet established** | must not be inferred from capture success |
| original-contract headline eligibility | **permanently false** | hard MFU and canary gates were waived, not passed |

Full R1 production should begin only after the exploratory Gemma report exists,
the Filestore evidence export is independently verified, and the Stage-8
repairs are committed, tested on TPU, and tagged. The final rows should be
updated with actual results, but the headline-eligibility row must not change.
