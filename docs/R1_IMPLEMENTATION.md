# DeepSeek-R1 conditional-SAE implementation

**Status:** protocol-oracle development, 2026-07-22; not production-qualified.

This document turns the Gemma 4 campaign into a deliberately stricter
DeepSeek-R1 experiment. It is both an implementation order and an evidence
contract. Nothing described as a gate is satisfied merely because code for it
exists.

The current branch deliberately cannot be treated as headline evidence. Its
canonical records, exact arithmetic, CAS controller, and adversarial unit tests
are a protocol oracle. Production qualification remains blocked until all of
the following are implemented and exercised against real artifacts:

- an operator-owned factory that binds exact parser implementations and their
  image/code attestations, rather than trusting a parser's self-reported ID;
- production deployment of the implemented atomic MFU-admission event under a
  separate controller principal, with trusted server-time provenance and a
  trace/parser path that echoes its accepted event digest and challenge;
- strict semantic parsing and cross-linking of scientific, implementation, and
  deployment identities;
- bounded, streaming parsers for real HLO, Xprof, input, routing-capacity,
  persistence, and host-metric formats; and
- a report command that accepts only a generation-pinned envelope reverified
  under an externally provisioned trust policy.

Synthetic parser fixtures and green protocol tests do not satisfy these
requirements and cannot be cited as a production MFU pass.

The protocol oracle now commits admission through the immutable event,
snapshot, and generation-matched head CAS; accepted-chain replay binds the
campaign, work, attempt, executor, fence, exact lease interval, full run
contract, and controller challenge. Renewal cannot extend that admitted
interval. This closes the state-machine race structurally, but the challenge
is only correlation material until a separately trusted controller and real
measurement parsers prove the executor observed it after admission.

## The claim we are trying to earn

The scientific claim remains narrow:

> At one fixed DeepSeek-R1 routed-MoE site, a router-conditioned dictionary
> recovers `G%` of the parameter-matched global dictionary's reconstruction
> gain over a candidate-matched global dictionary, at `C%` of the global
> dictionary's measured marginal SAE training cost.

The systems claim is now equally important:

> The activation prefix and an actually executed dense R1-activation SAE each
> sustain at least 51% model FLOP utilization on a real v6e-64. The production
> sparse F trainer separately reports only work it really executes and must
> reach either 51% actual-work MFU or at least 80% of its pre-registered
> measured roofline, never a counterfactual dense decoder.

The two claims are conjoined in the headline contract. A scientifically good
`G/C` result with sub-51% qualifying performance is still useful diagnostic
work, but it is not the intended R1 result. There is no R1 MFU waiver record.

The published 50.8% comparison comes from the original Gemma Scope report: a
131K-wide SAE, batch 4,096, one step in 45 ms on eight TPUv3 chips. Gemma Scope
2 reuses model-parallel latent sharding and adds sparse kernels, but its
technical report is not the source of the 50.8% number. We will state that
distinction exactly.

Primary references:

- [Gemma Scope report](https://aclanthology.org/anthology-files/pdf/blackboxnlp/2024.blackboxnlp-1.19.pdf)
- [Gemma Scope 2 technical report](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/gemma-scope-2-helping-the-ai-safety-community-deepen-understanding-of-complex-language-model-behavior/Gemma_Scope_2_Technical_Paper.pdf)
- [DeepSeek-R1-0528 source checkpoint](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528)
- [MaxText](https://github.com/AI-Hypercomputer/maxtext)
- [Cloud TPU v6e specification](https://docs.cloud.google.com/tpu/docs/v6e)
- [Pinned R1 checkpoint conversion contract](R1_CHECKPOINT_CONVERSION.md)

## What the Gemma run established

### What worked

- The aggregate routed-expert branch is the right common reconstruction
  target. It avoids inventing post-normalization per-expert vectors.
- A/B/F isolates conditional access: A and B expose the same candidate count;
  A and F have the same total dictionary width.
- Exact positive per-token TopK, sparse selected-column decode, tangent-space
  decoder updates, and exact L0 checks are sound reusable math.
- Immutable content-addressed bulk objects, exact input ranges, manifests
  sealed last, and independently checked hashes survived operational repairs.
- Marginal chip-seconds and persisted goodput expose real cost rather than a
  nominal candidate ratio.
- Fail-closed model, router, shard, checkpoint, and multi-process invariants
  caught several errors before accepted data were published.

### What failed or became a bodge

- Gemma prefix MFU was 2.124329898%, not remotely close to the planned 50%.
  That gate was explicitly waived. The run is permanently exploratory.
- A four-process data-by-feature mesh replicated large SAE state and forced
  expensive full-gradient data-parallel reductions. F currently demonstrates
  this failure most clearly.
- Runtime package installation, OS mutation, reboot, SSH-owned process
  lifetime, and a late `loginctl` repair made machine bring-up slow and
  operator-dependent.
- Mutable Filestore/NFS control records coupled correctness to one zonal
  filesystem. Immutable capture payloads in GCS were safer than their control
  plane.
- Every process initially revalidated the whole packed corpus. Work scaled as
  `hosts * corpus` instead of one proof plus cheap consumer verification.
- Host suffix, JAX process index, local device order, and global device order
  were initially treated as if they were interchangeable.
- Cache qualification was coupled to environment identity instead of being an
  explicitly selected content-addressed attestation. That does not survive
  repeated Spot replacement cleanly.
- One large sequential farm serialized independent A/B/F seeds. Global Spot
  capacity would not help if the scheduler retained that shape.

The lesson is not merely "use more chips." Scientific identity, executor
topology, object placement, and process lifecycle must be separate concepts.

## Frozen DeepSeek facts and choices

The checkpoint source is `deepseek-ai/DeepSeek-R1-0528` at immutable revision
`4236a6af538feda4548eca9ab308586007567f52`. Its configuration, index, model
source, safetensor headers, and exact block-31 prefix selection are pinned in
the [checkpoint conversion contract](R1_CHECKPOINT_CONVERSION.md). Its
released config has:

- 61 transformer blocks and hidden width 7,168;
- three initial dense blocks followed by routed MoE blocks;
- 256 routed experts, top-8 routing, and one shared expert;
- sigmoid router scores with eight routing groups and four selected groups;
- bias-adjusted scores for expert selection;
- unadjusted sigmoid scores for emitted route weights;
- normalized weights multiplied by routed scaling factor 2.5.

The MaxText architecture source pin inspected on 2026-07-22 is commit
`c03f8bd307d18049fa814fc8c104bd9ec8c292aa`. Its
`deepseek3-671b-batchsplit.yml` explicitly supplies `n_routing_groups=8`,
`topk_routing_group=4`, explicit sharding, and the batch-split schedule. This is
not yet an accepted runtime: it becomes the implementation identity only after
conversion, full-model parity, prefix parity, and v6e kernel qualification
pass.

The audit found architecture-level R1 support, but no accepted direct
conversion from the released raw checkpoint. The standalone MaxText
converters reject raw FP8 scale tensors, their documented dequantization stage
is fail-open on a missing scale, and the following DeepSeek converters cast to
FP16. The generic lazy converter maps FP8 weights without consuming their
scales. Production therefore requires the fail-closed streaming conversion in
the checkpoint contract; an upstream command merely exiting zero cannot seal
the checkpoint identity.

Consequences for the capture schema:

- expert IDs are `uint16`, not `uint8`;
- valid route weights sum to 2.5, not 1.0;
- group restriction and router bias are scientific semantics;
- the native MoE invocation must expose its actual IDs and weights; a second
  approximate router pass is not acceptable;
- the target remains the weighted routed-expert aggregate before the target
  block's shared expert and residual addition.

The tensor inventory confirms that one-indexed block 31 is Hugging Face layer
30 and unscanned MaxText `decoder.moe_layers_27`. It is the intended midpoint
MoE block. The scientific tap becomes fully accepted only after the full-model
oracle proves the routed aggregate immediately before the target shared expert
and residual addition.

The first geometry candidate keeps a 131,072-feature A/F dictionary by using
256 banks of width 512. B has 4,096 global features. Per token:

| condition | total features | candidates | nominal exposure |
|---|---:|---:|---:|
| A | 131,072 | 4,096 from eight visible banks | 3.125% |
| B | 4,096 | 4,096 | 100% of B |
| F | 131,072 | 131,072 | 100% of F |

All three retain exact positive TopK-64 unless a pre-data power and memory
analysis changes the contract. A and F each contain approximately 1.879
billion encoder/decoder/bias parameters at hidden width 7,168.

## Honest MFU contract

### Hardware denominator

A v6e chip has a documented BF16 peak of 918 TFLOP/s. A v6e-64 therefore has
a peak of:

```text
P_peak = 64 * 918e12 = 58.752e15 FLOP/s
P_51   = 0.51 * P_peak = 29.96352e15 FLOP/s
```

The denominator is the physical chip count multiplied by synchronized device
wall time. Queueing, compilation, artifact publication, and checkpoint I/O are
reported separately. They never disappear from the realized-cost ledger, but
they are not inserted selectively into one condition's MFU window.

### Useful-prefix numerator

The prefix numerator counts only floating-point matrix operations required by
scientific, non-padding tokens:

- embedding is a lookup and contributes zero matrix FLOPs;
- every executed dense projection contributes `2*M*N*K`;
- routed experts contribute only the eight experts selected for each valid
  token, never all 256 and never padded capacity rows;
- shared experts count in completed preceding blocks;
- the target block stops at the captured routed aggregate;
- causal attention QK and AV use the exact packed-segment lengths, not the
  rectangular padded sequence area;
- router projection matmuls execute in FP32 and are reported separately, but
  do not enter the useful-BF16 numerator divided by TPU BF16 peak; TopK,
  normalization, dispatch, gathers, activation functions, and object writes
  do not masquerade as matrix FLOPs;
- post-target blocks, final norm, vocabulary projection, MTP, and logits do
  not count because they do not execute.

The formula is generated from the pinned tensor inventory and execution
contract, then checked against compiler HLO/Xprof. A hand-maintained
"fraction of full model" estimate is not admissible.

As a planning cross-check—not the final ledger—the released R1 geometry gives
approximately 37.39 useful BF16 GFLOP per valid token through the proposed tap,
plus separately reported FP32 router work. Therefore 51% of a v6e-64 implies
about 801,000 valid tokens/s, or roughly 82 ms for a
65,536-token global batch. A 14,378-byte capture row at that rate is about 11.5
GB/s of aggregate device output. The writer must be designed for that number;
persisting directly in the synchronized prefix step would erase the MFU result.

### SAE numerators: dense calibration and sparse F

Gemma Scope's 50.8% example used a dense encoder and dense decoder. Its usual
model-FLOP convention is:

```text
F_scope = 12 * N * d * M
```

At `N=16,384`, `d=7,168`, and `M=131,072`, this is 184.718 TFLOP/update and
requires about 49.318 ms/update to reach 51% on one eight-v6e-chip SAE lane,
or 6.165 ms if a single job genuinely used all 64 chips. The leading
qualification layout executes eight independent eight-chip calibrations on
the v6e-64 and reports both per-lane and aggregate utilization. We will
implement and execute that dense R1-activation calibration. Only that real
program is called the Scope-comparable result. Because an autodiff compiler
may prune the SAE input gradient that the conventional `12*N*d*M` model count
includes, the report also publishes an HLO-reconciled executed-work numerator.
The 50.8% comparison uses the named Scope convention; it is never silently
relabeled as the executed-work metric.

Production F is a different, intentionally more efficient algorithm: it
encodes all 131,072 candidates but decodes only the selected 64 columns. A
conventional sparse model count is:

```text
F_sparse = 6 * N * d * (M + k)
```

which is 92.404 TFLOP/update at the same shape. It is admissible only if the
compiled graph really executes those terms. Removing the unneeded gradient to
constant SAE inputs gives about 61.618 TFLOP/update; exploiting the fact that
TopK sends encoder weight gradients only to selected columns can reduce real
work to about 30.846 TFLOP/update. Rematerialized work is real hardware work
but does not inflate the model-FLOP numerator. The exact F specification is
therefore frozen from the actual forward/backward program before timing. It
does not count inactive decoder columns, a pruned encoder-input gradient, or
multiply nominal parameter count by six if the compiled program does not
perform those products. Nor will we deliberately execute a useless dense
gradient solely to increase MFU; that would contradict the cost claim.

We publish two reconciled views:

1. **Contract MFU:** frozen useful-matmul FLOPs divided by v6e BF16 peak and
   synchronized numerical-step wall time. Dense calibration publishes both
   the explicitly named Scope `12*N*d*M` convention and HLO-executed work;
   production sparse conditions use only HLO-reconciled actual work.
2. **Profiler evidence:** Xprof/compiled-HLO operation inventory, MXU duty,
   memory stalls, collective time, and input wait for the same windows.
3. **Scope-equivalent throughput:** reported for sparse F only as an explicitly
   counterfactual throughput number, never relabeled as physical model MFU.

If the formula and profiler disagree materially, qualification stops. A number
above 51% without a matching operation inventory is not accepted.

### Non-waivable gates

The prefix and actual dense calibration each need all of the following for the
intended >=51% systems headline. Sparse F inherits every integrity, stability,
input, HBM, and evidence requirement, but its performance threshold is >=51%
actual-work MFU when feasible or >=80% of a pre-registered empirical roofline
when exact TopK/selected-gradient communication makes peak-FLOP MFU the wrong
bound:

- one static compile for the entire measured shape;
- five warmups followed by synchronized real-data steps;
- three independent qualifying windows, each at least 500 steps and 60 seconds;
- for prefix and dense calibration, every window and the aggregate are at or
  above 51.0%; each sparse-condition window instead meets its frozen 51%-MFU
  or 80%-of-roofline gate;
- instrumentation changes throughput by no more than two percentage points;
- real-data throughput at least 90% of density-matched synthetic throughput;
- input wait below 5% of numerical wall time;
- no OOM, dropped routed token, recompilation, capacity overflow, or NaN;
- peak HBM safety margin at least 15%;
- p95 step time at most 1.10 times p50 and slowest-host step time at most 1.05
  times the median host;
- routed-capacity amplification at most 1.10;
- for harvesting, durable goodput at least 90% of device-output goodput and a
  bounded writer queue.

The report includes per-window wall time, token count, useful FLOPs, peak
assumption, compiler version, topology, HLO digest, profiler digest, and
confidence interval. There is no `minimum_mfu_exception` field in the R1
protocol.

A and B may be fundamentally latency/selection-bound because they perform much
less dense work. They still publish raw MFU, but their optimization gate is at
least 80% of a pre-registered, measured condition-specific roofline plus proof
that no full dictionary or full gradient is replicated. Sparse F uses the same
rule if its actual-work program cannot physically reach 51%; the dense
calibration proves the hardware/software stack's Scope-comparable utilization.
Lower raw sparse MFU cannot be used to hide extra work in the A/F chip-second
comparison.

## Performance architecture

### Prefix harvesting

The Gemma prefix was wrapped around a training-oriented model path and then
recomputed router outputs. R1 instead needs a native inference prefix:

```text
tokens
  -> embedding
  -> blocks 1..30, inference-only
  -> block 31 attention and residual
  -> native router + grouped routed experts
       |-- aggregate activation
       |-- exact expert IDs
       `-- exact scaled weights
  -> stop
```

The benchmark starts from MaxText's supported DeepSeek sharding but does not
inherit its full-training mesh by faith. Candidate layouts are derived from
the actual prefix tensor inventory. Expert parallelism is expected to
dominate, but `EP64` already occupies all 64 physical devices: tensor/FSDP are
not extra independent axes that can simply be appended. The same physical
8x8 axes must be reused with layer-specific sharding for attention and dense
weights, and every resulting layout transition belongs in the profiler and
wall-time budget. Reducing EP to create another logical axis is tested only if
the resident-weight inventory still passes the HBM gate. At a global valid
batch of 65,536 tokens, eight routes per token provide 2,048 routes per expert
on average, enough to make expert GEMMs large if dispatch remains balanced.

The exact prefix inventory is 21,930 logical tensors and 324,837,104,640
parameters. It occupies 325,944,454,407 bytes as complete released source
shards 1--78 and 649,674,223,616 bytes after conversion with BF16 weights and
28 FP32 router biases. The converted tensor payload averages 10.151 GB/chip
under perfect 64-way partitioning. That makes v6e-64 arithmetically plausible,
but it is not a fit proof: with EP64, resident routed weights alone are about
9.865 GB/chip, and replicating the 27 preceding shared experts plus all routers
adds about 2.442 GB/chip before attention/dense weights, executable,
activations, dispatch, collective, or capture buffers. Every accepted layout
must still leave 15% measured HBM headroom. EP64 assigns four routed experts
per chip. Because each DeepSeek routing group contains 32 experts,
expert/group IDs must be mapped to the physical 8x8 torus deliberately rather
than trusting logical device order. Expert weights stay resident; circulating
them between chips is already slower than the whole target step under an
ideal-bandwidth lower bound.

There are 27 completed routed blocks before the target. Each creates a
dispatch/compute/combine dependency, and a forward-only prefix lacks the
backward pipeline that helps hide communication in training. MaxText's batch
split/ring-of-experts schedule is therefore the leading starting point:
double-buffer token chunks and prove attention/dispatch/expert/combine overlap
in Xprof instead of assuming raw expert GEMMs dominate.

The pinned MaxText audit found that its batch-split implementation is a useful
kernel source, not an executable harvesting path. The checked-in YAML inherits
`batch_split_factor=1`, while the scan indexes a second microbatch and leaves
microbatches beyond the second unfinished; the current schedule is coherent
only at exactly two. It also omits packed-sequence segment IDs from attention,
casts the FP32 router parameters to the global BF16 dtype, remaps global expert
IDs to shard-local IDs, computes the shared expert at the target, and carries
custom-VJP, backward-residual, and host-offload machinery. Upstream tests only
parse the configuration; they do not execute this path. None of these facts is
accepted implicitly because the model is listed as supported.

The ordinary MoE path is the independent performance candidate, but it has two
separate drop controls. Positive `capacity_factor` masks assignments, and
positive `ragged_buffer_factor` truncates the sparse routing buffer. Setting
only `capacity_factor=-1` is therefore not a dropless proof. A practical
positive ragged bound needs an explicit required-row counter and a compiled
hard failure; a nonpositive bound allocates the mathematical worst case, which
can reach 32 times balanced occupancy for top-8 over 256 experts.

Two model trees keep correctness and speed separable. The oracle is an
ordinary unscanned prefix through 30 complete blocks plus target attention and
the target routed-only MoE. The performance tree scans the three dense blocks,
scans the 27 completed MoE blocks, then executes one separate routed-only
target; it removes custom VJP/backward state, preserves FP32 routing and global
IDs, and adds exact packed-segment masking. HLO and restored-parameter
inventories must prove that target shared weights, suffix layers, final norm,
logits, backward computation, and host residual transfers are absent.

Static BF16 prefix state alone is approximately 26.24, 17.72, 13.45, and
11.32 GiB per chip for EP64/FSDP1, EP32/FSDP2, EP16/FSDP4, and EP8/FSDP8,
respectively. These numbers exclude gathered layer weights, executables,
attention, routing expansion, and capture buffers. Qualification therefore
starts with EP16/FSDP4, then EP32/FSDP2 and EP8/FSDP8; EP64/FSDP1 is an HBM and
communication boundary, not the presumed winner. Both the repaired
forward-only batch split and the instrumented ordinary ring/ragged path must
be benchmarked before one becomes production.

At 65,536 valid tokens, one-copy BF16 dispatch plus combine across the 28
routed invocations moves about 6.577 GB/chip. Even the optimistic 800 GB/s
lower bound is 8.22 ms, roughly 10.05% of the entire 81.78 ms 51%-MFU step
budget, before torus hops, bisection, imbalance, attention layout collectives,
or synchronization. If routed communication costs twice that lower bound and
other overhead consumes 10%, the counted BF16 kernels must average about 73%
of peak. Qualification therefore requires weight-stationary pipelining of
`dispatch(k+1)`, `expert GEMM(k)`, and `combine(k-1)`; serialized routing is
not a viable candidate.

Required kernel properties:

- ragged or grouped expert GEMMs with no token dropping;
- no worst-case dense `[tokens, experts, ...]` materialization;
- no host synchronization in the captured forward path;
- capture IDs/weights in the same native router invocation;
- bounded asynchronous device-to-host buffers and direct regional GCS upload;
- prefix-only weights so post-target layers do not consume HBM or load time.

Qualification proceeds from an uninstrumented synthetic kernel microbenchmark
to full synthetic prefix, instrumented synthetic prefix, real in-memory input,
and finally real persisted capture. A stage that misses its threshold is
profiled and repaired; it does not advance by operator waiver.

Formal real-data windows may publish provisional immutable capture objects
under their predeclared input ranges. They remain invisible to the production
manifest while qualification is pending and are promoted only if every window
and persistence gate passes. A failed qualification leaves diagnostic objects,
never accepted production rows. This avoids discarding nearly a full harvest
per 60-second v6e window without weakening the launch boundary.

### SAE training

The leading layout is `job=8, feature=8`: the physical 8x8 slice runs eight
independent eight-chip SAE lanes, one per torus row. Each lane preserves the
scientific effective batch of 16,384 tokens, shards one complete dictionary
over its feature axis, and performs no data-parallel full-gradient all-reduce.
This mirrors the central Gemma Scope lesson—maximal latent sharding with the
minimum data parallelism needed—without changing 50M-token training from 3,052
updates to roughly 382 merely to make a large GEMM.

Qualification uses homogeneous lanes and publishes both each lane's
eight-chip MFU and the aggregate useful work divided by all 64 chips. An idle,
duplicate, padding, or unrelated filler lane contributes neither useful FLOPs
nor a hidden cost credit. Production cost records the eight chips assigned to
each scientific job and separately records the realized cost of the whole
allocated slice. Concurrent jobs are never relabeled as one 64-chip training
job.

Single-job `data×feature` candidates `1×64`, `2×32`, `4×16`, and
`8×8` remain diagnostic fallbacks. Pure feature parallelism minimizes
gradient communication but replicates the R1 activation batch to every feature
shard; increasing data parallelism reduces that H2D pressure while introducing
parameter replication and gradient all-reduces. A larger effective batch can
enter production only as a new frozen scientific contract, not as a systems
tuning trick. Uninventoryed replication of the 1.879B-parameter state is
rejected.

The exact distributed TopK is hierarchical over the lane's physical feature
axis (eight chips in the leading layout, or the declared 2D feature mesh in a
single-job fallback):

1. each feature shard returns its exact local top 64;
2. candidates merge along the declared feature dimension(s) with global
   latent IDs as the tie-breaker;
3. a 2D fallback performs the second-dimension merge; the eight-chip lane does
   not invent a cross-row collective;
4. only the exact global 64 survive.

Selected decoder columns are gathered from their owning shards. Each owner
forms a partial `[batch, 7168]` reconstruction; only those outputs reduce
across the feature axis. The implementation never broadcasts the full
dictionary, dense latent vector, or full gradient.

Adam, tangent-space decoder-gradient projection, column renormalization, and
moment updates are fused by column owner where compiler evidence shows a
benefit. Within the frozen effective batch, microbatching and double buffering
are swept until H2D/input time is just below update time; unnecessary scans are
removed if HBM permits.

A uses an expert/bank ownership map aligned with the mesh. Router IDs drive
native ragged dispatch to the eight visible banks. Its one fixed-order capacity
inventory is computed during cache materialization, content-addressed, and
consumed as an explicit input. Overflow is a hard failure.

## Global Spot system

No collective crosses a zone. One v6e-64 is one atomic executor: 64 chips
exposed through exactly 16 four-chip TPU VMs. Physical-host placement is
provider-owned and is neither counted nor trusted by the protocol. All 16 TPU
VMs join or the attempt does no accepted scientific work.

```text
small non-Spot controller
  |-- immutable scientific/implementation registry
  |-- CAS campaign snapshot and append-only event chain
  |-- fenced, expiring work leases
  |-- regional capacity requests and retry budgets
  |-- progress watchdog and phone/email incidents
  `-- manifest/cost/evidence reducer
          |
          +-- zonal v6e-64: harvest range(s)
          +-- zonal v6e-64: condition/seed job
          `-- zonal v6e-64: independent replacement attempt
```

Scientific work IDs contain corpus ranges or condition/seed/order identities,
never zone, host, process count, or device order. Implementation identity binds
the repository/runtime fields to the lowercase SHA-256 of the observed image
manifest; it does not use an ambiguous container-style `image_digest` for a
GCE boot image. Deployment identity is deliberately only four content links:
the implementation digest, deployment-envelope digest, complete device-
inventory digest, and attempt ID. Project, zone, Spot policy, service account,
MIG identity, topology, hosts, and devices live once in the envelope or
inventory rather than being copied into the deployment identity.

Every lease contains a monotonically increasing fence. After expiry, a new
attempt increments the fence. The old slice may finish uploading immutable
bytes, but it cannot update progress, checkpoint authority, or completion.
Exact orphaned content may be adopted only after every logical hash and
replica identity matches the expected receipt.

GCS stores immutable payloads and generation-guarded snapshots. The event is
written first and the snapshot is changed with compare-and-swap; a failed race
can leave an unreachable immutable event, never a falsely committed state.
Replay follows explicit event-parent hashes from the accepted head and never
relies on object listing order.

Each source checkpoint, prefix checkpoint, activation shard, and SAE
checkpoint has a logical SHA-256 plus one or more physical regional replicas.
Replica URI, generation, CRC32C, size, and region are deployment facts. A
regional loss changes placement, not scientific identity.

Spot correctness assumes zero guaranteed preemption notice. An exact metadata
`TRUE`, ambiguous metadata, lease expiry, or member loss may close the local
new-step gate, but it never authorizes an emergency checkpoint. Recovery uses
only the last fully sealed checkpoint already published through the
controller's generation-pinned CAS pointer; partial, in-flight, and sealed but
unpublished writes remain diagnostic. A successor starts only after the old
lease expires, the fence advances, and a fresh whole-MIG attempt is admitted.
The additive V2 protocol tests this transition while preserving the historical
75-second V1 implementation unchanged and non-authoritative.

## Image, lifecycle, and alerts

Workers boot a qualified TPU-compatible custom OS image with pinned Python,
JAX, libtpu, MaxText, package lock, repository revision, system service, and
runtime user. Startup does not run package upgrades and does not reboot. Model
and capture bytes remain external content-addressed inputs.

The image chain is `recipe -> observed manifest -> GCE publication ->
deployment envelope -> host attestations -> device inventory`. Recipes and
publications record the exact base/custom GCE numeric image IDs and canonical
self-links. A host records its boot disk's GCE identity, the numeric source-
image ID, and the manifest SHA-256; no fake boot-disk content digest is
invented. The inventory contains exactly 16 hosts and 64 unique raw JAX
`(coords, core_on_chip)` device identities for declared topology `8x8`.

Image acceptance includes a real final-SSH-disconnect test: the system service,
owned `/dev/shm` sentinel, and numerical watchdog must remain alive beyond the
image's logind cleanup interval. Every replacement slice emits a fresh host
and cache qualification even when reusing the same logical artifact inventory.

The controller, not the researcher's laptop, monitors progress. Phone/email
alerts include the immutable incident digest and bounded actions such as
"retry exact work," "switch this request from Spot to Flex-start," or "halt."
No repair agent may change model revision, target, corpus, geometry, seed,
optimizer, MFU formula, acceptance threshold, or report contract.

## Implementation sequence

### Phase 0: evidence and protocol

**Implementation status, 2026-07-23:** additive preemption V2 records and a
thread-safe local step latch now bind the exact lease, deployment identity, and
placement V2 envelope; freeze the first zero-notice whole-slice incident; and
derive recovery only from a prepublished CAS checkpoint pointer under a fresh
fence/MIG. Lease renewal is allowed only while the gate is open. The metadata
poller, slice-wide fanout, durable incident sink, checkpoint-pointer adapter,
and live preempt/recreate chaos test remain unimplemented, so this is not yet
production authority.

1. Finish Gemma and export the complete Filestore control/evidence bundle to
   immutable GCS.
2. Implement strict scientific, implementation, deployment, content, work,
   lease, progress, and receipt records.
3. Implement and concurrency-test CAS snapshots, immutable event chains,
   fences, expiry, stale-writer rejection, and exact orphan adoption.

Exit: deterministic local tests and a GCS race/chaos test reconstruct one
campaign from its accepted event head.

### Phase 1: immutable image and data identities

**Implementation status, 2026-07-23:** additive placement V2 records now
require a certified dated Ubuntu 22.04 base-image resolution, Spot `DELETE`, a
full 16-VM v6e-64 MIG, a fresh generation-pinned launch reference/template/MIG
per attempt, and controller-side API evidence. They permit cross-region
successors while holding image/runtime/shape policy fixed. The cloud response
adapter and a live create/preempt/recreate qualification have not run; V1 is
preserved unchanged and is not production authority.

1. Bake and qualify the v6e image.
2. Restage the same bounded C4 source sample with the pinned R1 tokenizer.
3. Make sequence/range IDs topology-neutral and produce regional replicas.

Exit: a fresh slice reaches the service-ready state without apt/pip mutation,
reboot, human credentials, or SSH-owned lifetime.

### Phase 2: checkpoint and model oracle

**Implementation status, 2026-07-23:** the exact inventory/mapping authority,
an isolated R1 runtime lock, a deterministic allowlisted image context, and a
bounded reference conversion planner/executor exist. The executor uses
deterministic immutable chunk IDs, authenticating-source and create-only-sink
interfaces, FP32 blockwise FP8 multiplication followed by one BF16 cast,
statically checked transform-array limits, a one-shard source/sink protocol,
and exact crash/resume receipts. Its evidence is synthetic only. The shared
project lock still points MaxText at the Gemma commit and remains intentionally
separate; `deploy/r1/runtime/` is the only accepted static R1 lock. Its Packer
context and systemd entrypoint validate offline but are explicitly `UNBUILT`
and `production_qualified=false`. Production sealing remains disabled: no
published/qualified R1 image, live Compute placement receipt, atomic 16-VM
runtime admission, controller-authenticated weight prestaging,
TensorStore/Orbax sink and restore, independent real-payload R1 conversion
oracle, full-model logit parity, or prefix/full parity has passed. A bounded
local safetensor adapter now verifies already-prestaged blobs and executes the
conversion core end to end, but deliberately cannot attest how those blobs
arrived or seal a checkpoint. A diagnostic local immutable output adapter now
also proves exact chunk publication/recovery, plan-bound storage finalization,
retry-order-independent source-observation evidence, and bounded flat-region
reconstruction. Its path-derived store identity and POSIX durability model are
explicitly local-only; it is not the production GCS/TensorStore/Orbax sink and
cannot make the production seal eligible. A separate additive GCS immutable
byte-store primitive now implements regional-bucket attestation, create-only
generation convergence, exact allowlisted metadata, CRC32C/SHA-256 receipts,
reserved content-address validation, and generation-pinned bounded verification
without listing. It uses exact immutable `bytes`; selects server-prevalidated
CRC32C without the pinned client's resumable corruption-delete path; rejects a
default CMEK; and grants production identity only to its internally constructed
exact-3.13.0 client. Its one-time bucket attestation assumes trusted operators
do not update bucket configuration while the store is live. It is not yet a
`ConversionOutputSink`, does not publish the topology-neutral logical manifest
or regional envelope, and has no JAX/Orbax restore path, real-bucket chaos
evidence, full-scale throughput evidence, or measured Python RSS bound.

1. Use the exact Hugging Face and MaxText source identities and file hashes in
   the checkpoint conversion contract; neither repository's moving `main` is
   an input.
2. Implement the fail-closed streaming conversion of the required 325.944 GB
   in source shards directly into the 649.674 GB mixed-BF16/FP32 prefix. Do not
   materialize a full intermediate BF16 Hugging Face checkpoint.
3. Pass an R1-specific Hugging Face/MaxText full-model logit oracle at
   `max_kl_div <= 0.03`; the existing upstream V3 test is not this evidence.
4. Prove the prefix-only checkpoint matches the full checkpoint at the target
   attention residual, normalized MoE input, native router IDs/weights, and
   routed aggregate on a frozen validation set.

Exit: exact full/prefix identities, parity evidence, memory inventory, and a
confirmed target tap.

### Phase 3: prefix performance

1. Microbenchmark dense, MLA, router, dispatch, and grouped-expert kernels.
2. Sweep mesh and batch candidates on one v6e-64.
3. Require at least 55% uninstrumented synthetic prefix MFU to leave margin.
4. Add the native capture tap; instrumentation may cost at most two points.
5. Pass all three real-data 51% windows and the persistence gates.

Exit: one exact configuration and profiler-backed evidence bundle. Nothing is
captured for production before this exit.

### Phase 4: SAE performance

1. Port A/B/F math against the single-device oracle.
2. Implement exact hierarchical TopK and owner-local sparse decode.
3. Prove sharding inventories: no forbidden parameter/gradient replication.
4. Tune batch/input overlap and fused owner-local optimizer updates.
5. Execute and pass the actual dense Scope-comparable calibration at 51%.
6. Qualify sparse F using its actual useful-work ledger: pass 51% if physically
   attainable, otherwise its pre-registered 80%-of-roofline gate; require the
   same condition-specific roofline evidence for A/B.
7. Pass interrupted-versus-uninterrupted and cross-zone restore parity.

Exit: fixed per-condition mesh plans and profiler-backed timing definitions.

### Phase 5: distributed scientific campaign

1. Harvest disjoint topology-neutral ranges on as many independently leased
   slices as quota permits; seal one exact global manifest.
2. Schedule the nine condition/seed jobs independently. Pair or randomize A/F
   placement across zone and time so regional conditions cannot bias `C`.
3. Run the blind health gates, full training, held-out evaluation, and report.
4. Publish both marginal algorithmic cost and realized Spot cost including
   failed attempts, queueing, restore, checkpoint, and cross-region transfer.

Exit: the `G/C` result, the prefix/dense-calibration/sparse-F ≥51% systems
results, and a self-contained
evidence bundle reconstructable without a live VM or filesystem.

## What DeepMind should be able to audit

A grant reviewer should not need to trust an anecdote or terminal screenshot.
The released evidence must answer:

- Which exact R1 bytes, model code, tokenizer, data, target, and SAE contract
  produced the result?
- Did the prefix really stop at the stated tap and route tokens exactly?
- What BF16 matmuls entered the useful-FLOP numerator, and which operations
  were deliberately excluded?
- Did the same 500-step windows produce the wall time, profiler trace, HLO
  inventory, input wait, token count, and MFU claim?
- Did Spot retries preserve identical scientific work and matched A/F order?
- How much useful work, waste, queue time, checkpoint time, and data movement
  did every attempt consume?
- Can a stale or partially alive slice have influenced accepted state?
- Does A's reconstruction/cost advantage survive all three matched seeds and
  a held-out common SST?

That is the renewal story: not merely that conditional dictionaries are
cheaper, and not merely that free quota made a large run possible, but that a
new conditional representation was demonstrated on a 671B routed model with
auditable exactness and greater-than-Gemma-Scope-class hardware utilization on
globally preemptible infrastructure.
