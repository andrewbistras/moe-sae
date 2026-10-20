# Experiment 2: Expert-Aligned Contribution SAEs

Status: implementation contract, 2026-07-24

Protocol ID: `gemma4-contribution-v1-20260725`

## Decision

Experiment 1 answered its implemented question and falsified its implemented
form of condition A. It did not test the original expert-decomposition
hypothesis.

At the final checkpoints, the three-seed held-out means were approximately:

| evaluation | mean EV |
|---|---:|
| A, true routes | 0.704375 |
| B | 0.705844 |
| A, shuffled routes | 0.340308 |
| A, all banks visible | 0.001637 |

The immutable Gate-1 report is bound to:

```text
protocol_sha256 = 142102c565fe024e476d537f92e845dd149294c168b42324c16b619733bf094d
report_sha256   = edccff1ca9e57cc13febac0c9836d0df981f9b7fca6eb1cbbb299f29b8fd9a82
```

True routing beat inference-time shuffled routing by 0.364067 EV on average,
with a positive paired sequence-cluster interval in every seed. Exposing all
banks reduced EV by 0.702738 on average. A also trailed B on the fixed
stratified training sample, so ordinary train/eval overfit was not established
as the explanation.

These results show that the banks learned a real router-keyed representation.
They do not show that expert identity improved reconstruction. In Experiment
1, every visible bank encoded the same already-aggregated, post-RMSNorm
activation. The loss constrained only the sum of eight bank reconstructions;
it never required bank `i` to reconstruct expert `i`.

Experiment 2 moves the SAE boundary to the last point where Gemma's expert
decomposition is mathematically real.

## Question and falsifiable claims

The primary question is:

> When each selected expert's SAE reconstructs that expert's own weighted
> pre-normalization contribution, does stable expert identity improve
> held-out reconstruction over route-rank dictionaries, and how much of a
> full component dictionary's gain does it recover at what measured cost?

The experiment makes three separate claims and does not substitute one for
another:

1. **Contribution decomposition:** independently reconstructing the eight
   routed contributions should produce an accurate recomposed routed branch.
2. **Expert identity:** true expert-ID banks should outperform an otherwise
   identical per-sequence randomized bank assignment.
3. **Efficiency:** sparse expert-ID A should recover a positive, stable
   fraction of the full-width F-over-B gain at lower measured SAE chip-seconds.

A large true-versus-random routing effect is required for claim 2. A large
inference-time shuffle effect alone is insufficient. A must beat B on the
common held-out target for an architectural gain. If A only depends on its
router key but does not beat B, Experiment 1's conclusion is reproduced.

The actual shared-expert branch is a separate target. It was not captured or
trained in Experiment 1, and no claim about shared-versus-routed SAE properties
is valid until the shared analysis below is complete.

## Exact Gemma computation

The model, immutable checkpoint revision, C4 snapshot, tokenizer, target block,
and MaxText base commit remain those in `docs/EXPERIMENT.md`:

```text
model              google/gemma-4-26B-A4B
model revision     24548b62aa021d562695c04aaf7758a1ea47990b
MaxText base       48c7da3e28e8f57fe36ddf09d00eec53072a2ea0
target             one-indexed block 15
hidden width       2816
routed experts     128
selected experts   8
shared experts     1
```

For one token, let `h` be the block-15 post-attention residual. For selected
router slot `s`, let:

```text
i_s       selected expert ID
g_s       normalized router weight
e_s       selected expert's raw output after its learned output scale
c_s       FP32(g_s) * FP32(e_s)
u         sum_s c_s
r         routed_norm(u)
q_raw     raw shared-expert output
q         shared_norm(q_raw)
y         outer_ffn_norm(r + q)
```

The routed scientific target is the tuple `(c_1, ..., c_8)`. The exact
downstream routed reconstruction is:

```text
r_hat = routed_norm(sum_s c_hat_s)
```

The exact full-FFN reconstruction is:

```text
y_hat = outer_ffn_norm(r_hat + q_hat)
```

The residual addition and layer scalar are excluded from the scientific target
because the unchanged residual would dominate reconstruction statistics. They
remain part of the full-block instrumentation parity check.

The following are invalid:

- applying `routed_norm` to each expert separately;
- defining post-RMSNorm per-expert contributions;
- summing separately normalized expert vectors;
- recovering individual contributions from the old aggregate capture.

## MaxText capture seam

Pinned MaxText currently restores selected expert outputs to
`[batch, sequence, top_k, hidden]` inside the routed MoE `unpermute` path, then
immediately multiplies them by router weights in FP32 and reduces them. Its
public routed call returns only the aggregate, load-balancing loss, and bias
updates.

Experiment 2 will use a pinned fork whose parent is the exact MaxText base
commit. The fork adds one opt-in auxiliary return:

```text
return_selected_outputs: bool = False
```

The default three-result API and existing aggregation expression must remain
unchanged. Only the target block requests the auxiliary value. The auxiliary
path is initially qualified only for the existing non-ring sparse path; ring
of experts, fused-vLLM, or any other untested path fails closed.

This repository will not duplicate MaxText's private routing, ragged
all-to-all, grouped-matmul, and sharding implementation.

The pinned fork also carries the route-permutation optimization proved during
the DeepSeek systems work: the sender-major to expert-major permutation is
constructed directly from the small all-to-all count matrix, and inverse
permutations use a linear scatter. The measured path therefore does not sort
the padded dropless ragged buffer or run a second argsort merely to invert a
known permutation. Both transformations have exact parity tests and are part
of the fork source hash, StableHLO identity, and MFU receipt.

The shared expert already has an independent callable path and requires no
kernel modification.

## Capture schema and exactness

Experiment-2 capture is a new schema. A v1 aggregate shard must never be
auto-detected or reinterpreted as a contribution shard.

| field | wire type and shape |
|---|---|
| `expert_output` | BF16 `[N, 8, 2816]` |
| `expert_ids` | uint8 `[N, 8]` |
| `router_weights` | native BF16 `[N, 8]` |
| `shared_output` | BF16 `[N, 2816]` |
| `sequence_id` | int64 `[N]` |
| `position` | uint16 `[N]` |

`expert_output` is stored before multiplication by the router weight.
Pre-multiplying and then rounding every small contribution to BF16 would add a
second avoidable quantization. Consumers define:

```text
contribution = FP32(router_weights) * FP32(expert_output)
```

The selected output already includes Gemma's learned per-expert output scale.

A small, content-addressed recomposition sidecar stores the exact block-15
routed RMSNorm scale, shared RMSNorm scale, outer FFN RMSNorm scale, layer
scalar, dtype policy, accumulation order, MaxText fork identity, and source
fingerprints. SAE consumers must be able to recompose without loading the 26B
checkpoint.

Before a manifest can seal, all of the following must pass:

1. Capture mode's native aggregate is array-equal to the unchanged routed call.
2. FP32 recomposition from selected outputs and weights reproduces the native
   raw routed aggregate.
3. Applying the frozen routed RMSNorm reproduces the existing sealed v1
   aggregate target to the predeclared serialization noise floor.
4. Applying both branch RMSNorms reproduces the unmodified Gemma MLP output.
5. Outer normalization, residual addition, and the layer scalar reproduce the
   full unmodified target block.
6. BF16 safetensors round-trip preserves every stored operand bit-for-bit.
7. An independent tiny oracle evaluates the selected expert weights without
   the grouped routing path and agrees slot by slot.
8. Eight expert IDs are unique and in `[0, 127]`; weights are finite,
   non-negative, and normalized; routed tokens are never dropped.

The v2 manifest binds the corresponding sealed v1 capture manifest and exact
token ranges. This gives an independent post-RMSNorm target and prevents token
misalignment. Resume remains keyed by immutable input-range IDs.

The staged captures use one frozen logical-to-source range plan per split.
The sealed v1 artifact is arranged as contiguous per-host lanes, so each phase
takes the same prefix quota from every lane and deterministically interleaves
host-owned chunks into the logical v2 token order. Every assignment binds both
its logical v2 range and exact source-v1 range. This makes the 100k/25k smoke
exercise every TPU host rather than silently sampling only the first host's
lane.

Phase boundaries are permanent shard boundaries. Extending 100k to 2M never
widens a partial boundary shard or changes an earlier range ID. Each phase is
sealed under a new immutable control root, while retaining the same
content-addressed bulk-object prefix. The new root verifies and adopts the
prior phase's sealed receipts, then appends only the new suffix; it neither
mutates the earlier manifest nor copies or re-uploads its bulk shards. The
range-plan hash, source manifest hash, phase total, and adopted-parent manifest
hash are recorded in the phase evidence before a later gate can consume it.

The record is approximately 50,722 bytes per scientific token. At full scale:

```text
50M train       2.5361 TB
2M held-out     101.444 GB
total           2.637544 TB (about 2.399 TiB)
```

V2 shards contain at most 16,384 tokens, approximately 0.774 GiB each, rather
than reusing v1's 131,072-token shard limit.

Every shard carries FP64 counts and sufficient statistics for:

- one pooled raw-expert-output mean used by every routed condition;
- weighted contribution SSE/SST;
- router-rank and expert-ID counts and energy;
- the raw and normalized shared branch;
- exact routed and full-branch recomposition checks.

There are no per-expert centering means in the primary experiment. Giving A a
different mean for every expert would add side information that B and F do not
receive.

## Objective-matched A/R/B/F matrix

Selected experts are canonically ordered by descending router weight, with
expert ID as the deterministic tie-breaker.

| condition | bank identity | banks × width | total features | candidates/token |
|---|---|---:|---:|---:|
| A | true expert ID | 128 × 1,024 | 131,072 | 8,192 |
| R | per-sequence randomized expert ID | 128 × 1,024 | 131,072 | 8,192 |
| B | router rank | 8 × 1,024 | 8,192 | 8,192 |
| F | router rank | 8 × 16,384 | 131,072 | 131,072 |

Every condition reconstructs the same eight raw selected expert outputs.
Every contribution uses exact per-contribution TopK-8, so every valid source
token has exact total L0=64. Decoder columns reconstruct the raw expert output;
router weights are applied in FP32 when computing the scientific error and
recomposition.

This matrix preserves the original comparisons:

- A/B are candidate-matched.
- A/F are total-feature and parameter-matched.
- A evaluates 8,192 of 131,072 total features per token: 6.25% exposure.
- A and F have matched average examples per feature under balanced routing.

F is a full-width, route-rank-conditioned component dictionary. It is not an
unrestricted global aggregate oracle, and the report must say so.

R is a training-time causal control. For each sequence, a keyed deterministic
bijection maps expert IDs to bank IDs, preserving eight unique banks and
balanced load while destroying stable expert meaning. The same rule is used
during training and held-out evaluation. A single fixed global permutation
would merely rename banks and is invalid. Inference-only shuffling is also
invalid as the training-time causal control.

All four conditions use the same pooled mean, token order, initialization
policy, optimizer, schedule, dtype policy, update count, and checkpoint
cadence.

The optimizer contract is Adam with `beta1=0.9`, `beta2=0.999`,
`epsilon=1e-8`, no weight decay, FP32 moments and reductions, BF16 parameters
and matmuls, global gradient clipping at `1.0`, tangent-space decoder-gradient
projection, and decoder-column renormalization after every update. The
effective batch is 16,384 source tokens accumulated from four 4,096-token
microbatches. Learning rate warms linearly to `3e-4` and then decays by cosine
to `3e-5`.

Warmup length is phase-specific so that no pilot repeats Experiment 1's
all-warmup failure:

| training phase | tokens | updates | warmup updates | masked tail tokens |
|---|---:|---:|---:|---:|
| Phase 2 scaled seed 0 | 2,000,000 | 123 | 12 | 15,232 |
| Phase 3 full-width seed 0 | 10,000,000 | 611 | 61 | 10,624 |
| Phase 4 confirmatory | 50,000,000 | 3,052 | 250 | 3,968 |

Phase 1 has no SAE training schedule. Phase-2 scaled dictionaries start from
fresh initialization. Phase-3 full-width dictionaries also start from fresh
initialization because their shapes differ; they are not widened Phase-2
checkpoints.

The common training loss is normalized weighted-contribution SSE:

```text
SSE_c = sum_tokens sum_slots ||g_s * (e_s - e_hat_s)||^2
EV_c  = 1 - SSE_c / SST_c
```

The eight slots from one source token are never treated as eight independent
statistical samples. No condition-specific local loss, threshold calibration,
dead-feature auxiliary loss, or post-hoc score normalization exists.

## Shared-expert analysis

The shared expert is characterized rather than credited uniquely to A.

For each training seed, one global shared SAE `Q` reconstructs the actual
normalized shared term:

```text
q = shared_norm(shared_output)
```

The initial Q shape is 8,192 features with TopK-64, matching B's width and
sparsity. Q is trained once, frozen, and used identically with A, R, B, and F
for the learned-shared full-FFN metric. Its chip-seconds are reported
separately and in whole-system totals.

`Q0` in the Phase-2 pilot is this same 8,192-feature, TopK-64 shared
dictionary, not a width-scaled routed dictionary. It is common to every
condition, separately costed, and may continue training on the monotonically
extended capture in Phase 3. This choice cannot change a routed A/R/B/F
comparison.

The shared analysis reports EV/NMSE, dead rate, firing-frequency entropy and
Gini, decoder coherence, effective rank, and learning curve. Exposure-matched
subsamples are mandatory when comparing Q with a routed expert bank because
the shared expert fires on every token while one routed expert sees about
6.25% of tokens.

An independently trained, condition-specific global residual SAE is forbidden:
it could absorb routed error and erase the treatment difference.

If the routed experiment passes, a later hybrid condition will split the
8,192-candidate and L0-64 budget explicitly:

```text
8 routed banks × 512 candidates, TopK-4 each = 4,096 candidates / L0 32
one shared dictionary × 4,096 candidates, TopK-32 = 4,096 candidates / L0 32
```

The hybrid has 69,632 total features and is compared with a global 8,192-width
B and a global 69,632-width parameter-matched F on the exact full-FFN target.
This phase is not launched unless the routed contribution matrix passes.

## Evaluation and report

The primary endpoint is pooled weighted-contribution EV, `EV_c`, on the sealed
held-out contribution capture.

Mandatory downstream endpoints are:

```text
EV_r_oracle =
  EV(routed_norm(sum c_hat_s), existing routed target)

EV_y_oracle_shared =
  EV(outer_norm(routed_norm(sum c_hat_s) + exact q), exact full-FFN target)

EV_y_learned_shared =
  EV(outer_norm(routed_norm(sum c_hat_s) + q_hat), exact full-FFN target)
```

Component improvement is not assumed to imply full-branch improvement. The
sign of A-minus-B must agree on `EV_c` and `EV_r_oracle` for a headline.

The primary efficiency quantities are:

```text
G2 = 100 * (mean(EV_c_A) - mean(EV_c_B))
           / (mean(EV_c_F) - mean(EV_c_B))

C2 = 100 * median_seed(chipseconds(A) / chipseconds(F))

Delta_route = mean(EV_c_A - EV_c_R)
```

`G2` is never clipped. It is undefined when mean F-minus-B is non-positive.
If the paired interval for the denominator includes zero, the ratio is
unstable/exploratory.

Marginal SAE chip-seconds include all synchronized optimizer work, input
stalls, fixed capacity scans, and condition-specific preprocessing. They
exclude compilation, checkpoint serialization, evaluation, the common
component harvest, and common Q training. Those costs are separately reported,
plus one-seed and three-seed harvest-amortized and Q-amortized totals.

Every report includes:

- separate sealed train and held-out manifest hashes, exact split token counts,
  monotonic parent/adoption lineage for both splits, and every run's
  condition/layout/randomization/schedule input receipt bound to the train
  manifest;
- per-seed `EV_c`, recomposed `EV_r`, and full-FFN EV;
- exact per-contribution and total L0;
- dead rate and fired-feature coverage;
- micro contribution-energy-weighted and macro expert-balanced EV;
- EV by router rank, router-weight bin, and expert-frequency bin;
- A-minus-R and A-minus-B paired effects;
- candidates, total features, exposure, chip-seconds, tokens/s, and wall time;
- harvester useful-prefix MFU, device-output goodput, persisted goodput, and
  writer backlog;
- separate shared-Q properties and cost.

Statistics use 20,000 paired sequence-cluster bootstrap replicates. All eight
contributions from a token stay together, and all tokens from a packed
sequence stay together. Per-seed effects are primary; a hierarchical
seed-plus-sequence interval is secondary. Tokens are never pooled across seeds
as though they were independent training replicates.

## Staged execution and stopping gates

### Phase 0 — bounded existing-checkpoint forensics

On a fixed 250,000-token held-out subset per existing A seed, measure:

- in-route versus out-of-route score quantiles and native-rank margins;
- all-bank selected-bank entropy, Gini, effective bank count, and domination;
- per-bank decoded-vector norms, pairwise dot products, and cancellation ratio;
- leave-one-route-out and deterministic one-bank substitutions by router rank;
- a ranking-only in-domain CDF calibration diagnostic.

This phase gets a strict runtime bound and may run in parallel with adapter
work. It can distinguish score miscalibration from cancellation, but it cannot
replace fresh contribution capture and cannot gate whether the adapter is
built.

### Phase 1 — 100k numerical and writer smoke

Capture 100,000 predeclared training tokens and a disjoint 25,000 held-out
tokens. The user-authorized exploratory Experiment-2 workload may run directly
for at most 3,600 seconds. Qualification remains available and has a
600-second wall-clock kill deadline; work beyond one hour still requires an
exact self-hashed receipt for the compiled graph, hardware topology, runtime,
checkpoint/config identity, mesh, dtype policy, and static batch shape.

Then require:

- every exactness gate above;
- deterministic interrupted-versus-uninterrupted shard identity;
- multi-host no-gap/no-duplicate range accounting;
- bounded writer queue and no routed-token drops;
- persisted goodput at least 60% of device-output goodput;
- no unexpected recompilation after warmup;
- auxiliary capture tokens/s at least 85% of the same graph with its writer
  disabled.

The Gemma Experiment-2 useful-MFU admission floor is 3%. This narrowly scoped
override was frozen after the real B128 qualification measured 3.65598% useful
MFU, 63.88945% persisted/device goodput, and 90.08767%
auxiliary/writer-disabled throughput. The conservative exact floors are 3%,
60%, and 85%, respectively. They are cost-containment admission floors, not
scientific performance claims, and all absolute measurements remain recorded.
They do not modify or satisfy the separate DeepSeek >=51% MFU target.

### Phase 2 — seed-0 mechanism pilot

Capture exactly 2,000,000 training tokens and 250,000 held-out tokens using
versioned, immutable manifests. Seed 0 is permanently exploratory and excluded
from the headline.

Use the scaled matrix:

| condition | banks × width | total features | candidates/token | TopK |
|---|---:|---:|---:|---:|
| A0 | 128 × 256 | 32,768 | 2,048 | 2 per contribution |
| R0 | 128 × 256 | 32,768 | 2,048 | 2 per contribution |
| B0 | 8 × 256 | 2,048 | 2,048 | 2 per contribution |
| F0 | 8 × 4,096 | 32,768 | 32,768 | 2 per contribution |

Train A0/R0/B0/Q0 first. F0 must pass compile and a short numerical smoke
before its full pilot. Relative quality is not inspected before every
condition passes the finite-state, exact-L0, no-overflow, and falling-loss
health gate.

The pilot passes only when:

1. A0-minus-R0 and A0-minus-B0 are positive;
2. their paired sequence-cluster 95% intervals exclude zero;
3. F0-minus-B0 is positive with an interval excluding zero;
4. A0 recovers at least 10% of the stable F0-over-B0 `EV_c` gain;
5. A0-minus-B0 has the same sign on exact recomposed routed EV;
6. every condition meets coverage and numerical-health requirements.

Failure stops scaling. Diagnostics are reported; objectives, widths, and
normalization are not silently changed.

### Phase 3 — full-width 10M seed-0 pilot

On Phase-2 pass, publish immutable 10M-train and 1M-held-out manifests that
reuse already validated content-addressed shards. Run full-width A/R/B/F and Q
for seed 0 with checkpoints at 2M, 5M, and 10M tokens.

Require all Phase-2 directional gates, `G2 >= 10%`, and `C2 < 50%`. The A
learning curve must not show a material late collapse. A material late collapse
is frozen as final held-out `EV_c` more than `0.02` below the best earlier
checkpoint-bound held-out `EV_c` on the same sealed evaluation capture. The
curve and this decision are derived by the report from checkpoint evidence,
never supplied as an operator boolean. Only this gate authorizes confirmatory
seeds.

### Phase 4 — blind confirmatory production

On Phase-3 pass:

- extend via a new immutable manifest to 50M train and 2M held-out tokens;
- train seeds 1, 2, and 3 in matched order;
- retain 2M, 5M, 10M, 20M, 30M, 40M, and 50M checkpoints;
- do not inspect relative held-out quality until all conditions pass health;
- produce the frozen three-seed report without tuning on seeds 1–3.

Seed 0 and all pilot decisions remain visible but never enter the headline.

## Performance, storage, and cost controls

Every accelerator entry point has two modes:

1. a direct exploratory workload whose lease is killed at 3,600 seconds; and
2. an extended run that requires a passing receipt for its exact execution
   identity. The optional qualifier remains bounded to 600 seconds.

The receipt derives useful MFU from explicitly enumerated, non-padding FLOPs
and synchronized device time divided by the documented peak of the chips
actually present. It also binds compile count, executable/HLO identity,
topology, runtime image, source revision, MaxText fork, checkpoint, mesh,
condition/layout, dtype, static shapes, and instrumentation mode. It is invalid
after any identity drift. OOM, recompilation after warmup, token dropping,
non-finite state, capacity overflow, or an over-limit writer backlog fails the
qualification regardless of MFU. The supervisor enforces the wall deadline
and records termination rather than trusting cooperative worker exit.

Harvester admission tries only three serialized static-shape candidates, all
on the historically strongest `EP8/FSDP2/TP1` mesh:

```text
B128: per-device sequence batch 8
B64:  per-device sequence batch 4
B32:  per-device sequence batch 2
```

`B128` is first because it is the measured path that cleared the 3% floor; the
old B32-scale path peaked near 2.45% useful MFU and remains inadmissible. A
candidate receives at most 600 seconds including compilation.
Only an explicit OOM or static-headroom rejection advances to the next smaller
batch; identity, transport, parity, timeout, and unknown failures abort. The
candidates never overlap, and the first passing receipt ends the sweep.

The frozen attention setting is explicitly `autoselected`, which selects
MaxText's TPU-native Splash/Flash implementation at sequence length 1,024.
Both qualification and harvesting pass that value rather than inheriting a
mutable default, and the observed effective runtime must report the same value.
The attention kernel does not change the scientific capture boundary, but it
is part of the exact graph identity and materially affects the admission MFU.

The target shared branch adds only about 1.1% to the old prefix FLOPs. The main
risk is materializing and persisting the selected tensor:

```text
about 396 MiB total contribution output per chip per B128 step
about 1.55 GiB per controller per B128 step
about 6.19 GiB aggregate per B128 step
```

Those figures include all eight raw expert outputs, the shared output, router
IDs/weights, and token identity. The writer therefore splits each
controller-local step into independently validated shards of at most 16,384
tokens and bitcasts native BF16 directly to its exact uint16 wire
representation; it must not expand capture-sized tensors through FP32 merely
for serialization or finiteness checks.

Benchmark the exact auxiliary-return graph rather than extrapolating from the
aggregate harvester. Host assembly, FP64 statistics, hashing, and compiler
fusion are part of the observed goodput.

Bulk data remains region-affine and content-addressed. Host-local buffers are
bounded and refillable; GCS is authoritative for bulk shards; the strongly
consistent POSIX control plane stores only identities, locks, receipts, and
sealed manifests. No worker scans whole bucket prefixes to infer progress.
Resume uses immutable logical-plus-source range IDs and monotonic publication.
Partial shards never count as progress. Phase extension adopts hash-verified
receipts into a new control root and uploads only the newly authorized suffix,
so scientific scaling does not multiply GCS storage or inter-region traffic.

The capture stages prevent an immediate 2.64-TB commitment:

```text
100k smoke                  about 5.1 GB
2M + 250k pilot             about 114.1 GB
10M + 1M full-width pilot   about 557.9 GB
50M + 2M production         about 2.64 TB
```

Training reads one process-contiguous resident-cache lane per controller. At
the approximate 50,722-byte record size, the train payload is about 25.361 GB
(23.619 GiB) per controller in Phase 2, 126.805 GB (118.096 GiB) in Phase 3,
and 634.025 GB (590.482 GiB) in Phase 4, before filesystem headroom and at most
two boundary-shard overlaps. A 100-GiB cache or `/dev/shm` is therefore
categorically insufficient for Phase 4. Provision at least 700 GB of usable
local SSD per controller, preferably a 1-TB volume; add about 25.4 GB if the
held-out lane is colocated. The prefetch receipt records the exact
manifest-derived bytes and fails before downloading when the configured cache
bound cannot contain its lane.

GCS request counts, bytes written/read, storage class, regional placement, and
estimated storage/operation cost are recorded before each escalation.

## Test contract

Required tests include:

- pinned-fork default-path parity against unmodified MaxText;
- selected-output auxiliary-return sharding and slot-order parity on EP16;
- independent direct-expert oracle parity;
- exact routed, shared, MLP, and full-block recomposition;
- v2 BF16 serialization and sidecar hash validation;
- v1/v2 schema rejection in both directions;
- deterministic per-sequence randomized bijections with balanced loads;
- NumPy/JAX parity for A/R/B/F scoring and weighted contribution loss;
- exact per-contribution TopK-8 and total L0-64;
- tail padding, rare-expert coverage, capacity overflow, and no token drops;
- interrupted-versus-uninterrupted capture and training identity;
- preemption after upload but before receipt publication;
- common SST, sequence-cluster bootstrap, unstable denominator, and unclipped
  ratio tests;
- synthetic end-to-end smoke from selected outputs through the final report.

## Invalid shortcuts

The following do not produce confirmatory Experiment-2 evidence:

- comparing a newly contribution-trained A with old aggregate-trained B/F as
  the primary matched result;
- treating old all-bank failure as proof that the banks contain no
  complementary information;
- using evaluation-only route shuffling as the causal training control;
- assigning a fixed random global bank permutation;
- defining post-RMSNorm per-expert contributions;
- training a different shared SAE for each condition;
- treating eight contribution rows from one token as independent samples;
- tuning on seed 1 and later calling it confirmatory;
- scaling capture or F before the preceding gate passes.

Old B/F checkpoints may be used only as a clearly labeled, directional bridge
on the common post-RMSNorm routed target. They save time during debugging but
do not satisfy the objective-matched A/R/B/F contract.

## Implementation order

1. Pin the MaxText fork and add the opt-in selected-output return with
   default-path parity tests.
2. Add the component adapter and exact block-15 recomposition oracle.
3. Add an explicit v2 artifact schema, 16,384-token shards, statistics, and
   old-schema rejection.
4. Extend the harvester's sharding, compaction, bounded writer, manifests, and
   useful-FLOP accounting.
5. Implement A/R/B/F component scoring and training with one shared objective.
6. Implement Q and the oracle-shared/learned-shared recomposition metrics.
7. Implement the paired sequence-cluster report and stage gates.
8. Run Phase 1; proceed monotonically only on pass.

DeepSeek-R1 work consumes no production slice for this experiment until the
Gemma full-width seed-0 gate establishes that the representation boundary,
not merely the systems implementation, is worth scaling.

## Phase-2 mechanism-pilot execution record

The operator explicitly bypassed the staged qualification and escalation gates
on 2026-07-25 in order to obtain the direct 2M-train/250k-eval mechanism result.
This execution is therefore a single-seed pilot, not the frozen protocol's
multi-seed confirmatory result.

The sealed execution identity is:

```text
training namespace:
  phase2-direct-986193a60ad3
training code:
  986193a60ad310bba60aa4f5cd1c65b2247aeac5
evaluation/finalization code:
  67c2c75c3fa2acc4b65f9c67fd6660da7655a6c6
training capture:
  a2f23aef6b9121e81a5bfea473c64039ca5bf6b073e4f70f4703938a1d8a6903
held-out capture:
  31454e5ce8693521c41a2a95f2ef88d40c5d26c726a0efac5c26d882ed341f5e
scientific result:
  67d0e913537fa414ff23193ee3c9ca99e3ab9ea4bbdff9cccf240ccc1e04f7b0
common-SST receipt:
  cb96cb78957c8801cae67e2f50d541085099047c2c1af4dce19739080291d7cc
result file SHA-256:
  b7e39a29e510725e70835eb0dd88a1af7b159d505f97d6471e01acbd7d5b3011
```

All five checkpoints completed exactly 2,000,000 scientific training tokens.
A/R/B/F evaluated with total L0 16 and Q with L0 64 on the same 250,000
held-out tokens in 625 packed-sequence clusters. The four evaluator processes
exited zero, every checkpoint pointer and score-manifest object resolved, and
the result and denominator-receipt hashes independently recomputed.

The held-out component endpoint was:

| Condition | EV |
|---|---:|
| A | 0.3195047562 |
| R | 0.1000750821 |
| B | 0.2978687594 |
| F | 0.3109305831 |

The paired packed-sequence effects were:

| Contrast | EV effect | Paired 95% interval |
|---|---:|---:|
| A - R | +0.2194296741 | [+0.2143140416, +0.2243696004] |
| A - B | +0.0216359968 | [+0.0207357018, +0.0225976964] |
| F - B | +0.0130618237 | [+0.0125458622, +0.0135587845] |

Thus the F-over-B denominator is positive and stable, and:

```text
G2 = 165.64300097%
```

A also beats B at the recomposed routed endpoint by +0.1409539863 EV, with a
paired 95% interval of [+0.1357821629, +0.1459360019]. Q reaches held-out
EV 0.5912710448. Together with the large A-over-R effect, this is strong
evidence for the mechanism: route identity carries causally useful
factorization information, and the expert-aligned component objective turns
that information into reconstruction quality which neither a random alignment
nor the candidate-matched global baseline recovers.

The measured marginal cost result does **not** pass:

```text
A training chip-seconds = 37,015.638
F training chip-seconds = 19,434.918
C2 = 190.45945033%
```

The excess is explained by the legacy A capacity scan: 1,095.854 wall-seconds,
versus 1,217.624 seconds in A's optimizer interval. F's optimizer interval was
1,214.682 seconds. Removing the scan after the fact would give an optimizer-only
A/F ratio of about 100.24%, but that is diagnostic, not a replacement for the
sealed measured C2. Commit `f1536d0d62ba3bec504e753e273c3c3755d6f313`
accelerates future routed-capacity scans; it cannot retroactively change this
run's timing evidence.

Final aggregation initially rejected tiny target-only component-SST differences
between the grouped A/R graph and global B/F graph. The scientific token
inventory, routed SST, centering identity, and all hashes were exact; the
component difference was at most two binary32 ULPs after rounding. Commit
`67c2c75c3fa2acc4b65f9c67fd6660da7655a6c6` selects A's per-sequence
target-only SST as the deterministic common denominator, records a self-hashed
canonicalization receipt, and still fails closed outside a four-binary32-epsilon
tolerance.

The complete 936,208-byte evidence tree and operator logs are redundantly
retained under:

```text
gs://moe-sae-bucket/moe-sae/gemma4_v1/experiment2/evaluation/evidence/phase2-direct-986193a60ad3-seed0/
```

This pilot therefore establishes the representation mechanism but not the
efficiency headline. The decisive next measurement is a fresh, optimized A/F
timing run on the same sealed captures, with the accelerated scan in the timed
binary and no protocol expansion. If optimizer time remains at parity, the
next systems target is the A grouped encode/decode path itself rather than
additional scientific controls.
