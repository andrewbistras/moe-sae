# Experiment 3: Expert-Aligned Contribution Transcoders

Status: frozen seed-0 mechanism-pilot v2 reset, 2026-07-27

Protocol ID: `gemma4-contribution-transcoder-v2-20260727`

## Pre-result protocol reset

The first production attempt exposed a flaw in the v1 training contract
before any held-out evaluation or A/R/B/F contrast was produced. Condition F
completed its training loop under v1, but condition A stopped before update 21
after 20 healthy updates (327,680 tokens). The fail-closed error was:

```text
invalid transcoder inputs or fewer than k positive features
```

The paired input path had already validated finite source and target tensors,
router weights, bank ranges, bank uniqueness, and token alignment. Capacity
overflow has a separate hard error. The remaining failure mode was therefore
one v1 made possible: a 256-wide expert bank could present fewer than two
strictly positive ReLU activations for a valid component. The hard stop was
correct under v1; padding TopK with zero-valued features would not have been.

Before inspecting a scientific result, v1 was retired and replaced by this v2
contract. V2 uses softplus encoder activations for all four conditions. This
makes strict-positive TopK a model invariant instead of a probabilistic hope.
The implementation saturates underflowed softplus outputs at the smallest
normal positive FP32 value so accelerator subnormal handling cannot silently
reintroduce zero. Ties still resolve by ascending global latent ID.

This is a full training reset: the v1 F checkpoint is not mixed with v2 runs.
The source overlay remains valid because its
`gemma4-routed-input-overlay-v1` schema, capture site, token identities, and
payloads are independent of the downstream encoder activation. V2 pins and
reuses the exact sealed train manifest
`8d947629e57d64d054a3086e2729090eed1e9f6afc4568b76e5c90da5f378e3c`
and eval manifest
`f68e2fe6e4cf568e694925697dd5d65fe10160a52d357c641892de85ee5c3da7`;
the overlay harvester must not be pointed at either sealed production root.

## Decision

Experiment 2 established that Gemma block-15 expert identity is useful when
the eight routed expert outputs are reconstructed separately:

| held-out component endpoint | EV |
|---|---:|
| A, true expert ID | 0.3195047562 |
| R, randomized expert ID | 0.1000750821 |
| B, router-rank candidate control | 0.2978687594 |
| F, router-rank full-width control | 0.3109305831 |

The paired A-minus-R and A-minus-B effects were both positive with intervals
excluding zero. A also beat F. This is strong evidence that Gemma manufactures
a stable router-keyed factorization. It is not yet evidence that the
factorization remains useful when the sparse model must *predict* expert
computation from the expert's input rather than autoencode its output.

Experiment 3 changes exactly that boundary. It keeps the sealed Experiment-2
targets, token order, router metadata, recomposition parameters, layouts,
optimizer, and evaluation machinery. It replaces only the encoder operand:

```text
Experiment 2 SAE:          expert_output -> expert_output
Experiment 3 transcoder:   expert_input  -> expert_output
```

This is the smallest decisive transfer experiment.

## Primary question

> Does stable expert identity improve sparse prediction of Gemma's routed
> expert computation over a candidate-matched router-rank transcoder, and how
> much of the full-width gain does it recover at what measured marginal cost?

The experiment distinguishes three claims:

1. **Alignment:** A must beat the load-preserving randomized-ID control R.
2. **Architecture:** A must beat the candidate-matched router-rank control B.
3. **Efficiency:** A must recover a positive fraction of F-over-B at lower
   measured training cost.

The encoder is permitted to receive the routed expert ID only as a bank key.
It must not receive the expert output, router weight, router rank, shared
output, or any statistic derived from the target.

## Immutable parent artifacts

The experiment reuses the exact sealed Experiment-2 Phase-2 targets:

```text
training contribution manifest
  a2f23aef6b9121e81a5bfea473c64039ca5bf6b073e4f70f4703938a1d8a6903
  2,000,000 scientific tokens

held-out contribution manifest
  31454e5ce8693521c41a2a95f2ef88d40c5d26c726a0efac5c26d882ed341f5e
  250,000 scientific tokens
```

Those artifacts contain, for every retained token, the eight raw selected
expert outputs, canonical IDs and weights, the raw shared output, and exact
token identity. The frozen recomposition sidecar contains the routed,
shared, and outer RMSNorm parameters.

No target is recaptured or migrated. Experiment 3 adds a separate,
content-addressed input overlay and joins it to these immutable parents.

## Exact source and target

Let `h_t` be the one-indexed block-15 post-attention residual. The actual input
consumed by every selected routed expert is:

```text
x_t = target.mlp.pre_feedforward_layernorm_2(h_t)
```

This source is not the pre-attention block input, raw residual `h_t`, or the
distinct `gate_norm(h_t)` router operand.

For router slot `s`, the sealed target is:

```text
i_ts    selected expert ID
g_ts    normalized router weight
e_ts    selected expert's raw output after its learned output scale
```

The transcoder sees `x_t` and uses the condition's bank key. It predicts
`e_ts`. Router weight is used only in the loss and exact recomposition.

## Routed-input overlay

The companion schema is `gemma4-routed-input-overlay-v1`:

| field | wire type and shape |
|---|---|
| `routed_input` | BF16 `[N, 2816]` |
| `sequence_id` | int64 `[N]` |
| `position` | uint16 `[N]` |

At 2.25M tokens the tensor payload is about 12.672 GB decimal. Capturing the
shared-expert input would double this volume and is not required for the
primary routed transfer claim, so the first pilot leaves the shared branch
oracle-fixed.

The source-only adapter:

1. runs blocks 1 through 14;
2. runs block-15 attention through its residual;
3. applies only `mlp.pre_feedforward_layernorm_2`;
4. compacts the existing scientific token mask; and
5. stops before router recomputation, routed experts, or the shared expert.

Every overlay manifest binds the parent contribution-manifest SHA, model and
checkpoint identities, dataset/split, target layer, source site, logical
ranges, token count, per-shard content hashes, and FP64 source count/sum/
squared-norm statistics. It is sealed only after every range and hash
validates.

The paired reader zips source and target by immutable range ID and verifies
`sequence_id` and `position` elementwise. Equal lengths are insufficient.
Missing, reordered, duplicated, or extra tokens fail closed.

## Transcoder

Let `mu_x` be the one global routed-input mean from the training overlay and
`mu_e` the existing one pooled raw-expert-output mean from the sealed target.
There are no per-bank means and no trainable output bias.

For condition-specific bank `b_ts`:

```text
a_ts     = softplus(W_enc[b_ts] @ (x_t - mu_x) + beta[b_ts])
z_ts     = exact_positive_TopK_2(a_ts)
e_hat_ts = mu_e + W_dec[b_ts, z_ts] @ z_ts
```

Each selected slot is scored independently. The eight reconstructions are
combined only after decoding. Exact-positive TopK requires two strictly
positive activations per slot, uses global latent ID as the deterministic
tie-break, and never materializes a dense latent vector for decode. Softplus
is applied identically by the independent NumPy oracle, dense JAX oracle,
grouped A/R accelerator path, and rank-banked B/F accelerator path.

The numerical and optimizer contract remains matched to Experiment 2:

- BF16 parameters and matmuls;
- FP32 reductions and Adam moments;
- FP32 softplus outputs with a smallest-normal positive numerical floor;
- Adam beta1 0.9, beta2 0.999, epsilon 1e-8;
- no weight decay and global gradient clip 1.0;
- 12-update linear warmup to `3e-4`, cosine decay to `3e-5`;
- unit-norm decoder columns;
- encoder initialized from decoder transpose;
- zero encoder bias;
- tangent-space decoder-gradient projection; and
- decoder-column renormalization after every update.

Using the same initialization is intentional: this experiment isolates the
encoder operand, not a new optimizer recipe.

## Matched conditions

All conditions consume the same 2M-token order and predict the same eight
targets:

| condition | bank key | dictionary | total features | candidates/token | TopK |
|---|---|---:|---:|---:|---:|
| A | true expert ID | 128 x 256 | 32,768 | 2,048 | 2/slot |
| R | keyed per-sequence randomized expert ID | 128 x 256 | 32,768 | 2,048 | 2/slot |
| B | canonical router rank | 8 x 256 | 2,048 | 2,048 | 2/slot |
| F | canonical router rank | 8 x 4,096 | 32,768 | 32,768 | 2/slot |

A and B are candidate-matched. A and F are parameter/feature-matched. Every
token has exact total L0 16. R uses the same deterministic per-sequence
bijection during training and evaluation, preserving bank count and traffic
while destroying stable expert meaning.

Seed 0 is the frozen mechanism pilot. It is not described as a multi-seed
confirmatory result.

## Objective and exact recomposition

Training minimizes the unchanged normalized router-weighted component loss:

```text
SSE_c,batch = sum_t,s || FP32(g_ts) * (FP32(e_ts) - FP32(e_hat_ts)) ||^2
loss        = SSE_c,batch / SST_c,batch
```

The optimizer denominator is the target-only SST of the current matched
batch, exactly as in Experiment 2. It is mathematically independent of the
condition and keeps the inherited LR/loss scale unchanged. Batch target SST is
recorded so equality can be audited across A/R/B/F.

For held-out evaluation, one target-only common SST is computed once from the
sealed target inventory, bound by a receipt, and used by every condition. It
is not recomputed inside condition-specific evaluation graphs. This is the
common denominator required for all EV values and paired effects.

Held-out predictions are passed through Gemma's frozen routed computation:

```text
u_hat_t = sum_s FP32(g_ts) * FP32(e_hat_ts)
r_hat_t = routed_norm(u_hat_t)
```

The primary full-branch endpoint uses the one common *actual* shared branch:

```text
y_hat_t = outer_norm(r_hat_t + q_t)
```

This oracle-shared endpoint isolates routed prediction quality. A learned
shared transcoder is a later orthogonal follow-up; it cannot change A/R/B/F
ordering and is not allowed to delay this pilot.

## Schedule and accounting

Each condition trains on exactly 2,000,000 scientific tokens:

```text
effective batch     16,384 tokens
microbatch          4,096 tokens
optimizer updates   123
warmup updates      12
final tail mask     15,232 padded positions
```

Compilation, shared input harvest, checkpoint serialization, and evaluation
are reported separately. Marginal training chip-seconds include
condition-specific preprocessing/capacity work, optimizer synchronization,
and input stalls. The accelerated count-based A/R capacity scan from commit
`f1536d0d62ba3bec504e753e273c3c3755d6f313` is part of the measured binary;
no post-hoc scan subtraction is permitted.

The primary measured cost is:

```text
C3 = 100 * chipseconds(A) / chipseconds(F)
```

Optimizer-only timing may be shown diagnostically. It cannot replace C3.

The source harvest is expected to run for less than ten minutes after compile
and therefore falls below the previously declared long-job MFU threshold.
The SAE path already received explicit authorization after its measured
throughput estimate was reviewed. Useful MFU, tokens/s, wall time, and
chip-seconds are still retained.

## Metrics

For condition `j`:

```text
EV_c,j = 1 - SSE_c,j / SST_c,eval
```

The primary contrasts are:

```text
Delta_align_TC = EV_c,A - EV_c,R
Delta_arch_TC  = EV_c,A - EV_c,B
G3             = 100 * (EV_c,A - EV_c,B) / (EV_c,F - EV_c,B)
```

Do not clip G3. If F-minus-B is nonpositive, G3 is undefined. If its paired
interval includes zero, label the ratio unstable/exploratory.

Use 20,000 paired packed-sequence bootstrap replicates for A-minus-R,
A-minus-B, A-minus-F, and F-minus-B at both component and exact routed
recomposition endpoints. The eight slots of a token are not independent
observations.

For continuity with Experiment 2, report descriptive effect-retention ratios:

```text
T_align = Delta_align_TC / 0.2194296741
T_arch  = Delta_arch_TC  / 0.0216359968
```

Absolute transcoder EV is not expected to match SAE EV because prediction is
a harder task.

Always report:

- per-condition component, routed, and oracle-shared EV/NMSE;
- all four paired intervals, including the direct A-minus-F contrast;
- exact L0, dead rate, and feature coverage;
- macro expert-balanced EV plus expert-frequency and router-rank bins;
- source/target manifest hashes and common-SST receipt;
- training chip-seconds, optimizer-only time, throughput, useful MFU, and wall
  time; and
- shared capture cost and end-to-end cost separately.

## Verdict

The motif transfers in this pilot if:

1. A-minus-R component EV is positive and its paired interval excludes zero;
2. A-minus-B component EV is positive and its paired interval excludes zero;
3. A-minus-B remains positive after exact routed recomposition;
4. F-minus-B is positive and stable, so G3 is defined;
5. G3 is at least 10%; and
6. every condition used source-only encoding, exact L0 16, finite state, and
   healthy coverage.

The result is narrowed rather than cosmetically repaired:

- nonpositive A-minus-R: expert alignment does not transfer;
- positive A-minus-R but nonpositive A-minus-B: identity matters, but gives no
  candidate-matched architectural gain;
- positive component A-minus-B but nonpositive routed A-minus-B: the local
  objective does not survive native recomposition;
- nonpositive/unstable F-minus-B: G3 is undefined;
- C3 at or above 100%: the mechanism may transfer, but marginal efficiency
  does not.

No failed case authorizes adding auxiliary losses, per-expert offsets, a new
width, or target-derived encoder features after looking at quality.

## Acceptance tests

- source-only adapter parity at the exact post-attention routed-input site;
- proof that source capture calls neither routed nor shared experts;
- BF16 overlay round-trip and content-hash validation;
- immutable range resume with no gaps or duplicates;
- exact source/target token-identity join, including a permuted-token failure;
- NumPy/JAX parity with independent source and target arrays;
- leakage test: changing targets cannot change pre-update latent selections;
- A/R/B/F deterministic TopK and sparse-decode parity;
- exact-positive TopK for typical and strongly negative finite
  preactivations;
- interrupted-versus-uninterrupted checkpoint parity binding both manifests;
- exact tail-mask and L0 checks;
- exact recomposition against the sealed sidecar;
- paired metric tests for negative, greater-than-100%, and undefined G3; and
- synthetic end-to-end source-overlay through report.

## Known scope

This experiment tests conditional sparse approximation of a frozen MoE. A is
given expert identity as side information; it does not test whether expert
identity can be inferred from the common input. A win also does not by itself
establish human-interpretable features.

The shared branch remains an exact common oracle. A separate shared
transcoder can be trained from `target.pre_ffw_norm(h_t)` if a learned full-FFN
endpoint becomes scientifically necessary, but that adds no information to
the primary expert-alignment contrast and is outside the seed-0 critical path.

## Execution record

To be completed only from sealed run evidence:

```text
code commit:
routed-input training manifest:
routed-input held-out manifest:
parent training target manifest:
  a2f23aef6b9121e81a5bfea473c64039ca5bf6b073e4f70f4703938a1d8a6903
parent held-out target manifest:
  31454e5ce8693521c41a2a95f2ef88d40c5d26c726a0efac5c26d882ed341f5e
training namespace:
evaluation result:
evidence root:
```
