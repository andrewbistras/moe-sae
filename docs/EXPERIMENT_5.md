# Experiment 5: Routed–Shared Hybrid Sparse Models

Status: frozen H/R/B/F core contract, with a post-launch activation-ladder
extension added 2026-07-29

Protocol ID: `gemma4-routed-shared-hybrid-v1-20260728`

## Decision

Experiments 2–4 established that Gemma block 15 contains a stable,
router-keyed factorization. True expert-ID sparse models beat randomized-ID
controls and candidate-matched router-rank controls, and the advantage
transferred from autoencoding expert outputs to predicting them from the
common expert input. In the 10M-token learning-horizon study, the true-ID
condition also beat the full-width control in both the SAE and transcoder
panels.

Those experiments deliberately isolated the routed branch. They either used
the real shared branch as an oracle or evaluated routed reconstruction before
the full FFN output. Gemma itself does something more structured:

```text
routed experts -> weighted sum -> routed RMSNorm --+
                                                    +-> outer RMSNorm -> y
shared expert  ----------------> shared RMSNorm ----+
```

Experiment 5 asks whether the sparse model benefits when its own organization
matches that routed-plus-shared computation graph.

The hybrid allocation was written into `EXPERIMENT_2.md` before the
learning-horizon result:

```text
8 routed banks x 512 candidates, TopK-4 each = 4,096 candidates / L0 32
1 shared dictionary x 4,096 candidates, TopK-32 = 4,096 candidates / L0 32
```

The routed experiment has passed its launch condition. This experiment now
tests that frozen 32/32 allocation without a width sweep, revival loss,
pruning rule, or post-result repair.

## Pre-result SAE activation reset

The first SAE launch exposed an implementation-contract failure before a
complete panel or scientific verdict existed. Under the originally specified
ReLU encoder, H seed 0 accepted 73 updates (1,196,032 tokens) and H seed 1
accepted 70 updates (1,146,880 tokens), then each failed closed because a
bank-local selection had fewer than its required number of strictly positive
candidates. This was positive-support collapse under exact-positive TopK, not
a capacity overflow and not evidence about held-out reconstruction quality.

The ReLU SAE protocol is therefore retired. Its partial artifacts remain
audit evidence only and are excluded from every Experiment-5 metric, report,
and resume path. The SAE panel restarts H/R/B/F from scratch in a disjoint run
namespace. All four conditions use softplus with a smallest-normal positive
FP32 floor, so the change repairs the strict-L0 numerical contract without
giving the hybrid conditions a different encoder nonlinearity from their
controls. ReLU and softplus trajectories must never be combined.

## Primary question

> Does a sparse model whose dictionaries mirror Gemma's actual routed and
> shared FFN branches reconstruct the exact full FFN output better, or reach a
> global full-width model's quality sooner, at a matched sparse-compute budget?

The transcoder panel is primary because every condition receives the same
source activation. The SAE panel is a mechanistic companion: it tests whether
autoencoding the native operands is more effective than autoencoding their
already-composed output.

The experiment distinguishes four claims:

1. **Alignment:** true expert identity must beat a per-sequence randomized-ID
   hybrid with the same per-token route cardinality and sparse candidate
   budget.
2. **Architecture:** the complete routed/shared hybrid factorization,
   including its known Gemma recomposition, must beat a global
   candidate-matched dictionary.
3. **Capacity:** the hybrid is compared directly with a global
   parameter-matched dictionary.
4. **Efficiency:** the hybrid's earliest measured checkpoint reaching the
   global full-width model's 10M quality is compared by measured
   chip-seconds.

No claim may be substituted for another. In particular, a true-ID versus
randomized-ID effect establishes alignment but does not by itself establish a
candidate-matched architectural or cost advantage.

## Immutable model and parent data

The model and target site are unchanged:

```text
model              google/gemma-4-26B-A4B
model revision     24548b62aa021d562695c04aaf7758a1ea47990b
MaxText base       48c7da3e28e8f57fe36ddf09d00eec53072a2ea0
MaxText fork       73bafe53569eb17dc18542f2568dd591a8db9e28
target             one-indexed block 15
hidden width       2816
routed experts     128
selected experts   8
shared experts     1
```

No new model activation harvest is authorized. Experiment 5 reuses the sealed
10M Experiment-4 training target, its aligned routed-input overlay, the
250,000-token held-out target and overlay, and the exact recomposition
sidecar. An executable config must bind their canonical SHA-256 identities.
The checked-in configs bind the sealed manifests and recomposition sidecar
directly. They distinguish the sidecar's canonical self-identity
`sidecar_sha256` from the SHA-256 of its serialized file bytes.

The parent contribution artifact supplies, per retained token:

```text
expert_output[8, 2816]
expert_ids[8]
router_weights[8]
shared_output[2816]
sequence_id
position
```

The transcoder overlay supplies the aligned block-15 routed input
`x_t[2816]`. Source and target are joined by immutable range ID and
elementwise token identity, not by length.

## Exact Gemma computation

For token `t` and canonical router slot `s`:

```text
i_ts       selected expert ID
g_ts       normalized router weight
e_ts       selected expert raw output after learned output scale
q_raw_t    raw shared-expert output

u_t = sum_s FP32(g_ts) * FP32(e_ts)
r_t = routed_norm(u_t)
q_t = shared_norm(q_raw_t)
y_t = outer_norm(r_t + q_t)
```

`y_t`, before residual addition and the layer scalar, is the single
scientific target for every condition.

For hybrid H or R:

```text
u_hat_t = sum_s FP32(g_ts) * FP32(e_hat_ts)
r_hat_t = routed_norm(u_hat_t)
q_hat_t = reconstructed normalized shared branch
y_hat_t = outer_norm(r_hat_t + q_hat_t)
```

The shared decoder reconstructs `q_t` directly. It does not reconstruct
`q_raw_t` and then apply a second shared normalization. The routed decoder
reconstructs raw `e_ts`; router weights and the routed RMSNorm are applied
exactly after sparse decode.

The common training objective is full-FFN normalized SSE:

```text
SSE_y = sum_t ||FP32(y_t) - FP32(y_hat_t)||^2
loss  = SSE_y / SST_y
EV_y  = 1 - SSE_y_eval / SST_y_eval
```

The optimizer denominator for a matched batch is target-only and therefore
identical across H/R/B/F. Held-out evaluation computes one target-only SST
once and uses it for every condition, seed, and checkpoint. There is no
branch-local auxiliary loss.

## Frozen condition matrix

The only valid condition enum is `H/R/B/F`.

| condition | organization | total features | candidates/token | exact L0 |
|---|---|---:|---:|---:|
| H | true-ID routed/shared hybrid | 69,632 | 8,192 | 64 |
| R | randomized-ID routed/shared hybrid | 69,632 | 8,192 | 64 |
| B | candidate-matched global | 8,192 | 8,192 | 64 |
| F | parameter-matched global | 69,632 | 69,632 | 64 |

H and R contain:

```text
routed:
  128 banks x 512 features                 = 65,536 total features
  8 visible banks x 512 candidates         = 4,096 candidates/token
  independent TopK-4 for each routed slot  = L0 32/token

shared:
  1 dictionary x 4,096 features            = 4,096 total features
  4,096 candidates/token
  TopK-32                                   = L0 32/token
```

Thus H/B are candidate-matched. H/F are feature- and parameter-matched:

```text
69,632 * (2 * 2,816 + 1) = 392,237,056
```

trainable encoder/decoder/bias parameters. H exposes
`8,192 / 69,632 = 11.7647059%` of its dictionary per token.

R changes only the routed bank key. For every packed sequence and training
seed, a keyed deterministic bijection maps each true expert ID to a randomized
bank ID. It preserves total routed traffic, eight unique visible banks per
token, shared dictionary use, candidate count, parameter count, and L0 while
destroying stable expert meaning. Because the bijection changes by packed
sequence, it need not preserve each bank's aggregate exposure; H/R exposure is
reported explicitly. The same mapping is used in training and held-out
evaluation. A global fixed permutation, per-token remapping, or inference-only
shuffle is invalid.

B and F each use one global dictionary and decode `y_t` directly. B has 8,192
features and F has 69,632. Both use exact per-token TopK-64. No dense latent
vector is materialized for decode.

## Panel 5S: hybrid SAE

The valid panel identifier is `sae`.

For H/R:

- each routed encoder receives its own raw selected expert output `e_ts`;
- the shared encoder receives the normalized shared branch `q_t`;
- routed and shared sparse decodes are composed through the frozen Gemma
  function above; and
- gradients come only from the common full-FFN loss.

For B/F, the encoder receives `y_t` and reconstructs `y_t` directly.

All H/R/B/F conditions use softplus with a smallest-normal positive FP32
floor, exact-positive TopK, the frozen initialization and optimizer, and
decoder normalization. Routed source centering uses one pooled raw-expert-
output mean; shared source centering uses one global `q_t` mean; and B/F use
the one global `y_t` mean. There are no per-expert means.

Because H/R receive native decomposed operands while B/F receive the composed
output, this panel is evidence about representation boundaries, not a
same-input prediction comparison.

## Panel 5T: hybrid transcoder (primary)

The valid panel identifier is `transcoder`.

Every condition receives only the same routed expert input:

```text
x_t = target.mlp.pre_feedforward_layernorm_2(h_t)
```

For H/R, the same centered `x_t` is encoded eight times using the condition's
routed bank keys to predict the eight `e_ts`, and once through the shared
dictionary to predict `q_t`. For B/F, `x_t` is encoded once through the global
dictionary to predict `y_t`.

The encoder never receives expert output, router weight, router rank, shared
output, or target-derived statistics. Expert ID is available only by
selecting a bank. Router weights are used after decode in the known Gemma
recomposition.

This panel retains the Experiment-3 softplus encoder with a smallest-normal
positive FP32 floor, exact-positive TopK, and deterministic latent-ID
tie-breaking.

## Training contract

Both panels use seeds `0, 1, 2`. Within a seed, all H/R/B/F trajectories use
the same token order and seed-derived initialization stream. They start from
scratch and train on exactly 10,000,000 scientific tokens:

```text
effective batch        16,384 tokens
microbatch              4,096 tokens
optimizer updates            611
warmup updates                 61
masked tail tokens         10,624
scientific checkpoints  2M, 5M, 10M
```

Operational restart checkpoints may be saved every 1M tokens. They do not add
scientific looks.

The optimizer contract is:

- BF16 parameters and matmuls;
- FP32 reductions, nonlinear recomposition, and Adam moments;
- Adam beta1 `0.9`, beta2 `0.999`, epsilon `1e-8`;
- no weight decay;
- global gradient clipping at `1.0`;
- 61-update linear warmup to `3e-4`;
- cosine decay to `3e-5`;
- unit-normal decoder-column initialization;
- encoder initialization from decoder transpose;
- zero encoder bias;
- tangent-space decoder-gradient projection; and
- decoder-column renormalization after every update.

Every selection must contain exactly 64 strictly positive latents. Both
panels compute softplus in FP32 and floor its result at the smallest normal
positive FP32 value before deterministic TopK. Ties use ascending global
latent ID. Nonfinite state, insufficient positive candidates, capacity
overflow, token dropping, or L0 drift is a hard failure.

## Execution, performance, and hard cost ceiling

Existing sealed artifacts must be read from a region-local mirror or bounded
host cache. Experiment 5 may not create a new bulk capture, duplicate a sealed
capture into another region solely for convenience, or repeatedly read the
canonical cross-region GCS object on every step.

Useful MFU of 10% is an optimization target and diagnostic, not a launch gate.
Every trajectory records useful MFU, tokens/s, chip-seconds, compilation, and
input/preprocessing time. Obvious regressions should be investigated and
low-risk execution improvements applied, but measured MFU below 10% does not
block, truncate, or invalidate an otherwise correct scientific trajectory.
The direct resident campaign therefore requires no standalone qualification
run. Correctness, finiteness, exact L0, data identity, resumability, and the
cash ceiling remain fail-closed.

Cloud availability is a scheduler state, not a scientific attempt. Before
creating an attempt intent or reserving attempt cost, the controller records a
typed observation of the exact plan-bound TPU incarnation and requires both
`state=READY` and `health=HEALTHY`. An absent or forming node, an unhealthy but
recoverable node, or a transient control-plane failure leaves its work pending
and places only that resource worker on a bounded polling cooldown; another
eligible ready slice may claim the task. Malformed or unauthorized
observations remain hard errors. Terminal TPU states, accelerator drift, or
create-time/incarnation drift produce append-only
`terminal_replacement_required` evidence and stop promptly; this controller
does not mutate cloud resources. If an in-flight attempt fails through SSH or
resource loss, it is relabeled `spot_preemption` only when the pre-attempt
observation proved a ready spot node and the post-failure observation proves
that exact node terminal or absent. The attempt, conservative cost
reservation, and both observations remain in the audit trail, and the
campaign stops for fresh replacement evidence and replanning. Unconfirmed
transport failures retain the bounded transient-retry policy.

This experiment intentionally retains general sharded XLA primitives rather
than introducing a new hand-written routed grouped kernel. The measured bank
traffic and winning routed/shared geometry from this and the immediately
following confirmatory experiments will define that kernel's contract.

The user-authorized incremental billable cost ceiling is **USD 50.00**. TPU
time covered by the existing TRC allocation is recorded as quota-funded but
not invented as a cash charge. The budget ledger includes actual incremental
VM, persistent-disk, NFS/Filestore, GCS operation, storage, and network
charges. Before launch, the scheduler must reserve at most USD 45.00 against
the projected remaining work, leaving USD 5.00 contingency. It must not launch
a task whose conservative projection would cross USD 50.00. Cost pressure may
reduce parallelism or stop with an incomplete verdict; it may not silently
reduce tokens, seeds, conditions, or evaluation.

The controller binds a durable campaign clock and conservatively accounts
infrastructure at USD 1.25/hour. Its finite wall bound is derived from the USD
45 protected launch envelope after subtracting already-spent cost and the
baseline reserves for all 24 training and 18 evaluation attempts. A 15-second
watchdog uses the greater of the fixed reserve or accrued wall cost, protects
all not-yet-created baseline attempt reserves, and stops with an immutable
budget record before consuming the USD 5 contingency.

After the eighteenth evaluation, the controller seals the budget ledger and
the JSON/Markdown scientific report without operator intervention. Cash spend
is the plan's already-spent amount plus the greater of the fixed reserve or
USD 1.25/hour from the durable campaign start through the immutable evaluation
completion time; TPU attempt reservations are not cash charges. Quota-funded
TPU time is the sum of the cumulative 10M training chip-seconds for both
panels, all three seeds, and H/R/B/F exactly once. Evaluation TPU chip-seconds
are not present in the Experiment-5 scientific-result contract, so the final
campaign record explicitly marks them absent rather than estimating them.

H and F within one seed must use the same accelerator generation for a
headline chip-second ratio. Raw chip-seconds are always retained across
hardware, but cross-generation ratios are labeled non-comparable.

## Evaluation

Every seed and condition is evaluated at 2M, 5M, and 10M on the identical
250,000-token held-out capture.

The primary endpoint is exact full-FFN `EV_y`. Also report:

- NMSE and raw SSE under the common full-FFN SST;
- routed-only and shared-only reconstruction diagnostics for H/R;
- oracle-shared full-FFN EV, replacing `q_hat_t` with exact `q_t`;
- oracle-routed full-FFN EV, replacing `r_hat_t` with exact `r_t`;
- error-cancellation diagnostics comparing the learned full composition with
  both oracle substitutions;
- exact L0 and candidate exposure;
- marginal training chip-seconds, wall time, tokens/s, useful MFU, and
  accelerator generation;
- route exposure per expert bank; and
- firing coverage as two explicitly different quantities.

The coverage quantities must never be conflated:

```text
cumulative training coverage =
  features that fired at least once across the entire training prefix

held-out nonfiring rate =
  features that did not fire on this finite held-out evaluation
```

The latter is not called dead-feature rate. For routed banks, coverage is also
reported against bank-relative route exposure so rare-bank starvation can be
distinguished from within-bank winner lock-in.

Every curve point publishes a separate self-hashed paired-score sidecar with
the packed-sequence IDs, valid-token counts, one target-only SST vector, and
H/R/B/F SSE vectors. Its pooled sums must reproduce the scientific result.
All effects use 20,000 paired packed-sequence bootstrap replicates. Slots from
one token are never independent bootstrap observations. Per-seed effects are
reported, followed by a hierarchical paired interval for the mean effect that
resamples seeds and packed sequences.

Bootstrap randomness is reproducible. It uses NumPy PCG64 seeded by the first
64 bits, interpreted big-endian, of:

```text
SHA256(protocol_id | panel | seed | milestone | contrast | packed-sequence)
```

The hierarchical resample uses the analogous key ending in `hierarchical`.
Percentile limits use the linear 2.5th and 97.5th percentiles.

At 10M:

```text
Delta_HR = mean_seed(EV_y,H - EV_y,R)
Delta_HB = mean_seed(EV_y,H - EV_y,B)
Delta_HF = mean_seed(EV_y,H - EV_y,F)
Delta_FB = mean_seed(EV_y,F - EV_y,B)
```

Gain recovery is:

```text
G5 = 100 * mean_seed(EV_y,H - EV_y,B)
           / mean_seed(EV_y,F - EV_y,B)
```

G5 is never clipped. It is undefined when mean F-minus-B is nonpositive. If
the paired interval for F-minus-B includes zero, the ratio is labeled
unstable/exploratory rather than headline-ready.

## Fixed-quality cost endpoint

For each seed, find the earliest measured H checkpoint in `{2M, 5M, 10M}`
whose `EV_y,H` is at least that seed's `EV_y,F` at 10M. Do not interpolate.

```text
Q5_seed =
  100 * cumulative_chipseconds(H, earliest reaching checkpoint)
        / cumulative_chipseconds(F, 10M)
```

If H never reaches the reference, `Q5_seed` is undefined and the miss is
reported. The primary Q5 is the median of the three paired seed ratios only if
all three are defined. The report always includes the per-seed values and the
number of seeds reaching the target.

Equal-token H/F chip-second ratios at 2M, 5M, and 10M are separate
measurements. Compilation, checkpoint serialization, evaluation, and the
shared parent harvest are excluded from marginal training chip-seconds and
reported separately. An end-to-end ledger includes them without charging the
same existing harvest repeatedly.

## Verdict

The routed/shared motif is headline-ready in a panel only if:

1. all three seeds, four conditions, and three checkpoints are complete;
2. every trajectory used exact L0 64, remained finite, and preserved source
   and target identities;
3. H beats R at 10M in at least two of three seeds and the hierarchical
   paired mean interval excludes zero;
4. H beats B at 10M in at least two of three seeds and the hierarchical
   paired mean interval excludes zero; and
5. no result crossed the USD 50.00 hard incremental cash ceiling.

H versus F and Q5 are reported directly; H need not beat F to establish the
candidate-matched architecture. A fixed-quality efficiency headline requires
all three Q5 seed values and same-generation H/F comparisons.

The intended headline, if supported, is:

> The router-aligned routed/shared hybrid reaches the 10M-token global
> full-FFN baseline's held-out reconstruction at Q% of its measured marginal
> training cost while exposing 11.76% of the parameter-matched dictionary per
> token.

The outcome is narrowed honestly:

- H beats R but not B: stable identity matters, without candidate-matched
  architectural gain;
- H beats B but not R: hybrid decomposition helps, without evidence that
  stable expert identity caused it;
- H beats B/R but not F: a sparse-compute architectural win, not a
  parameter-matched quality win;
- H does not reach F's 10M quality: Q5 is undefined for the affected seeds;
- cross-generation H/F execution: raw cost is retained but no headline cost
  ratio is claimed; or
- budget stop: report an incomplete experiment, never a reduced protocol.

The transcoder panel is the primary architectural verdict. The SAE panel is
reported independently and cannot rescue a failed transcoder claim.

## Acceptance tests

- strict config validation for the only panel enum `sae/transcoder` and
  condition enum `H/R/B/F`;
- exact derived feature, parameter, candidate, exposure, and L0 counts;
- H/R per-sequence bijection determinism, uniqueness, load preservation, and
  seed separation;
- SAE and transcoder source-boundary tests with target-leakage failures;
- independent NumPy/JAX parity for routed decode, shared decode, exact Gemma
  recomposition, and full-FFN normalized SSE;
- sparse decode parity without dense latent materialization;
- exact-positive deterministic TopK for all four layouts, including extreme
  negative finite preactivations under the softplus FP32 floor;
- common target SST equality across conditions;
- interrupted-versus-uninterrupted checkpoint parity;
- exact 2M/5M/10M milestone restoration;
- cumulative training firing-mask checkpoint/resume parity;
- held-out nonfiring evidence stored separately from training coverage;
- paired and hierarchical bootstrap tests;
- negative, greater-than-100%, unstable, and undefined G5 tests;
- reached, missed, and no-interpolation Q5 tests;
- same-generation cost-comparability tests;
- fail-closed budget reservation and USD 50.00 ceiling tests; and
- synthetic end-to-end training through self-hashed report.

## Explicitly deferred from the frozen core

The original H/R/B/F Experiment-5 protocol does not include:

- alternative routed/shared L0 splits;
- traffic-proportional bank widths;
- bank-local BatchTopK;
- dead-latent revival or auxiliary loss;
- pruning based on the Experiment-4 held-out nonfiring observation;
- a wider or narrower hybrid;
- router weights as encoder inputs; or
- a new activation harvest.

Those are Experiment-6 optimization questions. Changing any of them after
viewing Experiment-5 quality creates a new protocol ID. The activation-ladder
extension below does exactly that: it has its own protocol ID, namespace, run
accounting, and interpretation, and does not retroactively alter this frozen
core.

## Post-launch routed-activation-ladder extension

Status: exploratory post-launch amendment, selected after observing initial H
training behavior

Extension protocol ID:
`gemma4-experiment5-routed-activation-ladder-v1-20260729`

This extension answers the narrower diagnostic question raised by the
Experiment-4 result:

> At fixed 128-by-512 routed geometry and TopK-4 per visible expert, does
> strictly positive Softplus encoding improve reconstruction over ReLU, and
> how do both routed-only systems relate descriptively to the complete
> routed/shared H model?

This question was not in the frozen H/R/B/F contract. It was added after the
initial H training losses had been inspected. Its results are therefore
exploratory and cannot repair, replace, or strengthen the preregistration
status of the original Experiment-5 verdict. The frozen H/R/B/F endpoints,
contrasts, completion counts, and report remain unchanged.

### Ladder conditions

The extension uses the unambiguous keys `A512ReLU`, `A512Softplus`, and `H`.
Neither A condition may be shortened to `A` or `A512` in an artifact because
those names would erase the activation treatment or collide with historical
condition names.

| condition | activation | routed features | candidates/token | routed L0 | shared learned features | total features |
|---|---|---:|---:|---:|---:|---:|
| A512ReLU | ReLU | 128 x 512 = 65,536 | 8 x 512 = 4,096 | 8 x 4 = 32 | 0 | 65,536 |
| A512Softplus | Softplus | 128 x 512 = 65,536 | 8 x 512 = 4,096 | 8 x 4 = 32 | 0 | 65,536 |
| H | Softplus, frozen Experiment-5 hybrid | 128 x 512 = 65,536 | 4,096 routed + 4,096 shared | 32 routed + 32 shared | 4,096 | 69,632 |

Every routed condition exposes eight true-expert-ID banks. Both A conditions
expose 6.25% of the routed dictionary per token and select the same fraction
inside each visible bank:

```text
A512ReLU:     TopK-4 / 512 = 0.78125%
A512Softplus: TopK-4 / 512 = 0.78125%
```

Their trainable parameter counts, including encoder, decoder, and bias, are:

```text
A512ReLU     = 65,536 * (2 * 2,816 + 1) = 369,164,288
A512Softplus = 65,536 * (2 * 2,816 + 1) = 369,164,288
H            = 69,632 * (2 * 2,816 + 1) = 392,237,056
```

H is the already-frozen core condition, not a newly tuned H trajectory. Its
routed half is geometrically identical to both A conditions. Its additional shared
dictionary contributes 4,096 features, 4,096 candidates, L0 32, and
23,072,768 parameters.

### Sources, activations, and training objectives

A512ReLU and A512Softplus are newly trained, extension-owned trajectories.
Within each panel they are identical in source tensors, targets, true-expert
bank identity, 128-by-512 geometry, TopK-4, centering, initialization,
optimizer, batch order, seed, and component objective. Their only treatment
difference is the encoder nonlinearity:

- in the SAE panel, each encoder receives and reconstructs its selected raw
  expert output `e_ts`;
- in the transcoder panel, each true-ID bank receives the common routed input
  `x_t` and predicts its `e_ts`;
- A512ReLU applies ReLU before the exact per-bank TopK-4 in both panels;
- A512Softplus applies Softplus before the same TopK-4 in both panels; and
- both conditions use the Experiment-4 normalized router-weighted component
  objective.

The strict-positive selected-support contract remains unchanged. If
A512ReLU cannot supply four positive features in a visible bank, that
trajectory fails closed; it is never repaired by switching activation.

For both A conditions, with the pooled raw-expert-output mean `m_e`, the
training objective is exactly:

```text
SSE_component = sum_t,s ||g_ts * (e_ts - e_hat_ts)||^2
SST_component = sum_t,s ||g_ts * (e_ts - m_e)||^2
loss_component = SSE_component / SST_component
```

Thus the normalized router weights are applied before squaring. This is the
Experiment-4 objective; it is not the full-FFN objective.

H remains exactly the frozen condition defined above:

- its SAE uses softplus for both routed and shared encoders, receives native
  `e_ts` and `q_t`, and reconstructs them through the exact Gemma graph;
- its transcoder uses softplus, receives only common `x_t`, and predicts the
  routed `e_ts` and normalized shared branch `q_t`; and
- both panels train only against normalized full-FFN SSE on `y_t`.

No A512ReLU or A512Softplus shared dictionary is implicit. Neither condition
may use a learned, zero, mean, or H-derived shared prediction during training.

### Common held-out endpoints

Every available 2M, 5M, and 10M checkpoint is rescored on the same sealed
250,000-token held-out capture. The extension publishes the following paired
endpoints for all three conditions wherever mathematically defined:

1. **Component EV:** raw expert-output SSE under the common Experiment-4
   router-weighted component SST.
2. **Routed EV:** reconstruct `u_hat_t = sum_s g_ts * e_hat_ts`, apply the
   frozen routed RMSNorm, and score `r_hat_t` against `r_t` under one common
   routed SST.
3. **Oracle-shared full-FFN EV:** use the exact held-out shared branch for all
   three conditions,
   `y_hat_oracle_t = outer_norm(r_hat_t + q_t)`, and score against `y_t` under
   the frozen common full-FFN SST.
4. **Learned full-FFN EV:** for H only, use its own `q_hat_t` and report the
   original Experiment-5 `EV_y`.

The oracle-shared endpoint is a diagnostic of routed reconstruction after the
real nonlinear recomposition. It is not an end-to-end model: A512ReLU and
A512Softplus receive unavailable target information through `q_t` at
evaluation. Their oracle-shared EV must never be presented as directly
beating H's learned full-FFN EV. H also publishes its oracle-shared endpoint
so routed quality can be compared with like-for-like shared information.

All conditions retain exact per-token additive score rows aggregated by
packed sequence. The 20,000-replicate paired bootstrap is authoritative only
for A512Softplus minus A512ReLU. H gaps are matched descriptive point
diagnostics with no interval. Component, routed, and full-FFN endpoints each
use their own single target-only SST shared across
A512ReLU/A512Softplus/H; SSTs may not be copied between different endpoint
spaces.

### Extension contrasts and limits

The planned diagnostic contrasts at each milestone and panel are:

```text
activation effect = EV(A512Softplus) - EV(A512ReLU)
descriptive joint-system gap = EV(H oracle shared) - EV(A512Softplus oracle shared)
H shared gap = EV(H oracle shared) - EV(H learned shared)
```

The activation contrast is the clean comparison: within each panel it changes
only ReLU to Softplus while holding geometry, TopK, L0, parameter count,
expert identity, source boundary, targets, initialization, optimizer, token
order, and component loss fixed. Seed-level intervals are paired by packed
sequence; the three-seed verdict uses the hierarchical paired bootstrap.

The H-versus-A512Softplus routed contrast is not a pure shared-dictionary
ablation. Although their routed geometry and activation match, H trains
routed and shared decoders jointly through full-FFN SSE; A512Softplus trains
routed components directly. The contrast localizes the transition to the
hybrid training system but cannot attribute it uniquely to the shared branch
or loss.

`H shared gap` is the descriptive reconstruction-quality penalty from
replacing the exact shared branch with H's learned shared reconstruction while
holding H's routed reconstruction fixed. It is not an A-versus-H treatment
effect. No direct end-to-end full-FFN contrast exists for either routed-only A
condition without choosing an arbitrary shared predictor.

The extension adds no randomized-ID control. Stable identity evidence
continues to come from Experiment-4 A/R and the frozen Experiment-5 H/R
contrasts. The activation ladder cannot independently establish that expert
identity caused either A result.

### Run and cost accounting

The analyzed trajectory matrix contains:

```text
12 newly trained A trajectories + 6 immutable pre-existing H trajectories
  = 18 analyzed trajectories
2 panels x 3 seeds x 3 milestones x 3 conditions = 54 evaluation cells
```

Of the 18 analyzed trajectories:

- six A512ReLU trajectories are new extension training work;
- six A512Softplus trajectories are new extension training work; and
- six H trajectories are read-only Experiment-5 core references.

Thus the extension launches 12 new trajectories. Historical Experiment-4 A
trajectories are pilot context, not members of this three-seed matrix. Their
width, activation policy, config, and source identities differ from the
extension contract, so they must not substitute for either activation arm.

Reuse is permitted only after exact checkpoint, model, capture, source-overlay,
seed, token-order, optimizer, and milestone identities validate. This reuse
rule applies to H. A missing or incompatible H artifact is not silently
regenerated under the old name; retraining it requires a new extension
protocol revision and condition identity.

Evaluation accounting is correspondingly:

```text
36 A512ReLU/A512Softplus evaluation executions
  + 18 extension-owned H checkpoint rescoring executions
  = 54 evaluation cells
```

H's historical compute is not assigned zero scientific cost merely because it
is reused operationally. Where validated parent timing records are supplied,
the report carries each trajectory's measured training chip-seconds and
accelerator generation. Newly incurred cash is reported separately from total
scientific compute. Any cross-protocol chip-second contrast remains
descriptive unless the paired trajectories used the same accelerator
generation and compatible timing boundaries.

This extension shares Experiment 5's cumulative USD 50.00 incremental cash
ceiling unless the user explicitly authorizes a replacement ceiling. The
extension scheduler may reserve only the unspent remainder after the core
ledger; reused H work does not create a fictional cash credit.

### Immutable namespace rule

The running or sealed H/R/B/F campaign is immutable. Its plan, controller
state, checkpoints, scores, report, completion counts, and protocol ID must
not be edited, resumed, or resealed to include A512ReLU or A512Softplus.

All extension-owned state lives under a disjoint logical prefix:

```text
experiment5/extensions/routed-activation-ladder-v1/<extension-run-identity>/
```

An extension manifest contains read-only content-hash references to the
Experiment-5 H artifacts; it never copies them into a mutable old-condition
path. The new A512ReLU and A512Softplus checkpoints, endpoint score vectors,
budget ledger, plan, controller state, report, and newly rescored H endpoint
vectors are create-only beneath the extension prefix. Historical Experiment-4
A trajectories may be linked as audit context but not imported as extension
results. The extension report is sealed separately and may link to, but may
not replace or amend, either parent report.
