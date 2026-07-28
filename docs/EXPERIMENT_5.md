# Experiment 5: Routed–Shared Hybrid Sparse Models

Status: frozen implementation contract, 2026-07-28

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

## Primary question

> Does a sparse model whose dictionaries mirror Gemma's actual routed and
> shared FFN branches reconstruct the exact full FFN output better, or reach a
> global full-width model's quality sooner, at a matched sparse-compute budget?

The transcoder panel is primary because every condition receives the same
source activation. The SAE panel is a mechanistic companion: it tests whether
autoencoding the native operands is more effective than autoencoding their
already-composed output.

The experiment distinguishes four claims:

1. **Alignment:** true expert identity must beat a load-preserving
   randomized-ID hybrid.
2. **Architecture:** the hybrid must beat a global candidate-matched
   dictionary.
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
bank ID. It preserves bank traffic, eight unique visible banks, shared
dictionary use, candidate count, parameter count, and L0 while destroying
stable expert meaning. The same mapping is used in training and held-out
evaluation. A global fixed permutation, per-token remapping, or
inference-only shuffle is invalid.

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

The panel retains the parent SAE's ReLU encoder semantics, exact-positive
TopK, initialization, optimizer, and decoder normalization. Routed source
centering uses one pooled raw-expert-output mean; shared source centering uses
one global `q_t` mean; and B/F use the one global `y_t` mean. There are no
per-expert means.

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

Every selection must contain exactly 64 strictly positive latents. Ties use
ascending global latent ID. Nonfinite state, insufficient positive
candidates, capacity overflow, token dropping, or L0 drift is a hard failure.

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
- exact-positive deterministic TopK for all four layouts;
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

## Explicitly deferred

Experiment 5 does not include:

- alternative routed/shared L0 splits;
- traffic-proportional bank widths;
- bank-local BatchTopK;
- dead-latent revival or auxiliary loss;
- pruning based on the Experiment-4 held-out nonfiring observation;
- a wider or narrower hybrid;
- router weights as encoder inputs; or
- a new activation harvest.

Those are Experiment-6 optimization questions. Changing any of them after
viewing Experiment-5 quality creates a new protocol ID.
