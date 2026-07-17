# Experiment contract

## Question

The experiment estimates how much of a large global SAE's held-out
reconstruction improvement can be recovered by exposing each token only to
feature banks named by Gemma 4's router, and what that restriction saves in
measured SAE training cost.

The base/PT checkpoint is `google/gemma-4-26B-A4B` at immutable revision
`24548b62aa021d562695c04aaf7758a1ea47990b`. The implementation is pinned to
an exact repository file/byte inventory, then to MaxText commit
`48c7da3e28e8f57fe36ddf09d00eec53072a2ea0`, and validates its 30 blocks,
hidden width 2,816, 128 routed experts, top-8 routing, and one shared expert
before compilation.

## Capture target

Sequences are deterministic length-1,024 packs from a bounded `allenai/c4`
snapshot at revision `1588ec454efa1a09f29cd18ddd04fe05fc8653a2`. The exact
one-file train and validation source inventories are size- and SHA-256-pinned.
The two-file Gemma tokenizer view is independently byte-pinned to the model
revision. Padding, BOS, and EOS are excluded from scientific token counts. The
prefix adapter:

1. embeds tokens and executes blocks 1–14;
2. executes block 15 attention and its residual;
3. runs the target routed experts and captures their aggregate after
   `post_feedforward_layernorm_2`;
4. recomputes the small router gate for exact top-8 IDs and normalized weights;
5. stops before the shared expert, outer FFN normalization/residual, blocks
   16–30, final normalization, and logits.

This aggregate is the only valid common target. Gemma 4 combines the weighted
routed output before a nonlinear branch RMSNorm, so normalized per-expert
contributions cannot be recovered by distributing that aggregate backward.

The sealed capture schema is:

| field | storage |
|---|---|
| `activation` | BF16 `[N, 2816]` |
| `expert_ids` | uint8 `[N, 8]` |
| `router_weights` | BF16 `[N, 8]` |
| `sequence_id` | int64 `[N]` |
| `position` | uint16 `[N]` |

Training has exactly 50,000,000 tokens and held-out evaluation exactly
2,000,000. Shards contain at most 131,072 scientific tokens, are named by
SHA-256, carry FP64 sufficient statistics, and are visible to readers only
after the merged manifest validates and seals the store.
The source fingerprint is the canonical hash of a strict staging manifest;
its ordered ArrayRecord SHA-256, size, and independently reopened record count
are reverified before harvesting, so staged bytes cannot be substituted after
the C4 staging declaration. Scientific-token counts are enforced again from
the actual packed batches during capture.
Every capture also binds the self-hashed successful HF/MaxText checkpoint
conversion attestation; downstream campaigns and reports reject provenance
that cannot be traced to the required logit-parity check.

## Conditions

All conditions use the same global training mean, normalized-SSE loss, and
strictly positive per-token TopK-64. Ties are broken by ascending global latent
ID.

| condition | dictionary | parameters | candidates/token |
|---|---:|---:|---:|
| A | 128 banks × 1,024 | 738,328,576 | 8,192 from top-8 banks |
| B | global 8,192 | 46,145,536 | 8,192 |
| F | global 131,072 | 738,328,576 | 131,072 |

A uses IDs only; router weights are diagnostic. Each visible bank encodes the
same globally centered target, and selected decoder columns sum into one
reconstruction. Dense latent tensors, BatchTopK, threshold calibration, local
expert losses, dead-feature losses, and conditions C/D/E do not exist.

## Optimization

Seeds 1, 2, and 3 use matched activation order. An update has 16,384 tokens in
four 4,096-token microbatches. There are 3,052 updates; the last 3,968 positions
are masked to retain exactly 50M tokens.

Parameters/matmuls are BF16; Adam moments and reductions are FP32. Adam uses
beta1=.9, beta2=.999, epsilon=1e-8, no weight decay, and global gradient
clipping at 1.0. The learning rate warms linearly for 250 updates to 3e-4, then
decays with a cosine to 3e-5. Decoder columns start at unit norm, the encoder
starts as their transpose, and bias starts at zero. Decoder gradients are
projected onto the unit-sphere tangent space and columns are renormalized after
every update.

The data axis spans processes and the feature axis spans chips local to each
process. Feature-sharded TopK gathers only each shard's local 64 candidates and
then performs the deterministic global TopK. Sparse decode gathers only the 64
selected columns. A uses a capacity fixed from each seed's deterministic route
order, with a rounded 10% margin; any overflow is fatal.
Evaluation derives an independent A capacity from the fixed held-out order, so
a legitimate held-out route imbalance is not treated as training overflow.

## Result contract

For condition `j`, with a common held-out SST:

```text
EV_j = 1 - SSE_j / SST_eval
G = 100 * (mean(EV_A) - mean(EV_B)) / (mean(EV_F) - mean(EV_B))
C = 100 * median_seed(chipseconds(A_seed) / chipseconds(F_seed))
```

Marginal chip-seconds include synchronized optimizer steps, input stalls, and
condition A's measured per-seed fixed-order capacity prepass. They exclude
compilation, checkpoint serialization, evaluation, and the shared harvest.
Those costs are recorded separately. Shared-harvest publication/verification
time is included in harvest chip-seconds, and harvest-amortized ratios for one
and three seeds are secondary results.

`G` is never clipped. It is undefined when mean F-minus-B EV is non-positive.
When the paired 95% interval for F-minus-B contains zero, the ratio is labeled
unstable/exploratory. A stable report renders exactly this claim:

> Router-conditioned A recovers G% of the F-over-B held-out reconstruction
> gain at C% of F's measured marginal SAE training cost (6.25% nominal
> candidate exposure).

Each report also carries per-seed EV/NMSE, exact L0, dead rate, exposure,
chip-seconds, tokens/s, harvester useful-prefix MFU, persisted goodput, and wall
time.
