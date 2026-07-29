# Experiment 5 routed-activation ladder v2

Status: corrected executable protocol

Protocol ID:
`gemma4-experiment5-routed-activation-ladder-v2-20260729`

## Why v2 exists

The first ladder contract contained only `A512ReLU`, `A512Softplus`, and
read-only `H`. That can isolate the encoder nonlinearity at one routed
geometry, but it cannot answer the width/overprovisioning question raised by
Experiment 4's dead-feature result.

Experiment 4's true-ID `A` was not the same model under another name. It used
128 banks of width 256 and TopK-2 in each of the eight visible banks. The first
ladder used width 512 and TopK-4. Omitting the smaller arm erased the scale
axis that motivated the follow-up.

V2 corrects the matrix before launch. It does not edit or reinterpret the
completed Experiment-4 or Experiment-5 protocols.

## Scientific matrix

All three A arms use true expert IDs, the same target, token order, optimizer,
centering, router-weighted objective, and exact branch recomposition.

| condition | activation | routed dictionary | candidates/token | routed L0 | parameters |
|---|---|---:|---:|---:|---:|
| `A256ReLU` | ReLU | 128 × 256 = 32,768 | 8 × 256 = 2,048 | 8 × 2 = 16 | 184,582,144 |
| `A512ReLU` | ReLU | 128 × 512 = 65,536 | 8 × 512 = 4,096 | 8 × 4 = 32 | 369,164,288 |
| `A512Softplus` | Softplus | 128 × 512 = 65,536 | 8 × 512 = 4,096 | 8 × 4 = 32 | 369,164,288 |
| `H` | Softplus | 128 × 512 routed + 4,096 shared | 8,192 total | 32 routed + 32 shared | 392,237,056 |

`H` is read-only Experiment-5 evidence. The ladder never retrains it.

The A256-to-A512 ReLU step holds the selected fraction within each visible
bank fixed:

```text
2 / 256 = 4 / 512 = 0.78125%
```

It doubles bank width, candidates, TopK, total L0, and trainable parameters.
It is therefore a frozen scale/density step, not a pure width-only,
L0-matched, parameter-matched, or compute-matched ablation. The report must
use that precise language.

## Seed provenance and the missing work

Experiment 4 produced exactly one compatible `A256ReLU` trajectory in the
SAE panel: seed 0 at 2M, 5M, and 10M checkpoints. Its production config
identity is:

```text
SAE: d2bc2a5f77ac303398b4570bd174ebb1afc4d69b9f94d1d5f601156b01ea8f50
```

Experiment 4's transcoder A256 seed 0 is not compatible: that panel used a
Softplus encoder. Calling it `A256ReLU` would silently change the treatment
and invalidate the scale contrast. V2 therefore excludes that checkpoint,
reuses only the SAE seed-0 checkpoints read-only, trains SAE A256ReLU seeds 1
and 2, and trains transcoder A256ReLU seeds 0, 1, and 2 from scratch. It
trains all three seeds for both A512 arms.

| panel/condition | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| SAE `A256ReLU` | immutable Experiment-4 reference | new v2 training | new v2 training |
| transcoder `A256ReLU` | new v2 training | new v2 training | new v2 training |
| both `A512ReLU` | new v2 training | new v2 training | new v2 training |
| both `A512Softplus` | new v2 training | new v2 training | new v2 training |
| both `H` | immutable Experiment-5 reference | immutable Experiment-5 reference | immutable Experiment-5 reference |

The training worker must reject an attempt to train
`SAE A256ReLU, seed=0` under the v2 identity. It must accept transcoder
`A256ReLU, seed=0` as new training. SAE reuse happens through a hash-pinned
reference manifest, never by resuming the historical checkpoint into a new
run namespace. The historical transcoder Softplus checkpoint must never bind
to any v2 A256ReLU cell.

Most importantly, one reused SAE seed is not three-seed evidence. The report
must not emit an across-seed A256 comparison unless that panel's three exact
cells are present at all three milestones.

## Authorized comparisons

V2 has two interval-bearing paired contrasts at every common endpoint:

```text
scale effect      = EV(A512ReLU)     - EV(A256ReLU)
activation effect = EV(A512Softplus) - EV(A512ReLU)
```

Each seed-level interval resamples packed sequences on the identical held-out
token rows and common SST. The across-seed interval hierarchically resamples
seed and packed sequence and exists only after all three seed effects
validate.

The scale contrast answers whether the larger constant-density routed model
earns its doubled sparse and parameter budget. The activation contrast
isolates ReLU versus Softplus at identical A512 geometry.

`H` remains matched descriptive context. Its oracle-shared and learned-shared
points do not turn either A comparison into a controlled shared-dictionary
effect, and no A-versus-H causal or efficiency headline is authorized here.

## Exact workload

New training:

```text
SAE A256ReLU:        2 missing seeds = 2 trajectories
transcoder A256ReLU: 3 new seeds     = 3 trajectories
A512ReLU:            2 panels × 3 seeds = 6 trajectories
A512Softplus:        2 panels × 3 seeds = 6 trajectories
total new training                       17 trajectories
```

Analyzed trajectories:

```text
17 new A trajectories
+ 1 historical SAE A256ReLU seed-0 trajectory
+ 6 read-only H trajectories
= 24 trajectories
```

Evaluation:

```text
2 panels × 3 seeds × 3 milestones × 4 conditions = 72 cells
```

The exact training budget remains 10,000,000 scientific tokens, effective
batch 16,384, 611 updates, and checkpoints at 2M, 5M, and 10M for scientific
evaluation. No new activation harvest or cross-region bulk copy is allowed.

Execution is work-conserving over one immutable pool of 3–18 already deployed
TRC resources. Each trajectory has exactly one owner at a time, but the 17
independent trajectories may occupy all compatible owners concurrently.
Panel/seed cohorts never cross accelerator, regional control, cache, or bulk
artifact planes. Preemption fences the failed owner and resumes the same
trajectory from its last committed checkpoint on a compatible replacement;
it never creates a second live owner.

The 72 evaluation cells remain separate scientific records but execute as six
resident `(panel, seed)` bundles. Each bundle waits for all three current-v2
training parents in that cohort (with historical SAE A256 seed 0 supplied by
its validated reference), restores process/runtime state once, evaluates the
12 condition-by-milestone cells in deterministic order, and seals both the 12
cell completions and one bundle completion. This execution change removes 66
redundant TPU bootstrap/cache cycles without changing a comparison, score, or
report row.

## Result provenance

Every reported A row states one of:

```text
historical_experiment4_seed0_reference
protocol_v2_new_training
```

The first label is valid only for `SAE A256ReLU, seed=0`. Every other A row,
including `transcoder A256ReLU, seed=0`, must carry the second label. Measured
historical chip-seconds remain historical cost; reuse does not make them zero.
Newly incurred wall time and cash are reported separately from total
scientific compute.

The v2 controller preserves the previously incurred Experiment-5 cash
estimate and every new attempt reservation in its sealed cost projection, but
those values are measurement-only. They do not stop, delay, or reject this
grant-funded TPU extension. The retired ready-three `$45` protected launch
watchdog is not an execution gate for v2.

Because SAE A256's three seeds cross two protocol execution identities, any
aggregate cost statement for that panel must expose the per-seed hardware and
timing rows. It may summarize scientifically compatible quality only after
exact model/data/order/objective/checkpoint compatibility has been proved.

## Fail-closed boundaries

The v2 run is invalid if any of the following occurs:

- an SAE A256 seed-0 training task is created instead of a read-only reference;
- the historical Softplus transcoder A256 checkpoint is bound to A256ReLU;
- any required A256 seed is absent but an across-seed result is reported;
- the historical source condition, config hash, phase, checkpoint, or seed
  differs;
- A256 is evaluated as width 512, TopK-4, L0 32, or Softplus;
- either paired contrast uses different token identities or target SST rows;
- an H checkpoint is retrained or silently substituted;
- v1 and v2 score/report schemas are merged under one protocol identity; or
- the scale contrast is described as width-only or compute-matched.

## Checked-in contracts

- `configs/gemma4_experiment5_routed_ladder_sae_v2.yaml`
- `configs/gemma4_experiment5_routed_ladder_transcoder_v2.yaml`
- `src/moe_sae/experiment5_routed_ladder_v2_config.py`
- `src/moe_sae/experiment5_routed_ladder.py`

Their canonical config identities are:

```text
SAE v2:        729902a48f48859eb0ce1aac386e4a5b16dd3c32417dc361af9633a41e8349b2
transcoder v2: 12062fbc4073bc93d72a55643ff3031d6166996e8454b35e557596e6361a9787
```

The v1 files remain historical implementation context. New campaign state,
scores, and reports must use the v2 protocol and schema identities.
