# Experiment 5 routed-activation policy v3

Status: executable replacement protocol

Protocol ID:
`gemma4-experiment5-routed-activation-ladder-v3-20260729`

V3 does not alter, resume, or reseal v2. Its completed trajectories may be
referenced read-only only under the explicit equivalence rules below;
partial/terminal artifacts remain evidence of a failed feasibility
assumption.

## Why the protocol reset is necessary

The v2 ReLU cells coupled two different claims:

1. the encoder nonlinearity is ReLU; and
2. every visible bank must always supply exactly `k` strictly positive
   activations.

The second claim is not implied by the first. The transcoder's three
`A256ReLU` trajectories remained numerically finite and reconstructed
normally, but eventually encountered a bank with fewer than two positive
ReLU values. Seed 1 accepted 42 updates (688,128 tokens) and seed 2 accepted
43 updates (704,512 tokens) before the terminal support check. Earlier
Experiment-3 and Experiment-5 ReLU trajectories exhibited the same class of
failure at different token counts. This is repeated structural evidence, not
a corrupt checkpoint, capacity overflow, or one unlucky seed.

Padding the missing slots with epsilon would make the run continue but would
falsely report inactive ReLU coordinates as positive features. Softplus
avoids the feasibility failure precisely because its output is positive, so
comparing exact-positive Softplus with epsilon-modified ReLU would conflate
the activation with an undocumented sparsity intervention.

V3 makes the actual intervention explicit. ReLU receives a fixed
*selected-slot budget*; the number of selected values that are positive is a
measured outcome. Softplus retains an exact-positive budget. The result is an
activation-policy comparison at matched geometry and maximum selected slots,
not a comparison at fixed exact positive L0.

## Scientific matrix

Both panels analyze the following two cells for seeds 0, 1, and 2:

| condition | activation | selection policy | dictionary | candidates/token | selected slots/token | positive L0 |
|---|---|---|---:|---:|---:|---|
| `A512ReLUInclusiveTopK` | ReLU | deterministic inclusive-zero TopK-4 per visible bank | 128 × 512 = 65,536 | 8 × 512 = 4,096 | exactly 32 | measured, 0–32 |
| `A512Softplus` | Softplus | deterministic exact-positive TopK-4 per visible bank | 128 × 512 = 65,536 | 8 × 512 = 4,096 | exactly 32 | exactly 32 |

Each cell has 369,164,288 trainable encoder/decoder/bias parameters. Both use
the same target, token order, centering, optimizer, router-weighted objective,
candidate count, and maximum number of decoded columns.

For `A512ReLUInclusiveTopK`, TopK ranks the real ReLU outputs.
Zero values are eligible when fewer than four positive values exist. Ties are
resolved by
ascending global latent ID. The selected decoder columns are multiplied by
their actual values, so a zero-valued slot contributes exactly zero. The
implementation may never turn zero into epsilon or count zero as positive
L0.

For `A512Softplus`, the existing smallest-normal-FP32 floor and
exact-positive check remain in force.

## Panel-native A256 context

`A256Native` means “the historical Experiment-4 A cell native to this panel.”
It is deliberately not one cross-panel synthetic treatment:

| panel | displayed source | activation | selection history | frozen v3 declaration |
|---|---|---|---|---|
| SAE | `A256ReLU` | ReLU | strict-positive TopK-2 | `035e1afc91f7351fda3128582ac52e2051fcb0741f6f71913f5141aaa1c7ab59` |
| transcoder | `A256Softplus` | Softplus | exact-positive TopK-2 | `1742ebc1b7993dffd0b35665695a074c3d10025f5d410021dd66ccb1bb221ec9` |

The transcoder value in the last column is a mistaken declaration in the
already-trained v3 metadata, not the observed identity of the historical
finalized config. Do not edit the frozen v3 config to hide that discrepancy.
Before A256 enters a contextual report, seal a create-only provenance
addendum from the original horizon plan:

```text
scripts/tpu/build_experiment5_routed_ladder_v3_a256_provenance_addendum.py \
  --horizon-plan /path/to/the/sealed/direct-horizon-plan.yaml \
  --output /new/create-only/a256native-provenance-addendum.yaml
```

The addendum hashes the real config bytes, derives the real semantic config
identity, records declared versus observed values, and pins the run contract,
training result, measured cost, and 2M/5M/10M checkpoint pointers for both
panels. It requires the SAE declaration to match and the transcoder
declaration to mismatch. Its scope flags permanently exclude A256 from the
matched A512 primary inference and estimand.

Pass the sealed addendum to both primary-report sealing and verification:

```text
scripts/tpu/seal_experiment5_routed_ladder_v3_report.py seal \
  --evaluation-plan /path/to/primary-evaluation-plan.yaml \
  --training-plan /path/to/v3-training-plan.yaml \
  --softplus-references /path/to/softplus-references.yaml \
  --a256-provenance-addendum /path/to/a256native-provenance-addendum.yaml \
  --output-json /new/create-only/primary-report.json \
  --output-markdown /new/create-only/primary-report.md
```

This attaches two validated parent lineages, their measured training costs,
and six checkpoint pointers under `optional_context.A256Native`. It does not
add A256 rows to the 36-cell A512 matrix, contrasts, primary verdict, or A512
cost comparison. The provenance addendum contains no A256 quality scores; a
quantitative A256-to-A512 curve requires a separately sealed context-score
manifest.

Only historical seed 0 is available. Both references are read-only, with
2M/5M/10M checkpoints. They are context points, not substitutes for three
new seeds.

The transcoder `A256Softplus` to `A512Softplus` curve is activation-matched
historical scale context. The SAE `A256ReLU` to
`A512ReLUInclusiveTopK` curve is descriptive until replay establishes that
the inclusive policy would not have changed the historical selected values.
Neither one-seed curve gets an across-seed confidence interval.

`H` remains the read-only Experiment-5 hybrid context. No A-versus-H causal
headline is authorized.

## Exact v2 reuse and new training

Import is allowed only for a completed trajectory whose immutable manifest,
config identity, condition, seed, and 2M/5M/10M checkpoints all validate.
Import means attaching read-only evidence to the corresponding v3 cell. It
never means resuming a v2 optimizer state under v3.

| panel/condition | imported v2 seeds | new v3 seeds | reason |
|---|---|---|---|
| SAE `A512ReLUInclusiveTopK` | none | 0, 1, 2 | ReLU selection policy changed; v2 ReLU never crosses the v3 policy boundary |
| SAE `A512Softplus` | 0, 1, 2 | none | activation, selection rule, geometry, objective, and training graph are identical |
| transcoder `A512ReLUInclusiveTopK` | none | 0, 1, 2 | ReLU selection policy changed, and the v2 attempts are failed/partial |
| transcoder `A512Softplus` | 0, 1, 2 | none | activation, selection rule, geometry, objective, and training graph are identical |

This reduces new training from twelve trajectories to six without changing
the estimand. If the import manifest cannot prove one listed completed role,
that role becomes missing work; it is never accepted approximately. Failed or
partial transcoder ReLU results cannot enter the v3 quality grid under any
name.

## Primary estimand and report labels

At each common checkpoint, panel, and seed, the primary paired contrast is:

```text
label:
  A512Softplus-A512ReLUInclusiveTopK

estimand:
  activation_policy_effect_at_matched_geometry_and_max_selected_slots

effect:
  EV(A512Softplus) - EV(A512ReLUInclusiveTopK)
```

The report must state all five matching facts:

- bank width is fixed;
- trainable parameter count is fixed;
- candidate count is fixed;
- maximum selected-slot count is fixed; and
- exact positive L0 is **not** fixed.

Every condition/checkpoint record must include:

```text
selected_slot_count
positive_l0
zero_selected_slot_fraction
EV
NMSE
SSE
SST
tokens
wall time
chip-seconds
```

`selected_slot_count` is 32 for both new conditions. `positive_l0` counts
strictly positive decoded values. `zero_selected_slot_fraction` is
`1 - positive_l0 / selected_slot_count`, aggregated with scientific-token
weights. Softplus should therefore report positive L0 32 and zero fraction
zero; ReLU is allowed to report fewer positives without failing.

Packed-sequence paired intervals use identical held-out rows and an exact
common SST within each panel/seed/endpoint six-cell group. Each group records
its SST hash and accelerator type. Cross-hardware SST representations need
not be bit-identical, but they must have aggregate-relative difference at
most `1e-8` and normalized-L1 difference at most `1e-7`; otherwise sealing
fails. The across-seed primary interval is emitted only after all three paired
seeds validate. Because every seed is evaluated on the same packed sequences,
the hierarchical interval treats seed and sequence as crossed factors: each
bootstrap replicate draws seeds and one common sequence-index sample, then
computes each sampled seed's effect using that seed's internally exact SST
denominator before averaging.

Raw marginal chip-seconds are retained for every trajectory. A paired
Softplus/ReLU cost ratio is emitted only when both trajectories used the same
accelerator type; cross-generation pairs are explicitly non-comparable.
The report CLI's `validate` command checks only self-hashes, internal
arithmetic, and deterministic Markdown and returns a non-authoritative
receipt. Only `verify`, supplied with the immutable evaluation/training plans
and reference manifests, authenticates the scientific report against source
evidence.

## Immutable v2 boundary

The checked-in v2 config identities are pinned as:

```text
SAE:        eec72b479ef470fa02917bef78a731503a4ad890e86a4571eae5ba64e9217f46
transcoder: 12062fbc4073bc93d72a55643ff3031d6166996e8454b35e557596e6361a9787
```

V3 may read terminal records for the statement “the strict-positive ReLU
policy was not feasible.” It may import only the completed condition/seed
roles enumerated above through a content-hashed reference manifest. All other
v2 quality metrics, partial checkpoints, and cells remain outside the v3
comparison grid. Every missing role starts from scratch under the v3
protocol/config identity.

## Workload and execution

New training is:

```text
SAE A512ReLUInclusiveTopK seeds 0–2:         3 trajectories
transcoder A512ReLUInclusiveTopK seeds 0–2:  3 trajectories
total new training:                           6 trajectories

read-only completed v2 imports:               6 trajectories
total v3 A512 analysis grid:                 12 trajectories
```

Each trajectory consumes exactly 10,000,000 scientific tokens in 611 updates
at effective batch 16,384, with the final 10,624 padded positions masked.
Evaluation checkpoints are 2M, 5M, and 10M. The campaign reuses the sealed
10M training target, 250k held-out target, transcoder source overlay, and
recomposition sidecar. No new activation harvest or cross-region bulk copy is
authorized.

The A512 evaluation grid contains 36 cells:

```text
2 panels × 2 conditions × 3 seeds × 3 checkpoints = 36

18 cells descend from 6 newly trained v3 trajectories
18 cells are read-only mappings from 6 completed equivalent v2 trajectories
```

Panel-native A256 and H checkpoints are resolved from their pinned historical
namespaces and reported separately as context. A preemption may resume one
new trajectory from its last committed v3 checkpoint. An imported v2
trajectory remains read-only evidence and is never restored into a v3
optimizer.

The durable training-only operator surface is:

```text
scripts/tpu/build_experiment5_routed_ladder_v3_training_plan.py
scripts/tpu/run_experiment5_routed_ladder_v3_training_campaign.py validate
scripts/tpu/run_experiment5_routed_ladder_v3_training_campaign.py status
scripts/tpu/run_experiment5_routed_ladder_v3_training_campaign.py run
```

The build step seals exactly six fresh ReLU roles onto six distinct admitted
resource owners. `validate` proves the plan/config/commit identities without
launching. `status` is read-only and classifies durable completions,
retryable attempts, and in-flight owners. `run` installs SIGINT/SIGTERM safe
stop handling and delegates retries/checkpoint resume to the durable
controller. Softplus imports and report assembly are not training
completions and cannot satisfy any of those six roles.

## Fail-closed rules

Stop and preserve diagnostics if:

- a selected ReLU value is negative;
- selected-slot count differs from 32;
- ReLU positive L0 lies outside `[0, 32]`;
- Softplus positive L0 differs from 32;
- deterministic global-ID tie-breaking changes on replay;
- an unlisted, incomplete, or hash-mismatched v2 role is offered as a v3
  quality cell;
- either panel binds the other panel's A256 historical config;
- any target/source/sidecar hash differs from the frozen parent contract; or
- a report labels the primary contrast as fixed-exact-L0.

Do not stop merely because `A512ReLUInclusiveTopK` has fewer than 32 positive
values. That is now a scientific measurement, not a runtime invariant.

## Checked-in contracts

- `configs/gemma4_experiment5_routed_ladder_sae_v3.yaml`
- `configs/gemma4_experiment5_routed_ladder_transcoder_v3.yaml`
- `src/moe_sae/experiment5_routed_ladder_v3_config.py`

Their semantic identities are computed from canonical parsed content. The
validator also projects the unchanged model/artifact/optimizer/runtime/H
surface through the audited v1 validator, so v3 cannot silently relax those
parts of the protocol.

```text
SAE v3:        725c1d846bc2189096f5742a758e6845db4d7cad1fdc203d950e50de152bafc5
transcoder v3: 059d845bbb7b1140ac50f6866efc7727fd2ee88fcc4e788d147ddcfa86c9798f
```
