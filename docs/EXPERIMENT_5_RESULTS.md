# Experiment 5 v3 results

Status: **sealed and source-verified against every pinned parent**

Protocol:
`gemma4-experiment5-routed-activation-ladder-v3-20260729`

The numerical authority is the sealed JSON report, not this summary.

## Verdict

At matched A512 geometry, parameter count, candidate count, and maximum
selected-slot count, `A512ReLUInclusiveTopK` reconstructed the held-out
component better than `A512Softplus` in both panels.

The protocol-defined effect is:

```text
EV(A512Softplus) - EV(A512ReLUInclusiveTopK)
```

A negative value therefore favors ReLU.

| panel | ReLU mean EV | Softplus mean EV | Softplus − ReLU | crossed 95% interval | seeds favoring ReLU | verdict |
|---|---:|---:|---:|---:|---:|---|
| SAE | 0.51844384 | 0.47874741 | -0.03969643 | [-0.04250111, -0.03620697] | 3/3 | `relu_inclusive_topk_better` |
| transcoder | 0.43174280 | 0.39993109 | -0.03181171 | [-0.03273180, -0.03095912] | 3/3 | `relu_inclusive_topk_better` |

The JSON field `positive_seed_count` is 0/3 because it counts positive
Softplus-minus-ReLU effects. “Seeds favoring ReLU” above is the relabeled
complement.

The intervals use 20,000 deterministic crossed seed-by-packed-sequence
bootstrap replicates. No multiplicity adjustment was applied.

This is an activation-policy result, not an activation-only or
fixed-positive-L0 result. Both conditions select exactly 32 slots per token.
Softplus makes all 32 positive. ReLU may select real zeros when a visible
bank has fewer than four positive values; those zeros decode to exactly zero
and are not counted as positive features.

## Learning curve

Component EV is averaged over the three paired training seeds.

| panel | checkpoint | ReLU EV | Softplus EV | Softplus − ReLU | crossed 95% interval |
|---|---:|---:|---:|---:|---:|
| SAE | 2M | 0.44515859 | 0.40031946 | -0.04483912 | [-0.04633402, -0.04349307] |
| SAE | 5M | 0.50745299 | 0.46606004 | -0.04139296 | [-0.04414297, -0.03793838] |
| SAE | 10M | 0.51844384 | 0.47874741 | -0.03969643 | [-0.04250111, -0.03620697] |
| transcoder | 2M | 0.35788867 | 0.31174928 | -0.04613939 | [-0.04760562, -0.04472059] |
| transcoder | 5M | 0.42167327 | 0.38610852 | -0.03556475 | [-0.03653021, -0.03465193] |
| transcoder | 10M | 0.43174280 | 0.39993109 | -0.03181171 | [-0.03273180, -0.03095912] |

The advantage appears by 2M tokens, survives to 10M, and has the same sign
in all three seeds at every component checkpoint.

## Ten-million-token sparsity

These values are three-seed means from the common 250,000-token held-out
capture.

| panel | condition | selected slots | positive L0 | zero-selected fraction | held-out nonfiring rate |
|---|---|---:|---:|---:|---:|
| SAE | ReLU | 32 | 32.0000 | 0.0000 | 0.5326 |
| SAE | Softplus | 32 | 32.0000 | 0.0000 | 0.0254 |
| transcoder | ReLU | 32 | 31.4600 | 0.0169 | 0.0253 |
| transcoder | Softplus | 32 | 32.0000 | 0.0000 | 0.0240 |

“Held-out nonfiring rate” is the finite-evaluation fraction of dictionary
features that did not fire. It is not a cumulative training dead-feature
rate.

The transcoder result directly shows that selected-slot budget and positive
support are different quantities: ReLU reconstructs better while averaging
about 31.46 positive values in its 32 selected slots.

The SAE ReLU dictionary has a high held-out nonfiring rate but still has the
better of the two matched A512 reconstructions. That motivates a later width
or pruning sweep; it does not prove that narrowing would be lossless.

## Marginal training cost

| panel | seed | hardware | ReLU chip-s | Softplus chip-s | Softplus/ReLU |
|---|---:|---|---:|---:|---:|
| SAE | 0 | v5litepod-16 | 21,246.661 | 21,419.297 | 100.81% |
| SAE | 1 | v5litepod-16 | 21,599.317 | 21,057.569 | 97.49% |
| SAE | 2 | v5litepod-16 | 21,303.397 | 21,629.863 | 101.53% |
| transcoder | 0 | v5litepod-16 | 22,079.245 | 22,135.411 | 100.25% |
| transcoder | 1 | v4-32 | 21,952.429 | 21,332.971 | 97.18% |
| transcoder | 2 | v4-32 | 20,855.676 | 20,524.119 | 98.41% |

| panel | median paired Softplus/ReLU | comparison status |
|---|---:|---|
| SAE | 100.81% | same accelerator type within every seed |
| transcoder | 98.41% | same accelerator type within every seed |

The observed marginal costs were close to parity: median paired
Softplus/ReLU ratios were 100.81% for SAE and 98.41% for the transcoder. No
cost-equivalence interval was estimated. These chip-seconds include the
fixed-capacity scan and synchronized optimizer execution, including input
wait. They exclude shared harvesting, prefetch, qualification, lowering,
compilation, checkpoint serialization, and evaluation.

## Historical A256 context

`A256Native` is panel-native historical context, not one synthetic
cross-panel treatment. Only completed historical seed 0 is admissible in
this context.

| panel | historical source | policy | checkpoint | component EV | total L0 | held-out nonfiring |
|---|---|---|---:|---:|---:|---:|
| SAE | `A256ReLU` | strict-positive TopK-2 | 2M | 0.41024758 | 16 | 0.2174 |
| SAE | `A256ReLU` | strict-positive TopK-2 | 5M | 0.47002126 | 16 | 0.5270 |
| SAE | `A256ReLU` | strict-positive TopK-2 | 10M | 0.47833751 | 16 | 0.5707 |
| transcoder | `A256Softplus` | exact-positive TopK-2 | 2M | 0.30295650 | 16 | 0.0232 |
| transcoder | `A256Softplus` | exact-positive TopK-2 | 5M | 0.39058841 | 16 | 0.0263 |
| transcoder | `A256Softplus` | exact-positive TopK-2 | 10M | 0.39944633 | 16 | 0.0257 |

The transcoder A256-to-A512 Softplus comparison is activation-matched
historical scale context: for seed 0, doubling width and selected L0 changed
10M EV from 0.39945 to 0.39975, only about +0.00030. The A512 Softplus
three-seed mean was 0.39993, while the A512 ReLU-policy mean reached 0.43174.
This is descriptive evidence that doubling width and selected L0 under
Softplus did not reproduce the ReLU-sized gain in this seed-0 comparison.

The SAE historical comparison is less clean because A256 enforced
strict-positive ReLU selection while A512 uses inclusive-zero selection.
Neither A256 curve has an across-seed interval, the two A256 panels may not
be pooled, and no A256 row entered the primary A512 inference.

`H` is attached only as hash-pinned read-only lineage. No H score enters the
A512 estimand, interval, or verdict.

## Interpretation and boundary

The result supports three bounded conclusions:

1. At Gemma block 15, on this target and C4 sample, inclusive-zero ReLU is
   materially better than exact-positive Softplus at matched A512 geometry.
2. The advantage is early, seed-consistent, and accompanied by observed
   marginal training costs close to parity.
3. Strict-positive ReLU selection was structurally infeasible in the earlier
   trajectories. Inclusive-zero TopK allowed honest completion and produced
   better reconstruction than the matched Softplus policy.

The experiment does not isolate the scalar nonlinearity alone: ReLU uses
inclusive-zero TopK while Softplus uses exact-positive TopK. The authorized
claim is therefore an activation-policy comparison. Generalization beyond
Gemma block 15, this target family, C4, and three seeds remains untested.

Target-only SST is bit-exact within each panel/seed evaluation group. The
v4 and v5e reductions produce distinct but numerically equivalent
representations: the largest component aggregate-relative difference is
1.103e-9 and the largest normalized-L1 difference is 4.675e-8, both inside
the sealed 1e-8 and 1e-7 tolerances.

## Integrity and provenance

The create-only primary report was sealed and then deterministically rebuilt
by a separate verification invocation from every pinned parent. Verification
returned `status: primary_report_verified`.

| artifact | immutable controller path | semantic/self SHA-256 | file SHA-256 |
|---|---|---|---|
| primary JSON | `/mnt/moe-sae-control/experiment5/extensions/routed-activation-ladder-v3/reports/600e4522f7e2/primary-report.json` | `692e12de616091c6768776f97471d0705fbcb14f7b3ab6ff90d12c3b5bc17c9d` | `5202e4d3472b034d791865fd0e80dade5f243905117b886a4df275f40bab4359` |
| primary Markdown | `/mnt/moe-sae-control/experiment5/extensions/routed-activation-ladder-v3/reports/600e4522f7e2/primary-report.md` | — | `73b1e350fdd5f9f63ceac9e0074991fc9fa7d3e905e52604e2b555e8166a838a` |
| evaluation plan | `/mnt/moe-sae-control/experiment5/extensions/routed-activation-ladder-v3/contracts/primary-evaluation-plan-8389720686b4-a1.json` | `d06ff7debc624ba1eb29f8fec74ed927ade15c73c63d67e46ed2edc82d1fab95` | `613dc8bd69df3be6e187a9a9f73927a9cf9f72a7f88a8cbdb1a40fe33ff0a7f2` |
| evaluation completion | `/mnt/moe-sae-control/experiment5/extensions/routed-activation-ladder-v3/primary-evaluation/controller/8389720686b4/primary-evaluation/primary-evaluation-complete.json` | `ba3bc34dda9ec133cc4601ecfc4b08ffec6d6b1b819bb119021c0cfc8a594fac` | `7069238006c5fbeb2f912b2cf27d6f146fe78b85db7156afc793af71c454960d` |
| training plan | `/mnt/moe-sae-control/experiment5/extensions/routed-activation-ladder-v3/contracts/training-plan-8dcfadee4a51-a1.yaml` | `eb52be80dd7a98cf508e529f28f487d5aca1cf0c4c6169f11f2d8bc1194dac43` | `dd4f1c8aed2d5fb51f703ff69b304abcaf59577fcf14787a6933a4cb690cd155` |
| Softplus references | `/mnt/moe-sae-control/experiment5/extensions/routed-activation-ladder-v3/contracts/softplus-references-c02e96ef3170-a1.yaml` | `f4b6d30cadd55d5a9feedb5dbbd12e40e4e0fa6fac0ea237c648d18de3e5cd29` | `30c6b31a4411c4660b2fe8518704e06a799f7485b6547bdafb22efa10c5f3267` |
| A256 provenance | `/mnt/moe-sae-control/experiment5/extensions/routed-activation-ladder-v3/contracts/a256native-provenance-addendum-bafbb2c93828-a1.yaml` | `d9d2dde064f36c332ac1cde9499198810a99ca4da4741e673e116db374abc6fe` | `4632d2995c155eae906e17624a4d9af1659580e658f6631a2dda427195f22e5c` |
| A256 context scores | `/mnt/moe-sae-control/experiment5/extensions/routed-activation-ladder-v3/contracts/a256native-context-scores-bafbb2c93828-a1.yaml` | `567392932d08797d414765f43d0b1c8579b0132ba4deb4999108a230d9c6282f` | `792996e4dd26e03fca20fe9ee7e361969d09371a5be1244a17ae67c05f17cab0` |
| H references | `/mnt/moe-sae-control/experiment5/extensions/routed-activation-ladder-v2/contracts/h-references-a1c04eb79050-a1.yaml` | `c5e019060ae690ae49348c67028649d06ba9dd07e95a1f36067843e5264dec91` | `abac6632b894169492c125dc358ac8e943cfdf4170b0721eda76f822304408d5` |

Source revisions:

- report and sealer: `600e4522f7e2efed6c0c644fac1030b88102269a`
- evaluation: `8389720686b4f449395ac230cad58383d0867872`
- fresh v3 ReLU training: `8dcfadee4a517029b3c1c0c7f2bb830e9d35f12f`
- imported v2 Softplus training: `a1c04eb790508367a96224ecd19c356ba7b6db26`
- imported v2 source plan: `c365da30a6a1f82c3809ea7cff3bee9a0ad3d3dc3e6c015001b8dc82274cd1df`
