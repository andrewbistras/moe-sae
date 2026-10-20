# A512 component-closure report

Evidence: `4b8bf760befc6b8dbca2efe209e5de9c1ed47ce67f46e3a1cd9e0dbc9370329f`

Primary endpoint: component-target explained variance. Intervals use paired hierarchical seed/packed-sequence bootstrap.

Common target identity: all 72 cells share exactly 250,000 ordered evaluation tokens in 625 packed sequences, the same evaluation-target manifest, the same separately bound training-centering inputs, and recomposition sidecar. Token identity canonically binds ordered sequence IDs and positions; evaluation/training source identities are exact within panel (null for SAE, present for transcoder).
The JSON common-target audit inventories every cell, including its result and centering-receipt identities; centering-receipt hashes may differ because protocol/config provenance differs.

Common SST policy: condition A is canonical within each panel/seed/checkpoint; sequence IDs must match exactly. Other conditions may differ only within aggregate-relative <= 1e-08 and normalized-L1 <= 1e-07, with maximum per-sequence relative difference <= 1e-06.
Per-sequence relative differences divide by absolute A SST; a zero A entry maps equal zero to 0 and any unequal value to infinity. Scientific SST rows are strictly positive.
Maximum observed SST differences: aggregate-relative 1.466e-10; normalized-L1 1.982e-08; max-absolute 4.678e-04; per-sequence-relative 1.641e-07; unequal-sequence count 621.
The JSON audit inventories all 72 condition representations with role/source bindings, sequence identities, canonical-A and domain-separated SST identities, raw little-endian float64 hashes, totals, and drift metrics.

## sae

| tokens | A EV | R EV | B EV | F EV | A-R | A-B | A-F | F-B | G |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2,000,000 | 0.44516 | 0.26240 | 0.40686 | 0.41765 | 0.18276 | 0.03830 | 0.02751 | 0.01079 | 355.0% (defined) |
| 5,000,000 | 0.50745 | 0.37793 | 0.44909 | 0.47004 | 0.12952 | 0.05836 | 0.03742 | 0.02094 | 278.7% (defined) |
| 10,000,000 | 0.51844 | 0.38735 | 0.45764 | 0.48093 | 0.13110 | 0.06080 | 0.03751 | 0.02329 | 261.1% (defined) |

Cost ratios are emitted only where accelerator generation and timing scope/exclusions match exactly.

## transcoder

| tokens | A EV | R EV | B EV | F EV | A-R | A-B | A-F | F-B | G |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2,000,000 | 0.35789 | 0.01161 | 0.31685 | 0.27567 | 0.34628 | 0.04104 | 0.08222 | -0.04118 | undefined_nonpositive_denominator |
| 5,000,000 | 0.42167 | 0.10344 | 0.36480 | 0.33558 | 0.31823 | 0.05688 | 0.08609 | -0.02921 | undefined_nonpositive_denominator |
| 10,000,000 | 0.43174 | 0.11826 | 0.37146 | 0.34678 | 0.31348 | 0.06028 | 0.08496 | -0.02468 | undefined_nonpositive_denominator |

Cost ratios are emitted only where accelerator generation and timing scope/exclusions match exactly.

