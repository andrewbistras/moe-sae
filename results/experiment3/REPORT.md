# Experiment 3: expert-aligned contribution transcoders

Status: **expert alignment transferred; preregistered cost-efficiency result
not established**

This is the sealed seed-0 mechanism pilot for protocol
`gemma4-contribution-transcoder-v2-20260727`.

## Result

| condition | component EV | routed EV | oracle-shared EV | dead rate | chip-seconds |
|---|---:|---:|---:|---:|---:|
| A | 0.187430 | -0.404303 | 0.483760 | 2.2308% | 22,818.1 |
| R | -0.047151 | -0.523583 | 0.430604 | 0.0000% | 22,973.0 |
| B | 0.155317 | -0.491658 | 0.449652 | 0.0000% | 22,013.9 |
| F | 0.134082 | -0.484048 | 0.425905 | 0.0000% | 22,074.8 |

Paired packed-sequence effects:

| endpoint | contrast | effect | 95% interval |
|---|---|---:|---:|
| component | A minus R | 0.234581 | [0.229390, 0.239810] |
| component | A minus B | 0.032113 | [0.031058, 0.033294] |
| component | A minus F | 0.053348 | [0.052149, 0.054632] |
| component | F minus B | -0.021235 | [-0.021764, -0.020686] |
| routed | A minus R | 0.119280 | [0.115923, 0.122533] |
| routed | A minus B | 0.087355 | [0.084782, 0.089881] |
| routed | A minus F | 0.079745 | [0.076144, 0.083198] |
| routed | F minus B | 0.007610 | [0.005272, 0.010125] |
| oracle-shared | A minus R | 0.053156 | [0.052268, 0.054136] |
| oracle-shared | A minus B | 0.034108 | [0.033518, 0.034719] |
| oracle-shared | A minus F | 0.057855 | [0.057044, 0.058748] |
| oracle-shared | F minus B | -0.023747 | [-0.024337, -0.023187] |

The architecture result is strong. A predicts unseen expert contributions
better than the load-preserving randomized-ID control R, the candidate-matched
router-rank control B, and the parameter-matched full-width control F. All
three direct component intervals exclude zero by a wide margin. Stable expert
identity therefore remains useful when the sparse model receives only the
expert input and must predict the expert output.

The registered G3 ratio is correctly undefined because component F minus B is
negative. This is not a zero gain: A directly beats F by 0.05335 EV. Rather,
the nominal full-width numerator used as the denominator never materialized at
this fixed 2M-token budget. C3 is 103.37%, and the one-seed end-to-end A/F
ratio including the shared harvest is 103.00%. A is consequently not cheaper
than F in this grouped-kernel implementation.

This is best read as a positive architecture and fixed-budget sample-efficiency
result wrapped in a negative cost-efficiency result. It is a single seed and
does not establish a scaling law.

## Health and performance

- Every condition trained for 123 updates and exactly 2,000,000 scientific
  tokens, then scored the same 250,000 held-out tokens.
- Every condition used exact per-slot TopK-2 and exact total L0 16.
- All states and scores were finite; no token was dropped and no capacity
  overflow occurred.
- A fired 97.77% of its features and supported all 128 experts.
- Throughput was 1,402.4 tokens/s for A and 1,449.6 tokens/s for F.
- The SAE-arithmetic useful-MFU diagnostic was 0.00113% for A and 0.01825%
  for F. Measured chip-seconds remain the primary cost measure, but these
  values expose severe accelerator underutilization and explain why nominal
  6.25% candidate exposure did not become a cost advantage.
- An independent replay validated all 16 process artifacts, exact cross-
  condition token identities, 625 packed-sequence clusters, 20,000 bootstrap
  replicates, every parent/run self-hash, and all four clean controller exits.

## Provenance

- Scientific result SHA:
  `c33e83789af3120f42687f2d73a6300ce401138d5966a47068592a6109194074`
- Paired-score SHA:
  `c31129e9ab514bf68dffda8cbfec1298ad6fc3eee93d43802ac8869f8e5c0714`
- Training commit:
  `e54e76b25ec53e3c6bdc7f54f8f695dc4dad5af4`
- Evaluator commit:
  `7125d850d0e825184c14d5355f74fafda9c85ad6`
- Train/eval source manifests:
  `8d947629e57d64d054a3086e2729090eed1e9f6afc4568b76e5c90da5f378e3c`,
  `f68e2fe6e4cf568e694925697dd5d65fe10160a52d357c641892de85ee5c3da7`
- Train/eval target manifests:
  `a2f23aef6b9121e81a5bfea473c64039ca5bf6b073e4f70f4703938a1d8a6903`,
  `31454e5ce8693521c41a2a95f2ef88d40c5d26c726a0efac5c26d882ed341f5e`

Direct immutable result:

```text
gs://moe-sae-bucket/moe-sae/gemma4_v1/experiment3/results/c33e83789af3120f42687f2d73a6300ce401138d5966a47068592a6109194074/
```

Verified evidence archives:

```text
control:
  gs://moe-sae-bucket/moe-sae/gemma4_v1/experiment3/evidence/v2/control/53b4e3eed4ef30f5a4321284332d06577134bb24c4700a11fabcdfb90a28f0fd.tar.gz
operator:
  gs://moe-sae-bucket/moe-sae/gemma4_v1/experiment3/evidence/v2/operator/de3c19df4d9f1262c04621029afb49f1d10fdb62db4a27ad1ebf1815d2926517.tar.gz
failed evaluator attempt:
  gs://moe-sae-bucket/moe-sae/gemma4_v1/experiment3/attempt-evidence/evaluation-v2-capacity-mismatch-20260727T231300Z/
```
