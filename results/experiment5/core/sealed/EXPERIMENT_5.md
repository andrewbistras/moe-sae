# Experiment 5: Routed–Shared Hybrid Sparse Models

Experiment 5 completed; consult the transcoder H-R/H-B effects and fixed-quality status before making an efficiency claim.

Verdict: `mechanism_only`.

## SAE

| train tokens | H EV | R EV | B EV | F EV |
|---:|---:|---:|---:|---:|
| 2,000,000 | 0.361255 | 0.302771 | 0.440579 | 0.454423 |
| 5,000,000 | 0.442556 | 0.377763 | 0.513652 | 0.527845 |
| 10,000,000 | 0.454678 | 0.385502 | 0.522540 | 0.537748 |

### 10M paired effects

| contrast | mean | paired 95% interval | positive seeds |
|---|---:|---:|---:|
| H-R | 0.069176 | [0.067944, 0.070489] | 3/3 |
| H-B | -0.067862 | [-0.068702, -0.067018] | 0/3 |
| H-F | -0.083069 | [-0.084056, -0.082072] | 0/3 |
| F-B | 0.015208 | [0.014792, 0.015719] | 3/3 |

G5: -446.23% (defined).
Fixed-quality Q5: undefined; 0/3 seeds reached F-at-10M.

### 10M coverage

| condition | cumulative training fired | held-out nonfiring |
|---|---:|---:|
| H | 99.99% | 8.18% |
| R | 100.00% | 0.76% |
| B | 100.00% | 0.00% |
| F | 100.00% | 0.00% |

## TRANSCODER

| train tokens | H EV | R EV | B EV | F EV |
|---:|---:|---:|---:|---:|
| 2,000,000 | 0.255100 | 0.120662 | 0.253569 | 0.230952 |
| 5,000,000 | 0.361781 | 0.277870 | 0.371601 | 0.379742 |
| 10,000,000 | 0.375554 | 0.295926 | 0.383825 | 0.393285 |

### 10M paired effects

| contrast | mean | paired 95% interval | positive seeds |
|---|---:|---:|---:|
| H-R | 0.079628 | [0.078575, 0.080640] | 3/3 |
| H-B | -0.008271 | [-0.008788, -0.007734] | 0/3 |
| H-F | -0.017731 | [-0.018540, -0.016729] | 0/3 |
| F-B | 0.009460 | [0.008463, 0.010221] | 3/3 |

G5: -87.42% (defined).
Fixed-quality Q5: undefined; 0/3 seeds reached F-at-10M.

### 10M coverage

| condition | cumulative training fired | held-out nonfiring |
|---|---:|---:|
| H | 99.99% | 8.54% |
| R | 100.00% | 3.46% |
| B | 100.00% | 0.00% |
| F | 100.00% | 0.76% |

## Cost and identity

Incremental cash spend: $40.12 of $50.00; within limit: `true`.
Quota-funded TPU chip-seconds: 668981.4.
Budget ledger: `de5bf5337a98adf1cbe965c208f53815796aaf742e3178cfecc38a47259c1a00`
Report SHA-256: `572633c5db9808f4544daf9d751f714f8cde4845d25a86ad8a1a4941dd883570`
