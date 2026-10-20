# Pilot-width learning horizon: transcoder curve point

Verdict: **curve_point_measured**

Evaluator commit: `2f778bb0c28cf4496f159311e4e68325810810fc`
Training commit: `fa141831645e21b8adc5eb13d576bbc03a5ce1e0`

Checkpoint: 2,000,000 nominal training tokens.

Router-conditioned A recovered undefined of the full-width F-over-B component gain at 156.78% of F's measured marginal training chip-seconds.

| condition | component EV | routed EV | oracle-shared EV | dead rate | chip-seconds |
|---|---:|---:|---:|---:|---:|
| A | 0.302957 | -0.364849 | 0.494797 | 2.3193% | 4976.4 |
| R | -0.023841 | -0.514223 | 0.432521 | 0.0000% | 5013.7 |
| B | 0.260955 | -0.504934 | 0.453838 | 0.0000% | 2827.0 |
| F | 0.239840 | -0.487864 | 0.431994 | 0.0000% | 3174.2 |

Paired packed-sequence effects (left minus right):

| endpoint | contrast | effect | 95% interval |
|---|---|---:|---:|
| component | A-R | 0.326798 | [0.318820, 0.334927] |
| component | A-B | 0.042002 | [0.040231, 0.044068] |
| component | A-F | 0.063116 | [0.061209, 0.065292] |
| component | F-B | -0.021115 | [-0.021630, -0.020565] |
| routed | A-R | 0.149374 | [0.144629, 0.153895] |
| routed | A-B | 0.140085 | [0.135894, 0.144261] |
| routed | A-F | 0.123015 | [0.118236, 0.127648] |
| routed | F-B | 0.017070 | [0.014141, 0.020296] |
| oracle_shared | A-R | 0.062276 | [0.061410, 0.063253] |
| oracle_shared | A-B | 0.040959 | [0.040311, 0.041638] |
| oracle_shared | A-F | 0.062802 | [0.061993, 0.063675] |
| oracle_shared | F-B | -0.021844 | [-0.022437, -0.021279] |

Result SHA-256: `02754a6dbf5ffb3321de7e6aa1ba70db2d06612381bde36150bdc1cab4202d6c`
Paired-score SHA-256: `70e2cc864a375ac5768ef00ffa0c899e8f860a970df8fe9eea696700c5d4fa22`
