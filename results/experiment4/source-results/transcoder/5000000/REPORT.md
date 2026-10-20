# Pilot-width learning horizon: transcoder curve point

Verdict: **curve_point_measured**

Evaluator commit: `2f778bb0c28cf4496f159311e4e68325810810fc`
Training commit: `fa141831645e21b8adc5eb13d576bbc03a5ce1e0`

Checkpoint: 5,000,000 nominal training tokens.

Router-conditioned A recovered undefined of the full-width F-over-B component gain at 125.29% of F's measured marginal training chip-seconds.

| condition | component EV | routed EV | oracle-shared EV | dead rate | chip-seconds |
|---|---:|---:|---:|---:|---:|
| A | 0.390588 | -0.192102 | 0.529304 | 2.6276% | 9963.3 |
| R | 0.059336 | -0.516112 | 0.438781 | 0.0000% | 10013.5 |
| B | 0.324875 | -0.562885 | 0.465953 | 0.0000% | 7114.6 |
| F | 0.303150 | -0.535781 | 0.445906 | 0.0000% | 7952.0 |

Paired packed-sequence effects (left minus right):

| endpoint | contrast | effect | 95% interval |
|---|---|---:|---:|
| component | A-R | 0.331252 | [0.321864, 0.341140] |
| component | A-B | 0.065714 | [0.062260, 0.069519] |
| component | A-F | 0.087438 | [0.083347, 0.091966] |
| component | F-B | -0.021724 | [-0.022786, -0.020796] |
| routed | A-R | 0.324010 | [0.313421, 0.334067] |
| routed | A-B | 0.370784 | [0.360733, 0.380863] |
| routed | A-F | 0.343680 | [0.333457, 0.353903] |
| routed | F-B | 0.027104 | [0.022775, 0.031134] |
| oracle_shared | A-R | 0.090524 | [0.088637, 0.092642] |
| oracle_shared | A-B | 0.063352 | [0.062019, 0.064781] |
| oracle_shared | A-F | 0.083398 | [0.081672, 0.085311] |
| oracle_shared | F-B | -0.020047 | [-0.020876, -0.019279] |

Result SHA-256: `891df4e8e5346fbbe8a4198344c82a577c35b6fb2c4cdf963d241f439a832ef7`
Paired-score SHA-256: `2a4d53c00ed49d625332015a2fed344a2f0b4a9ef6857d9587b355630f0d95a0`
