# Pilot-width learning horizon: transcoder curve point

Verdict: **curve_point_measured**

Evaluator commit: `2f778bb0c28cf4496f159311e4e68325810810fc`
Training commit: `fa141831645e21b8adc5eb13d576bbc03a5ce1e0`

Checkpoint: 10,000,000 nominal training tokens.

Router-conditioned A recovered undefined of the full-width F-over-B component gain at 114.88% of F's measured marginal training chip-seconds.

| condition | component EV | routed EV | oracle-shared EV | dead rate | chip-seconds |
|---|---:|---:|---:|---:|---:|
| A | 0.399446 | -0.170846 | 0.535263 | 2.5665% | 18254.2 |
| R | 0.073597 | -0.518662 | 0.438660 | 0.0000% | 18303.2 |
| B | 0.331932 | -0.568222 | 0.470025 | 0.0000% | 14282.0 |
| F | 0.311875 | -0.527926 | 0.450743 | 0.0000% | 15890.3 |

Paired packed-sequence effects (left minus right):

| endpoint | contrast | effect | 95% interval |
|---|---|---:|---:|
| component | A-R | 0.325849 | [0.316195, 0.336170] |
| component | A-B | 0.067515 | [0.064565, 0.070839] |
| component | A-F | 0.087571 | [0.084596, 0.090914] |
| component | F-B | -0.020056 | [-0.020737, -0.019360] |
| routed | A-R | 0.347816 | [0.336398, 0.358633] |
| routed | A-B | 0.397376 | [0.386207, 0.408425] |
| routed | A-F | 0.357080 | [0.346214, 0.367831] |
| routed | F-B | 0.040296 | [0.037024, 0.043474] |
| oracle_shared | A-R | 0.096603 | [0.094294, 0.099208] |
| oracle_shared | A-B | 0.065238 | [0.063852, 0.066770] |
| oracle_shared | A-F | 0.084520 | [0.083141, 0.086053] |
| oracle_shared | F-B | -0.019281 | [-0.019932, -0.018671] |

Result SHA-256: `c501c20f9e2f5d97a2a76dc58d856078d26db677a8cc96d3beaf38676072e300`
Paired-score SHA-256: `fd13223d1be746d4f10c6b31d39231d86c9b89e1fd0ebe641d47bcedf80bfa85`
