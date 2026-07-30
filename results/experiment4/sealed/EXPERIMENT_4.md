# Experiment 4: SAE/transcoder learning horizon

Held-out reconstruction is measured jointly at 2M, 5M, and 10M training tokens. Raw training chip-seconds retain their hardware labels; mixed v4/v5e/v6e ratios are not interpreted as equivalent.

## SAE

| condition | accelerator | generation | run contract |
|---|---|---|---|
| A | v5litepod-16 | v5e | `aa45e048b489c64cd618261c257959dac0b06084d06c0de4c0f4c1e6cc4de462` |
| R | v5litepod-16 | v5e | `b4a45c51b26325601781a5e827d9d757137d867d94741703b19005b97b078ca5` |
| B | v4-32 | v4 | `edd30d6d5c332317e25473d4d5fdbdfbf80d9b69fdd9d1dd68eee8ddbcc53102` |
| F | v5litepod-16 | v5e | `ba2f8c6667d5d294a60d234b534f72c5dc36ec796897044deaf038e0e7b8510f` |

**Cost warning:** this model used mixed accelerator generations. Raw chip-seconds are shown, but affected cross-condition ratios are not scientific comparisons.

### 2,000,000 training tokens

| condition | actual tokens | component EV | routed EV | marginal chip-s | train tok/s | useful MFU |
|---|---:|---:|---:|---:|---:|---:|
| A | 2,015,232 | 0.410248 | -0.125494 | 4805.6 | 6709.6 | — |
| R | 2,015,232 | 0.188562 | -0.565044 | 4860.6 | 6633.7 | — |
| B | 2,015,232 | 0.379876 | -0.289405 | 2745.6 | 11743.6 | — |
| F | 2,015,232 | 0.388352 | -0.261691 | 3040.2 | 10605.7 | — |

Gain recovery: 358.34%; A/F marginal cost: 158.07%.
Source result: `46d7ee167c1174d3e291a06031b6bfd0745e462a5f918ffca298d56941d352a2`

### 5,000,000 training tokens

| condition | actual tokens | component EV | routed EV | marginal chip-s | train tok/s | useful MFU |
|---|---:|---:|---:|---:|---:|---:|
| A | 5,013,504 | 0.470021 | 0.035928 | 9496.0 | 8447.3 | — |
| R | 5,013,504 | 0.348133 | -0.431169 | 9617.0 | 8341.0 | — |
| B | 5,013,504 | 0.429072 | -0.155155 | 6864.5 | 11685.7 | — |
| F | 5,013,504 | 0.448466 | -0.085301 | 7588.3 | 10571.0 | — |

Gain recovery: 211.15%; A/F marginal cost: 125.14%.
Source result: `085f69af4f2cbdaa591ce9fdeae0fb8461fd9258a3c78f5921a416e0fbcc9058`

### 10,000,000 training tokens

| condition | actual tokens | component EV | routed EV | marginal chip-s | train tok/s | useful MFU |
|---|---:|---:|---:|---:|---:|---:|
| A | 10,000,000 | 0.478338 | 0.048394 | 17451.4 | 9168.3 | — |
| R | 10,000,000 | 0.363058 | -0.389171 | 17520.6 | 9132.1 | — |
| B | 10,000,000 | 0.435350 | -0.171523 | 13685.9 | 11690.8 | — |
| F | 10,000,000 | 0.455411 | -0.070766 | 15233.0 | 10503.5 | — |

Gain recovery: 214.29%; A/F marginal cost: 114.56%.
Source result: `f9864abebb2abb48a930c03b766f3b13336327d5cb6ccfdbb334e80a79defeaa`

## TRANSCODER

| condition | accelerator | generation | run contract |
|---|---|---|---|
| A | v5litepod-16 | v5e | `d8798b92e2bbc4f4e9a1a8b5452936de635a6e8394ee4bc0ac7cccb089a70997` |
| R | v5litepod-16 | v5e | `00b2794be309836c9fd7ad7d1aa965f5cbb81a0f5a05d3e367182049fe383eda` |
| B | v4-32 | v4 | `607a52153a973059eddecf6b16a44f43854445096c9bb47ca7f7a19e1c103b4e` |
| F | v5litepod-16 | v5e | `b5589c810135b4b140553b7c570558a06a4e22bd98424ebf6eab244ef44af4e3` |

**Cost warning:** this model used mixed accelerator generations. Raw chip-seconds are shown, but affected cross-condition ratios are not scientific comparisons.

### 2,000,000 training tokens

| condition | actual tokens | component EV | routed EV | marginal chip-s | train tok/s | useful MFU |
|---|---:|---:|---:|---:|---:|---:|
| A | 2,015,232 | 0.302957 | -0.364849 | 4976.4 | 6479.3 | 0.01% |
| R | 2,015,232 | -0.023841 | -0.514223 | 5013.7 | 6431.1 | 0.01% |
| B | 2,015,232 | 0.260955 | -0.504934 | 2827.0 | 11405.5 | 0.01% |
| F | 2,015,232 | 0.239840 | -0.487864 | 3174.2 | 10158.2 | 0.18% |

Gain recovery: undefined; A/F marginal cost: 156.78%.
Source result: `02754a6dbf5ffb3321de7e6aa1ba70db2d06612381bde36150bdc1cab4202d6c`

### 5,000,000 training tokens

| condition | actual tokens | component EV | routed EV | marginal chip-s | train tok/s | useful MFU |
|---|---:|---:|---:|---:|---:|---:|
| A | 5,013,504 | 0.390588 | -0.192102 | 9963.3 | 8051.2 | 0.01% |
| R | 5,013,504 | 0.059336 | -0.516112 | 10013.5 | 8010.8 | 0.01% |
| B | 5,013,504 | 0.324875 | -0.562885 | 7114.6 | 11274.9 | 0.01% |
| F | 5,013,504 | 0.303150 | -0.535781 | 7952.0 | 10087.6 | 0.18% |

Gain recovery: undefined; A/F marginal cost: 125.29%.
Source result: `891df4e8e5346fbbe8a4198344c82a577c35b6fb2c4cdf963d241f439a832ef7`

### 10,000,000 training tokens

| condition | actual tokens | component EV | routed EV | marginal chip-s | train tok/s | useful MFU |
|---|---:|---:|---:|---:|---:|---:|
| A | 10,000,000 | 0.399446 | -0.170846 | 18254.2 | 8765.1 | 0.01% |
| R | 10,000,000 | 0.073597 | -0.518662 | 18303.2 | 8741.6 | 0.01% |
| B | 10,000,000 | 0.331932 | -0.568222 | 14282.0 | 11203.0 | 0.01% |
| F | 10,000,000 | 0.311875 | -0.527926 | 15890.3 | 10069.0 | 0.18% |

Gain recovery: undefined; A/F marginal cost: 114.88%.
Source result: `c501c20f9e2f5d97a2a76dc58d856078d26db677a8cc96d3beaf38676072e300`

## Identity

Report SHA-256: `18908abf1c22512d5f4dc9180ea52c28f57953c0046193f4e8957f9c72cdcb88`
