# Aggregate results

| File | Contents |
|---|---|
| `component-closure-report.json` | The sealed report (report identity `fc1d6d41…`) |
| `component-closure-report.md` | The same report as a table |
| `component-closure-score-grid.json` | The evidence (evidence identity `4b8bf760…`): per-sequence errors for all 72 cells and training costs for all 24 runs |
| `endpoints.csv` | Three-seed component explained variance (EV) for each panel, checkpoint and layout |
| `contrasts.csv` | A−R, A−B, A−F and F−B, with 95% paired hierarchical bootstrap intervals |
| `per_seed.csv` | EV for each seed, the values behind Table A1 |
| `training_costs.csv` | Accelerator, chip count, wall time and chip-seconds for each run |
| `cost_ratios.csv` | Per-seed chip-second ratios of A to each control, where hardware and timing scope match |
| `identities.json` | Report and evidence identities, evaluation-set identity, and training source revisions |

The three `component-closure-*` files are the originals, unmodified. The CSV
files and `identities.json` are generated from the evidence by
[`../scripts/export_results.py`](../scripts/export_results.py), which also
rebuilds the report from the evidence. The rebuilt report is identical to the
sealed one, bootstrap intervals included. Explained variance is
`1 − SSE/SST`, with condition A's SST as the common denominator within each
panel, seed and checkpoint.
