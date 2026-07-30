# Scientific result registry

`REGISTRY.json` is a generated, create-only index over sealed scientific
artifacts. It is intentionally not a spreadsheet of labels such as `A` and
`F`. The registry keeps five independent identities:

- `source_id`: the immutable report or result artifact that supplied it.
- `score_space_id`: model, layer, target, evaluation-token identity,
  centering rule, and SST basis. EV differences are valid only inside this
  identity.
- `cell_id`: the fully qualified condition, seed, checkpoint, geometry,
  activation policy, scores, cost, and provenance.
- `cost_space_id`: accelerator type/generation/chip count and timing scope.
  A measured cost ratio is interpreted only when every paired cell shares it.
- `contrast_id`: the exact paired cells, endpoint, aggregation, interval, and
  interpretation status for one scientific comparison.

The validator rejects bare `A`, `B`, `F`, `H`, and `R` condition identities,
cross-score-space quality claims, interpreted cross-hardware cost claims,
incorrect EV/NMSE/chip-second/throughput arithmetic, broken source references,
and noncanonical hashes.

Build a registry from Experiment 4, v3, and a future closure fragment:

```zsh
uv run python scripts/results/build_result_registry.py build \
  --experiment4-report /path/to/experiment4/REPORT.json \
  --v3-report /path/to/experiment5/v3/PRIMARY_REPORT.json \
  --fragment /path/to/a512-closure.registry-fragment.json \
  --output results/REGISTRY.json
```

Core Experiment 5 additionally requires the report and all eighteen exact
parents named by that report. In zsh, construct those repeated arguments
without an unreviewable glob:

```zsh
core_results=(
  /path/to/result-01.json
  /path/to/result-02.json
  # List result-03 through result-18 explicitly.
)
core_args=()
for artifact in $core_results; do
  core_args+=(--experiment5-core-result "$artifact")
done

uv run python scripts/results/build_result_registry.py build \
  --experiment5-core-report /path/to/experiment5/REPORT.json \
  "${core_args[@]}" \
  --output results/REGISTRY.json
```

Validate a sealed registry without changing it:

```zsh
uv run python scripts/results/build_result_registry.py validate \
  results/REGISTRY.json
```

Future experiments should emit the generic
`moe-sae-scientific-result-registry-fragment-v1` schema and use the public
builders in `moe_sae.result_registry`. This keeps ingestion generic while the
condition, score, cost, and contrast identities remain fail-closed.
