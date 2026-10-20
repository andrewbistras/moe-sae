# A512 ReLU-inclusive component closure

This protocol is a new, commit-scoped successor to Experiment 5 v3. It does
not rewrite v3 configs, checkpoints, results, or reports.

## Question and matrix

The primary question is whether true expert identity explains component-target
quality after separating it from a random identity key, visible candidate
count, and total parameter count.

| condition | grouping | banks × width | total features | candidates/token | TopK/component | selected slots/token |
|---|---|---:|---:|---:|---:|---:|
| A | true expert ID | 128 × 512 | 65,536 | 4,096 | 4 | 32 |
| R | per-sequence randomized expert ID | 128 × 512 | 65,536 | 4,096 | 4 | 32 |
| B | router rank | 8 × 512 | 4,096 | 4,096 | 4 | 32 |
| F | router rank | 8 × 8,192 | 65,536 | 65,536 | 4 | 32 |

All four cells use ReLU with inclusive-zero TopK. Selected-slot count is
exactly 32; positive L0 may be lower and must not be relabelled as exact L0.
A/R/F match total features and parameter scale. A/R/B match visible
candidates. Every condition matches the selected-slot ceiling.

A is imported read-only from the completed Experiment 5 v3
`A512ReLUInclusiveTopK` trajectories at seeds 0–2 and 2M/5M/10M. The importer
hash-binds the original config, training result, run contract, milestone
pointer, source revision, payload root, resource, and accelerator without
copying or relabelling the parent. R/B/F are trained from scratch for both SAE
and transcoder panels: 18 fresh 10M-token trajectories total.

## Execution boundaries

The closure configs reuse the sealed 10M target and routed-input captures,
optimizer, 16,384-token effective batch, 611-update schedule, 1M restart
checkpoints, and mature contribution A/R/B/F kernels. A narrow new adapter
loads the successor schema into those generic kernels and preserves itself
across the worker supervisor’s child-process boundary. The resident evaluator
uses grouped-capacity scans only for A/R; rank-bank B/F restore and evaluate
with an explicit not-applicable capacity contract.

Evaluation is six resident bundles, one per panel/seed. Each bundle evaluates
A/R/B/F at 2M/5M/10M on the same 250k-token held-out capture. The mixed-parent
adapter uses the original v3 config and source condition for A, the closure
config for R/B/F, and retains the logical A mapping only in the new,
self-hashed closure manifest. The report collector reduces token scores to
paired packed-sequence component SSE/SST rows.

## Frozen report

The primary endpoint is component-target explained variance with common SST.
Every panel/milestone reports A-R, A-B, A-F, and F-B with a paired hierarchical
seed/packed-sequence bootstrap. Define:

`G = 100 × (A - B) / (F - B)`.

G is undefined when the point estimate F-B is nonpositive and is labelled
unstable whenever the paired F-B interval includes zero. The report still
shows the numerator, denominator, interval, and positive-denominator bootstrap
fraction.

Training cost ratios are emitted only when accelerator generation,
chip-seconds scope, and the exclusion list match exactly. Historical A cost is
not silently compared with a fresh F run on another generation or timing
boundary.

## Bounded systems profile

The optional A/F profile is four tasks: A and F, seed 0, in each panel. A and F
must use the same physical resource within a panel. Each task is capped at 600
seconds with one warmup and three measured steps and may not publish training
or checkpoint artifacts. Required outputs are compile time, steady step time,
useful MFU, peak HBM, input-wait fraction, collective fraction, and top HLO ops
by time. Its role is diagnostic optimization evidence, not a replacement for
10M quality or cost measurements.

## Intended launch order

1. Validate both checked-in closure configs.
2. Build the read-only A reference manifest from the completed v3 training
   plan.
3. Build the 18-task training plan from explicit admitted resources and
   controller/remote artifact mappings.
4. Train R/B/F, then build the six mixed-parent evaluation bundles only after
   all parent results and milestone pointers validate.
5. Collect the 72 component score cells plus 24 trajectory cost rows and seal
   JSON and Markdown reports.
6. Run the bounded same-slice A/F profile as a diagnostic after the scientific
   closure is sealed; it must not delay or substitute for the quality result.

The builders and validators are local and perform no cloud discovery. The
training executor then runs an append-only readiness/retry queue over exactly
five admitted slices, with a $40 hard ceiling and 12-hour watchdog. All nine
SAE tasks are eligible on two central1 v5 workers sharing one artifact plane;
transcoder seeds 0/1/2 are pinned to v5/on-demand-v4/spot-v4 respectively.
F is scheduled before R and B without changing the serialized R/B/F matrix.

## Controller commands

After replacing the checked-in template placeholders:

```bash
uv run python scripts/tpu/build_component_closure_a_references.py \
  --v3-training-plan /ABS/v3-training-plan.yaml \
  --output /ABS/component-closure-a-references.json

uv run python scripts/tpu/run_component_closure_campaign.py build \
  --source configs/gemma4_component_closure_campaign.template.yaml \
  --a-references /ABS/component-closure-a-references.json \
  --project-commit FULL_40_CHARACTER_COMMIT \
  --output /ABS/component-closure-training-plan.yaml

uv run python scripts/tpu/run_component_closure_campaign.py commands \
  --plan /ABS/component-closure-training-plan.yaml

uv run python scripts/tpu/run_component_closure_campaign.py run \
  --plan /ABS/component-closure-training-plan.yaml

uv run python scripts/tpu/evaluate_component_closure.py build \
  --training-plan /ABS/component-closure-training-plan.yaml \
  --a-references /ABS/component-closure-a-references.json \
  --runtime configs/gemma4_component_closure_evaluation_runtime.template.yaml \
  --evaluation-source-revision FULL_40_CHARACTER_COMMIT \
  --state-root /ABS/component-closure/evaluation-state \
  --output /ABS/component-closure-evaluation-plan.json

uv run python scripts/tpu/evaluate_component_closure.py run-all-operators \
  --plan /ABS/component-closure-evaluation-plan.json \
  --training-plan /ABS/component-closure-training-plan.yaml \
  --maximum-wall-seconds 43200

uv run python scripts/tpu/evaluate_component_closure.py collect-costs \
  --training-plan /ABS/component-closure-training-plan.yaml \
  --a-references /ABS/component-closure-a-references.json \
  --output /ABS/component-closure-costs.json

uv run python scripts/tpu/evaluate_component_closure.py collect \
  --plan /ABS/component-closure-evaluation-plan.json \
  --costs /ABS/component-closure-costs.json \
  --output /ABS/component-closure-score-grid.json

uv run python scripts/tpu/seal_component_closure_report.py \
  --evidence /ABS/component-closure-score-grid.json \
  --output-json /ABS/component-closure-report.json \
  --output-markdown /ABS/component-closure-report.md
```

`run-all-operators` creates one controller queue per physical resource, so the
two SAE bundles assigned to the same v5 slice are serialized. Each bundle
fans out four simultaneous SSH workers according to the sealed
controller-to-JAX mapping, records immutable per-controller logs, publishes
all cell bytes to the controller plane, and seals a retry-safe bundle
completion. A retry skips already-complete canonical cells. `run-bundle` and
`run-bundle-json` are accelerator process boundaries, not controller launch
commands.

For an unattended production run, use the durable end-to-end controller after
the training plan and evaluation runtime have been sealed:

```bash
uv run python scripts/tpu/run_component_closure_end_to_end.py \
  --training-plan /ABS/component-closure-training-plan.yaml \
  --a-references /ABS/component-closure-a-references.json \
  --evaluation-runtime /ABS/component-closure-evaluation-runtime.yaml \
  --output-root /ABS/component-closure/final \
  --state-root /ABS/component-closure/end-to-end-state \
  --project-commit FULL_40_CHARACTER_COMMIT \
  --archive-gcs-prefix gs://BUCKET/component-closure/evidence \
  --maximum-wall-seconds 43200
```

This one persisted clock covers training, evaluation, report construction,
and compact evidence archival. The 160 GiB worker cache is fixed to
`/dev/shm/moe-sae-local/horizon-10m/cache`; Filestore remains only the control
plane.
