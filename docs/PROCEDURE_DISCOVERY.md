# Discovery-run procedure (spec §20 steps 4–10)

The single ordered checklist from "committee sign-off" to "discovery results".
Every step is artifact-gated and idempotent: rerunning a phase is always safe.
Cross-cutting rules: launch long jobs `nohup ... < /dev/null`; every phase
transition gets a `scripts/logbook.py --event` entry; anomalies HALT, never
skip (docs/OPS.md §2); nothing in Phase 3+ runs before Phase 0 completes.

## Phase 0 — protocol freeze (human; blocks everything)
1. Committee answers Q1–Q7 (`docs/briefing_spec_committee_v0.1.2.md`);
   amendments (if any) land in the addendum with changelog rows.
2. `git tag spec-v0.1.2 && git push --tags`; fill `[TAG]`/`[COMMIT HASH]` in
   `docs/preregistration_osf.md`.
3. Post the OSF registration; record URL + timestamp in the LOGBOOK.
   **Invariant: no discovery shard may predate this timestamp.**

## Phase 1 — infrastructure up (~45 min, parallelizable)
- Bucket + v4-32: `scripts/tpu/00_provision.sh` stanzas (bucket in
  us-central2; queued-resource; poll ACTIVE).
- Spot A100-40 box: clone repo at the tag, `uv sync`, `gcloud auth login`.
- TPU hosts: `scripts/tpu/01_setup_hosts.sh` via `--worker=all`.
- **Preflight on both substrates** — go/no-go:
  `uv run python scripts/preflight.py --out reports/preflight_$(hostname).json`
  → requires `PREFLIGHT PASS`; sync reports to the bucket.

## Phase 2 — capture-backend gate (A7; ~40 min on the A100)
```
uv run python scripts/run_capture.py --export-validation-inputs validation_inputs
uv run python scripts/run_validate_gate.py --inputs validation_inputs \
  --out reports/hook_validation_cuda.json --device cuda
```
FP32 gate must PASS on the CUDA path before any discovery shard counts.

**A3 pre-capture contamination re-verification (consequence freeze):** before
any discovery capture, re-verify zero train/dev/eval contamination under the
keyed-hash membership function —
`uv run --no-sync pytest tests/test_data_splits.py -q` must be green at the
tag (the A3 regression test drives a composer over shards spanning all
buckets and asserts zero rows with `split_bucket ∈ 80–99`); logbook the
verification alongside the FP32 gate + BF16 report for the capture backend.

## Phase 3 — discovery capture (~3 h, A100, 3 workers)
```
for W in 0 1 2; do
  nohup uv run python scripts/run_capture.py \
    --out captures/discovery_50m_w$W --start $((W*32000)) --examples 32000 \
    --visual-cap 512 --seed 7 --device cuda < /dev/null \
    > logs/capture_w$W.log 2>&1 &
done
# preempted/crashed worker: rerun the same command + --resume
```
Then: `merge_captures.py w0 w1 w2 --out captures/discovery_50m` →
`run_stats.py --capture captures/discovery_50m --out stats/discovery_50m` →
rsync captures+stats to the bucket. Verify: 3× `progress.json` complete,
merged ≈50M tokens, stats.json bank loads sane (no unexplained starvation).
**A11 note:** the statistics pass computes μ_e / μ_route under the
modality-balanced (50/50) measure restricted to the train split — verify
`stats.json` records `"measure": "balanced_train_split"` before any training
run consumes the artifact.

## Phase 4 — training suite (~35 min TPU; A12 backend map)
- TPU hosts: rsync bucket → `/dev/shm/moe_sae/discovery_50m`.
- **A12 backend map (frozen before launch, recorded in the registration):
  ALL 14 training runs — A/B/C/F × seeds 1–3, plus D and E seed 1 — run on
  the TPU/JAX lane.** This satisfies A12's constraint that all conditions
  within a matched seed share one backend, and SUPERSEDES the earlier
  runbook note assigning condition D to the GPU: D runs on TPU accepting the
  ~2–3 h host-transfer selection path (RUNBOOK "Lane assignment note",
  superseded). Capture and fidelity stay on CUDA per A7. The planned
  cross-backend A-vs-C replication (torch/CUDA) is kept as a robustness
  check (and is the A10 audit instrument on gate failure).
- Per worker W: `scripts/tpu/03_discovery_suite.sh $W` — including the D slot.
- Watch: `logs/tpu_*.log`, ops_metrics.jsonl (MFU ≥ ~40% on A-shapes; A19
  load/budget dispersion diagnostics), LOGBOOK for anomaly halts. Each run
  ends with `final/theta_star.json` (condition D: `theta_star_per_bank`, A4);
  rsync `runs/` to the bucket as they land.

## Phase 5 — §20.5 BINDING GATE (first science, ~30 min)
```
run_eval.py --run runs/discovery_A_seed1 --capture captures/discovery_50m \
  --split dev --dump dumps/discovery_A_seed1/dev
run_eval.py --run runs/discovery_C_seed1 ... --dump dumps/discovery_C_seed1/dev
python -c "from moe_sae.eval.sanity_gate import compare_sanity; \
  compare_sanity('runs/discovery_A_seed1','runs/discovery_C_seed1', \
  'reports/sanity_gate_discovery.json')"
```
PASS (A ≤ 0.9·C) → proceed. FAIL → **audit-first (A10), not
halt-by-definition**: (1) run the full regression suite plus a cross-backend
replication of the A-vs-C comparison on the second compute lane with the
parity suite green; (2) fault found → fix, document, re-run the gate;
(3) no fault and both backends agree → **the study continues**, the result is
recorded as evidence against the stable-router-partition hypothesis, and the
A18 trigger is evaluated normally at its appointed step. Logbook the verdict
either way.

## Phase 6 — evals + dumps, all runs (~2–3 h)
Per run: dev-split eval + dump; additionally `--split train` dumps for the
probe-fit inputs (§16.4/A3); fidelity on the A100
(`--fidelity matched,full`). Rsync everything.

## Phase 7 — §20.7–8 analyses (code locked by the tag)
`run_analysis.py --dump dumps/<run>/dev --train-dump dumps/<run>/train \
 --stats stats/discovery_50m \
 [--compare-dump dumps/discovery_C_seed1/dev] [--seed-dumps ...] \
 [--draw-epoch-robustness --capture captures/discovery_50m \
  --draw-counts runs/<run>/final/text_draw_counts.npz]`
Load floor: A13 split-proportional by default (10% of the analysis split's
uniform-load expectation, counts from the dump's own host routing);
`--load-floor` is an absolute override for audits only. Order discipline:
per-bank loads + floor applied and analysis code confirmed locked (the tag)
**before** modality results are examined.

## Phase 7.5 — A18 confirmatory-trigger evaluation (after §20 step 8, before step 9's launch decision context)
Immediately after the step-8 analyses, **on the development split only, with
the evaluation split still untouched**:
```
uv run python scripts/evaluate_trigger.py \
  --gate reports/sanity_gate_discovery.json \
  --a-metrics runs/discovery_A_seed{1,2,3}/eval_dev/metrics.json \
  --c-metrics runs/discovery_C_seed{1,2,3}/eval_dev/metrics.json \
  --b-metrics runs/discovery_B_seed{1,2,3}/eval_dev/metrics.json \
  [--a-fidelity ... --b-fidelity ...] [--a-probes ... --c-probes ...] \
  --out reports/confirmatory_trigger.json
```
The two-route rule (architecture / representation) is applied mechanically;
§20 step 11's discretionary clause is VOID. The inputs + verdict are recorded
and posted as a registration addendum BEFORE any confirmatory launch. If
neither route triggers: discovery is reported as final; the negative
interpretation remains unclaimed under the power clause.

## Phase 8 — §20.9–10 interventions
`run_intervention.py --run runs/discovery_A_seed1 --dev-dump ... --capture ...`
— enriched sets defined on dev dumps, locked, then run on the untouched eval
buckets, with the identical pipeline on condition C as the negative control.
(Load floor: A13 split-proportional over the dev dump by default.)

## Phase 9 — wrap
Teardown slice + GPU box (the bucket holds everything). Results into the
committee report; the confirmatory decision is exactly the recorded A18
verdict from Phase 7.5 (no discretion remains). Deviations, if any,
documented with spec-version discipline.

**Failure playbook** (docs/OPS.md): preemption → rerun with `--resume`
(training resume is bitwise; capture resume loses ≤5 min); anomaly halt →
emergency checkpoint exists, investigate before resuming; backend fault →
A7 substitution policy; >5% capture failures → systematic, stop.
