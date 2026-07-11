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

## Phase 4 — training suite (~35 min TPU ∥ ~15 min A100)
- TPU hosts: rsync bucket → `/dev/shm/moe_sae/discovery_50m`.
- Per worker W: `scripts/tpu/03_discovery_suite.sh $W` — **skip the D slot**;
  concurrently on the A100:
  `uv run python scripts/run_train.py --config configs/discovery_D_seed1.yaml --device cuda`
  (lane note in RUNBOOK; §10 matched order holds — same shards, same data_seed).
- Watch: `logs/tpu_*.log`, ops_metrics.jsonl (MFU ≥ ~40% on A-shapes),
  LOGBOOK for anomaly halts. Each run ends with `final/theta_star.json`;
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
PASS (A ≤ 0.9·C) → proceed. FAIL → **the study halts for an implementation
audit** (spec §20.5); logbook the verdict either way.

## Phase 6 — evals + dumps, all runs (~2–3 h)
Per run: dev-split eval + dump; additionally `--split train` dumps for the
probe-fit inputs (§16.4/A3); fidelity on the A100
(`--fidelity matched,full`). Rsync everything.

## Phase 7 — §20.7–8 analyses (code locked by the tag)
`run_analysis.py --dump dumps/<run>/dev --train-dump dumps/<run>/train \
 --stats stats/discovery_50m --load-floor 500000 \
 [--compare-dump dumps/discovery_C_seed1/dev] [--seed-dumps ...]`
Order discipline: per-bank loads + floor applied and analysis code confirmed
locked (the tag) **before** modality results are examined.

## Phase 8 — §20.9–10 interventions
`run_intervention.py --run runs/discovery_A_seed1 --dev-dump ... --capture ...`
— enriched sets defined on dev dumps, locked, then run on the untouched eval
buckets, with the identical pipeline on condition C as the negative control.

## Phase 9 — wrap
Teardown slice + GPU box (the bucket holds everything). Results into the
committee report; the confirmatory decision per the pre-committed trigger
(briefing Q6 resolution). Deviations, if any, documented with spec-version
discipline.

**Failure playbook** (docs/OPS.md): preemption → rerun with `--resume`
(training resume is bitwise; capture resume loses ≤5 min); anomaly halt →
emergency checkpoint exists, investigate before resuming; backend fault →
A7 substitution policy; >5% capture failures → systematic, stop.
