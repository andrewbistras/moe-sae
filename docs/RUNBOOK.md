# Runbook — frozen spec v0.1.1, §20 execution order

Operational manual mapping spec §20's execution order to concrete commands.
The science is frozen in `spec/frozen_spec_v0.1.1.md`; shared formats and
constants in `docs/INTERFACES.md`. Nothing here adds a degree of freedom: every
scientific choice below is a citation, not a decision.

All commands run from the repo root with `uv run`. Configs in `configs/` are
exact `RunConfig` serializations; do not hand-edit frozen fields —
`RunConfig.validate()` aborts the run if they drift.

Hardware split:

| Stage | Where | Why |
|---|---|---|
| §20.1–§20.3 (validation, stats, 1M pipeline) | local M1 Pro 16 GB (MPS) | small enough; proves plumbing before renting anything |
| §20.4–§20.10 (discovery training + analysis) | **cloud GPU required** | 50M-token capture and 14 training runs; see sizing below |

---

## §20.1 — Hook validation (HARD GATE)

Export prompt-only validation inputs (≥100k language-stack token positions,
mixed across the 4 domains; `moe_sae.data.evalsets.export_validation_inputs`,
seed 7 per contract):

```sh
uv run python scripts/run_capture.py --export-validation-inputs \
    --out captures/validation_inputs --n-tokens 100000 --seed 7
```

Run the gate (`moe_sae/capture/validate.py` behind the script):

```sh
uv run python scripts/run_validate_gate.py \
    --inputs captures/validation_inputs \
    --report runs/validation/gate_report.json
```

Pass criteria (spec §3, changelog #4):

- **Gate (FP32 reference forward):** relative error of
  ‖Σ_slots c_te − model's routed aggregate‖ / ‖aggregate‖ < 1e-5 for **every**
  token in the 100k set. Any failure halts everything; do not proceed, do not
  rationalize.
- **Report (BF16 production path):** max and median discrepancy are recorded in
  the report, *not* gated. Expected scale 1e-3–1e-2 (kernel reduction order).
  Inspect the report for *systematic* structure (e.g. correlation with slot
  ordering) before proceeding.

Note: the FP32 reference is a layer-streamed forward (see
`moe_sae.model_loading.load_model` docstring) — a second full FP32 model copy
does not fit in 16 GB.

## §20.2 — Statistics pass (μ_e, routing stats, bank loads)

Capture the 1M-token pipeline stream (50/50 visual/text prompt positions from
the same examples, answer tokens excluded — spec §9):

```sh
uv run python scripts/run_capture.py \
    --out captures/pipeline_1m --tokens 1000000 --seed 1
```

Then compute per-expert means μ_e (frozen thereafter, spec §5), routing
frequency / gate-mass baselines (spec §15), and per-bank load counts:

```sh
uv run python scripts/run_stats.py \
    --capture captures/pipeline_1m \
    --out stats/pipeline_1m/stats.safetensors
```

## §20.3 — 1M pipeline run (plumbing only, no scientific conclusions)

```sh
uv run python scripts/run_train.py --config configs/pipeline_1m_A.yaml
uv run python scripts/run_train.py --config configs/pipeline_1m_B.yaml
uv run python scripts/run_train.py --config configs/pipeline_1m_C.yaml
```

f=1024, k̄=32, 1M tokens, seed 1 (spec §10). B/C exist at pipeline scale so the
A-vs-C sanity gate (§20.5) and the condition dispatch are exercised end-to-end
before any GPU money is spent.

Optional one-time LR sanity check (spec §11): compare 1e-4 / 3e-4 / 1e-3 on
this run only, then the LR is frozen for every subsequent comparison. If you
skip it, LR stays at the spec default 3e-4.

**Expected wall-clock on M1 Pro (MPS):** capture dominates. At roughly
1.5–2 s/example and ~500–700 usable positions/example, the 1M-token capture is
on the order of **1–2 hours**. SAE training itself is small: 1M tokens /
4096-token batches = 244 steps — **minutes** per condition. Budget the
afternoon, not the week.

## §20.4 — Discovery training (CLOUD GPU REQUIRED)

### Sizing

- **Activation storage:** one captured token costs ≈15.4 KB
  (c [6,1280] bf16 = 15,360 B + experts/gates/ids ≈ 35 B). 50M tokens ≈
  **770 GB** of shards. Alternatives: (a) capture once to object storage
  (S3/GCS) and stream — recommended; (b) capture-on-the-fly inside the trainer
  — saves storage but re-pays the model forward for all 14 runs and makes
  "same training-data order across matched conditions" (spec §10) harder to
  guarantee. Capture once.
- **SAE memory (f=4096):** condition A weight params = 64 × 2 × 4096 × 1280 =
  671,088,640 ≈ **1.34 GB BF16**; FP32 Adam moments 2 × 671M × 4 B ≈
  **5.4 GB**. Condition F is exactly parameter-matched (spec §12F). A single
  A100/H100 80 GB is ample for any condition; activations at B=4096 are
  negligible next to the moments.
- **Compute:** capture forward passes dominate (a ~3.4B-param multimodal
  forward per example vs a ~0.7B-param SAE fwd+bwd per 4096-token step ×
  12,207 steps). Do the 50M-token capture **once**, then train all 14
  discovery runs off the same shards; each SAE run is hours, not days, on one
  80 GB card.

### Commands

Capture and stats at discovery scale:

```sh
uv run python scripts/run_capture.py --out captures/discovery_50m --tokens 50000000 --seed 1
uv run python scripts/run_stats.py --capture captures/discovery_50m \
    --out stats/discovery_50m/stats.safetensors
```

Train all 14 runs — A, B, C, F × seeds {1,2,3} + D, E × seed 1 (spec §10, §12):

```sh
for cfg in configs/discovery_*.yaml; do
  uv run python scripts/run_train.py --config "$cfg"
done
```

θ_b is logged every batch and θ* is frozen per condition/seed at end of
training (spec §6; `runs/<run_name>/final/theta_star.json`). Matched conditions
see identical batch sequences via `data_seed = 1000 + seed` (contract).

## §20.5 — A-vs-C implementation sanity gate (HARD GATE)

Evaluate seed-1 A and C, then run the gate:

```sh
uv run python scripts/run_eval.py --run runs/discovery_A_seed1 --capture captures/discovery_50m
uv run python scripts/run_eval.py --run runs/discovery_C_seed1 --capture captures/discovery_50m
uv run python scripts/sanity_gate.py \
    --a runs/discovery_A_seed1 --c runs/discovery_C_seed1
```

Gate: A must beat C on **local micro-averaged NMSE by ≥10% relative** under
frozen-θ* inference. Failure semantics (**A10, v0.1.2 addendum — audit-first,
not halt-by-definition**): an implementation audit runs (full regression
suite + cross-backend A-vs-C replication on the second lane with the parity
suite green); a found fault is fixed, documented, and the gate re-run; if no
fault is found and both backends agree, **the study continues**, the result
is recorded as evidence against the stable-router-partition hypothesis, and
the A18 confirmatory trigger is evaluated normally at its appointed step.

## §20.6 — Reconstruction + downstream fidelity

For every run:

```sh
uv run python scripts/run_eval.py --run runs/<run_name> \
    --capture captures/discovery_50m \
    --downstream matched  # primary: matched-position replacement (spec §13)
uv run python scripts/run_eval.py --run runs/<run_name> \
    --downstream full     # secondary; flagged train/eval shift, answer spans unseen
```

All metrics under frozen-θ* inference (spec §6/§13). `run_eval` also produces
the FeatureDump shards (`dumps/<run_name>/<split>/`) consumed by §20.7–8.
Splits are by `example_id % 100`: 0–79 train, 80–89 development, **90–99
evaluation — untouched until §20.10**.

## §20.7 — Load floor + analysis code lock

```sh
uv run python scripts/run_analysis.py --stage loads --run runs/<run_name>
```

Reports per-bank routed-token counts for all 64 banks in all conditions and
applies the frozen 500,000-token load floor (spec §16.3; pipeline scale:
10,000). **Lock the expert-concentration analysis code (commit/tag it) before
examining any modality result** (spec §20.7). Results seen before the lock
contaminate the pre-registration.

## §20.8 — Modality / enrichment / stability / incremental-information

```sh
uv run python scripts/run_analysis.py --stage modality    --run runs/<run_name>   # §16.1 JSD + bootstrap
uv run python scripts/run_analysis.py --stage enrichment  --run runs/<run_name>   # §15 R_M(e) vs router baselines
uv run python scripts/run_analysis.py --stage stability   --runs runs/discovery_A_seed{1,2,3}  # §16.3
uv run python scripts/run_analysis.py --stage information --run runs/<run_name>   # §16.4 incl. feature-identity probe
```

All on train + development splits only. Probes are held out at the **example**
level, never position level (spec §16.4).

## §20.9 — Enriched-set definition + intervention code lock

```sh
uv run python scripts/run_analysis.py --stage enriched-sets \
    --run runs/discovery_A_seed1 --split development
```

Enriched expert/feature sets are defined on the **development split only**
(`example_id % 100` in 80–89). Then **lock the intervention code before any
evaluation-split data is touched** (spec §17/§20.9). Redefining enriched sets
after evaluation data has been seen is an explicit non-goal violation
(spec §19).

## §20.10 — Subtraction interventions + condition-C negative control

```sh
uv run python scripts/run_intervention.py --run runs/discovery_A_seed1 \
    --enriched-sets runs/discovery_A_seed1/analysis/enriched_sets.json \
    --split evaluation
uv run python scripts/run_intervention.py --run runs/discovery_C_seed1 \
    --enriched-sets runs/discovery_C_seed1/analysis/enriched_sets.json \
    --split evaluation   # identical pipeline = the negative control (spec §17)
```

Intervention is **feature subtraction from the true activation** (spec §17,
changelog #2): encode true routed contributions under θ*, subtract the named
latents' decoder contributions, continue the forward — no reconstruction error
injected. Matched-control ablation sets (count / mass / frequency / load /
norm) are produced by the same script. A "visual subsystem" claim requires
condition A to exceed condition C on visual-task selectivity.

## §20.11 — Only then

Decide on layer replication, width, model scale, SparseCore, and whether the
200M confirmatory run proceeds. No metric or hypothesis changes (spec §10).

---

## Known scale realities

- **Text-position reuse.** The 4 training datasets supply roughly ~6M *unique*
  prompt-text token positions, but the 50/50 batch composition needs 25M text
  positions at discovery scale. Positions therefore repeat across epochs; this
  is sanctioned by §9's shuffling rules (examples and positions shuffled
  independently within each half; the `BatchComposer` contract permits
  sampling with replacement). Do not "fix" this by relaxing the 2048/2048
  composition — batch composition is frozen (spec §9).
- **Statistical power.** Discovery scale gives 64 × 50M = 3.2e9 activation
  events over 262,144 features ≈ **12k fires per feature** — ~16× leaner than
  frontier compute-optimal SAE allocations (spec §10). Expect substantial dead
  rates and per-bank variance *by design*; a discovery-scale null is
  **underpowered, not informative**, and the negative interpretation may only
  be claimed from the 200M confirmatory run (spec §18 power clause).
- **M1 Pro 16 GB.** The BF16 model (~6.7 GB) plus activations fits; a second
  FP32 copy does not, hence the layer-streamed FP32 reference forward in the
  validation gate. Capture throughput on MPS is roughly 1.5–2 s/example —
  fine for §20.1–§20.3, prohibitive at 50M tokens. Do not attempt discovery
  capture locally.
- **Script availability.** Command lines above are the canonical CLIs from the
  engineering contract; if a script grows flags, `--help` wins on spelling but
  the *order of operations and the two hard gates* in this runbook are fixed
  by spec §20.

## Resilient local execution (interrupt/sleep-safe)

Long local stages run detached and self-heal; safe to close the terminal or
put the laptop to sleep (work pauses and resumes on wake; `caffeinate -i`
prevents idle sleep while running):

```bash
# capture (§20 step 2 data): checkpoints progress.json every 100 examples;
# auto-restarts with --resume; ≤~5 min of work ever redone; kill-safe
nohup scripts/run_pipeline_capture.sh >> logs/capture_runner.log 2>&1 &

# everything after capture: stats → train A/B/C → eval dumps → §20.5 gate;
# waits for capture to complete; every step idempotent — rerun anytime
nohup scripts/run_pipeline_night.sh >> logs/night_runner.log 2>&1 &
```

Progress: `tail -f logs/capture_pipeline.log logs/pipeline_night.log`, or
`python3 -m json.tool captures/pipeline_1m/progress.json`.

Verified failure modes: SIGKILL of the worker (runner restarts `--resume`,
stray un-checkpointed shards deleted), orphaned worker after wrapper death
(pre-attempt sweep kills it), per-example data failures (logged + skipped;
>5% failure rate aborts loudly), interpreter-exit pyarrow deadlock (hard
exit paths).

## TPU v4-32 execution plan (discovery §20 step 4)

**Region**: TPU v4 lives in GCP `us-central2-b`. The GCS bucket MUST be in
`us-central2` — in-region reads are free; cross-region/cloud egress on the
~770 GB shard set is the one large avoidable I/O cost (~$70/TB, repeatedly).

**Division of labor** (keeps the §3-gate-validated capture path untouched):
- Capture stays PyTorch on a rented GPU (scripts/cloud/capture_discovery.sh,
  ~8 h ≈ $10–25 spot on A100, shards rsync straight to the bucket).
- SAE training runs in JAX on the slice (src/moe_sae/tpu/, checkpoint format
  identical to torch — the whole eval/analysis stack works unchanged).
- Data order: the JAX trainer reuses the torch BatchComposer (CPU) so batch
  sequences are bit-identical across backends.

**Why the whole suite is ~30 min on a v4-32**: SAE compute is tiny (~19
PFLOP·s for an A-run, 10.7× for F). The binding resource is activation
streaming (~90 GB/s for one run at 50% MFU) — solved by two design moves:
1. *One stream, many consumers*: spec §10 requires matched data order across
   conditions, so all 14 runs consume the identical batch sequence — fetch
   once, broadcast; per-stream need drops to ~6 GB/s.
2. *Host-RAM cache*: 770 GB < ~1.6 TB of host RAM across the 4 TPU-VMs; after
   one free in-region GCS pass, every read is memory-speed and the A-runs are
   compute-bound (≥50% MFU). Grouped-GEMM capacity padding costs ~2× on A's
   encode (Cap=768 vs mean load 384) — still ~5× cheaper than F.

**Order of operations**: 00_provision.sh → 01_setup_hosts.sh →
02_smoke_ladder.sh (rungs 1–2 run free locally first; rung 4–5 replicate the
pipeline run and the A-vs-C sanity gate cross-backend before any discovery
spend) → capture_discovery.sh on the GPU box → 03_discovery_suite.sh.

**Cost picture**: GPU capture $10–25 + storage ~$15/mo + in-region reads $0 +
~1 h of v4-32 (free under TRC). MFU >50% target applies to the A-shape
training steps; F-runs are the wall-clock stragglers by construction.

## Bucketless A100 lane (current discovery plan, 2026-07-08)

No GCS/Drive. `notebooks/colab_discovery.ipynb` (rewritten) runs five phases
on one A100-40: prefix stats (2k train examples → μ_e/baselines) → retained
dev/eval capture (buckets 80–99, ~48 GB local, resumable) → **streaming
co-training** (`scripts/run_stream_discovery.py`: rotating ~50 GB windows;
all 14 runs replay each window with one condition GPU-resident at a time;
matched data order is bitwise; capture paid once; seeds vary init only —
user decision 2026-07-08) → eval + binding §20.5 gate → artifact tarball
(~25 GB) downloaded to the laptop. Correctness: windowed+swapped ==
loop.train bit-equal (tests/test_streaming.py). Accepted risk: phase 3 is
not resumable; a dead session reruns it (~5–7 h, ~70 CU).

## Lane history: Colab windowed-streaming (attempted, retired)

Between 2026-07-08 and 07-10 a third lane was built and debugged: 50M-token
discovery entirely on one Colab VM — no GCS, no Drive — streaming activations
through a rotating 2×31 GB on-disk capture window pipelined with training
(design preserved at git tag `colab-streaming-campaign`). It fought Colab's
disk (ENOSPC at the original sizing) and session model through five fix
rounds and never completed a discovery run; no artifacts were produced. It is
retired in favor of: **TPU v4-32 training (primary) + slice-host-CPU or spot
A100 capture into the us-central2 bucket**, with the GCS Colab notebook as
backup. Lesson recorded: capture-and-train-in-place on session-limited,
disk-limited VMs couples the two most failure-prone phases; the
capture-once-to-durable-storage design decouples them.

## Lane assignment note: condition D — SUPERSEDED by A12 (v0.1.2 addendum)

**Superseded (2026-07-12, A12 backend map):** the earlier plan below assigned
condition D to the GPU lane. A12 freezes the (condition × seed) → backend
assignment with the constraint that all conditions within a matched seed
share one backend; the frozen map is **ALL 14 training runs (A/B/C/F ×
seeds 1–3, plus D/E seed 1) on the TPU/JAX lane**. D therefore runs on TPU,
accepting the ~2–3 h host-transfer selection path (per-bank budgets computed
host-side under the static-shape port; pinned f32 transfers). Capture and
fidelity remain on CUDA per A7; the cross-backend A-vs-C replication on the
torch/CUDA lane is kept as a robustness check (and is the A10 audit
instrument on gate failure).

*Original note (historical):* D's per-bank variable budgets are host-side
under the JAX static-shape port (~400 MB/step PCIe → hours). The torch/CUDA
path selects on-device natively: train D on the A100 (~15 min) in parallel
with the TPU suite. Same shards, same composer seed → matched data order
(§10) holds; A7 + parity legitimize the mixed backend. TPU-only fallback (no
GPU available): accept ~2-3 h with pinned f32 transfers, or host-CPU torch
for D.
