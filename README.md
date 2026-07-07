# moe-sae — Router-Tied BatchTopK SAEs for Multimodal MoE Models

This repository implements a frozen interpretability experiment: train one
sparse-autoencoder dictionary per routed expert of DeepSeek-VL2-Tiny, tied to
the model's own router, with a single batch-wide BatchTopK competition across
all visible expert banks — and test, against pre-registered control
architectures, whether explanatory feature mass concentrates stably and
causally on a small set of experts for visual tokens beyond what the router
itself already explains. The experimental design is frozen before results are
seen; every scientific constant in this codebase is a citation to
`spec/frozen_spec_v0.1.1.md`, not a tunable.

The frozen one-sentence specification (spec v0.1.1):

> Train a jointly optimized bank of 64 expert-specific BatchTopK SAEs on the
> gated routed-expert contributions at block 6 of DeepSeek-VL2-Tiny, with
> 4,096 features per expert and a single batch-wide budget averaging 64 active
> features per original token; test — against compute-matched,
> capacity-matched, random-routing, independent-budget, and per-token controls
> — whether explanatory feature mass concentrates stably and causally on a
> small set of experts for visual tokens beyond what is explained by the
> model's router itself.

## Repo map

```
spec/frozen_spec_v0.1.1.md   The constitution. Nothing may contradict it.
docs/INTERFACES.md           Engineering contract: formats, APIs, frozen constants.
docs/RUNBOOK.md              Operational manual for spec §20's execution order.
configs/                     RunConfig YAMLs: 3 pipeline + 14 discovery runs.
src/moe_sae/
  model_loading.py           Frozen model load + spec §2 startup assertions.
  vendor/deepseek_vl2/       Vendored DeepSeek-VL2 modeling code (see PATCHES.md).
  capture/                   Activation-site hooks, FP32 validation gate, shard writer.
  data/                      Batch composition (§9), eval example provider, splits.
  sae/                       Conditions A–F, BatchTopK engine, SpecAdam, dead tracking.
  train/                     RunConfig + training loop (θ EMA, aux loss, artifacts).
  eval/                      Reconstruction metrics, downstream replacement, dumps.
  analysis/                  §14–§16 mass/enrichment/stability/information analyses.
scripts/                     CLI entry points (run_capture, run_train, run_eval, ...).
tests/                       CPU-only unit tests vs naive reference implementations.
```

## Quickstart

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/). All heavy lifting
is `torch` (MPS on Apple silicon, CUDA in the cloud); `transformers` is pinned
to 4.38.2 for the vendored DeepSeek-VL2 code.

```sh
uv sync                      # create the environment (installs moe_sae editable)
uv run pytest -q             # unit tests: CPU-only, seconds, no model download
uv run python scripts/smoke_load.py   # downloads deepseek-vl2-tiny (~7 GB) and
                                      # runs the spec §2 architecture assertions
```

Then follow **`docs/RUNBOOK.md`**, which maps spec §20's execution order to
commands: (1) hook-validation hard gate, (2) statistics pass, (3) 1M pipeline
run on a laptop, (4+) 50M-token discovery runs on a cloud GPU, evaluation,
analysis, and the subtraction-intervention causal test.

## Status

- [x] Frozen spec v0.1.1 and engineering contract
- [x] Vendored model code + frozen loading/assertions (`smoke_load.py`)
- [x] Capture hooks (instrumented + FP32 reference paths) with bit-identity tests
- [x] Capture shard writer / reader
- [x] Run configurations (pipeline + discovery) and runbook
- [ ] FP32 validation gate run on 100k tokens (spec §20.1)
- [ ] Statistics pass (μ_e, routing baselines, bank loads)
- [ ] 1M pipeline run, conditions A/B/C
- [ ] Discovery runs A/B/C/F × 3 seeds + D/E (cloud)
- [ ] A-vs-C implementation sanity gate
- [ ] Evaluation, analyses, interventions (§20.6–§20.10)

## Licenses

- **Vendored model code** (`src/moe_sae/vendor/deepseek_vl2/`) comes from the
  upstream [DeepSeek-VL2](https://github.com/deepseek-ai/DeepSeek-VL2)
  repository, whose code is MIT-licensed; `modeling_deepseek.py` additionally
  carries an Apache-2.0 header from its HuggingFace GPT-NeoX/OPT lineage. Both
  headers are preserved in the files. Local patches are execution-backend
  only, never math, and are logged in
  `src/moe_sae/vendor/deepseek_vl2/PATCHES.md`.
- **Model weights** (`deepseek-ai/deepseek-vl2-tiny`) are distributed under the
  [DeepSeek Model License](https://github.com/deepseek-ai/DeepSeek-VL2/blob/main/DeepSeek-MODEL-LICENSE),
  which permits commercial use with use-based restrictions; they are downloaded
  from HuggingFace at runtime and never redistributed by this repository.
- Original code in this repository: research code accompanying the frozen
  spec; no separate license file yet.
