# moe-sae

This repository runs one focused Gemma 4 experiment:

> Router-conditioned A recovers **G%** of the full-dictionary reconstruction
> gain at **C%** of its measured marginal SAE training cost.

The active model is `google/gemma-4-26B-A4B`, text only. Activations are the
aggregate routed-MoE branch at one-indexed block 15, immediately after
`post_feedforward_layernorm_2`. Conditions A, B, and F reconstruct exactly the
same globally centered target with exact per-token TopK-64.

The previous DeepSeek-VL2/spec-era implementation is intentionally absent from
the active tree. It is preserved at Git tag
`deepseek-vl2-final-a291707`; ignored historical `runs/`, `captures/`, and
`stats/` directories are never read or migrated.

## Active surface

- [`configs/gemma4_26b_a4b.yaml`](configs/gemma4_26b_a4b.yaml) — the single
  production configuration.
- `moe-sae` — the single CLI, with `preflight`, `convert-checkpoint`,
  `benchmark-harvest`, `harvest`, `train`, `eval`, `report`, and `run`.
- [`src/moe_sae/model.py`](src/moe_sae/model.py) — pinned MaxText prefix adapter.
- [`src/moe_sae/artifacts.py`](src/moe_sae/artifacts.py) — content-addressed
  capture shards, atomic manifests, and deterministic resume.
- [`src/moe_sae/sae.py`](src/moe_sae/sae.py) — exact A/B/F scoring, distributed
  TopK, sparse decode, and optimizer math.
- [`src/moe_sae/metrics.py`](src/moe_sae/metrics.py) — reconstruction and cost
  report contract.

## Local development

Python 3.12 and [uv](https://docs.astral.sh/uv/) are required.

```sh
uv sync
uv run pytest -q
uv run moe-sae --help
```

TPU hosts additionally install the pinned MaxText runtime:

```sh
uv sync --extra tpu
```

Checkpoint conversion uses a separate CPU-only Torch environment:

```sh
uv sync --extra conversion
```

MaxText resolves from commit
`48c7da3e28e8f57fe36ddf09d00eec53072a2ea0`. Torch is not a training-runtime
dependency; install CPU Torch only in the isolated checkpoint-conversion
environment.

On multi-host TPU runs, `artifacts/gemma4_v1/` must be a shared POSIX mount;
preflight proves cross-host visibility. The converter writes a self-hashed
parity attestation and checkpoint inventory digest that all later commands
require, and the `run`
subcommand resumes only stages whose config, checkpoint, capture, code, and
environment provenance still match. Production commands reject artifact roots
outside this namespace and independently recompute the staged checkpoint
inventory before trusting the supplied digest. The attestation hash is bound
into benchmark, capture, campaign, and report metadata. C4 provenance is likewise
derived from a verified staging manifest and ordered ArrayRecord SHA-256
inventory; no operator-supplied dataset label is trusted.

Read [`docs/EXPERIMENT.md`](docs/EXPERIMENT.md) for the scientific contract and
[`docs/RUNBOOK.md`](docs/RUNBOOK.md) for the fail-closed execution sequence.
