# Historical boundary

The repository began as a governed DeepSeek-VL2-Tiny study: multimodal data,
64 routed experts with top-6 routing, block-6 gated expert contributions,
BatchTopK, A–F controls, and visual-token analyses. Its most reusable work was
operational rather than model-specific: Torch/JAX parity checks, grouped GEMMs,
capacity adaptation after real routes violated IID estimates, atomic/resumable
capture, fingerprints, MFU/goodput telemetry, and corruption halts.

The Colab streaming campaign then demonstrated the cost of too many live state
surfaces: local disk pressure, upload races, partially recoverable shards, and
complex restart logic. Moving capture to staged storage and TPU execution was
the right conclusion. Spec v0.1.2 later added keyed splits, per-bank thresholds,
auxiliary diagnostics, triggers, and governance for a visual-cortex hypothesis;
those mechanisms do not answer the present reconstruction-cost question.

The completed 1M A/B/C artifacts were plumbing tests. They stopped after 244
optimizer steps, entirely inside a 1,000-step warmup; evaluation sparsity did
not match training; F was absent; and TPU MFU was not measured. They cannot
support the new efficiency headline.

Accordingly, the active tree is a clean Gemma 4 rewrite. Committed DeepSeek-era
code is preserved at annotated tag `deepseek-vl2-final-a291707`, pointing to
commit `a291707`. Ignored `runs/`, `captures/`, and `stats/` remain untouched,
but the new code neither detects nor migrates their schemas. Every new artifact
lives below `artifacts/gemma4_v1/` and old capture shapes fail closed.
