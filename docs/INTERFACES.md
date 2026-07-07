# Implementation contract — frozen spec v0.1.1

> **v0.1.2 note:** the operational decisions recorded here that carry scientific
> weight (aux-loss normalization, condition-C construction, splits, θ_b
> conventions, optimizer constants, deadness boundaries, backend-substitution
> policy, capture/eval masks) are now FROZEN in
> `spec/frozen_spec_v0.1.2_addendum.md` (A1–A10), which supersedes this file
> for those items. This file remains the engineering reference for formats
> and APIs.

This document binds every module to shared formats and frozen constants. It is
an *engineering* contract; the science is frozen in `spec/frozen_spec_v0.1.1.md`.
If this document and the spec disagree, **the spec wins** and this file has a bug.

## Frozen constants (`moe_sae.model_loading`)

| Constant | Value |
|---|---|
| `MODEL_ID` | `deepseek-ai/deepseek-vl2-tiny` |
| `HIDDEN_SIZE` (d) | 1280 |
| `NUM_LANGUAGE_LAYERS` | 12 |
| `NUM_ROUTED_EXPERTS` (E) | 64 |
| `NUM_SHARED_EXPERTS` | 2 |
| `TOP_K_ROUTED` | 6 |
| `TARGET_LAYER_ZERO_INDEXED` | 5 (== "block 6, one-indexed", spec §2) |

Frozen SAE constants (spec §5–§11), in `moe_sae.train.config`:

| Name | Discovery | Pipeline validation |
|---|---|---|
| `f` (per-bank width) | 4096 | 1024 |
| `k_bar` | 64 | 32 |
| `B` (batch, tokens) | 4096 (2048 visual + 2048 text) | 4096 (2048+2048) |
| tokens | 50M | 1M |
| LR | 3e-4, 1000-step linear warmup, cosine to 3e-5 | same |
| Adam | β1=0.9, β2=0.999, no weight decay, grad-clip 1.0 | same |
| aux α | 1/32, per-token budget 512 dead candidates across visible banks | same |
| deadness window | 250,000 bank-relative tokens | 250,000 |
| θ EMA decay | 0.999, init = first batch's θ_b | same |
| load floor (analysis) | 500,000 routed training tokens/bank | scaled: 10,000 |

## Capture record format (spec §3 manifest)

Directory layout, one per capture run:

```
captures/<name>/
  meta.json                      # model info, spec version, counts, config
  visual/shard_{i:05d}.safetensors
  text/shard_{i:05d}.safetensors
```

Safetensors fields per shard (N = tokens in shard):

| key | shape | dtype | meaning |
|---|---|---|---|
| `c` | [N, 6, 1280] | bf16 | gated contributions c_te = g_te·y_te, **slot order = gate topk order** |
| `experts` | [N, 6] | uint8 | S_t — routed expert id per slot |
| `gates` | [N, 6] | float32 | g_te per slot (FP32 as produced by MoEGate) |
| `example_id` | [N] | int32 | globally unique example id |
| `domain` | [N] | uint8 | 0=coco, 1=textvqa, 2=chartqa, 3=ai2d, 255=text-only control |

Token type (visual / prompt-text) is encoded by which pool (subdirectory) the
shard lives in. Answer-generation tokens are never captured for training
(spec §9). Writer/reader API in `moe_sae.capture.writer`:
`ShardWriter(root, pool, shard_size)` / `iter_shards(root, pool)` /
`CaptureMeta`.

## Capture hooks (`moe_sae.capture.hooks`)

- `instrumented_capture(moe) -> ContextManager[CaptureSink]` — production path.
  Replaces `moe.moe_infer` with a line-identical clone that additionally records
  `(topk_idx [N,6] i64, topk_weight [N,6] f32, per_slot_contrib [N,6,d] f32)`
  per forward into the sink. Bit-identical output to the original (unit-tested).
- `reference_capture(moe) -> ContextManager[ReferenceSink]` — validation path.
  Pre-hook captures module input x; gate hook captures (topk_idx, topk_weight);
  a wrapper records `moe_infer`'s return (the model's internal routed aggregate)
  without altering it. `recompute_contributions(sink, moe)` then *independently*
  recomputes c_te = g_te · expert_e(x_t) from x and the expert weights.
- The FP32 gate (spec §3): rel err of ‖Σ_slots c_te − aggregate‖/‖aggregate‖
  < 1e-5 per token over 100k validation tokens (`capture/validate.py`).

## SAE training batch (all conditions)

The trainer consumes `TrainBatch`:

```python
@dataclass
class TrainBatch:
    c: Tensor        # [B, 6, d] bf16 — gated contributions, slot order
    experts: Tensor  # [B, 6] int64 — routed expert ids (bank ids under condition A)
    gates: Tensor    # [B, 6] float32
    token_type: Tensor  # [B] uint8 — 0 text, 1 visual
    example_id: Tensor  # [B] int32
    domain: Tensor     # [B] uint8
```

Composed by `moe_sae.data.batches.BatchComposer`: exactly B/2 visual + B/2 text
positions per batch, each half shuffled independently (spec §9); sampling with
replacement across the pool is permitted (positions may repeat across epochs).

## Conditions (spec §12) — operational definitions

All implement `SAECondition` (`moe_sae.sae.conditions`) with
`encode_select_decode(batch) -> StepOutput` and share the BatchTopK engine.

- **A** router-tied: token's slot i encodes into bank `experts[t,i]`.
- **B** compute-matched global: input `c_route = c.sum(dim=1)` [B, d]; single
  dictionary F=6f (24,576 discovery / 6,144 pipeline); k̄=64.
- **C** random-routing: like A, but bank ids = σ_ex(experts[t,i]) where σ_ex is
  a uniformly random permutation of {0..63} drawn per example, seeded by
  `hash(example_id, FIXED_SALT_C)` — independent of training seed (spec:
  six distinct banks per token ✓ bijection; approx. load preserved ✓ expected
  uniform; varies across examples ✓; fixed across matched seeds ✓).
- **D** independent-expert budgets: per-batch, bank e's budget is
  `round_to_total(k̄·B · load_e / Σ load)` where load_e = # slots routed to e in
  this batch; top-k within each bank's candidates; total selected == k̄·B
  (largest-remainder rounding).
- **E** per-token TopK: exactly k̄ features per token across its 6 visible banks.
- **F** capacity-matched global: like B with F=262,144 (discovery) / 65,536
  (pipeline), k̄=64.

θ_b logging + EMA applies to every condition (for D: the global-pool threshold
is still recorded from the union of selected candidates: the smallest selected
score; for E: per-token k-th largest — logged, and θ* frozen the same way).

## Optimizer (spec §11) — `moe_sae.sae.optim.SpecAdam`

Custom Adam: params BF16, `exp_avg`/`exp_avg_sq` FP32, step math FP32, update
cast back to BF16. Order per step, all conditions:
1. global grad-norm clip 1.0 (over all params, FP32 accumulation);
2. for decoder matrices: project out gradient component parallel to each unit
   column: `g_j ← g_j − (g_j·d_j) d_j` (tangent-space projection);
3. Adam moment update + parameter step;
4. renormalize decoder columns to unit ℓ2 norm.

## Deadness (spec §8) — `moe_sae.sae.dead.DeadTracker`

Per bank e: `tokens_seen_e` (int64) increments by #tokens routed to bank e in
the batch; `last_fire_e[j]` stores `tokens_seen_e` at latent j's last main-TopK
selection. Dead iff `tokens_seen_e − last_fire_e[j] > 250_000` AND
`tokens_seen_e ≥ 250_000`. Until a bank has seen a full window, all its latents
are alive. Global conditions (B/F): the single dictionary is one "bank" whose
token count advances by B per batch — same rule, window 250,000 · 6? **No** —
window stays 250,000 *tokens routed to the bank*; for B/F every token visits
the dictionary, so the window is 250k tokens ≈ 61 batches. Bank-relative by
construction (spec changelog #3).

## Aux loss (spec §8) — decisions log

- Candidate pool: latents dead *now*, among the token's 6 visible banks
  (A/C/D/E) or the global dictionary (B/F).
- Selection: per token, top `min(512, #dead_visible)` dead candidates by
  pre-activation a (ReLU'd), token-level competition.
- Target: r_te = c_te − ĉ_te (detached); selected dead latents' activations
  reconstruct r̂_te through their own bank's decoder.
- Normalization (decision): L_aux = Σ_te ‖r_te − r̂_te‖² / Σ_te ‖r_te‖²
  (residual-energy normalized, per-expert sums — no cross-bank cancellation),
  coefficient α = 1/32. Guard: batches where Σ‖r‖² < 1e-12 contribute 0.
- Dead latents selected by aux do NOT reset deadness (only main-TopK selection
  fires a latent).

## Seeding

- Training seeds: 1, 2, 3 (conditions A, B, C, F); seed 1 for D, E (discovery).
- Condition C permutation salt: `0xC0FFEE_C` (fixed, never seed-dependent).
- Data order: capture shard order + composer RNG seeded by `data_seed = 1000 +
  training_seed` so matched conditions see identical batch sequences
  (spec: "Same training-data order across matched conditions where possible").

## Run artifact layout

```
runs/<run_name>/
  config.yaml            # resolved config, frozen constants echoed
  model_info.json
  metrics.jsonl          # per-step: losses, L0, theta_b, dead counts, lr
  theta_trajectory.npy
  checkpoints/{step}.safetensors   # SAE weights + mu_e + theta state
  final/theta_star.json, bank_loads.npy, dead_final.npy
```

## Numerics

- SAE matmuls in BF16 with FP32 accumulation (torch default matmul accumulation
  on MPS/CUDA is FP32 for BF16 inputs).
- All loss reductions / norms / EMA in FP32.
- BatchTopK scores compared in FP32 (preacts upcast before topk) to make
  selection deterministic across backends.

## FeatureDump format (θ*-inference dumps consumed by analysis §14–§16)

Produced by `moe_sae.eval.reconstruction.dump_features` over a capture split:

```
dumps/<run_name>/<split>/shard_{i:05d}.safetensors
```

Sparse activation fields (M = nonzero activations in shard):

| key | shape | dtype | meaning |
|---|---|---|---|
| `act_token` | [M] | int64 | token row index within this shard |
| `act_slot`  | [M] | uint8 | slot 0..5; 0 for global conditions |
| `act_bank`  | [M] | uint8 | assigned dictionary bank — equals experts[token, slot] for A/D/E, the *permuted* bank for C, 0 for B/F |
| `act_latent`| [M] | int32 | latent index j within the bank / global dictionary |
| `act_value` | [M] | float32 | z value (post θ* threshold) |

Per-token parallel fields (N tokens): `experts` [N,6] u8, `gates` [N,6] f32,
`token_type` [N] u8, `example_id` [N] i32, `domain` [N] u8, `l0` [N] i32,
`energy` [N,6] f32 (per-slot ‖D_e z_te‖², slot 0 for global conditions — the
§14 m_energy source; analysis skips the energy robustness check when absent).
Consumers prefer `act_bank` for bank attribution and fall back to
experts[token, slot] (exact for condition A only).

## Frozen-θ* inference API (integration decision)

`SAECondition.encode_thresholded(c [N,6,d], experts [N,6], example_id [N],
theta_star) -> ThresholdedEncoding` (exported from `moe_sae.sae`): sparse
act_token/act_slot/act_bank/act_latent/act_value + l0 [N] + recon_slots
[N,S,d] (ĉ = μ + Dz) + mu_slots [N,S,d]; S=6 router-tied, S=1 global (B/F).
`example_id` is REQUIRED — condition C's bank assignment is a per-example
permutation even at inference. Conditions also expose `num_banks` (64
router-tied / 1 global) and `decoder_cols(bank [M], latent [M]) -> [M, d]`.

## Stats artifact (`train/stats_pass.py` → stats.safetensors)

Keys: `mu_e` [64,1280] f32, `mu_route` [1280] f32, `bank_counts`,
`bank_counts_visual`, `bank_counts_text` [64] i64, `p_route_visual`,
`p_route_text`, `p_gate_visual`, `p_gate_text` [64] f32. The image modality is
named **"visual"** in stored artifacts (matching capture pool names); analysis
code maps its "image" naming onto the `_visual` keys when loading.

## Eval example provider (`moe_sae.data.evalsets`)

```python
@dataclass
class EvalExample:
    input_ids: Tensor            # [T] includes prompt AND teacher-forced answer
    images: Tensor | None        # as produced by DeepseekVLV2Processor
    images_seq_mask: Tensor | None
    images_spatial_crop: Tensor | None
    answer_mask: Tensor          # [T] bool — answer-span token positions
    visual_mask: Tensor          # [T] bool — projected visual positions
    example_id: int
    domain: str                  # coco|textvqa|chartqa|ai2d|text_control
    answer_text: str

def iter_eval_examples(domain: str, n: int, seed: int, split: str = "validation") -> Iterator[EvalExample]
```

Matched-position replacement mask (spec §13 primary) = NOT answer_mask (and not
BOS); full replacement = all positions.

## Validation-input export (consumed by capture/validate.py)

`moe_sae.data.evalsets.export_validation_inputs(out_dir, n_tokens_target=100_000, seed=7)`
writes `examples_{i:04d}.pt` (torch.save of a dict with the processor outputs:
input_ids, images, images_seq_mask, images_spatial_crop, example_id, domain) —
prompt-only (no answers), mixed 4 domains, enough examples to cover ≥100k
language-stack token positions.

## Splits (frozen convention)

`example_id % 100`: 0–79 train, 80–89 development (enriched-set definition,
§17), 90–99 evaluation (untouched until intervention run). Example-level, never
position-level (spec §16.4). SAE training (`BatchComposer`) and the statistics
pass (`compute_stats`) consume ONLY the train buckets 0–79 by default
(`split='train'`; `split=None` disables filtering for debugging only).
