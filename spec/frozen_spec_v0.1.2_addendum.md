# Frozen Research Spec v0.1.2 — Addendum to v0.1.1

**Status:** Frozen. v0.1.2 = `frozen_spec_v0.1.1.md` (unmodified, byte-frozen) **plus this addendum**. On conflict, this addendum wins. Supersedes the engineering-contract decisions log (`docs/INTERFACES.md`) for every item restated here.

**Scope:** No architectural change. The model, activation site, widths, sparsity, batch composition, data, losses, optimizer core, conditions, metrics, analyses, and interpretations of v0.1.1 are untouched. This bump freezes *operational* degrees of freedom that v0.1.1 left open and that were closed during implementation — **before any discovery-scale data exists**. Freezing them now, in the spec proper, removes them as post-hoc researcher degrees of freedom.

**Timing note (for the pre-registration record):** at the time of this freeze, the only training executed is the §10 pipeline-validation run (1M tokens, f=1024, k̄=32 — pre-declared "plumbing only; no scientific conclusions"). Its observed results are disclosed in the pre-registration. No discovery-scale capture or training has begun.

---

## 0. Changelog v0.1.1 → v0.1.2

| # | Section | Class | Change | Written reason |
|---|---------|-------|--------|----------------|
| 1 | §8 | freeze | Aux-loss operational definition completed: normalization, candidate semantics, reset semantics (A1). | v0.1.1 froze selection budget and target but left the loss normalization and deadness-interaction semantics open — an unfrozen degree of freedom in an optimized loss. |
| 2 | §12.C | freeze | Condition C permutation *construction* pinned to an exact PRNG scheme (A2). | v0.1.1 froze four required properties; two implementations satisfying them could still differ. Cross-seed and cross-backend reproducibility requires one construction. |
| 3 | §9/§16.4/§17 | freeze | Example-level split construction pinned: `example_id % 100` buckets; training and statistics consume train buckets only (A3). | v0.1.1 required example-level held-out splits without constructing them; training-set contamination of "held-out" metrics was caught in implementation review and is closed here. |
| 4 | §6 | freeze | θ_b conventions for conditions D and E (A4). | §6 defines θ_b only for the global competition; D and E need a logged θ_b and frozen θ\* by the same procedure. |
| 5 | §11 | freeze | Optimizer residual constants: Adam ε, bias correction, clip formula, projection detail (A5). | §11 froze the step order but not ε/bias-correction; these alter trajectories and must not be tunable. |
| 6 | §8 | freeze | Deadness boundary semantics: initialization, pre-window rule, reset source (A6). | The window was frozen in v0.1.1; the boundary conditions (never-fired latents, aux selections) were not. |
| 7 | §3 | clarification | Execution-backend substitutions that preserve selection/loss semantics are permitted and must be recorded; candidate dropping is never permitted (A7). | Two backend issues surfaced at production scale (MPS top-k kernel fault; TPU static-shape grouping). The class of permissible response needs a frozen boundary. |
| 8 | §3/§13 | clarification | Capture/eval masks pinned: BOS exclusion, token-type definition, matched-position mask (A8). | §13's "matched positions" needs an executable definition; off-by-one choices here silently move the primary fidelity metric. |
| 9 | §11 | record | The optional LR sanity check was **waived**; LR is frozen at 3×10⁻⁴ (A9). | §11 permitted one pre-run comparison; the option is hereby exercised as "not run" and closed. |
| 10 | §20 | record | The pipeline run included an A-vs-C *rehearsal* of the step-5 gate machinery; the binding gate remains discovery seed 1 (A10). | Prevents the rehearsal from being conflated with, or substituted for, the frozen gate. |

---

## A1. Aux loss — completed frozen definition (§8)

- **Candidate pool:** latents dead *at the step's start* (per the tracker state entering the step), among the token's six visible banks (A/C/D/E) or the single dictionary (B/F).
- **Selection:** per token, the top `min(512, n_candidates)` dead candidates by ReLU pre-activation, a token-level competition over dead-masked scores; a candidate is selected iff its masked score exceeds −∞ (i.e., only genuinely dead latents are ever selected).
- **Target and loss:** with detached per-expert residuals r_te = c_te − ĉ_te, the selected dead latents reconstruct r̂_te through their own banks' decoders without μ, and
  L_aux = Σ_te ‖r_te − r̂_te‖² / Σ_te ‖r_te‖²,
  with the batch contributing 0 when Σ‖r‖² < 10⁻¹²; coefficient α = 1/32 (unchanged).
- **Interactions:** aux selection never resets deadness (only main-BatchTopK selection does); a latent may be both main- and aux-selected in one batch; when no visible latent is dead, L_aux = 0 exactly.

## A2. Condition C permutation construction (§12.C)

σ_ex = a uniform random permutation of {0,…,63} drawn from `numpy.random.Generator(numpy.random.PCG64(SALT ^ (example_id & 0xFFFFFFFF)))` with frozen salt `SALT = 0xC0FFEEC`; bank of slot i = σ_ex(S_t[i]). The stream is independent of every training seed and of the ML framework (pure NumPy), satisfying v0.1.1's four properties by construction. Under condition C, the deadness clock advances on *assigned* (permuted) banks; the per-bank training-load artifact (`bank_loads`) records *host-expert* routing in all conditions (it feeds the §16.3 load floor, a property of the model's router).

## A3. Data splits (§9, §16.4, §17)

Frozen split function: `example_id % 100` → buckets 0–79 **train**, 80–89 **development**, 90–99 **evaluation**. SAE training batches and the §5/§15 statistics pass consume train buckets only. Probes (§16.4) fit on train-bucket tokens and report on development buckets; evaluation buckets are not scored before the intervention stage. Enriched sets (§17) are defined on development buckets; interventions run on evaluation buckets only. Example IDs are stable, dataset-derived integers assigned by the frozen stream order.

## A4. θ_b conventions for D and E (§6)

- **Condition D:** θ_b = the smallest selected score across the union of per-bank selections.
- **Condition E:** θ_b = the mean over tokens of each token's k̄-th largest candidate score.
Both feed the same frozen EMA (decay 0.999, init = first batch) and freeze θ\* at end of training, as in §6. These θ_b are log-only conventions; they do not alter selection.

## A5. Optimizer residual constants (§11)

Adam ε = 10⁻⁸ inside the square-root denominator; standard bias correction (m̂ = m/(1−β₁ᵗ), v̂ = v/(1−β₂ᵗ)); global clip coefficient min(1, 1.0/(‖g‖₂ + 10⁻⁶)) over all parameters in FP32; the tangent-space projection re-normalizes the stored decoder column direction before projecting (so BF16 rounding of the unit-norm invariant cannot leave a parallel residue).

## A6. Deadness boundary semantics (§8)

Per-bank clocks start at 0 with all last-fire stamps 0 ("never fired"); a latent is dead iff (clock − last_fire) > 250,000 AND clock ≥ 250,000 — so every latent is alive until its bank has seen a full window, including never-fired latents. Only main-BatchTopK selection stamps last_fire; a zero-valued selected candidate (possible when fewer than K scores are positive) still stamps. For B/F the single dictionary's clock advances by B per batch.

## A7. Execution-backend substitutions (§3 extension)

Backend substitutions are permitted iff they preserve selection and loss *semantics* exactly (same fp32 comparison math; same selected sets up to floating-point ties; no candidate ever dropped), and each is recorded with its trigger. Recorded to date: (1) BatchTopK computed on CPU for MPS-resident tensors (MPS top-k kernel fault at ≥25.2M elements); (2) capacity-padded grouped GEMMs on static-shape backends, where capacity overflow *grows the capacity and retries* — token dropping is forbidden; (3) a JAX port of the training step, held to the PyTorch reference by a parity suite (exact selection/permutation/optimizer-semantics equality in fp32; documented fp-accumulation tolerances end-to-end). The FP32 validation gate (§3) and its production cross-check bind every backend that produces capture data.

## A8. Capture and evaluation masks (§3, §13)

Position 0 (BOS) is never captured and never replaced. Token type is frozen at capture: *visual* = positions under `images_seq_mask` (including tile-layout separator embeddings); *prompt-text* = all other non-BOS prompt positions. §13 matched-position replacement = all non-answer, non-BOS positions; full replacement = all non-BOS positions. Answer spans are teacher-forced ground-truth continuations; answer-token cross-entropy is measured on exactly the answer-span positions.

## A9. Learning-rate option closed (§11)

The optional one-time LR sanity check (10⁻⁴ vs 3×10⁻⁴ vs 10⁻³ on the pipeline run) was not exercised. LR = 3×10⁻⁴ with the §11 schedule is frozen for every condition, seed, and scale.

## A10. Pipeline-scale gate rehearsal (§20)

The §20.3 pipeline run was additionally used to rehearse the step-5 A-vs-C machinery end-to-end (capture → stats → train → eval → gate). Its outcome is disclosed in the pre-registration. The **binding** implementation gate remains as written: discovery seed 1, A must beat C by ≥10% relative on local micro-averaged NMSE. The rehearsal cannot substitute for it, and its result was not used to alter any frozen choice.
