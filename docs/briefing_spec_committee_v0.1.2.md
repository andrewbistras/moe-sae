# Implementation & Pre-Registration Briefing — spec v0.1.1 → v0.1.2

**To:** the co-authors of frozen spec v0.1.1
**From:** implementation (Andrew Bistras; built with Claude as implementation agent)
**Date:** 2026-07-07
**Requested actions:** (1) review the proposed v0.1.2 operational freezes (§4, items A1–A10) — objections now or never, the tag follows this review; (2) review the pre-registration draft (§5), with particular attention to the disclosure section; (3) answer the numbered questions in §6 — several close degrees of freedom that become unclosable once discovery data exists. **Discovery has not started and will not start until this document clears review and the registration is posted.**

---

## 1. Execution status against §20

| §20 step | Status | Evidence |
|---|---|---|
| 1. Capture implementation + FP32 hard gate | **DONE — PASS** | Pure-FP32 reference forward over 100,933 tokens: max per-token rel. err **0.0** (bit-exact); production-path cross-check 0.0; BF16 report max 1.99e-3 / median 1.66e-3, no systematic structure (all \|ρ\| < 0.03 vs gate magnitude, position, modality) |
| 2. Statistics pass (μ_e, router baselines, loads) | DONE (pipeline scale) | 3,500 examples / 1.74M visual + 77k text tokens; re-runs automatically for the discovery capture |
| 3. Pipeline run (1M tokens, f=1024, k̄=32) | **DONE** | conditions A, B, C trained end-to-end; LR sweep waived (A9) |
| 5. A-vs-C gate | **Rehearsed** at pipeline scale — see §3 | Binding gate remains discovery seed 1 (A10) |
| 4, 6–10 | Not started; fully implemented and reviewed | 258 automated tests; independent adversarial review (§2) |

## 2. Implementation report (condensed)

**Validation architecture.** Production capture is a line-identical instrumented clone of the model's `moe_infer` (bit-identity unit-tested); the §3 gate compares an *independent* recomputation (module input + gate outputs → per-expert re-execution) against the model's own aggregate. The FP32 gate returning literal 0.0 means the capture path reproduces the model's arithmetic exactly, not merely within tolerance.

**Adversarial review.** A 4-lens review (SAE math, capture/eval semantics, analysis statistics, runtime-at-scale) with independent verification produced 16 confirmed findings, all fixed with regression tests before any training. The ones with scientific teeth: training-stream contamination of the held-out example buckets (now frozen as A3); global-condition dictionaries mis-sized at eval (would have crashed/corrupted the A–B–F triangle); probes fitting on a split the pipeline could never produce; analysis-stage scoring of the frozen evaluation split (now forbidden until post-intervention). Automated tests were green throughout — the review is what caught these.

**Production-scale events (both now frozen policy, A7).** (i) MPS `topk` kernel fault at ≥25.2M elements — semantics-identical CPU fallback. (ii) Per-batch bank loads burst far beyond iid predictions: aggregate-hottest bank predicts ~500±22 slots/batch, but a real batch put 909 slots on a bank outside the aggregate top-3 — routing is example-correlated (an image's ~500 tokens route together). Handled by grow-and-retry capacity (candidate dropping forbidden). This observation has analysis implications — see Q1.

**Cross-backend story.** Discovery training may run on CUDA (A100) and/or TPU (JAX port). The JAX port is held to the PyTorch reference by a 21-test parity suite: exact fp32 equality on selection masks/θ_b/condition-C permutations/losses, exact optimizer-semantics parity under identical gradients, checkpoint format readable by the registered torch eval stack. Capture remains on the PyTorch path on every backend; per A7, the §3 FP32 gate + BF16 report re-run on any new capture backend before its shards are used (planned for the A100 before discovery capture).

## 3. Results so far — pipeline validation (1M tokens; pre-declared plumbing-only)

Disclosed verbatim; spec §10 forbids scientific conclusions from this run, and we draw none. Single seed, f=1024, k̄=32, 244 steps (all inside LR warmup), deadness window never crossed (aux loss inert by construction at this scale).

| | A (router-tied) | B (compute-matched global) | C (random routing) |
|---|---|---|---|
| Held-out local micro NMSE (θ\*-inference, dev buckets) | **0.242** | — | 0.450 |
| Final-step aggregate NMSE (train) | **0.328** | 0.461 | 0.648 |
| Final-step mean L0, visual/text | 36.5 / 27.5 | 48.2 / 15.8 | 43.3 / 20.7 |

- **Gate rehearsal:** A/C = **0.538** against the ≤0.9 rule — the implementation transmits routing information to the dictionaries, decisively. This validates plumbing; it is not the §20.5 gate (A10).
- Directional teasers we are *not* entitled to: A beat B on the aggregate metric (B's own objective — the "strong evidence" direction under the §18 asymmetry rule) even at 1/40th of discovery scale and mid-warmup LR; and the BatchTopK budget allocates features across modalities very differently per condition — the global SAE starves text hardest, router-tied allocates most evenly. Both are exactly the shapes the discovery run will measure properly.
- Honest deflation available: at 244 steps, none of these dictionaries is remotely converged; relative orderings at 1M can invert at 50M. The disclosure exists so no reader (including us) can pretend the discovery priors are flat.

## 4. Proposed spec v0.1.2 — addendum text for review

*(Full frozen text as committed; v0.1.1 remains byte-frozen. On conflict the addendum wins.)*

> **Scope:** No architectural change. Freezes operational degrees of freedom closed during implementation, before any discovery-scale data exists.

| # | Section | Class | Change |
|---|---|---|---|
| A1 | §8 | freeze | Aux loss completed: candidates = dead-at-step-start among visible banks; per-token top min(512,·) of dead-masked preacts (selected iff > −∞); target = detached per-expert residuals through own decoders, no μ; **L_aux = Σ‖r−r̂‖²/Σ‖r‖²**, 0 below 1e-12 energy; aux never resets deadness; main+aux co-selection permitted; L_aux = 0 exactly when nothing visible is dead |
| A2 | §12.C | freeze | σ_ex = numpy PCG64(0xC0FFEEC ^ (example_id & 0xFFFFFFFF)) uniform permutation of {0..63}; framework- and seed-independent; C's deadness clock on assigned banks; bank_loads records host routing in all conditions |
| A3 | §9/§16.4/§17 | freeze | Splits: example_id % 100 → 0–79 train / 80–89 dev / 90–99 eval; training and statistics consume train only; probes fit train, report dev; eval untouched until intervention; enriched sets on dev |
| A4 | §6 | freeze | θ_b for D = smallest selected across union; for E = mean per-token k̄-th largest; same EMA/θ\* procedure |
| A5 | §11 | freeze | Adam ε=1e-8, bias correction on, clip coef = min(1, 1/(‖g‖+1e-6)); projection renormalizes stored column direction first |
| A6 | §8 | freeze | Deadness boundaries: clocks/stamps init 0; alive until bank sees a full window; only main-selection stamps (zero-valued selections stamp); B/F clock advances by B per batch |
| A7 | §3 ext. | clarification | Backend substitutions permitted iff selection/loss semantics exact; recorded per event; candidate dropping never permitted; §3 gate + BF16 report bind every capture backend; parity suite binds training backends |
| A8 | §3/§13 | clarification | BOS never captured/replaced; visual = images_seq_mask positions; matched-position mask = non-answer ∧ non-BOS; answer CE on exact answer spans |
| A9 | §11 | record | LR sanity option exercised as *waived*; 3e-4 frozen everywhere |
| A10 | §20 | record | Pipeline A-vs-C rehearsal happened (results in §3); binding gate remains discovery seed 1 |

## 5. Pre-registration draft (OSF)

*(Text as committed to `docs/preregistration_osf.md`; placeholders `[TAG]`/`[COMMIT HASH]` fill at tag time.)*

**Structure:** research questions (spec §1) · hypotheses = the four frozen §18 interpretations with all three reading rules pre-committed (objective asymmetry; power clause — negative claims from the confirmatory run only; the (A−B)/(F−B) headline template) · design and budgets (§2–§12; seeds 1–3 for A/B/C/F, 1 for D/E; 50M discovery / 200M confirmatory) · all measures and analyses by spec section (§13–§17), including both implementation gates · **data-collected-to-date disclosure**: the §3 gate result and the full pipeline table above, verbatim, with the statement that every choice these could have influenced was frozen 2026-07-06, before the run — and the explicit concession that a reader may treat discovery A-vs-B/C priors as favorable · materials = the tagged commit (spec + all analysis code + 258 tests), locked per §20.7/§20.9 · deviations policy = the spec's own §0 version-bump discipline.

## 6. Questions for the committee — closable now, not later

**Q1 — Example-correlated load bursts and condition D.** Real per-batch bank loads are far heavier-tailed than iid (909 observed vs ~500±22 predicted). D's budgets are per-batch loads by largest-remainder — so D inherits this variance directly, and §16.3 stability statistics inherit it indirectly. My reading: this is *intended* — the variance is part of what D isolates, and the load floor handles the analysis side. Confirm, or amend D now (e.g., EMA-smoothed loads) — after discovery it's unamendable.

**Q2 — Text-position reuse.** At discovery scale the text half of the stream re-draws from ~10⁶-scale unique prompt-text tokens (~4–8 effective epochs) while the visual half is effectively unique. Disclosed. Cheap to add now if wanted: a pre-registered robustness check that recomputes §16.1/§16.4 on the text half stratified by draw-epoch, to show reuse doesn't drive the modality signal. Add, or decline explicitly?

**Q3 — Gate-folding and the primacy of m_act.** §14 makes coefficient mass primary; but m_act inherits the gate-magnitude confound (§15.2) most directly of the three masses, while m_count is the least gate-contaminated. The identity probe and intervention matching are the frozen defenses. Question: keep m_act primary (my recommendation — it's frozen, and switching primacy post-hoc would be worse than the confound), or promote m_count to co-primary *now* with a written reason?

**Q4 — §17's fifth matching criterion.** Removed-vector norm is implemented as: per-latent coefficient-mass proxy at construction (exact under unit decoder columns) + post-hoc per-token removed-norm measurement with a 20% tolerance flag on every control set. Exact pre-hoc token-level norm matching is impossible (norms depend on co-firing patterns). Confirm this reading satisfies §17, or specify an alternative construction now.

**Q5 — Small analysis lacunae to bless.** (a) Probe 2 (64-dim bank-mass) is undefined for B/F — reported as N/A, probes 1/3/4 run for all conditions. (b) For condition C, all bank attribution (masses, energy, identity probe, interventions) uses the *assigned* bank via the dump's act_bank field. (c) JSD in base 2; enrichment log natural; ε = 1e-6. Objections?

**Q6 — Sharpen the confirmatory trigger?** §20.11 currently says "decide whether the confirmatory run proceeds" after discovery. That's a researcher degree of freedom a cynical reader will notice. Option: pre-commit now, e.g. *"the confirmatory run proceeds iff the §20.5 gate passes and A ≥ B on aggregate NMSE in ≥2 of 3 seeds; otherwise the project reports discovery as final with the power clause invoked."* I mildly favor pre-committing. Decide.

**Q7 — Outcome priors, on the record.** So a later reader can score us: I put architecture-only as the most likely §18 outcome (A > C decisively; A ≥ B on aggregate; X = (A−B)/(F−B) meaningfully positive), partial-positive second (enrichment present, top-6 stability marginal under Jaccard), strong-positive perhaps one-in-four (the intervention-selectivity-beyond-C bar is high), discovery-negative unlikely and — per the power clause — uninterpretable anyway. If the committee's priors differ materially, say so now; divergent priors are exactly what §18's reading rules are for.

## 7. Mechanics after sign-off

1. Apply any committee amendments → freeze → `git tag spec-v0.1.2` → fill `[TAG]`/`[COMMIT HASH]`.
2. Post the registration (OSF).
3. Re-run the §3 gate + BF16 report on the discovery capture backend (A100), per A7.
4. Launch discovery: capture (~96k examples → 50M tokens, 3 parallel workers) → 14 training runs → §20 steps 5–10 in order. Two independent compute lanes exist (CUDA/torch and TPU/JAX-parity); the A-vs-C gate will additionally be replicated cross-backend as a robustness check.

*Artifacts referenced: `spec/frozen_spec_v0.1.1.md`, `spec/frozen_spec_v0.1.2_addendum.md`, `docs/preregistration_osf.md`, `reports/hook_validation.json`, `reports/sanity_gate_pipeline.json`, `docs/INTERFACES.md`, test suite (258 passing).*
