# Pre-registration: Router-Tied BatchTopK Sparse Autoencoders for Multimodal Mixture-of-Experts Models

**Registration type:** OSF Prereg (adapted template). **Draft v1 — not yet posted.**
**Author:** Andrew Bistras. *(Implementation and drafting assistance: Claude (Anthropic); all frozen scientific choices predate implementation.)*
**Date of registration:** [POST DATE]
**Frozen protocol:** spec v0.1.2 = `spec/frozen_spec_v0.1.1.md` + `spec/frozen_spec_v0.1.2_addendum.md`, at repository tag `[TAG]`, commit `[COMMIT HASH]`. The registered analysis code is the same commit (258 automated tests passing at tag time).

---

## 1. Research questions

**RQ1 (primary).** When a multimodal MoE model (DeepSeek-VL2-Tiny) processes visual and textual tokens, does a global BatchTopK competition over router-exposed, expert-specific dictionaries reveal stable concentrations of explanatory feature mass in a small subset of experts — beyond what is explained by the model router's own assignment and gate magnitude?

**RQ2 (representation-learning).** Does the pretrained router partition activation space usefully enough that a router-tied SAE bank outperforms (a) an equally expensive global SAE, (b) an equally sized global SAE, and (c) a randomly routed SAE bank?

Full motivation, activation-site definition, and architecture: frozen spec §1–§8.

## 2. Hypotheses and pre-registered interpretations

The four permitted outcome readings are frozen verbatim in spec §18 ("strong positive", "partial positive", "architecture-only", "negative"), each with necessary conditions. Three reading rules are pre-committed (spec §18):

1. **Objective asymmetry.** Condition A optimizes a strictly harder decomposed loss and is scored on the aggregate metric B/F optimize directly; narrow A losses on aggregate NMSE are ambiguous, A wins are strong evidence.
2. **Power clause.** The negative interpretation may only be claimed from the confirmatory run (200M tokens). A discovery-scale null (~12k fires/feature; 16× leaner than compute-optimal reference allocations — spec §10) is reported as underpowered, not informative.
3. **Headline template.** The architecture result is stated as: "router-tying recovers X% of the full-dictionary gain (F − B) at 1/10.7 the encoder cost," X = (A−B)/(F−B) on the pre-registered aggregate metric.

## 3. Design

- **Model/site (frozen):** DeepSeek-VL2-Tiny, language block 6 (one-indexed; asserted at startup), gated routed-expert contributions c_te = g_te·y_te captured before cross-expert summation; shared experts excluded (spec §2–§4).
- **Conditions (spec §12):** A router-tied (64 banks × 4096); B compute-matched global (24,576); C random-routing control; D independent per-bank budgets; E per-token TopK; F capacity-matched global (262,144; parameter count exactly equal to A).
- **Training (spec §5–§11):** BatchTopK k̄=64 over B=4096 token batches (2048 visual + 2048 text); frozen θ\* inference; dead-latent aux loss α=1/32; Adam with frozen schedule; all operational definitions frozen in v0.1.2 addendum A1–A6.
- **Seeds/budget (spec §10):** discovery = 50M tokens; seeds {1,2,3} for A, B, C, F; seed 1 for D, E. Confirmatory = 200M tokens, 3 seeds, all conditions, triggered per §20.11 only.
- **Data (spec §9):** COCO-style captions, TextVQA, ChartQA, AI2D; even example-level sampling; 50/50 visual/prompt-text stream from the same examples; answer tokens excluded from training. Known scale reality disclosed in advance: unique prompt-text tokens (~10⁶-scale) are fewer than text-position draws at discovery scale, so text positions repeat across the stream; visual positions are effectively unique.
- **Splits (frozen, addendum A3):** example_id % 100 → 80/10/10 train/development/evaluation; training and statistics use train buckets only; enriched sets defined on development; interventions evaluated on untouched evaluation buckets.

## 4. Measures and analysis plan (all pre-specified)

- **Reconstruction (spec §13):** local micro/macro NMSE, aggregate NMSE, EV, L0 distributions by token type, dead %, firing-frequency distributions — all under frozen-θ\* inference.
- **Downstream fidelity (spec §13):** matched-position replacement (primary; masks frozen in A8) and full replacement (secondary, flagged as containing a designed train/eval shift); answer-token CE degradation and accuracy change, per domain plus text-only controls.
- **Expert mass and enrichment (spec §14–§15):** m_count/m_act (primary)/m_energy; q_M(e); enrichment R_M(e) = log((q+ε)/(p_gate+ε)) against router-frequency and gate-mass baselines; three pre-acknowledged pre-tilts (shared-expert absorption, gate-folding, modality-as-easiest-organization) stated in spec §15.
- **Statistics (spec §16):** JSD(q_image, q_text) with example-level permutation null and bootstrap CIs; normalized entropy/top-k mass/effective expert count; stability of top-6 image-enriched experts across four domains, two random halves, three seeds (Jaccard, rank correlation, bootstrap selection frequency) above a frozen 500k-token load floor; incremental-information probes (router gates; bank-mass; L1 feature-identity; combinations) with example-level held-out scoring.
- **Causal validation (spec §17):** feature-subtraction intervention (no reconstruction-error injection; single-shot), against control sets matched on count, coefficient mass, firing frequency, expert load, and removed-vector norm (per-latent proxy at construction + post-hoc per-token measurement, addendum-recorded); the identical pipeline run on condition C as the negative control. A "visual subsystem" claim requires preferential visual degradation exceeding the condition-C control.

**Implementation gates (not hypothesis tests):** (i) capture validated by a pure-FP32 reference forward, per-token rel. err < 10⁻⁵ over ≥100k tokens (spec §3); (ii) at discovery seed 1, A must beat C by ≥10% relative on local micro NMSE, else the study halts for an implementation audit (spec §20.5).

## 5. Data collected to date — full disclosure

**No discovery-scale data exists.** Discovery capture and training begin only after this registration is posted. Completed to date, in accordance with spec §20 steps 1–3:

1. **Hook validation (spec §3 hard gate): PASS.** FP32 reference forward, 100,933 tokens: max per-token relative error 0.0; production-path cross-check 0.0; BF16 production report max 1.99×10⁻³ / median 1.66×10⁻³ with no systematic structure (all |ρ| < 0.03).
2. **Pipeline-validation run (spec §10: 1M tokens, f=1024, k̄=32, one seed — pre-declared "plumbing only; no scientific conclusions").** Observed, disclosed verbatim:
   - Step-5 gate *rehearsal* (addendum A10): held-out local micro NMSE A = 0.242 vs C = 0.450 (ratio 0.538; rule ≤ 0.9). PASS.
   - Final-step training metrics: aggregate NMSE A 0.328 / B 0.461 / C 0.648; mean L0 (visual/text): A 36.5/27.5, B 48.2/15.8, C 43.3/20.7.
   - Engineering findings fixed and frozen before this registration: an MPS top-k kernel fault (backend fallback, A7) and example-correlated per-batch bank-load bursts (capacity handling, A7).

   These observations were **not** used to select hypotheses, metrics, thresholds, or architecture; every scientific choice they could have influenced was frozen in v0.1.1 (2026-07-06) before the run executed. We disclose them because they are directionally informative (a reader may treat the discovery-scale A-vs-B/C comparisons as having a favorable prior).

## 6. Materials, code, and deviations policy

- Repository (spec, implementation, 258 automated tests, this document) at tag `[TAG]` / commit `[COMMIT HASH]`; private during execution, shared with reviewers on request and made public with the report. Analysis code for §14–§17 is in the registered commit and is locked per spec §20.7/§20.9 (any post-registration change requires a spec version bump with written reason, per the spec's own §0 discipline; the git history documents all changes).
- Compute: capture + training on commodity accelerators (A100-class and/or TPU v4); a cross-backend parity suite (fp32-exact selection/permutation/optimizer semantics) binds every backend to the registered semantics; capture backends are additionally bound by the §3 FP32 gate.
- **Deviations:** will be reported in the final write-up with the spec-version changelog; the frozen non-goals list (spec §19) enumerates what will *not* be done in v0.1.x.

## 7. Timeline

Discovery capture (~96k examples → 50M tokens) and the 14 discovery training runs commence immediately after posting; discovery analysis follows §20 steps 5–10 in order. The confirmatory run is triggered only per §20.11, with no metric or hypothesis changes permitted.

---

*Posting notes (not part of the registration text): use OSF's generic "Open-Ended Registration" or "OSF Preregistration" template; paste §1–§7; attach the two spec files and this document; fill `[TAG]`/`[COMMIT HASH]` after tagging; consider a short embargo if reviewer anonymity matters. TRC/Google Cloud and Colab acknowledgments go in the eventual paper, not here.*
