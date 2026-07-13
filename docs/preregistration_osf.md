# Pre-registration: Router-Tied BatchTopK Sparse Autoencoders for Multimodal Mixture-of-Experts Models

**Registration type:** OSF Prereg (adapted template). **Draft v1 — not yet posted.**
**Author:** Andrew Bistras. *(Implementation and drafting assistance: Claude (Anthropic); all frozen scientific choices predate implementation.)*
**Date of registration:** [POST DATE]
**Frozen protocol:** spec v0.1.2 = `spec/frozen_spec_v0.1.1.md` + `spec/frozen_spec_v0.1.2_addendum.md`, at repository tag `[TAG]`, commit `[COMMIT HASH]`. The registered analysis code is the same commit (355 automated tests passing at tag time).

## 0. Provenance statement (addendum §0, verbatim — required disclosure)

> v0.1.1's architecture, hypotheses, primary metrics, model/layer/site, widths, sparsity, controls, and interpretation rules were frozen before the pipeline validation run. v0.1.2 was drafted after implementation and pipeline validation and contains only operational freezes, backend semantics, split mechanics, auxiliary-loss details, evaluation pins, committee dispositions, and disclosure of pilot results. No discovery-scale data had been collected. Because pipeline-scale A/B/C results were known before v0.1.2 was tagged, readers should treat discovery A-vs-B/C priors as favorable and should evaluate confirmatory claims under the pre-registered power clause. Committee amendments in this addendum (Reviewers F and S) were likewise adopted post-pipeline, pre-discovery; all are operationally outcome-neutral, and each carries a written reason.

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
- **Seeds/budget (spec §10):** discovery = 50M tokens; seeds {1,2,3} for A, B, C, F; seed 1 for D, E. Confirmatory = 200M tokens, 3 seeds, all conditions, triggered only by the pre-committed A18 two-route rule (§20 step 11's discretionary clause is void; the trigger's inputs and verdict are posted as a registration addendum before any confirmatory launch).
- **Backend map (addendum A12):** `[BACKEND MAP]` <!-- fill at tag time; prepared content: ALL 14 discovery training runs — conditions A/B/C/F × seeds 1–3, plus D and E at seed 1 — run on the TPU v4 / JAX lane (all conditions within a matched seed share one backend, satisfying A12's constraint; condition D accepts the ~2–3 h host-transfer selection path). Capture and downstream-fidelity evaluation run on the CUDA lane per A7 (bound by the §3 FP32 gate + BF16 report). The cross-backend A-vs-C replication on the torch/CUDA lane is retained as a robustness check. Seed scope = parameter initialization + data order; backend recorded as run metadata. -->.
- **Data (spec §9):** COCO-style captions, TextVQA, ChartQA, AI2D; even example-level sampling; 50/50 visual/prompt-text stream from the same examples; answer tokens excluded from training. Known scale reality disclosed in advance: unique prompt-text tokens (~10⁶-scale) are fewer than text-position draws at discovery scale, so text positions repeat across the stream; visual positions are effectively unique.
- **Splits (frozen, addendum A3):** keyed hash `int(sha256(f"{dataset_key}:{example_id}").hexdigest()[:8],16) % 100` with dataset keys `coco`/`textvqa`/`chartqa`/`ai2d` → 80/10/10 train/development/evaluation (language-native `hash()` forbidden; the raw `example_id % 100` convention is superseded); training and statistics use train buckets only; enriched sets defined on development; interventions evaluated on untouched evaluation buckets. Zero contamination is re-verified under the keyed-hash membership function before discovery capture begins (A3 consequence freeze).
- **Statistics measure (addendum A11):** all per-expert means μ_e, every NMSE normalization denominator, and any dataset-level statistic feeding training or evaluation are computed over the modality-balanced (50/50) training-stream measure restricted to the train split.

## 4. Measures and analysis plan (all pre-specified)

- **Reconstruction (spec §13):** local micro/macro NMSE, aggregate NMSE, EV, L0 distributions by token type, dead %, firing-frequency distributions — all under frozen-θ\* inference.
- **Downstream fidelity (spec §13; A8 evaluation pins):** matched-position replacement (primary; masks frozen in A8) and full replacement (secondary, flagged as containing a designed train/eval shift); answer-token CE evaluated **teacher-forced**; task accuracy evaluated under **greedy generation** with per-step replacement applying only at the frozen matched-position mask (generated tokens excluded by construction), per domain plus text-only controls.
- **Expert mass and enrichment (spec §14–§15):** m_count/m_act (primary)/m_energy; q_M(e); enrichment R_M(e) = log((q+ε)/(p_gate+ε)) against router-frequency and gate-mass baselines; three pre-acknowledged pre-tilts (shared-expert absorption, gate-folding, modality-as-easiest-organization) stated in spec §15.
- **Statistics (spec §16):** JSD(q_image, q_text) with example-level permutation null and bootstrap CIs; normalized entropy/top-k mass/effective expert count; stability of top-6 image-enriched experts across four domains, two random halves, three seeds (Jaccard, rank correlation, bootstrap selection frequency) above the frozen load floor — split-proportional per addendum A13: a bank qualifies only if its routed token count in the split under analysis is ≥ 10% of that split's uniform-load expectation (split_tokens × 6)/64, per-bank counts reported for all 64 banks regardless; incremental-information probes (router gates; bank-mass; L1 feature-identity; combinations) with example-level held-out scoring. Pre-registered A15 robustness: §16.1 and §16.4 recomputed on the text half stratified by training draw count (monotonicity across strata reported); example counts by modality are disclosed alongside token counts, and example-level bootstrap is the pre-registered unit partly for this reason.
- **Causal validation (spec §17):** feature-subtraction intervention (no reconstruction-error injection; single-shot), against control sets matched on count, coefficient mass, firing frequency, expert load, and removed-vector norm (per-latent proxy at construction + post-hoc per-token measurement, addendum-recorded); the identical pipeline run on condition C as the negative control. A "visual subsystem" claim requires preferential visual degradation exceeding the condition-C control.

**Implementation gates:** (i) capture validated by a pure-FP32 reference forward, per-token rel. err < 10⁻⁵ over ≥100k tokens (spec §3); (ii) at discovery seed 1, A must beat C by ≥10% relative on local micro NMSE (spec §20.5). Gate (ii) failure semantics are audit-first (addendum A10): an implementation audit runs (full regression suite + cross-backend A-vs-C replication with the parity suite green); a found fault is fixed and the gate re-run; if no fault is found and both independent backends agree, the study continues, the result is recorded as evidence against the stable-router-partition hypothesis, and the A18 trigger is evaluated normally.

## 5. Data collected to date — full disclosure

**No discovery-scale data exists.** Discovery capture and training begin only after this registration is posted. Completed to date, in accordance with spec §20 steps 1–3:

0. **Infrastructure campaign without data (2026-07-08 → 07-10), disclosed for
   completeness:** a Colab-based "windowed streaming" discovery harness was
   built and debugged (five engineering commits; design preserved at git tag
   `colab-streaming-campaign`) but **no discovery capture or training run ever
   completed and no discovery-scale artifacts exist** (verified: no artifacts
   locally, in the linked Drive, or in any project bucket). No analysis code
   ran on any discovery-scale data. The frozen protocol was not altered by or
   during this campaign.
1. **Hook validation (spec §3 hard gate): PASS.** FP32 reference forward, 100,933 tokens: max per-token relative error 0.0; production-path cross-check 0.0; BF16 production report max 1.99×10⁻³ / median 1.66×10⁻³ with no systematic structure (all |ρ| < 0.03).
2. **Pipeline-validation run (spec §10: 1M tokens, f=1024, k̄=32, one seed — pre-declared "plumbing only; no scientific conclusions").** Observed, disclosed verbatim:
   - Step-5 gate *rehearsal* (addendum A10): held-out local micro NMSE A = 0.242 vs C = 0.450 (ratio 0.538; rule ≤ 0.9). PASS.
   - Final-step training metrics: aggregate NMSE A 0.328 / B 0.461 / C 0.648; mean L0 (visual/text): A 36.5/27.5, B 48.2/15.8, C 43.3/20.7.
   - Engineering findings fixed and frozen before this registration: an MPS top-k kernel fault (backend fallback, A7) and example-correlated per-batch bank-load bursts (capacity handling, A7).

   These observations were **not** used to select hypotheses, metrics, thresholds, or architecture; every scientific choice they could have influenced was frozen in v0.1.1 (2026-07-06) before the run executed. We disclose them because they are directionally informative (a reader may treat the discovery-scale A-vs-B/C comparisons as having a favorable prior).

   **Mechanism-level deflators (addendum §2.2, verbatim, alongside the pipeline table):**

   > At 244 steps, condition A holds 10.7× the parameters of condition B, and larger dictionaries reduce loss faster early in training even under the null hypothesis that the additional capacity buys nothing at convergence; the pipeline A-over-B ordering is therefore consistent with what that null predicts mid-warmup.

   > The L0-allocation pattern admits at least two readings at this scale — "the global SAE starves text" and "visual tokens are higher-variance and costlier to reconstruct" are observationally identical here — and is disclosed as a shape the discovery run will measure, not as evidence.

## 6. Materials, code, and deviations policy

- Repository (spec, implementation, 258 automated tests, this document) at tag `[TAG]` / commit `[COMMIT HASH]`; private during execution, shared with reviewers on request and made public with the report. Analysis code for §14–§17 is in the registered commit and is locked per spec §20.7/§20.9 (any post-registration change requires a spec version bump with written reason, per the spec's own §0 discipline; the git history documents all changes).
- Compute: capture + training on commodity accelerators (A100-class and/or TPU v4); a cross-backend parity suite (fp32-exact selection/permutation/optimizer semantics) binds every backend to the registered semantics; capture backends are additionally bound by the §3 FP32 gate.
- **Deviations:** will be reported in the final write-up with the spec-version changelog; the frozen non-goals list (spec §19) enumerates what will *not* be done in v0.1.x.

## 7. Timeline

Discovery capture (~96k examples → 50M tokens) and the 14 discovery training runs commence immediately after posting; discovery analysis follows §20 steps 5–10 in order, with the A18 trigger evaluated immediately after step 8 (dev split only, evaluation split untouched). The confirmatory run is triggered only by the recorded A18 verdict — posted as a registration addendum before any confirmatory launch — with no metric or hypothesis changes permitted. Disclosed asymmetry (A18): under this rule, confirmation is reachable only from positive-trending discovery.

## 8. Frozen constants (registration-inline; addendum §4 — self-contained even if repositories rot)

| Constant | Value |
|---|---|
| Split function | `int(sha256(f"{dataset_key}:{example_id}").hexdigest()[:8],16) % 100`; 0–79/80–89/90–99 |
| Dataset keys | `coco`, `textvqa`, `chartqa`, `ai2d` |
| θ estimation | EMA decay 0.999 of per-batch selection threshold; frozen θ\* per condition/seed; D: per-bank θ\*ₑ; E: mean per-token k̄-th largest |
| §20.5 gate | A beats C by ≥10% relative, local micro NMSE, discovery seed 1; audit-first on failure (A10) |
| Confirmatory trigger | A18 two-route; evaluated post-step-8 on dev; verdict recorded pre-launch |
| Representation criterion | probe {2,3,4} advantage over probe 1: 95% example-level bootstrap CI > 0 in ≥2/3 seeds, and > condition-C advantage |
| Load floor | 10% of split-uniform expectation, E = (split tokens × 6)/64 |
| Deadness window | 250,000 bank-relative tokens; alive-until-full-window |
| Aux | per-token top min(512,·) dead across visible banks; raw preacts; α = 1/32; detached local residuals |
| Norm matching | proxy construction; ±20% tolerance; 95% validity rule binding on strong-positive; norm covariate; all-pass sensitivity subset |
| Masses | m^act primary; m^count robustness panels; Jaccard<0.5 divergence diagnostic |
| Probes | fit train, report dev; example-level splits and bootstrap; L1-logistic identity probe |
| JSD / enrichment / ε | base 2 / natural log / 10⁻⁶ |
| Optimizer | Adam(0.9, 0.999, ε=1e-8, bias-correct); clip min(1, 1/(‖g‖+1e-6)); LR 3e-4 → 3e-5 cosine, 1k warmup; tangent projection then renorm |
| Batch | B = 4096 (2048 visual / 2048 text); k̄ = 64; f = 4096; F = 262,144 |
| Seeds | {1, 2, 3}; scope = init + data order; backend per A12 map |
| Evaluation | CE teacher-forced; accuracy generated; replacement at matched-position mask in both regimes; BOS never replaced |

## 9. Personnel, attestations, and committee priors (addendum §2.4, §2.5, §2.8)

**Personnel statement:** one human implementer (A. Bistras); an AI implementation agent (Claude); a two-reviewer AI committee (Reviewers F and S) whose reports and reconciliation are part of the project record; the independent adversarial review (16 confirmed findings, fixed pre-training) and the 258-test suite named.

**A9 waiver-timing attestation:** the LR sanity option is exercised as **waived**; 3×10⁻⁴ is frozen everywhere. The waiver decision was made **before** observing pipeline training behavior — it was made at config-authoring time on 2026-07-06, before any pipeline training executed (see the git history of `configs/`).

**Trigger-verdict recording requirement (addendum §2.7):** the A18 inputs and verdict will be posted as a registration addendum before any confirmatory launch (`reports/confirmatory_trigger.json`, produced by `scripts/evaluate_trigger.py`).

**Outcome priors, on the record (addendum §2.8 — so a later reader can score the committee):**

- *Implementer:* architecture-only most likely; partial-positive second; strong-positive ≈ one-in-four; discovery-negative unlikely and uninterpretable per the power clause.
- *Reviewer F:* architecture-only 0.40; partial-positive 0.25; strong-positive 0.20; discovery-trends-negative 0.15; §20.5 gate passes ≈ 0.90. Holds extra partial-positive mass riding on the identity probe.
- *Reviewer S:* architecture-only modal; partial-positive second; strong-positive 20–30%; discovery-negative possible but not decisive given the power clause.

---

*Posting notes (not part of the registration text): use OSF's generic "Open-Ended Registration" or "OSF Preregistration" template; paste §0–§9; attach the two spec files and this document; fill `[TAG]`/`[COMMIT HASH]`/`[BACKEND MAP]` after tagging (the backend-map content is prepared inline as a comment at the §3 "Backend map" bullet); consider a short embargo if reviewer anonymity matters. TRC/Google Cloud and Colab acknowledgments go in the eventual paper, not here.*
