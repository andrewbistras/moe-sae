# Frozen Spec v0.1.2 — Operational Addendum

**Applies to:** *Router-Tied BatchTopK Sparse Autoencoders for Multimodal Mixture-of-Experts Models*
**Status:** Frozen upon tagging as `spec-v0.1.2`. v0.1.1 remains byte-frozen; on conflict, this addendum wins. No architectural change is made or permitted here.
**Provenance class of every item:** operational freeze, evaluation pin, disclosure requirement, or committee disposition — each with a written reason, per the spec's own §0 discipline.
**Committee:** A. Bistras (implementer, with Claude as implementation agent); Reviewer F ("Fable"); Reviewer S ("5.5"). Both reviewers' reports and the reconciliation are part of the project record.

Fill-at-tag placeholders: `[TAG]`, `[COMMIT HASH]`, `[WAIVER-TIMING]`, `[BACKEND MAP]`.

---

## 0. Provenance statement (required verbatim in the registration)

> v0.1.1's architecture, hypotheses, primary metrics, model/layer/site, widths, sparsity, controls, and interpretation rules were frozen before the pipeline validation run. v0.1.2 was drafted after implementation and pipeline validation and contains only operational freezes, backend semantics, split mechanics, auxiliary-loss details, evaluation pins, committee dispositions, and disclosure of pilot results. No discovery-scale data had been collected. Because pipeline-scale A/B/C results were known before v0.1.2 was tagged, readers should treat discovery A-vs-B/C priors as favorable and should evaluate confirmatory claims under the pre-registered power clause. Committee amendments in this addendum (Reviewers F and S) were likewise adopted post-pipeline, pre-discovery; all are operationally outcome-neutral, and each carries a written reason.

---

## 1. Operational freezes A1–A20 (final, committee-amended form)

### A1 — Auxiliary loss, completed *(amended: raw-preact pin + required diagnostics)*

Frozen as proposed in the briefing: candidates are latents dead at step start among the token's visible banks; per token, the top $\min(512,\cdot)$ dead candidates are selected from the dead-masked pre-activations; targets are **detached** per-expert residuals reconstructed through the latents' own decoders, with no $\mu$ term; $\mathcal L_{\mathrm{aux}} = \sum\|r-\hat r\|^2 / \sum\|r\|^2$, defined as 0 below $10^{-12}$ residual energy; aux selection never resets deadness clocks; main+aux co-selection is permitted; $\mathcal L_{\mathrm{aux}} = 0$ exactly when no visible latent is dead. Coefficient $\alpha_{\mathrm{aux}} = 1/32$.

**Committee pin (Reviewer F):** aux-selected dead latents contribute their **raw (pre-ReLU) pre-activation values** to the aux reconstruction. *Reason:* gradient must reach below-threshold latents to revive them; if aux values were ReLU'd, most dead latents would contribute exactly zero and the revival mechanism would be decorative. This is the reading implied by "selected iff > −∞," now stated rather than implied.

**Required diagnostics (Reviewer S), logged every eval interval, all conditions:** (i) aux/primary loss ratio; (ii) aux/main gradient-norm ratio; (iii) mean aux-selected latents per token; (iv) fraction of aux-selected latents also selected by main; (v) per-bank aux pressure (aux selections per bank per window). *Reason:* 512 aux candidates per token is large relative to $\bar k = 64$; these logs establish whether the aux term dominated early training, without leaving it to memory.

### A2 — Condition C permutation *(amended: explicit mapping sentence)*

Frozen as proposed: $\sigma_{\mathrm{ex}} = \mathrm{PCG64}\big(\texttt{0xC0FFEEC} \oplus (\mathrm{example\_id}\ \&\ \texttt{0xFFFFFFFF})\big)$ generating a uniform permutation of $\{0,\dots,63\}$; framework- and seed-independent; C's deadness clocks run on assigned banks; `bank_loads` records host routing in all conditions.

**Committee sentence (Reviewer S):** for each contribution from host expert $e$ in example $x$, condition C uses assigned bank $\sigma_x(e)$; the mapping is therefore stable within an example and independent across examples. **Committee note (Reviewer F):** because $\sigma_x$ is a per-example uniform permutation, C's expected per-bank loads are the uniformization of A's while preserving within-example routing clumps — within the spec's "approximate per-bank load," and stated here so no reader mistakes it for an accident.

### A3 — Data splits *(amended: keyed hash replaces raw modulo)*

Split assignment is frozen as:

```
split(x) = int(sha256(f"{dataset_key}:{example_id}".encode("utf-8")).hexdigest()[:8], 16) % 100
```

with `dataset_key` ∈ {`coco`, `textvqa`, `chartqa`, `ai2d`} exactly as lowercase strings; 0–79 train / 80–89 dev / 90–99 eval. Language-native `hash()` is forbidden (process-salted, non-reproducible). Training and all statistics passes consume train only; probes fit on train and report on dev; eval is untouched until the intervention stage; enriched sets are defined on dev.

*Reason (Reviewer S):* sequential `example_id % 100` can encode shard/source/collection order into the split, and correlates buckets across datasets sharing IDs. **Consequence freeze:** the A3 contamination regression test is updated to the new membership function, and zero train/dev/eval contamination is re-verified under it before discovery capture begins.

### A4 — Inference thresholds for D and E *(amended: per-bank thresholds for D)*

**E:** as proposed — $\theta_b$ is the mean per-token $\bar k$-th largest score; same EMA(0.999) and single frozen $\theta^\*$.
**D (committee amendment, Reviewer F):** D maintains **per-bank thresholds** $\theta^\*_e$ — for each bank, the EMA(0.999) of that bank's per-batch selection threshold under its own budget — frozen at end of training and used for all D inference.

*Reason:* D trains under per-bank budgets, so per-bank selection scales diverge by construction; a single global $\theta^\*$ re-imposed at evaluation systematically mis-thresholds low-scale banks and biases evaluation against exactly the control whose purpose is per-bank independence. Cost: a 64-float vector.

### A5 — Adam details

Frozen as proposed: $\epsilon = 10^{-8}$; bias correction on; clip coefficient $\min(1, 1/(\|g\|+10^{-6}))$; the decoder tangent-space projection renormalizes the stored column direction before projecting.

### A6 — Deadness boundaries

Frozen as proposed: clocks and stamps initialize to 0; latents are alive until their bank has seen a full 250k-token window; only main-selection stamps deadness (zero-valued selections stamp); B/F deadness clocks advance by $B$ per batch.

### A7 — Backend substitutions

Frozen as proposed: substitutions permitted iff selection/loss semantics are exact; every substitution event recorded; candidate dropping is never permitted under any pressure; the §3 FP32 gate + BF16 report bind every capture backend before its shards are used; the 21-test parity suite binds every training backend.

### A8 — Position semantics and downstream-evaluation pins *(extended)*

Frozen as proposed: BOS is never captured or replaced; visual positions are `images_seq_mask` positions; the matched-position mask is non-answer ∧ non-BOS; answer CE is computed on exact answer spans.

**Committee extension (Reviewer S; mechanics Reviewer F):**
- Answer-token **cross-entropy** is evaluated **teacher-forced**.
- **Task accuracy** is evaluated under **normal generation**.
- During free-running generation, per-step replacement applies only at the frozen matched-position mask (visual + prompt-text, non-BOS); generated tokens are excluded by construction, so the primary metric's replacement scope is identical in both regimes.

*Reason:* CE and generation settings otherwise drift apart in subtle, unfrozen ways.

### A9 — LR sanity option *(extended: timing attestation)*

The LR sanity option is exercised as **waived**; $3\times10^{-4}$ is frozen everywhere. **Required attestation at tag time:** the waiver decision was made `[WAIVER-TIMING: before | after]` observing pipeline training behavior; the true value is stated in the registration. *Reason:* the waiver's provenance class differs across the two cases and the record must say which occurred.

### A10 — A-vs-C gate: rehearsal record and audit-first semantics *(amended)*

The pipeline rehearsal (A/C = 0.538 against the ≤0.9 rule) is recorded; the binding §20.5 gate remains discovery seed 1.

**Replacement gate semantics (Reviewer S; protocol Reviewer F):** if A fails to beat C by ≥10% relative on local micro-averaged NMSE at discovery seed 1:

1. An **implementation audit** runs: the full regression suite, plus a cross-backend replication of the A-vs-C comparison on the second compute lane with the parity suite green.
2. If a fault is found: fix, document, re-run the gate.
3. If no fault is found and both independent backends agree: **the study continues**, the result is recorded as evidence against the stable-router-partition hypothesis, and the confirmatory trigger (A18) is evaluated normally at its appointed step.

*Reason:* "failure means bug, by definition" makes the partition hypothesis unfalsifiable through the gate. Audit-first converts the gate from a tautology into a measurement.

### A11 — Statistics measure *(new, Reviewer F)*

All per-expert means $\mu_e$, every NMSE normalization denominator, and any dataset-level statistic feeding training or evaluation are computed over the **modality-balanced (50/50) training-stream measure restricted to the train split** — not over the raw capture distribution. The statistics pass re-runs under this measure for discovery.

*Reason:* the pipeline statistics pass captured ~1.74M visual against ~77k text tokens (~22:1); the training stream is 50/50. These are different measures, and $\mu_e$ and every denominator change under them. This degree of freedom was open and silently favored visual statistics.

### A12 — Backend map *(new, Reviewer F)*

Before discovery launch, the (condition × seed) → backend assignment is frozen and recorded in the registration:

`[BACKEND MAP: table, conditions A/B/C/F × seeds 1–3, plus D/E seed 1]`

Constraints: all conditions within a matched seed run on one backend; backend is recorded as run metadata; seed values are {1, 2, 3} with scope = parameter initialization + data order; capture backends are governed by A7; the planned cross-backend A-vs-C replication remains as a robustness check.

*Reason:* with two compute lanes, an unfrozen backend choice is a hidden covariate inside "seed."

### A13 — Load floor made split-proportional *(new, Reviewer F — bug fix)*

The v0.1.1 absolute floor (500k tokens) is replaced by: a bank enters enrichment rankings and stability statistics only if its routed token count in **the split under analysis** is at least **10% of the uniform-load expectation for that split**, $E_{\mathrm{uniform}} = (\text{tokens in split} \times 6)/64$. Per-bank counts remain reported for all 64 banks, all conditions, all splits.

*Reason:* the 500k figure was calibrated against training loads (~4.7M/bank expected); A3 sends enriched-set definition and stability analysis to dev, where uniform expectation is ≈490k/bank. As written, the frozen floor would have disqualified roughly half the banks from §16.3 by arithmetic accident.

### A14 — Provenance and amendment-timing disclosure *(new)*

The §0 provenance statement of this addendum appears verbatim in the registration's disclosure section. Every A-item's amendment provenance (proposed-in-briefing vs committee-amended, and by whom) is preserved as written here.

### A15 — Text draw-epoch robustness check *(new; Q2 adopted)*

Pre-registered robustness analysis: recompute §16.1 (JSD) and §16.4 (probes) on the text half stratified by draw-epoch, and report whether image/text separation or incremental probe performance changes monotonically with text reuse count. The registration additionally discloses example counts by modality alongside token counts, and notes that example-level bootstrap is the pre-registered unit partly for this reason.

*Reason:* at discovery scale the text half re-draws from a smaller unique pool (~4–8 effective epochs) while the visual half is effectively unique; this check pre-empts the most predictable objection to any modality asymmetry.

### A16 — Mass primacy and divergence diagnostic *(Q3 disposition)*

$m^{\mathrm{act}}$ remains the primary mass, as frozen. Every main-text concentration figure carries an $m^{\mathrm{count}}$ robustness panel ($m^{\mathrm{energy}}$ where useful). **Pre-registered divergence diagnostic:** if the top-6 enriched-expert sets under $m^{\mathrm{act}}$ and $m^{\mathrm{count}}$ disagree at Jaccard < 0.5 for any modality/seed, the discrepancy is reported prominently and an enrichment-versus-mean-gate regression is run as the diagnostic for gate-magnitude contamination.

*Reason:* switching primacy after seeing pipeline numbers would be post-hoc by construction; the confound is instead disclosed (§15.2), defended (identity probe; matched interventions), and converted into a measurement.

### A17 — Removed-norm matching: validity rule *(Q4 disposition)*

Control sets are constructed by the per-latent coefficient-mass proxy (exact under unit decoder columns). Exact removed-vector norm is measured post-hoc per token. **Validity rule (binding):** if fewer than 95% of intervention instances fall within the ±20% removed-norm tolerance across all matched control sets, the causal comparison is reported as **norm-unmatched and cannot support a strong-positive conclusion**. Exact removed norm is included as a covariate in the intervention analysis. **Sensitivity recomputation:** all intervention conclusions are additionally computed on the token subset where every matched control passes the tolerance.

*Reason:* exact pre-hoc token-level norm matching is impossible under co-firing; the tolerance must therefore have consequences rather than decorate.

### A18 — Confirmatory trigger, pre-committed *(Q6 disposition; two-route)*

§20.11's "decide whether the confirmatory run proceeds" is replaced by the following frozen rule, **evaluated immediately after §20 step 8, on the development split only, with the evaluation split still untouched**; the trigger's inputs and verdict are recorded in a registration addendum before any confirmatory launch.

The confirmatory run proceeds iff the §20.5 gate (per A10 semantics) has passed and at least one route holds:

- **Architecture route:** A beats C on local micro NMSE by ≥10% relative in ≥2 of 3 seeds, **and** A ≥ B on aggregate NMSE or on matched-position downstream fidelity in ≥2 of 3 seeds.
- **Representation route:** A beats C on local micro NMSE by ≥10% relative in ≥2 of 3 seeds, **and** the frozen representation criterion holds: the held-out (dev, example-level) accuracy advantage of probe 2, 3, or 4 over probe 1 (router-only) has a 95% example-level bootstrap confidence interval excluding zero in ≥2 of 3 seeds, **and** that advantage exceeds the corresponding condition-C advantage (point estimate) in those seeds.

If neither route triggers: discovery is reported as final; the negative interpretation remains **unclaimed** under the power clause; confirmation is deferred to future work. **Disclosed asymmetry:** under this rule, confirmation is reachable only from positive-trending discovery; the registration states this plainly.

*Reason:* the loose clause was a researcher degree of freedom; a single reconstruction-only trigger would amputate the representational branch that probe 3 exists to protect; a mechanical rule evaluated on dev is not a degree of freedom even after unblinding.

### A19 — Condition D confirmed; load-dispersion battery *(Q1 disposition)*

D remains per-batch-load-proportional by largest remainder; no smoothing. *Reason:* heavy-tailed, example-correlated routing bursts are the actual regime the method sees; D is supposed to inherit them, and EMA-smoothing would test an allocator nobody proposed while making D's selection history-dependent.

**Required diagnostics, all conditions:** per-batch max bank load; per-batch load entropy or Gini; D's per-bank allocated-budget distribution; fraction of banks with zero routed tokens per batch; the relation between burstiness and reconstruction error. The observed 909-vs-~500±22 burst is retained for the paper as a systems observation about routing coherence.

### A20 — Analysis lacunae blessed *(Q5 disposition)*

(a) Probe 2 (64-dim bank-mass) is undefined for B/F and reported as N/A; probes 1, 3, 4 run for all conditions. (b) All condition-C bank attribution — masses, energies, identity probe, interventions — uses the **assigned** bank via the dump's `act_bank` field; the assigned bank *is* C's dictionary structure. (c) JSD in base 2; enrichment in natural log; $\epsilon = 10^{-6}$ (≈10⁻⁴ of uniform mass). All frozen.

---

## 2. Registration requirements (content that must appear in the OSF document)

1. **Provenance statement** (§0 above), verbatim.
2. **Pilot disclosure with mechanism-level deflators**, verbatim alongside the pipeline table:
   > At 244 steps, condition A holds 10.7× the parameters of condition B, and larger dictionaries reduce loss faster early in training even under the null hypothesis that the additional capacity buys nothing at convergence; the pipeline A-over-B ordering is therefore consistent with what that null predicts mid-warmup.
   > The L0-allocation pattern admits at least two readings at this scale — "the global SAE starves text" and "visual tokens are higher-variance and costlier to reconstruct" are observationally identical here — and is disclosed as a shape the discovery run will measure, not as evidence.
3. **Frozen constants inline** (§4 table below), so the registration is self-contained even if repositories rot.
4. **Personnel statement:** one human implementer; an AI implementation agent (Claude); a two-reviewer AI committee (Reviewers F and S) whose reports and reconciliation are part of the record; the independent adversarial review (16 confirmed findings, fixed pre-training) and the 258-test suite named.
5. **A9 waiver-timing attestation** with the true value of `[WAIVER-TIMING]`.
6. **A12 backend map**, filled.
7. **Trigger-verdict recording requirement:** the A18 inputs and verdict will be posted as a registration addendum before any confirmatory launch.
8. **Outcome priors, on the record** (so a later reader can score the committee):
   - *Implementer:* architecture-only most likely; partial-positive second; strong-positive ≈ one-in-four; discovery-negative unlikely and uninterpretable per the power clause.
   - *Reviewer F:* architecture-only 0.40; partial-positive 0.25; strong-positive 0.20; discovery-trends-negative 0.15; §20.5 gate passes ≈ 0.90. Holds extra partial-positive mass riding on the identity probe.
   - *Reviewer S:* architecture-only modal; partial-positive second; strong-positive 20–30%; discovery-negative possible but not decisive given the power clause.

---

## 3. Amendments to §20 execution order (consolidated)

- Step 5 gate: A10 audit-first semantics replace "indicates an implementation fault" as the failure interpretation.
- After step 8: **A18 trigger evaluation and verdict recording** inserted, before any step-11 confirmatory decision; step 11's discretionary clause is void.
- Step 2 statistics pass: runs under the A11 balanced train-split measure.
- Pre-capture: A3 hash-split contamination re-verification; §3 FP32 gate + BF16 report on the discovery capture backend (per A7).

---

## 4. Frozen constants (registration-inline table)

| Constant | Value |
|---|---|
| Split function | `int(sha256(f"{dataset_key}:{example_id}").hexdigest()[:8],16) % 100`; 0–79/80–89/90–99 |
| Dataset keys | `coco`, `textvqa`, `chartqa`, `ai2d` |
| θ estimation | EMA decay 0.999 of per-batch selection threshold; frozen θ\* per condition/seed; D: per-bank θ\*ₑ; E: mean per-token k̄-th largest |
| §20.5 gate | A beats C by ≥10% relative, local micro NMSE, discovery seed 1; audit-first on failure (A10) |
| Confirmatory trigger | A18 two-route; evaluated post-step-8 on dev; verdict recorded pre-launch |
| Representation criterion | probe {2,3,4} advantage over probe 1: 95% example-level bootstrap CI > 0 in ≥2/3 seeds, and > condition-C advantage |
| Load floor | 10% of split-uniform expectation, $E = (\text{split tokens} \times 6)/64$ |
| Deadness window | 250,000 bank-relative tokens; alive-until-full-window |
| Aux | per-token top min(512,·) dead across visible banks; raw preacts; α = 1/32; detached local residuals |
| Norm matching | proxy construction; ±20% tolerance; 95% validity rule binding on strong-positive; norm covariate; all-pass sensitivity subset |
| Masses | $m^{\mathrm{act}}$ primary; $m^{\mathrm{count}}$ robustness panels; Jaccard<0.5 divergence diagnostic |
| Probes | fit train, report dev; example-level splits and bootstrap; L1-logistic identity probe |
| JSD / enrichment / ε | base 2 / natural log / $10^{-6}$ |
| Optimizer | Adam(0.9, 0.999, ε=1e-8, bias-correct); clip min(1, 1/(‖g‖+1e-6)); LR 3e-4 → 3e-5 cosine, 1k warmup; tangent projection then renorm |
| Batch | B = 4096 (2048 visual / 2048 text); k̄ = 64; f = 4096; F = 262,144 |
| Seeds | {1, 2, 3}; scope = init + data order; backend per A12 map |
| Evaluation | CE teacher-forced; accuracy generated; replacement at matched-position mask in both regimes; BOS never replaced |

---

## 5. Q-disposition index

Q1 → A19 (confirmed + battery). Q2 → A15 (added). Q3 → A16 ($m^{\mathrm{act}}$ primary + diagnostic). Q4 → A17 (confirmed + validity rule). Q5 → A20 (blessed). Q6 → A18 (two-route, pre-committed). Q7 → §2.8 (priors filed, divergence noted).

---

## 6. Sign-off

- **A. Bistras (implementer):** ________________
- **Reviewer F:** reviewed the briefing; proposed A11–A14, the A1 pin, the A4-D amendment, and the A18 mechanics; answered Q1–Q7; priors filed; **approved for discovery** conditional on this addendum as written.
- **Reviewer S:** accepted v0.1.2 in spirit; contributed the hash-split, audit-first gate, two-route trigger, validity rule, evaluation pins, aux diagnostics, and provenance language; priors filed; closing instruction — *stop editing and run discovery* — adopted by the committee as unanimous.

**Mechanics after sign-off:** apply → `git tag spec-v0.1.2` → fill `[TAG]`, `[COMMIT HASH]`, `[WAIVER-TIMING]`, `[BACKEND MAP]` → post the OSF registration → re-run the §3 gate + BF16 report on the discovery capture backend → launch discovery: capture → 14 training runs → §20 steps 5–10 in order → A18 trigger evaluation and recorded verdict.

The next document produced by this project is the discovery report.
