# Frozen Research Spec v0.1.1

**Working title:** *Router-Tied BatchTopK Sparse Autoencoders for Multimodal Mixture-of-Experts Models*

**Status:** Frozen for the first complete experiment. Supersedes v0.1.
Any architectural change requires a version bump and a written reason. We do not alter the setup after seeing results.

**Scope of this bump:** v0.1.1 changes no architecture. The main model (A), the activation site, all frozen widths, sparsity, batch composition, data, and optimizer core are identical to v0.1. This bump fixes four internal inconsistencies that would have made the freeze unenforceable, adds two pre-registered conditions/analyses that sharpen the triangle of comparisons, and records reading rules for the interpretation section. Every change is listed in §0 with its reason.

---

## 0. Changelog v0.1 → v0.1.1

| # | Section | Class | Change | Written reason |
|---|---------|-------|--------|----------------|
| 1 | §13 | blocker | Downstream fidelity split into **matched-position replacement** (primary) and **full replacement** (secondary). | v0.1 trained the SAE with answer-generation tokens excluded (§9) but evaluated with replacement at every position (§13), conflating reconstruction quality with a designed-in train/eval distribution shift. |
| 2 | §17 | blocker | Intervention convention changed from encode → zero → decode → insert to **feature subtraction from the true activation**. | The v0.1 convention injects the SAE's full reconstruction error at every intervened position, confounding "remove this feature" with "add this SAE's residual error." Subtraction removes only the thing claimed to be removed. |
| 3 | §8 | blocker | Aux loss re-scaled and fully specified: per-token budget of 512 dead candidates across the token's six visible banks (not 512 per bank); deadness defined in **bank-relative tokens** with a frozen window. | Top-512 per 4,096-wide bank is 12.5% of the bank — the cited convention used 512 against dictionaries of hundreds of thousands. Global-token deadness windows systematically misclassify latents in low-load banks, hammering exactly the banks whose statistics the study depends on. |
| 4 | §3 | blocker | Hook validation gate moved to a **pure FP32 reference forward** (rel. err < 1e-5). BF16 production-path discrepancy is reported, not gated. | Fused/grouped expert kernels sum in implementation-defined orders; a 1e-3 BF16 gate would fail on numerical ghosts that are not bugs. |
| 5 | §11 | freeze | Decoder-norm handling frozen: gradient component parallel to each decoder column is projected out before the Adam step; columns renormalized after the step. | Adam's moments fight bare post-step renormalization; the two variants converge differently. One must be frozen; this is the established convention [4]. |
| 6 | §6 | freeze | BatchTopK inference-threshold estimation frozen: EMA (decay 0.999) of the per-batch selection threshold; final value θ\* used for all inference and evaluation. | The training threshold is fit at 50/50 modality composition; evaluation streams are not 50/50. A frozen θ\* makes per-token inference composition-independent by construction and removes an unfrozen degree of freedom. |
| 7 | §12 | addition | **Condition F**: capacity-matched global SAE, F = 262,144, k̄ = 64. | A vs B isolates capacity at matched compute; A vs F isolates the routing restriction at matched parameters and sparsity (parameter counts are exactly equal). Without F, the "conditional-capacity benefit" has no measured upper bound. |
| 8 | §16.4 | addition | **Feature-identity probe** added to the incremental-information analysis. | The modality signal may live in *which* features fire rather than in the 64-dim bank-mass marginal. Without this probe, a well-organized dictionary one level below the instrument's resolution reads as a false negative. Remains an aggregate statistic; §19 intact. |
| 9 | §16.3 | addition | Frozen per-bank load floor for stability and enrichment rankings; per-bank counts reported. | Load balancing is approximate; starved banks inject noise into Jaccard/rank statistics. |
| 10 | §17 | addition | Random-routing SAE (condition C) added as the **negative control** for the causal intervention, run through the identical pipeline. | Establishes the baseline effect of "any structured mass deletion" before attributing visual selectivity to router-tied structure. |
| 11 | §18 | addition | Reading rules recorded: (a) objective-asymmetry rule for A vs B/F on aggregate metrics; (b) power clause — the negative interpretation may only be claimed from the confirmatory run; (c) quantitative headline template from the A–B–F triangle. | Prevents post-hoc reasoning about asymmetric objectives and underpowered nulls. Discovery-scale data is ~16× leaner per feature than the frontier compute-optimal allocation. |
| 12 | §15 | addition | "Known pre-tilts acknowledged in advance" recorded: shared-expert absorption, gate-folding magnitude conflation, modality as the easiest organization. | These biases exist by construction; stating them before results is cheaper than discovering them in review. |
| 13 | §2 | copyedit | Block indexing pinned to executable startup assertions. | The dense first layer makes off-by-one errors near-mandatory without an assertion. |
| 14 | §20 | addition | Execution order gains an A-vs-C implementation sanity gate and explicit θ\*/load-floor freeze steps. | A does not merely need to beat C eventually; failure to beat C decisively at seed 1 indicates an implementation fault, not a scientific result. |
| 15 | Refs | copyedit | Reference [2] tracking parameter stripped; duplicate reference replaced with the actual sources of the aux-loss and decoder-norm conventions [3][4]. | Hygiene; conventions should cite their origin. |

---

## 1. Core research question

> When a multimodal MoE model processes visual and textual tokens, does a global BatchTopK competition over router-exposed, expert-specific dictionaries reveal stable concentrations of explanatory feature mass in a small subset of experts?

The associated representation-learning question:

> Does the pretrained model router partition activation space usefully enough that a router-tied SAE bank outperforms an equivalently expensive global SAE, an equivalently sized global SAE, or a randomly routed SAE bank?

The systems question is deferred:

> At what local dictionary width should the same architecture move from grouped dense execution to SparseCore?

SparseCore is **not** part of v0.1.1.

---

## 2. Frozen model and layer

### Model

**DeepSeek-VL2-Tiny.**

Reasons unchanged from v0.1: a genuinely multimodal MoE (not a text MoE with a bolted-on dense vision stack); small enough for rapid, repeated, controlled runs; fully explicit architecture — hidden width d = 1280, 12 language-model layers, 64 routed experts, 2 shared experts, top-6 routed selection per token; first language layer dense; ≈3.37B total / ≈1.0B active parameters [1].

### Layer

**Language transformer block 6, one-indexed.** No layer sweep occurs before the primary experiment is complete. A positive result is later replicated at block 10; that replication is not part of the discovery run.

Reasons unchanged: safely inside the MoE stack; deep enough for modality information to have entered the language backbone; early enough that the representation is not dominated by answer-generation behavior.

### Index pinning (new)

"Block 6, one-indexed" means zero-indexed entry 5 of the checkpoint's language-model layer list — the **5th MoE block**, since entry 0 is the dense block. Implementation must assert at startup, and abort otherwise:

1. `len(language_layers) == 12`;
2. entry 0's MLP is dense (no router);
3. entry 5's MLP is an MoE module with 64 routed experts, 2 shared experts, and top-6 routing.

---

## 3. Exact activation site

For token $t$, the model router selects $S_t \subseteq \{1,\dots,64\}$, $|S_t| = 6$, with router weights $g_{t,e}$. Let $y_{t,e} \in \mathbb{R}^{1280}$ be the post-down-projection output of routed expert $e$.

The SAE input is the **gated routed-expert contribution**:

$$c_{t,e} = g_{t,e}\, y_{t,e}.$$

Captured: after the routed expert's final projection; after multiplication by its router weight; before summation across routed experts; before addition of shared-expert output; before residual addition. The routed MoE contribution is $c_t^{\mathrm{route}} = \sum_{e \in S_t} c_{t,e}$.

Shared experts are excluded from training. During downstream replacement tests, their original outputs remain untouched.

### Capture manifest (clarified)

Each captured record stores: $c_{t,e}$ for all $e \in S_t$; the index set $S_t$; gates $g_{t,e}$; token-type tag (visual / prompt-text); example ID; dataset domain. (§15–§17 require all of these; the example ID is required for example-level bootstrap and example-level held-out splits.)

### Hook validation gate (revised)

Before any SAE training:

1. **Gate (FP32 reference):** in a pure FP32 forward pass of the language stack, $\left\| c_t^{\mathrm{route}} - \sum_{e\in S_t} c_{t,e} \right\|$ relative to the model's internal routed aggregate must be below $10^{-5}$ for every token in a 100k-token validation set.
2. **Report (BF16 production path):** the same discrepancy under the normal BF16 execution path is measured and reported (max and median). It is **not a gate**. Fused/grouped expert kernels reduce in implementation-defined orders; discrepancies at the $10^{-3}$–$10^{-2}$ scale are expected numerical behavior, not capture bugs. Any *systematic* structure in the discrepancy (e.g., correlation with $|S_t|$ ordering) is investigated before proceeding.

---

## 4. Why the gated contribution is the correct object

Unchanged from v0.1. We decompose what each expert actually contributes, not the common pre-expert residual:

1. Every local target lives in the common residual-stream coordinate system.
2. The router weight is already included, so candidate magnitude is not detached from actual model contribution.
3. Expert attribution remains meaningful: a feature in bank $e$ reconstructs output contributed by expert $e$.
4. Summing the reconstructions directly approximates the model's full routed contribution.

This avoids separately encoding ungated outputs and calibrating their activation scales afterward. Two known biases this choice creates are stated in advance in §15.

---

## 5. SAE architecture

One local SAE bank per routed host expert. For expert $e$: $W^{\mathrm{enc}}_e \in \mathbb{R}^{f \times d}$, $D_e \in \mathbb{R}^{d \times f}$, untied.

### Frozen dictionary width

$$f = 4096, \qquad F_{\mathrm{total}} = 64 \times 4096 = 262{,}144, \qquad F_{\mathrm{visible/token}} = 6 \times 4096 = 24{,}576.$$

The central conditional-computation advantage: 262K total features with 24.6K encoder candidates per token.

### Centering

Per-expert fixed mean $\mu_e = \mathbb{E}[\,c_{t,e} \mid e \in S_t\,]$, computed in an initial statistics pass and frozen; not learned jointly. Encoder input: $\tilde c_{t,e} = c_{t,e} - \mu_e$.

### Encoder / decoder

$$a_{t,e} = \operatorname{ReLU}\!\left(W^{\mathrm{enc}}_e \tilde c_{t,e} + b^{\mathrm{enc}}_e\right), \qquad \hat c_{t,e} = \mu_e + D_e z_{t,e},$$

with every decoder column constrained to unit norm, $\|D_{e,j}\|_2 = 1$. The optimization mechanics of this constraint are frozen in §11.

---

## 6. The frozen BatchTopK operation

For a batch of $B$ original token positions, the ragged candidate set is
$\mathcal A_{\mathcal B} = \{\, a_{t,e,j} : t \in \mathcal B,\ e \in S_t,\ 1 \le j \le 4096 \,\}$.
Exactly $K_{\mathrm{batch}} = B\bar k$ activations are selected across the entire set.

### Frozen average sparsity

$$\bar k = 64 \ \text{active features per original token, averaged over the batch:}\quad \sum_{t,e,j} \mathbf 1[z_{t,e,j} > 0] = 64B.$$

There is **no constraint** on per-token $k_t$ or per-token-per-expert $k_{t,e}$. A visual token may use 150 features while a trivial text token uses three; within the visual token, most may come from two banks. That allocation is the phenomenon under investigation. This follows BatchTopK's batch-level budget with per-sample variability [2].

### Frozen inference threshold (new)

During training, after each batch, record the batch selection threshold $\theta_b$ — the value of the $K_{\mathrm{batch}}$-th largest candidate score. Maintain $\theta \leftarrow 0.999\,\theta + 0.001\,\theta_b$, initialized to the first batch's value. The final value $\theta^\*$ is frozen at the end of training.

All evaluation, downstream replacement, mass analysis, and intervention encoding uses **thresholded inference**: $z_{t,e,j} = a_{t,e,j}\cdot\mathbf 1[a_{t,e,j} > \theta^\*]$. This makes per-token inference independent of evaluation batch composition by construction, consistent with BatchTopK's fixed-threshold inference [2]. Per-batch $\theta_b$ trajectories are logged for all conditions.

---

## 7. No extra gate weighting during SAE selection

Unchanged. The BatchTopK score is $s_{t,e,j} = a_{t,e,j}$. We do **not** multiply by $g_{t,e}$ again: the encoder input is already $g_{t,e} y_{t,e}$; gating twice would improperly suppress low-gate contributions.

Cross-expert comparability is controlled through: gated input vectors; joint training; one shared BatchTopK competition; unit-norm decoder columns; identical architecture and optimizer settings across banks. No learned expert-level calibration scalar in v0.1.x. (A known side effect of gate-folding is recorded in §15.)

---

## 8. Training objective

### Primary reconstruction loss (unchanged)

$$\mathcal L_{\mathrm{local}} = \frac{\sum_{t}\sum_{e\in S_t} \|c_{t,e} - \hat c_{t,e}\|_2^2}{\sum_{t}\sum_{e\in S_t} \|c_{t,e} - \mu_e\|_2^2}.$$

This prevents one bank from using its decoder to cancel errors made by another bank.

### Aggregate diagnostic loss (unchanged)

$$\mathcal L_{\mathrm{aggregate}} = \frac{\sum_t \|c_t^{\mathrm{route}} - \hat c_t^{\mathrm{route}}\|_2^2}{\sum_t \|c_t^{\mathrm{route}} - \bar c^{\mathrm{route}}\|_2^2}, \qquad \hat c_t^{\mathrm{route}} = \sum_{e\in S_t}\hat c_{t,e}.$$

Reported as an evaluation metric; never optimized directly.

### Dead-latent auxiliary loss (revised — changelog #3)

**Deadness (frozen definition):** latent $j$ of bank $e$ is *dead* if it has not been selected by the main BatchTopK within the most recent **250,000 tokens routed to bank $e$** (bank-relative window). At B = 4096 and top-6/64 routing, a bank sees ≈384 tokens per batch, so the window spans ≈650 batches ≈ 2.7M global tokens; at uniform usage a latent is expected to fire ≈650 times per window, making the classification robust for low-load banks. Deadness is undefined (treated as alive) until a bank has seen a full window.

**Aux selection (frozen):** per token $t$, take the top $\min(512, \cdot)$ *dead* candidates by pre-activation $a_{t,e,j}$ across the union of the token's six visible banks — a token-level competition mirroring the main selection, replacing v0.1's per-bank 512 (which was 12.5% of each bank; the 512 convention originates against dictionaries three orders larger [3]).

**Aux target (frozen):** the selected dead latents reconstruct the **per-expert local residuals** $r_{t,e} = c_{t,e} - \hat c_{t,e}$ through their own banks' decoders, summed per expert as in the primary loss — keeping the no-cross-bank-cancellation property. Coefficient:

$$\alpha_{\mathrm{aux}} = \tfrac{1}{32} \quad [3].$$

---

## 9. Training data

Unchanged from v0.1.

| Domain | Dataset role |
|---|---|
| Natural photographs | COCO-style caption and question prompts |
| OCR and scene text | TextVQA |
| Charts | ChartQA |
| Diagrams | AI2D |

Datasets sampled evenly at the example level. The SAE training stream contains **50% projected visual token positions** and **50% textual prompt token positions from the same multimodal examples**; answer-generation tokens are excluded from the discovery training run. Both token types coming from the same examples keeps the image-versus-text comparison less confounded by subject matter.

**Batch composition:** every BatchTopK batch contains exactly $B = 4096$ positions — 2048 visual, 2048 textual — because BatchTopK's threshold is batch-dependent during training and accidental modality ratios must not determine which candidates win. Examples and positions shuffled independently within each half.

---

## 10. Training budget

| Run | f | k̄ | Tokens | Seeds | Purpose |
|---|---|---|---|---|---|
| Pipeline validation | 1024 | 32 | 1M | 1 | plumbing only; no scientific conclusions |
| Frozen discovery | 4096 | 64 | 50M | 3 (A, B, C, F) / 1 (D, E) | primary experiment |
| Confirmatory | 4096 | 64 | 200M | 3 (all conditions) | triggered only after discovery analysis completes; identical architecture; no metric or hypothesis changes |

Same training-data order across matched conditions where possible.

**Power note (new — changelog #11):** the discovery run supplies $64 \times 50\mathrm{M} = 3.2{\times}10^9$ activation events over 262,144 features — **12k fires per feature**, roughly 16× leaner than the ≈2×10⁵ fires-per-live-feature allocation observed on frontier compute-optimal SAE sweeps; the confirmatory run is still ≈4× lean. Consequence recorded in §18: dead rates and per-bank variance at discovery scale will be substantial by design, and a discovery-scale null is underpowered rather than informative.

---

## 11. Optimization

- SAE weights stored in BF16; matmuls BF16 with FP32 accumulation; Adam states FP32.
- Adam $\beta_1 = 0.9$, $\beta_2 = 0.999$; no weight decay; global grad-norm clip 1.0.
- LR $3{\times}10^{-4}$, 1,000-step linear warmup, cosine decay to $3{\times}10^{-5}$. One pre-run LR sanity check may compare $10^{-4}$, $3{\times}10^{-4}$, $10^{-3}$ on the 1M validation run; once chosen, frozen for every comparison.
- **Decoder-column constraint (frozen — changelog #5):** before each Adam update, the gradient component parallel to each decoder column is projected out (tangent-space projection); after the update, columns are renormalized to unit norm. Both steps, in that order, every step, in all conditions [4].

---

## 12. Required comparison conditions

All conditions share: data, batch composition, token budget, optimizer settings, θ-estimation procedure, and (where meaningful) matched seeds.

### A. Main model: router-tied global BatchTopK
The full architecture above. All visible candidates across tokens and experts compete in one BatchTopK.

### B. Active-compute-matched global BatchTopK SAE
One shared SAE reconstructs $c_t^{\mathrm{route}}$. $F_{\mathrm{global}} = 6f = 24{,}576$, $\bar k = 64$. Approximately the same candidate count and encoder work per token as A, with far less total capacity (24,576 vs 262,144). Measures the conditional-capacity benefit at matched compute.

### C. Random-routing control
Same architecture, parameters, batches, and BatchTopK budget as A, but each contribution vector is assigned to a randomly selected bank rather than its host expert's stable bank. The assignment must: use six distinct banks per token; preserve approximate per-bank load; change independently across examples; remain fixed across matched seeds. Tests whether the host router provides a useful *stable* partition rather than merely conditional parameters.

### D. Independent-expert BatchTopK control
Each bank receives a separate budget proportional to its routed assignments; no cross-bank competition; total selected features still $64B$. Isolates the value of global competition. One seed at discovery.

### E. Per-token TopK ablation
Router-tied banks with exactly 64 features per token selected across its six visible dictionaries. Isolates the value of variable per-token complexity. One seed at discovery; three at confirmation.

### F. Capacity-matched global BatchTopK SAE (new — changelog #7)
One shared SAE on $c_t^{\mathrm{route}}$ with $F_{\mathrm{global}} = 262{,}144$, $\bar k = 64$. Weight parameter count is **exactly** equal to A's ($2 \times 1280 \times 262{,}144 = 671{,}088{,}640$ on both sides; bias counts differ negligibly), at $262{,}144 / 24{,}576 = 10.7\times$ A's encoder cost per token. A vs F isolates the *routing restriction* at matched parameters and sparsity; A vs B isolates *capacity* at matched compute. Three seeds; at d = 1280 the added cost is immaterial.

---

## 13. Evaluation metrics

### Reconstruction
Local micro-averaged NMSE; local macro-averaged NMSE across experts; aggregate routed-output NMSE; explained variance; average $L_0$; $L_0$ distribution by token type; dead-feature percentage (per the §8 definition); feature firing-frequency distribution. All under frozen-θ\* inference.

### Downstream fidelity (revised — changelog #1)
Replace only the routed expert aggregate, $c_t^{\mathrm{route}} \rightarrow \hat c_t^{\mathrm{route}}$; attention, shared experts, and all other layers unchanged.

- **Primary — matched-position replacement:** replacement applied only at positions whose contributions match the SAE's training distribution: projected visual tokens and prompt-text tokens. Answer-span positions receive their true activations. Measured: answer-token cross-entropy degradation and task-accuracy change — i.e., the causal effect of SAE-reconstructing the *context* the answer attends to.
- **Secondary — full replacement:** replacement at every position including answer spans. Reported and plotted, explicitly flagged as containing a train/eval distribution shift (the SAE never saw answer-position contributions).

Results reported separately for natural images, OCR, charts, and diagrams, plus text-only control prompts.

---

## 14. Definition of expert-level SAE mass

Unchanged. For token $t$: count mass $m^{\mathrm{count}}_{t,e} = \sum_j \mathbf 1[z_{t,e,j} > 0]$; coefficient mass $m^{\mathrm{act}}_{t,e} = \sum_j z_{t,e,j}$ (primary); reconstructed-vector energy $m^{\mathrm{energy}}_{t,e} = \|D_e z_{t,e}\|_2^2$ (robustness). For modality $M$:

$$q_M(e) = \frac{\sum_{t\in M} m^{\mathrm{act}}_{t,e}}{\sum_{t\in M}\sum_{e'} m^{\mathrm{act}}_{t,e'}}.$$

---

## 15. Router baselines for concentration

The model router itself may already concentrate image tokens. Every SAE concentration result is therefore compared against routing frequency $p_M^{\mathrm{route}}(e)$ and gate mass $p_M^{\mathrm{gate}}(e)$ as defined in v0.1, with SAE enrichment

$$R_M(e) = \log\frac{q_M(e) + \epsilon}{p_M^{\mathrm{gate}}(e) + \epsilon}.$$

The "visual cortex" result cannot be established merely by showing that image tokens are routed to certain experts; SAE feature mass must reveal concentration or selectivity **beyond** router assignment and gate magnitude.

### Known pre-tilts acknowledged in advance (new — changelog #12)

1. **Shared-expert absorption.** DeepSeekMoE-family shared experts exist to absorb common features so routed experts specialize. Our training target — the routed contribution with shared experts excluded — is therefore the *specialized residue by architectural construction*, which pre-tilts the measurement toward concentration. The gate-mass baselines above are the defense; this mechanism is stated in any write-up.
2. **Gate-folding magnitude conflation.** Because $g_{t,e}$ is folded into the input, one semantic event occurring at two gate values presents to the SAE as two magnitudes. Possible consequence: gate-tracking features. Watch item for eventual feature-level analysis (v0.2); no v0.1.1 mitigation.
3. **Modality is the easiest organization.** Multimodal routers are known to separate modalities strongly at the routing level; the router baselines will likely show heavy concentration before the SAE says anything. The scientifically live quantity is the enrichment residue $R_M(e)$ and the incremental information of §16.4, not raw concentration.

---

## 16. Primary statistical analyses

### 16.1 Image-versus-text separation
$\operatorname{JSD}(q_{\mathrm{image}}, q_{\mathrm{text}})$, compared against: shuffled modality labels; the random-routing SAE control; JSD computed from host router gate mass. Example-level bootstrap confidence intervals.

### 16.2 Concentration
Normalized entropy $H_{\mathrm{norm}}(q_M) = -\sum_e q_M(e)\log q_M(e) / \log 64$; top-1, top-3, top-6 mass; effective expert count $\exp(H(q_M))$.

### 16.3 Stability (revised — changelog #9)
Identify the six most image-enriched experts independently on: natural images; OCR; charts; diagrams; two random dataset halves; all three seeds. Report pairwise Jaccard overlap, rank correlation, bootstrap selection frequency.

**Load floor (frozen):** banks that received fewer than **500,000 routed training tokens** in the relevant split (≈11% of the ≈4.7M expected under uniform load at 50M tokens) are excluded from enrichment rankings and stability statistics. Per-bank token counts are reported for all 64 banks in all conditions.

### 16.4 Incremental modality information (revised — changelog #8)
Regularized linear classifiers predict visual-vs-text token type from:

1. host router weights alone;
2. SAE expert-mass vector ($q$-style 64-dim) alone;
3. **binary feature-identity vector** (which of the 262,144 features fired under θ\*-inference), via L1-regularized logistic regression — *new*;
4. combinations of the above.

Held-out at the **example** level, never randomly held-out positions from seen examples. Probe (3) exists because modality organization may reside in feature identity below the resolution of the bank-mass marginal; it remains an aggregate statistic and does not involve manual interpretation of individual features (§19 intact). The SAE signal is scientifically stronger when it adds predictive information beyond the host router.

---

## 17. Causal validation

Causal work begins only after enriched expert/feature sets are defined on a development split, with analysis code locked before evaluation data is touched.

### Intervention (revised — changelog #2)

For each held-out token: encode the **true** routed contributions under frozen θ\*; choose the ablation set $J_t$ of selected image-enriched latents; and form

$$c_t^{\mathrm{route}\,\prime} = c_t^{\mathrm{route}} - \sum_{(e,j)\in J_t} z_{t,e,j}\, D_{e,j},$$

inserted into the frozen model, forward pass continued. The token's activation is otherwise the model's own — no reconstruction error is injected anywhere; the only modification is the removal of the named features' contributions. Single-shot: no re-encoding of the modified vector.

Compared against ablation sets matched for: number of latents; total coefficient mass; activation frequency; expert load; and removed-vector norm $\|\sum_{(e,j)\in J_t} z_{t,e,j} D_{e,j}\|_2$.

### Negative control (new — changelog #10)

The identical pipeline — enriched-set definition on the development split, identical matching criteria, identical subtraction intervention — is run on the **random-routing SAE (condition C)**. This establishes the effect size of structured mass deletion absent router-tied organization. A "visual subsystem" claim requires the condition-A intervention to exceed the condition-C intervention on visual-task selectivity.

### Evaluation
Effects measured on visual question answering; OCR; chart reasoning; diagram reasoning; and text-only language-modeling control prompts. A "visual subsystem" claim requires **preferential** degradation on visual tasks, not merely general damage.

---

## 18. Pre-registered interpretations

### Reading rules (new — changelog #11)

- **Objective asymmetry.** Condition A optimizes the strictly harder decomposed local loss and is *scored* on the aggregate metric that B and F optimize directly. Therefore: narrow A losses on aggregate NMSE are ambiguous; A wins on aggregate NMSE are strong evidence. Stated here so it cannot become a post-hoc rationalization.
- **Power clause.** The **negative** interpretation below may only be claimed from the confirmatory run. A discovery-scale null (12k fires/feature; §10) is reported as *underpowered, not informative*.
- **Headline template.** With the A–B–F triangle, the architecture result is stated quantitatively: "router-tying recovers **X%** of the full-dictionary gain (F − B) at **1/10.7** the encoder cost," where X = (A − B)/(F − B) on the pre-registered aggregate metric.

### Strong positive result
All of: (1) A beats B on aggregate reconstruction or downstream fidelity; (2) A beats C; (3) SAE expert mass or feature identity distinguishes visual from text tokens beyond router weights (§16.4); (4) the same small expert set is stable across datasets and seeds (§16.3, above the load floor); (5) feature-level subtraction ablation selectively damages visual behavior beyond the condition-C negative control.

> Permitted conclusion: the host MoE router exposes a stable, functionally meaningful visual expert subsystem whose internal explanatory features are revealed by router-tied BatchTopK dictionary learning.

### Partial positive result
Concentration is stable and predictive, but causal selectivity is weak.

> Permitted conclusion: the model contains modality-associated expert-feature organization, though evidence for a discrete functional subsystem is incomplete.

### Architecture-only result
Router-tied reconstruction/fidelity improves (A ≥ B, A > C, and X materially > 0), but modality concentration is weak.

> Permitted conclusion: host routing is a useful conditional partition for scalable dictionary learning, but does not yield cortex-like expert specialization at the examined layer.

### Negative result *(confirmatory run only)*
Random routing performs similarly and no stable concentration appears.

> Permitted conclusion: the host router's computational partition does not provide a correspondingly useful or modality-specific dictionary partition at the examined layer and scale.

---

## 19. Explicit non-goals for v0.1.1

We do not: write a SparseCore kernel; sweep transformer layers; train on multiple models; alter the host router; learn a new SAE router; include shared experts in the SAE bank; assign fixed $k/6$ budgets to experts; tune per-expert dictionary widths; add expert calibration scalars; optimize the aggregate loss directly; train on answer-generation tokens; re-estimate θ per evaluation composition; redefine enriched sets after evaluation data has been touched; interpret individual features manually before establishing the aggregate effect; or claim a visual cortex from correlational routing plots alone.

These are follow-on experiments, not hidden degrees of freedom.

---

## 20. Execution order

1. Implement and validate expert-contribution capture (FP32 gate; BF16 report). **Hard gate.**
2. Collect per-expert means $\mu_e$, routing statistics, and per-bank load counts.
3. Run the 1M-token pipeline model (f = 1024, k̄ = 32); optional LR sanity check; freeze LR.
4. Train discovery conditions A, B, C, F (three seeds) and D, E (one seed); log $\theta_b$ throughout; freeze θ\* per condition/seed at end of training.
5. **Implementation sanity gate:** at seed 1, A must beat C on local micro-averaged NMSE by ≥10% relative. Failure halts the study for an implementation audit — this is an implementation gate, not a hypothesis test, and is recorded as such.
6. Evaluate reconstruction and downstream fidelity (matched-position primary; full replacement secondary).
7. Compute per-bank loads; apply the 500k load floor; lock the expert-concentration analysis code **before** examining modality results.
8. Run modality, enrichment, stability, and incremental-information analyses (including the identity probe).
9. Define enriched expert/feature sets on the development split; lock intervention code.
10. Run subtraction interventions on untouched evaluation data, with the condition-C negative control.
11. Only then decide whether to scale dictionary width, layers, model size, or SparseCore — and whether the confirmatory run proceeds.

---

## Frozen one-sentence specification

> Train a jointly optimized bank of 64 expert-specific BatchTopK SAEs on the gated routed-expert contributions at block 6 of DeepSeek-VL2-Tiny, with 4,096 features per expert and a single batch-wide budget averaging 64 active features per original token; test — against compute-matched, capacity-matched, random-routing, independent-budget, and per-token controls — whether explanatory feature mass concentrates stably and causally on a small set of experts for visual tokens beyond what is explained by the model's router itself.

---

## References

[1] DeepSeek-VL2: Mixture-of-Experts Vision-Language Models for Advanced Multimodal Understanding. arXiv:2412.10302.
[2] BatchTopK Sparse Autoencoders. arXiv:2412.06410.
[3] Scaling and Evaluating Sparse Autoencoders. arXiv:2406.04093. (Origin of the dead-latent auxiliary loss, $k_{\mathrm{aux}}$, and $\alpha_{\mathrm{aux}} = 1/32$ conventions.)
[4] Towards Monosemanticity: Decomposing Language Models With Dictionary Learning. Transformer Circuits, 2023. (Origin of the decoder-column tangent-space projection + renormalization convention.)
