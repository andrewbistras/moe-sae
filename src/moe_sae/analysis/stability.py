"""Separation, concentration, and stability statistics (spec §16.1–§16.3).

§16.1 — JSD(q_image, q_text) with base-2 logarithms (bounded in [0, 1]),
compared against (a) an example-level shuffled-modality-label null, (b) the
JSD of the router's own gate-mass baselines, and (c) the same statistic on a
condition-C dump; example-level percentile-bootstrap CIs throughout.

§16.2 — normalized entropy H_norm(q) = −Σ q ln q / ln 64 (base-independent),
top-1/3/6 mass, effective expert count exp(H) (natural-log H).

§16.3 — top-6 image-enriched experts (by R_image vs gate mass, banks above
the load floor only) identified independently per domain, per random example
half, and per seed; pairwise Jaccard, Spearman rank correlation over the
floor-restricted enrichment vectors, and bootstrap selection frequency.

Resampling conventions (frozen here):
- The exchangeable unit is the *example* (spec §16.1). Both modality halves of
  an example move together in the bootstrap.
- Shuffled-label null: because every example contributes a paired
  (image-half, text-half) of mass vectors, the label permutation is the paired
  one — each shuffle independently swaps the two halves within each example
  with probability 1/2. This is the exchangeability-preserving permutation of
  modality labels at the example level. n_shuffles = 1000.
- Bootstrap: resample examples with replacement, n_boot = 1000, percentile
  (2.5, 97.5) CIs, numpy default_rng with a fixed caller-supplied seed.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

from moe_sae.analysis.enrichment import (
    EPSILON,
    RouterStats,
    enrichment,
    load_floor_mask,
)
from moe_sae.analysis.mass import (
    TOKEN_TYPE_VISUAL,
    BankMasses,
    ExampleMasses,
    FeatureDump,
    domain_masks,
    per_example_masses,
    q_from_mass,
)

N_SHUFFLES = 1000
N_BOOT = 1000
TOP_K_ENRICHED = 6


# --------------------------------------------------------------------------- #
# Core statistics                                                              #
# --------------------------------------------------------------------------- #


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen–Shannon divergence with base-2 log: 0 ≤ JSD ≤ 1.

    Inputs are defensively renormalized; zero entries contribute 0 by the
    x·log x → 0 continuity convention.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def kl(a: np.ndarray) -> float:
        nz = a > 0
        return float(np.sum(a[nz] * np.log2(a[nz] / m[nz])))

    return 0.5 * kl(p) + 0.5 * kl(q)


def entropy_stats(q: np.ndarray) -> dict:
    """Spec §16.2 concentration statistics for one q vector."""
    q = np.asarray(q, dtype=np.float64)
    q = q / q.sum()
    nz = q > 0
    h = float(-np.sum(q[nz] * np.log(q[nz])))  # natural log
    desc = np.sort(q)[::-1]
    return {
        "H_norm": h / np.log(q.size),
        "top1_mass": float(desc[:1].sum()),
        "top3_mass": float(desc[:3].sum()),
        "top6_mass": float(desc[:6].sum()),
        "effective_experts": float(np.exp(h)),
    }


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return float(sp_stats.spearmanr(x, y).statistic)


def top_enriched(
    q: np.ndarray,
    baseline: np.ndarray,
    floor_mask: np.ndarray,
    k: int = TOP_K_ENRICHED,
    eps: float = EPSILON,
) -> list[int]:
    """Top-k banks by enrichment R = ln((q+ε)/(baseline+ε)) among banks above
    the load floor, in descending-R order (spec §16.3)."""
    r = enrichment(q, baseline, eps)
    r = np.where(floor_mask, r, -np.inf)
    order = np.argsort(-r, kind="stable")
    return [int(e) for e in order[: min(k, int(floor_mask.sum()))]]


# --------------------------------------------------------------------------- #
# Example-level resampling                                                     #
# --------------------------------------------------------------------------- #


def _q_pair(image_total: np.ndarray, text_total: np.ndarray) -> tuple:
    ti, tt = image_total.sum(), text_total.sum()
    qi = image_total / ti if ti > 0 else None
    qt = text_total / tt if tt > 0 else None
    return qi, qt


def observed_q(ex: ExampleMasses) -> tuple:
    return _q_pair(ex.image.sum(axis=0), ex.text.sum(axis=0))


def shuffled_label_jsds(
    ex: ExampleMasses, n_shuffles: int = N_SHUFFLES, seed: int = 0
) -> np.ndarray:
    """Null distribution of JSD under example-level modality-label permutation
    (paired within-example swap; see module docstring). NaN where a shuffle
    leaves one side with zero total mass (excluded by callers via nan-aware
    reductions)."""
    rng = np.random.default_rng(seed)
    out = np.empty(n_shuffles, dtype=np.float64)
    n = ex.example_ids.shape[0]
    for s in range(n_shuffles):
        swap = rng.random(n) < 0.5
        img = np.where(swap[:, None], ex.text, ex.image).sum(axis=0)
        txt = np.where(swap[:, None], ex.image, ex.text).sum(axis=0)
        qi, qt = _q_pair(img, txt)
        out[s] = jsd(qi, qt) if qi is not None and qt is not None else np.nan
    return out


def bootstrap_analysis(
    ex: ExampleMasses,
    p_gate_image: np.ndarray,
    floor_mask: np.ndarray,
    n_boot: int = N_BOOT,
    seed: int = 0,
    k: int = TOP_K_ENRICHED,
    eps: float = EPSILON,
) -> dict:
    """One seeded example-level bootstrap pass yielding both the §16.1 JSD CI
    and the §16.3 top-k selection frequency per bank (same replicates)."""
    rng = np.random.default_rng(seed)
    n, e = ex.image.shape
    qi_obs, qt_obs = observed_q(ex)
    if qi_obs is None or qt_obs is None:
        raise ValueError("cannot bootstrap: one modality carries zero mass")
    obs = jsd(qi_obs, qt_obs)

    jsds = np.empty(n_boot, dtype=np.float64)
    selection = np.zeros(e, dtype=np.int64)
    for b in range(n_boot):
        counts = np.bincount(rng.integers(0, n, size=n), minlength=n).astype(np.float64)
        img = counts @ ex.image
        txt = counts @ ex.text
        qi, qt = _q_pair(img, txt)
        jsds[b] = jsd(qi, qt) if qi is not None and qt is not None else np.nan
        if qi is not None:
            for bank in top_enriched(qi, p_gate_image, floor_mask, k, eps):
                selection[bank] += 1

    lo, hi = np.nanpercentile(jsds, [2.5, 97.5])
    return {
        "jsd_observed": obs,
        "jsd_ci_low": float(lo),
        "jsd_ci_high": float(hi),
        "n_boot": n_boot,
        "selection_frequency": (selection / n_boot).tolist(),
    }


def random_half_q_image(
    ex: ExampleMasses, seed: int = 0
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """q_image over two disjoint random example halves (seeded)."""
    rng = np.random.default_rng(seed)
    n = ex.example_ids.shape[0]
    perm = rng.permutation(n)
    halves = (perm[: n // 2], perm[n // 2 :])
    out = []
    for h in halves:
        tot = ex.image[h].sum(axis=0)
        out.append(tot / tot.sum() if tot.sum() > 0 else None)
    return out[0], out[1]


# --------------------------------------------------------------------------- #
# Full §16.1–§16.3 report                                                      #
# --------------------------------------------------------------------------- #


def stability_analysis(
    dump: FeatureDump,
    masses: BankMasses,
    stats: RouterStats,
    load_floor: int,
    seed: int = 0,
    n_shuffles: int = N_SHUFFLES,
    n_boot: int = N_BOOT,
    compare_example_masses: ExampleMasses | None = None,
    seed_q_image: dict[str, np.ndarray] | None = None,
    top_k: int = TOP_K_ENRICHED,
    eps: float = EPSILON,
) -> dict:
    """Assemble the full JSON-ready §16.1–§16.3 report for a banked dump.

    `compare_example_masses`: example table from a condition-C dump — its JSD
    (point + bootstrap CI) is reported alongside for the §16.1 comparison.
    `seed_q_image`: name → q_image from replicate-seed dumps of the same
    condition, entering the §16.3 per-seed top-k stability.
    Domain-restricted enrichment uses the global per-modality gate baseline
    from `stats` (the stats artifact carries per-modality baselines).
    """
    floor_mask = load_floor_mask(stats.bank_counts, load_floor)
    baseline = stats.p_gate["image"]
    ex = per_example_masses(masses.m_act, dump)
    qi, qt = observed_q(ex)
    if qi is None or qt is None:
        raise ValueError("stability analysis needs both modalities present")

    # §16.1 separation
    null = shuffled_label_jsds(ex, n_shuffles, seed)
    obs = jsd(qi, qt)
    valid = null[~np.isnan(null)]
    p_value = (1.0 + float((valid >= obs).sum())) / (1.0 + valid.size)
    boot = bootstrap_analysis(ex, baseline, floor_mask, n_boot, seed + 1, top_k, eps)
    gate_jsd = jsd(stats.p_gate["image"], stats.p_gate["text"])

    compare = None
    if compare_example_masses is not None:
        cqi, cqt = observed_q(compare_example_masses)
        if cqi is not None and cqt is not None:
            cboot = bootstrap_analysis(
                compare_example_masses, baseline, floor_mask, n_boot, seed + 2, top_k, eps
            )
            compare = {
                "jsd_observed": cboot["jsd_observed"],
                "jsd_ci_low": cboot["jsd_ci_low"],
                "jsd_ci_high": cboot["jsd_ci_high"],
            }

    # §16.3 named enrichment vectors and top-k sets
    named_r: dict[str, np.ndarray] = {}
    named_sets: dict[str, list[int]] = {}

    def register(name: str, q_img: np.ndarray | None) -> None:
        if q_img is None:
            return
        named_r[name] = enrichment(q_img, baseline, eps)
        named_sets[name] = top_enriched(q_img, baseline, floor_mask, top_k, eps)

    register("all", qi)
    vis = dump.token_type == TOKEN_TYPE_VISUAL
    for dom, dmask in domain_masks(dump).items():
        register(f"domain:{dom}", q_from_mass(masses.m_act, dmask & vis))
    h0, h1 = random_half_q_image(ex, seed + 3)
    register("half:0", h0)
    register("half:1", h1)
    for name, q_img in (seed_q_image or {}).items():
        register(f"seed:{name}", np.asarray(q_img, dtype=np.float64))

    names = list(named_sets)
    jaccard_pairs = {
        f"{a}|{b}": jaccard(set(named_sets[a]), set(named_sets[b]))
        for i, a in enumerate(names)
        for b in names[i + 1 :]
    }
    fm = floor_mask
    spearman_pairs = {
        f"{a}|{b}": spearman(named_r[a][fm], named_r[b][fm])
        for i, a in enumerate(names)
        for b in names[i + 1 :]
    }

    return {
        "seed": seed,
        "load_floor": int(load_floor),
        "floor_mask": [bool(b) for b in floor_mask],
        "separation": {
            "jsd_observed": obs,
            "jsd_ci_low": boot["jsd_ci_low"],
            "jsd_ci_high": boot["jsd_ci_high"],
            "n_boot": n_boot,
            "shuffled_null_mean": float(np.nanmean(null)),
            "shuffled_null_p95": float(np.nanpercentile(null, 95)),
            "shuffled_null_p_value": p_value,
            "n_shuffles": n_shuffles,
            "gate_mass_jsd": gate_jsd,
            "compare_dump": compare,
        },
        "concentration": {"image": entropy_stats(qi), "text": entropy_stats(qt)},
        "top_enriched": {name: s for name, s in named_sets.items()},
        "jaccard": jaccard_pairs,
        "spearman": spearman_pairs,
        "selection_frequency": boot["selection_frequency"],
    }
