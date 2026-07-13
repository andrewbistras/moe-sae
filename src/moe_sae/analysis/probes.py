"""Incremental modality-information probes (spec §16.4).

Regularized linear classifiers predict visual-vs-text token type from:

1. `router_gates`       — the host router's gate vector: 64-dim sparse vector
                          with g_te at the token's routed banks (all conditions);
2. `sae_bank_mass`      — the SAE expert-mass vector m_act[t] (64-dim; banked
                          conditions only — global dictionaries have no banks);
3. `feature_identity`   — binary which-latents-fired vector over the full
                          dictionary (F_total dims, scipy CSR), L1-regularized
                          logistic regression (all conditions);
4. combinations (1)+(2) and (1)+(3).

Splits (frozen, A3 keyed hash — moe_sae.data.splits): split_bucket(domain,
example_id) — 0–79 train, 80–89 dev, 90–99 eval. Probes are fit on
train-split tokens and scored on dev-split tokens; held-out at the *example*
level, never at the position level (spec §16.4).

Classifier choices (frozen here): sklearn LogisticRegression.
- Dense probes (1, 2, 1+2): penalty='l2', solver='lbfgs'. The spec asks only
  for "regularized linear classifiers"; L2/lbfgs is the deterministic default
  for low-dimensional dense inputs.
- Identity probes (3, 1+3): penalty='l1', solver='liblinear' (spec §16.4
  mandates L1 for the identity probe; liblinear over saga because it is
  deterministic and fast for binary problems on sparse inputs of this size).
- C is configurable (default 1.0), identical across probes within one run,
  reported in the output. Seeds fixed; class balance reported per split.

Decisions:
- Two-dump flow: `run_eval` produces single-split FeatureDumps, so run_probes
  takes a TRAIN-split dump (+ masses) to fit on and a separate score dump
  (+ masses) whose DEV-split tokens are scored. Passing the same object for
  both is valid for mixed-split dumps (e.g. synthetic fixtures); split
  membership is always re-derived from example_id, never trusted from meta.
- The frozen EVAL split (A3 split_bucket in 90–99) is scored only when
  include_eval_split=True (run_analysis --include-eval-split), a
  POST-INTERVENTION-ONLY switch: spec §17/§20 require evaluation data
  untouched until the intervention has run. By default the per-probe "eval"
  entry is a skipped marker and no eval-split class balance is reported. The
  former "heldout" (dev∪eval) aggregate is removed for the same reason; dev
  is the pre-intervention reporting split.
- Bank attribution for the identity probe uses FeatureDump.act_banks()
  (act_bank field when present, exact for condition C's per-example bank
  permutation; experts[token, slot] fallback). F_total comes from meta
  `f_dict` when present, else is derived from the per-bank width `f`.
- A18 representation-route producer: probes.json carries
  `advantages_vs_router` — for probes 2, 3, and the two probe-4 combinations,
  the DEV-split accuracy advantage over probe 1 (router-only) with a 95%
  example-level percentile-bootstrap CI (n_boot=1000, seeded; same
  conventions as analysis.stability: resample examples with replacement,
  recompute BOTH probes' accuracies on the resampled token multiset, take the
  difference; the replicate weights are shared across probes within one
  call). A20: probe 2 and (1)+(2) are undefined for B/F → null entries.
  Schema consumed by scripts/evaluate_trigger.py:
  {probe_name: {"point": x, "ci_low": lo, "ci_high": hi, "n_boot": n} | null}.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

from moe_sae.analysis.mass import (
    BANKED_CONDITIONS,
    TOKEN_TYPE_VISUAL,
    BankMasses,
    FeatureDump,
)
from moe_sae.data.splits import split_buckets
from moe_sae.model_loading import NUM_ROUTED_EXPERTS
from moe_sae.train.config import global_width

SPLIT_TRAIN, SPLIT_DEV, SPLIT_EVAL = 0, 1, 2

EVAL_SPLIT_NOTE = (
    "eval split (A3 split_bucket in 90–99) untouched pre-intervention "
    "(spec §17/§20); rerun with --include-eval-split after the intervention"
)

# A18 representation-route inputs: probes 2, 3, and the two probe-4
# combinations, each compared against probe 1 (router_gates).
ADVANTAGE_PROBES = (
    "sae_bank_mass",  # probe 2 (A20: N/A for B/F → null entry)
    "feature_identity",  # probe 3
    "gates_plus_bank_mass",  # probe 4, (1)+(2) (A20: N/A for B/F)
    "gates_plus_identity",  # probe 4, (1)+(3)
)
N_BOOT_ADVANTAGE = 1000  # same convention as analysis.stability (N_BOOT)


def example_split(domain: np.ndarray, example_id: np.ndarray) -> np.ndarray:
    """Frozen split codes from the A3 keyed hash over (domain, example_id)."""
    r = split_buckets(np.asarray(domain), np.asarray(example_id))
    return np.where(r < 80, SPLIT_TRAIN, np.where(r < 90, SPLIT_DEV, SPLIT_EVAL))


def total_features(condition: str, f: int) -> int:
    """Dictionary size F_total: 64·f for banked conditions, the global width
    (6f for B, 64f for F) otherwise."""
    if condition in BANKED_CONDITIONS:
        return NUM_ROUTED_EXPERTS * f
    return global_width(condition, f)


# --------------------------------------------------------------------------- #
# Feature builders                                                             #
# --------------------------------------------------------------------------- #


def gate_features(dump: FeatureDump) -> np.ndarray:
    """[N, 64] float32 — g_te scattered to the token's routed banks."""
    n = dump.n_tokens
    x = np.zeros((n, NUM_ROUTED_EXPERTS), dtype=np.float32)
    rows = np.repeat(np.arange(n, dtype=np.int64), dump.experts.shape[1])
    np.add.at(x, (rows, dump.experts.reshape(-1)), dump.gates.reshape(-1))
    return x


def identity_features(dump: FeatureDump) -> sparse.csr_matrix:
    """[N, F_total] binary CSR — which latents fired under θ*-inference.

    Global latent index: bank·f + latent for banked conditions, with bank
    from FeatureDump.act_banks() (permutation-aware for condition C); the
    raw latent index for global conditions. F_total is meta `f_dict` when
    present, else derived from the per-bank width `f`.
    """
    n = dump.n_tokens
    if dump.is_banked:
        f = dump.f
        if f is None and dump.f_dict is not None:
            f = dump.f_dict // NUM_ROUTED_EXPERTS
        if f is None:
            raise ValueError(
                "identity probe needs the per-bank width: dump meta.json provides "
                "neither 'f' nor 'f_dict' — write them, or pass f= to load_dump "
                "(--f on scripts/run_analysis.py)"
            )
        f_total = dump.f_dict if dump.f_dict is not None else NUM_ROUTED_EXPERTS * f
        cols = dump.act_banks() * f + dump.act_latent
    else:
        if dump.f_dict is not None:
            f_total = dump.f_dict
        elif dump.f is not None:
            f_total = global_width(dump.condition, dump.f)
        else:
            raise ValueError(
                "identity probe needs the dictionary width: dump meta.json provides "
                "neither 'f_dict' nor 'f' — write them, or pass f= to load_dump "
                "(--f on scripts/run_analysis.py)"
            )
        cols = dump.act_latent
    x = sparse.csr_matrix(
        (np.ones(len(cols), dtype=np.float32), (dump.act_token, cols)),
        shape=(n, f_total),
    )
    x.data = np.minimum(x.data, np.float32(1.0))  # binarize any duplicates
    return x


# --------------------------------------------------------------------------- #
# Probe runner                                                                 #
# --------------------------------------------------------------------------- #


def _evaluate(clf: LogisticRegression, x, y: np.ndarray) -> dict | None:
    if y.shape[0] == 0:
        return None
    pred = clf.predict(x)
    scores = clf.decision_function(x)
    auc = float(roc_auc_score(y, scores)) if np.unique(y).size == 2 else None
    return {"n": int(y.shape[0]), "accuracy": float(accuracy_score(y, pred)), "auc": auc}


def advantages_vs_router(
    dev_preds: dict[str, np.ndarray],
    y_dev: np.ndarray,
    example_id_dev: np.ndarray,
    n_boot: int = N_BOOT_ADVANTAGE,
    seed: int = 0,
) -> dict:
    """A18 representation-route producer: for each probe in ADVANTAGE_PROBES,
    the dev accuracy advantage over probe 1 (router_gates) with a 95%
    example-level percentile-bootstrap CI.

    Bootstrap unit = EXAMPLE (spec §16.4; stability.py conventions): each
    replicate resamples example ids with replacement (numpy default_rng,
    `rng.integers(0, n_ex, n_ex)` → bincount weights), recomputes BOTH
    probes' accuracies on the resampled token multiset, and takes the
    difference; percentile (2.5, 97.5) CI over n_boot=1000 replicates. The
    replicate weights are shared across probes so paired advantages see the
    same resamples. Probes absent from `dev_preds` (A20: bank-mass probes for
    global conditions B/F) get a null entry.
    """
    correct = {
        name: (pred == y_dev).astype(np.float64) for name, pred in dev_preds.items()
    }
    ids, inverse = np.unique(np.asarray(example_id_dev), return_inverse=True)
    n_ex = ids.shape[0]
    tok_per_ex = np.bincount(inverse, minlength=n_ex).astype(np.float64)
    sums = {
        name: np.bincount(inverse, weights=c, minlength=n_ex)
        for name, c in correct.items()
    }

    rng = np.random.default_rng(seed)
    weights = np.stack(
        [
            np.bincount(rng.integers(0, n_ex, size=n_ex), minlength=n_ex)
            for _ in range(n_boot)
        ]
    ).astype(np.float64)  # [n_boot, n_ex]
    denom = weights @ tok_per_ex  # [n_boot] — every example has ≥1 token
    acc_router_boot = (weights @ sums["router_gates"]) / denom
    acc_router_point = float(correct["router_gates"].mean())

    out: dict[str, dict | None] = {}
    for name in ADVANTAGE_PROBES:
        if name not in dev_preds:
            out[name] = None  # A20: probe undefined for this condition
            continue
        boot = (weights @ sums[name]) / denom - acc_router_boot
        lo, hi = np.percentile(boot, [2.5, 97.5])
        out[name] = {
            "point": float(correct[name].mean()) - acc_router_point,
            "ci_low": float(lo),
            "ci_high": float(hi),
            "n_boot": int(n_boot),
        }
    return out


def run_probes(
    train_dump: FeatureDump,
    train_masses: BankMasses | None,
    score_dump: FeatureDump,
    score_masses: BankMasses | None,
    c: float = 1.0,
    seed: int = 0,
    max_iter: int = 2000,
    include_eval_split: bool = False,
    n_boot: int = N_BOOT_ADVANTAGE,
) -> dict:
    """Fit the §16.4 probes on `train_dump`'s TRAIN-split tokens and score them
    on `score_dump`'s DEV-split tokens. `*_masses` are None for global
    conditions; the bank-mass probes are then reported as skipped. EVAL-split
    scoring is off unless include_eval_split=True (post-intervention only —
    see module Decisions). The output additionally carries the A18
    representation-route inputs under `advantages_vs_router` (see
    `advantages_vs_router`; `n_boot` bootstrap replicates, seeded by `seed`)."""
    if train_dump.condition != score_dump.condition:
        raise ValueError(
            f"probe train/score dumps disagree on condition: "
            f"{train_dump.condition!r} vs {score_dump.condition!r}"
        )
    y_tr = (train_dump.token_type == TOKEN_TYPE_VISUAL).astype(np.int64)
    y_sc = (score_dump.token_type == TOKEN_TYPE_VISUAL).astype(np.int64)
    train = example_split(train_dump.domain, train_dump.example_id) == SPLIT_TRAIN
    sc_split = example_split(score_dump.domain, score_dump.example_id)
    dev = sc_split == SPLIT_DEV
    ev = sc_split == SPLIT_EVAL
    if not train.any() or np.unique(y_tr[train]).size < 2:
        raise ValueError(
            "train dump has no train-split tokens (A3 split_bucket < 80) or a "
            "single class; cannot fit probes — pass a TRAIN-split FeatureDump "
            "(--train-dump on scripts/run_analysis.py)"
        )
    if not dev.any():
        raise ValueError(
            "score dump has no dev-split tokens (A3 split_bucket in 80–89); "
            "probes are scored on the dev split — pass a DEV-split FeatureDump "
            "(--dump on scripts/run_analysis.py)"
        )

    x_gate_tr, x_gate_sc = gate_features(train_dump), gate_features(score_dump)
    x_id_tr, x_id_sc = identity_features(train_dump), identity_features(score_dump)
    if x_id_tr.shape[1] != x_id_sc.shape[1]:
        raise ValueError(
            f"identity feature spaces differ: train dump F_total={x_id_tr.shape[1]}, "
            f"score dump F_total={x_id_sc.shape[1]} — check the dumps' meta.json "
            "'f'/'f_dict' fields (or the --f override)"
        )
    have_mass = train_masses is not None and score_masses is not None
    probes: dict[str, tuple[object, object, str]] = {
        "router_gates": (x_gate_tr, x_gate_sc, "l2")
    }
    if have_mass:
        xm_tr = train_masses.m_act.astype(np.float32)
        xm_sc = score_masses.m_act.astype(np.float32)
        probes["sae_bank_mass"] = (xm_tr, xm_sc, "l2")
    probes["feature_identity"] = (x_id_tr, x_id_sc, "l1")
    if have_mass:
        probes["gates_plus_bank_mass"] = (
            np.hstack([x_gate_tr, xm_tr]),
            np.hstack([x_gate_sc, xm_sc]),
            "l2",
        )
    probes["gates_plus_identity"] = (
        sparse.hstack([sparse.csr_matrix(x_gate_tr), x_id_tr], format="csr"),
        sparse.hstack([sparse.csr_matrix(x_gate_sc), x_id_sc], format="csr"),
        "l1",
    )

    results: dict[str, dict | None] = {}
    dev_preds: dict[str, np.ndarray] = {}
    for name, (x_tr, x_sc, penalty) in probes.items():
        # sklearn >= 1.8 deprecates penalty= in favor of l1_ratio (1.0 == pure
        # L1, 0.0 == pure L2); verified equivalent to the old penalty kwarg.
        if penalty == "l1":
            clf = LogisticRegression(
                l1_ratio=1.0, solver="liblinear", C=c, random_state=seed, max_iter=max_iter
            )
        else:
            clf = LogisticRegression(l1_ratio=0.0, solver="lbfgs", C=c, max_iter=max_iter)
        clf.fit(x_tr[train], y_tr[train])
        dev_preds[name] = np.asarray(clf.predict(x_sc[dev]))
        entry = {
            "penalty": penalty,
            "n_features": int(x_tr.shape[1]),
            "dev": _evaluate(clf, x_sc[dev], y_sc[dev]),
            "eval": (
                _evaluate(clf, x_sc[ev], y_sc[ev])
                if include_eval_split
                else {"skipped": EVAL_SPLIT_NOTE}
            ),
        }
        if penalty == "l1":
            entry["n_nonzero_coef"] = int(np.count_nonzero(clf.coef_))
        results[name] = entry
    if not have_mass:
        note = "global condition: no bank structure (spec §14)"
        results["sae_bank_mass"] = {"skipped": note}
        results["gates_plus_bank_mass"] = {"skipped": note}

    # A18 representation-route inputs: dev accuracy advantage over probe 1
    # with an example-level bootstrap CI (A20: null for undefined probes).
    advantages = advantages_vs_router(
        dev_preds, y_sc[dev], score_dump.example_id[dev], n_boot=n_boot, seed=seed
    )

    def balance(dump: FeatureDump, y: np.ndarray, mask: np.ndarray) -> dict:
        return {
            "n_tokens": int(mask.sum()),
            "n_examples": int(np.unique(dump.example_id[mask]).size),
            "frac_visual": float(y[mask].mean()) if mask.any() else None,
        }

    splits = {
        "train": balance(train_dump, y_tr, train),
        "dev": balance(score_dump, y_sc, dev),
    }
    if include_eval_split:
        splits["eval"] = balance(score_dump, y_sc, ev)

    return {
        "condition": train_dump.condition,
        "C": c,
        "seed": seed,
        "include_eval_split": include_eval_split,
        "splits": splits,
        "probes": results,
        # A18 schema consumed by scripts/evaluate_trigger.py
        "advantages_vs_router": advantages,
    }
