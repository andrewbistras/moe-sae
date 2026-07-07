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

Splits (frozen contract, docs/INTERFACES.md §Splits): example_id % 100 —
0–79 train, 80–89 dev, 90–99 eval. Probes are fit on train-split tokens and
reported on dev and eval tokens (and their union, "heldout"); held-out at the
*example* level, never at the position level (spec §16.4).

Classifier choices (frozen here): sklearn LogisticRegression.
- Dense probes (1, 2, 1+2): penalty='l2', solver='lbfgs'. The spec asks only
  for "regularized linear classifiers"; L2/lbfgs is the deterministic default
  for low-dimensional dense inputs.
- Identity probes (3, 1+3): penalty='l1', solver='liblinear' (spec §16.4
  mandates L1 for the identity probe; liblinear over saga because it is
  deterministic and fast for binary problems on sparse inputs of this size).
- C is configurable (default 1.0), identical across probes within one run,
  reported in the output. Seeds fixed; class balance reported per split.
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
from moe_sae.model_loading import NUM_ROUTED_EXPERTS
from moe_sae.train.config import global_width

SPLIT_TRAIN, SPLIT_DEV, SPLIT_EVAL = 0, 1, 2


def example_split(example_id: np.ndarray) -> np.ndarray:
    """Frozen split codes from example_id % 100 (contract §Splits)."""
    r = np.asarray(example_id) % 100
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

    Global latent index: bank·f + latent for banked conditions (bank =
    experts[act_token, act_slot]); the raw latent index for global conditions.
    """
    if dump.f is None:
        raise ValueError("dump meta must provide per-bank width 'f' for the identity probe")
    n = dump.n_tokens
    f_total = total_features(dump.condition, dump.f)
    if dump.is_banked:
        banks = dump.experts[dump.act_token, dump.act_slot]
        cols = banks * dump.f + dump.act_latent
    else:
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


def run_probes(
    dump: FeatureDump,
    masses: BankMasses | None,
    c: float = 1.0,
    seed: int = 0,
    max_iter: int = 2000,
) -> dict:
    """Fit and score the §16.4 probes. `masses` is None for global conditions;
    the bank-mass probes are then reported as skipped."""
    y = (dump.token_type == TOKEN_TYPE_VISUAL).astype(np.int64)
    split = example_split(dump.example_id)
    train = split == SPLIT_TRAIN
    dev = split == SPLIT_DEV
    ev = split == SPLIT_EVAL
    heldout = dev | ev
    if not train.any() or np.unique(y[train]).size < 2:
        raise ValueError("train split is empty or single-class; cannot fit probes")

    x_gate = gate_features(dump)
    x_id = identity_features(dump)
    probes: dict[str, tuple[object, str]] = {"router_gates": (x_gate, "l2")}
    if masses is not None:
        x_mass = masses.m_act.astype(np.float32)
        probes["sae_bank_mass"] = (x_mass, "l2")
    probes["feature_identity"] = (x_id, "l1")
    if masses is not None:
        probes["gates_plus_bank_mass"] = (np.hstack([x_gate, x_mass]), "l2")
    probes["gates_plus_identity"] = (
        sparse.hstack([sparse.csr_matrix(x_gate), x_id], format="csr"),
        "l1",
    )

    results: dict[str, dict | None] = {}
    for name, (x, penalty) in probes.items():
        # sklearn >= 1.8 deprecates penalty= in favor of l1_ratio (1.0 == pure
        # L1, 0.0 == pure L2); verified equivalent to the old penalty kwarg.
        if penalty == "l1":
            clf = LogisticRegression(
                l1_ratio=1.0, solver="liblinear", C=c, random_state=seed, max_iter=max_iter
            )
        else:
            clf = LogisticRegression(l1_ratio=0.0, solver="lbfgs", C=c, max_iter=max_iter)
        clf.fit(x[train], y[train])
        entry = {
            "penalty": penalty,
            "n_features": int(x.shape[1]),
            "dev": _evaluate(clf, x[dev], y[dev]),
            "eval": _evaluate(clf, x[ev], y[ev]),
            "heldout": _evaluate(clf, x[heldout], y[heldout]),
        }
        if penalty == "l1":
            entry["n_nonzero_coef"] = int(np.count_nonzero(clf.coef_))
        results[name] = entry
    if masses is None:
        note = "global condition: no bank structure (spec §14)"
        results["sae_bank_mass"] = {"skipped": note}
        results["gates_plus_bank_mass"] = {"skipped": note}

    def balance(mask: np.ndarray) -> dict:
        return {
            "n_tokens": int(mask.sum()),
            "n_examples": int(np.unique(dump.example_id[mask]).size),
            "frac_visual": float(y[mask].mean()) if mask.any() else None,
        }

    return {
        "condition": dump.condition,
        "C": c,
        "seed": seed,
        "splits": {
            "train": balance(train),
            "dev": balance(dev),
            "eval": balance(ev),
            "heldout": balance(heldout),
        },
        "probes": results,
    }
