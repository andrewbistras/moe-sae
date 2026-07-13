"""A15 text draw-epoch robustness check (v0.1.2 addendum; Q2 adopted).

Pre-registered robustness analysis: recompute §16.1 (image-vs-text JSD) and
§16.4 (probes) on the text half STRATIFIED BY TRAINING DRAW COUNT, and report
whether image/text separation or incremental probe performance changes
monotonically with text reuse count. At discovery scale the text half
re-draws from a smaller unique pool (~4-8 effective epochs) while the visual
half is effectively unique; this check pre-empts the most predictable
objection to any modality asymmetry.

Interpretation decision (recorded): draw counts are defined only for tokens
the training stream could draw — the TRAIN split — and dev tokens are NEVER
drawn in training, so the pre-registered stratification runs on the
TRAIN-SPLIT FeatureDump, where the counts are defined. The reported statistic
is the MONOTONICITY (Spearman rank correlation over strata) of (a) the
image-vs-stratum-text JSD and (b) per-stratum probe performance, across
strata of increasing draw count. Probe scoring per stratum: probes 1
(router_gates) and 3 (feature_identity) are fit ONCE on the full train split
(both modalities), then scored on each stratum's text tokens pooled with ALL
train-split image tokens; per-stratum class balance is reported alongside.

Draw-count alignment: the training loop persists final/text_draw_counts.npz —
one int64 array per capture text shard (``shard_{i:05d}``), indexed by the
row's ORIGINAL position in that shard (moe_sae.data.batches, A15). The
FeatureDump producer walks capture shards in pool order (visual then text)
and shard order, keeping split rows in order, so dump text tokens align with
the concatenation of each text shard's kept rows; alignment is verified
against example_id before counts are accepted.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from moe_sae.analysis.mass import (
    TOKEN_TYPE_VISUAL,
    BankMasses,
    FeatureDump,
    q_from_mass,
)
from moe_sae.analysis.probes import gate_features, identity_features
from moe_sae.analysis.stability import jsd, spearman
from moe_sae.capture.writer import iter_shards
from moe_sae.data.splits import split_mask


def load_text_draw_counts(npz_path: str | Path) -> dict[int, np.ndarray]:
    """Load final/text_draw_counts.npz → {shard_idx: int64 per-row counts}."""
    data = np.load(str(npz_path))
    return {int(name.split("_")[1]): data[name].astype(np.int64) for name in data.files}


def per_token_draw_counts(
    dump: FeatureDump,
    capture_root: str | Path,
    draw_counts: dict[int, np.ndarray],
    split: str = "train",
) -> np.ndarray:
    """Per-dump-token training draw counts: text tokens get their capture
    row's count; visual tokens get −1 (undefined — the visual pool is not
    recorded; it is effectively unique at discovery scale)."""
    out = np.full(dump.n_tokens, -1, dtype=np.int64)
    n_visual = int((dump.token_type == TOKEN_TYPE_VISUAL).sum())
    # producer order: all visual shards first, then text shards (POOLS order)
    assert (dump.token_type[:n_visual] == TOKEN_TYPE_VISUAL).all(), (
        "dump token order does not match the producer contract (visual pool "
        "first); cannot align draw counts"
    )
    cursor = n_visual
    for i, shard in enumerate(iter_shards(Path(capture_root), "text")):
        keep = split_mask(shard["domain"], shard["example_id"], split).numpy()
        if not keep.any():
            continue
        rows = np.nonzero(keep)[0]
        n = rows.shape[0]
        got = dump.example_id[cursor : cursor + n]
        want = shard["example_id"].numpy()[rows]
        assert np.array_equal(got, want), (
            f"draw-count alignment failed at text shard {i}: dump example_ids "
            "do not match the capture shard's kept rows"
        )
        if i in draw_counts:
            out[cursor : cursor + n] = draw_counts[i][rows]
        else:
            out[cursor : cursor + n] = 0  # shard never touched by training
        cursor += n
    assert cursor == dump.n_tokens, (cursor, dump.n_tokens)
    return out


def _strata(counts: np.ndarray, max_strata: int) -> list[tuple[str, np.ndarray]]:
    """Group text tokens by draw count; the highest counts collapse into one
    '>=k' stratum when there are more than `max_strata` distinct values."""
    values = np.unique(counts)
    if values.size <= max_strata:
        return [(str(int(v)), counts == v) for v in values]
    head = values[: max_strata - 1]
    tail_min = int(values[max_strata - 1])
    out = [(str(int(v)), counts == v) for v in head]
    out.append((f">={tail_min}", counts >= tail_min))
    return out


def draw_epoch_robustness(
    dump: FeatureDump,
    masses: BankMasses | None,
    token_draw_counts: np.ndarray,
    c: float = 1.0,
    seed: int = 0,
    max_iter: int = 2000,
    max_strata: int = 6,
) -> dict:
    """A15 stratified recomputation (see module docstring for the recorded
    interpretation decision). `dump` must be the TRAIN-split FeatureDump and
    `token_draw_counts` its per-token counts (−1 on visual tokens)."""
    is_vis = dump.token_type == TOKEN_TYPE_VISUAL
    is_txt = ~is_vis
    assert (token_draw_counts[is_txt] >= 0).all(), (
        "text tokens must carry defined draw counts (train-split dump only)"
    )

    # --- probes fit once on the full train split ---
    y = is_vis.astype(np.int64)
    x_gate = gate_features(dump)
    x_id = identity_features(dump)
    clf_gate = LogisticRegression(l1_ratio=0.0, solver="lbfgs", C=c, max_iter=max_iter)
    clf_gate.fit(x_gate, y)
    clf_id = LogisticRegression(
        l1_ratio=1.0, solver="liblinear", C=c, random_state=seed, max_iter=max_iter
    )
    clf_id.fit(x_id, y)

    # --- §16.1 baseline over all text ---
    q_image = q_from_mass(masses.m_act, is_vis) if masses is not None else None

    strata = _strata(token_draw_counts[is_txt], max_strata)
    txt_index = np.nonzero(is_txt)[0]
    rows = []
    for name, sub in strata:
        sel = txt_index[sub]  # dump token indices of this stratum's text
        entry: dict = {"stratum": name, "n_text_tokens": int(sel.size)}
        # §16.1: JSD(q_image over ALL image tokens, q_text over the stratum)
        if masses is not None and q_image is not None:
            sel_mask = np.zeros(dump.n_tokens, dtype=bool)
            sel_mask[sel] = True
            q_txt = q_from_mass(masses.m_act, sel_mask)
            entry["jsd_image_vs_stratum_text"] = (
                jsd(q_image, q_txt) if q_txt is not None else None
            )
        else:
            entry["jsd_image_vs_stratum_text"] = None
        # §16.4: probe scores on stratum text ∪ ALL image tokens
        pool = np.concatenate([np.nonzero(is_vis)[0], sel])
        y_pool = y[pool]
        entry["frac_visual_in_pool"] = float(y_pool.mean())
        entry["probe1_router_gates_acc"] = float(
            accuracy_score(y_pool, clf_gate.predict(x_gate[pool]))
        )
        entry["probe3_feature_identity_acc"] = float(
            accuracy_score(y_pool, clf_id.predict(x_id[pool]))
        )
        entry["probe3_minus_probe1_acc"] = (
            entry["probe3_feature_identity_acc"] - entry["probe1_router_gates_acc"]
        )
        rows.append(entry)

    def _mono(key: str) -> float | None:
        vals = [r[key] for r in rows]
        if any(v is None for v in vals) or len(vals) < 2:
            return None
        rho = spearman(np.arange(len(vals), dtype=np.float64), np.asarray(vals))
        return None if rho != rho else rho  # NaN (constant series) → None

    return {
        "interpretation": (
            "A15: stratification runs on the TRAIN-split dump (dev tokens are "
            "never drawn in training, so draw counts are undefined there); "
            "the reported statistic is monotonicity across strata of "
            "increasing draw count (Spearman over stratum order)."
        ),
        "n_strata": len(rows),
        "strata": rows,
        "monotonicity_spearman": {
            "jsd_image_vs_stratum_text": _mono("jsd_image_vs_stratum_text"),
            "probe1_router_gates_acc": _mono("probe1_router_gates_acc"),
            "probe3_feature_identity_acc": _mono("probe3_feature_identity_acc"),
            "probe3_minus_probe1_acc": _mono("probe3_minus_probe1_acc"),
        },
    }
