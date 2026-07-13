"""SAE enrichment over router baselines (spec §15) and the §16.3 load floor.

    R_M(e) = log( (q_M(e) + ε) / (p_M(e) + ε) ),   ε = 1e-6, natural log.

The spec writes `log` without a base; we freeze the natural log (base only
rescales R by a constant, leaving every ranking, sign, and correlation in
§16.3 invariant). ε = 1e-6 guards empty banks on both sides of the ratio.

Baselines come from the router-statistics artifact (`stats.safetensors`):

| key             | shape | meaning                                          |
|-----------------|-------|--------------------------------------------------|
| `p_route_image` | [64]  | routing frequency per bank, image tokens         |
| `p_route_text`  | [64]  | routing frequency per bank, text tokens          |
| `p_gate_image`  | [64]  | gate-mass fraction per bank, image tokens        |
| `p_gate_text`   | [64]  | gate-mass fraction per bank, text tokens         |
| `bank_counts`   | [64]  | routed *training* tokens per bank                 |

Load floor (A13, v0.1.2 addendum — supersedes the v0.1.1 absolute floor):
a bank enters enrichment rankings and stability statistics only if its routed
token count IN THE SPLIT UNDER ANALYSIS is ≥ 10% of that split's uniform-load
expectation E_uniform = (split_tokens × 6)/64. The per-split counts come from
the analysis dump's host-expert routing (`bank_counts` argument of
`enrichment_analysis`; computed by `moe_sae.eval.intervention.
dump_split_bank_counts`); the floor value itself comes from
`split_proportional_floor`, with an absolute override retained on the
scripts. Per-bank counts are reported for all 64 banks regardless.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from safetensors.torch import load_file

from moe_sae.model_loading import NUM_ROUTED_EXPERTS

EPSILON = 1e-6  # frozen ε of spec §15

MODALITIES = ("image", "text")

# The stats pass (train/stats_pass.py) names the image modality "visual"
# (matching the capture pool names); analysis uses spec §14's "image".
_STATS_KEY_SUFFIX = {"image": "visual", "text": "text"}


@dataclass(frozen=True)
class RouterStats:
    """Router baselines + per-bank training load, from the stats pass."""

    p_route: dict[str, np.ndarray]  # modality -> [64] float64
    p_gate: dict[str, np.ndarray]  # modality -> [64] float64
    bank_counts: np.ndarray  # [64] int64 — routed training tokens per bank


def load_stats(path: str | Path) -> RouterStats:
    """Load stats.safetensors (or a directory containing one)."""
    p = Path(path)
    if p.is_dir():
        p = p / "stats.safetensors"
    raw = {k: v.numpy() for k, v in load_file(str(p)).items()}
    e = NUM_ROUTED_EXPERTS

    def vec(key: str) -> np.ndarray:
        v = raw[key].astype(np.float64).reshape(-1)
        assert v.shape == (e,), (key, v.shape)
        return v

    return RouterStats(
        p_route={m: vec(f"p_route_{_STATS_KEY_SUFFIX[m]}") for m in MODALITIES},
        p_gate={m: vec(f"p_gate_{_STATS_KEY_SUFFIX[m]}") for m in MODALITIES},
        bank_counts=raw["bank_counts"].astype(np.int64).reshape(e),
    )


def enrichment(q: np.ndarray, p: np.ndarray, eps: float = EPSILON) -> np.ndarray:
    """R(e) = ln((q(e)+ε)/(p(e)+ε)), elementwise over banks."""
    q = np.asarray(q, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    return np.log(q + eps) - np.log(p + eps)


def load_floor_mask(bank_counts: np.ndarray, load_floor: float) -> np.ndarray:
    """True for banks at or above the floor ("fewer than" excluded, spec §16.3)."""
    return np.asarray(bank_counts) >= load_floor


def enrichment_analysis(
    q_by_mod: dict[str, np.ndarray | None],
    stats: RouterStats,
    load_floor: float,
    eps: float = EPSILON,
    bank_counts: np.ndarray | None = None,
) -> dict:
    """JSON-ready §15 report: R_M(e) vs gate mass (primary) and vs routing
    frequency, with sub-floor banks reported as null (excluded from rankings)
    while per-bank counts are always reported for all 64 banks.

    `bank_counts` (A13): the per-split routed counts of the dump under
    analysis — the floor source. Falls back to `stats.bank_counts` for
    callers that predate A13 (absolute-floor audits)."""
    counts = np.asarray(
        bank_counts if bank_counts is not None else stats.bank_counts
    )
    mask = load_floor_mask(counts, load_floor)

    def masked(values: np.ndarray) -> list[float | None]:
        return [float(v) if ok else None for v, ok in zip(values, mask)]

    report: dict = {
        "epsilon": eps,
        "log_base": "e",
        "load_floor": float(load_floor),
        "load_floor_source": (
            "split_counts (A13 split-proportional)" if bank_counts is not None
            else "stats.bank_counts (explicit absolute floor)"
        ),
        "bank_counts": [int(c) for c in counts],
        "floor_mask": [bool(b) for b in mask],
        "n_banks_above_floor": int(mask.sum()),
        "R": {},
    }
    for mod, q in q_by_mod.items():
        if q is None:
            report["R"][mod] = None
            continue
        report["R"][mod] = {
            "vs_gate": masked(enrichment(q, stats.p_gate[mod], eps)),
            "vs_route": masked(enrichment(q, stats.p_route[mod], eps)),
        }
    return report
