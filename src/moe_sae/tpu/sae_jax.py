"""JAX mirrors of the SAE hot path (spec §5–§8, §11–§12) — static shapes.

Faithful ports of, in order: banks.py (grouped encode/decode → sort +
capacity-padded batched GEMMs), batchtopk.py (exact-K selection, fp32 scores),
losses.py, optim.py (SpecAdam order: clip → project → Adam → renorm), and the
per-condition training step of conditions.py.

TPU-specific decisions (recorded):
- Grouped GEMMs use capacity padding: tokens-slots sorted by bank, scattered
  into an [E, cap, ·] buffer, one batched einsum, gathered back. Padding rows
  never re-enter results (the gather uses only real destinations) — semantics
  are exact; padding costs FLOPs only (cap=2× mean load ⇒ ~2× encode FLOPs,
  still ~5× cheaper than condition F). Capacity overflow is a hard host-side
  error (never token dropping — that would violate spec §6).
- The grouping plan (sort order / destinations) is computed host-side in
  numpy from the batch's bank ids and passed in as arrays, so the jitted
  functions see static shapes only.
- Condition D's per-bank selection reuses the torch engine on host (one run
  at discovery; correctness over device residency — decision recorded).
- Aux-loss "min(512, #dead visible)" mirrors torch `_aux_select`: top-512 of
  dead-masked scores, selections with score > -inf.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

_NEG_INF = float("-inf")
_DEN_EPS = 1e-12


# --------------------------------------------------------------------------- #
# Grouping plan (host-side; mirrors banks.py's per-bank row grouping)          #
# --------------------------------------------------------------------------- #


@dataclass
class GroupingPlan:
    """Sort/scatter plan for one batch's bank assignment (static shapes)."""

    order: np.ndarray  # [N] int32 — flat indices sorted by bank (stable)
    dest: np.ndarray  # [N] int32 — padded-buffer slot of each sorted element
    bank_flat: np.ndarray  # [N] int32 — bank of each flat (token, slot)
    cap: int
    n_banks: int


def grouping_plan(bank_ids: np.ndarray, n_banks: int, cap: int) -> GroupingPlan:
    """bank_ids: [B, S] int. Raises on capacity overflow (hard error;
    spec §6 forbids dropping candidates)."""
    flat = np.asarray(bank_ids, dtype=np.int64).reshape(-1)
    counts = np.bincount(flat, minlength=n_banks)
    if int(counts.max()) > cap:
        worst = int(counts.argmax())
        raise RuntimeError(
            f"bank {worst} received {int(counts.max())} slots > capacity {cap}; "
            "raise the capacity (config; see measured bank loads in the stats pass)"
        )
    order = np.argsort(flat, kind="stable").astype(np.int32)
    starts = np.zeros(n_banks, dtype=np.int64)
    starts[1:] = np.cumsum(counts)[:-1]
    sorted_banks = flat[order]
    rank_in_bank = np.arange(flat.shape[0], dtype=np.int64) - starts[sorted_banks]
    dest = (sorted_banks * cap + rank_in_bank).astype(np.int32)
    return GroupingPlan(
        order=order, dest=dest, bank_flat=flat.astype(np.int32), cap=cap, n_banks=n_banks
    )


# --------------------------------------------------------------------------- #
# Encode / decode (banks.py mirrors)                                           #
# --------------------------------------------------------------------------- #


@partial(jax.jit, static_argnames=("cap", "n_banks", "out_f"))
def encode_routed(
    W_enc: jnp.ndarray,  # [E, f, d]
    b_enc: jnp.ndarray,  # [E, f]
    mu: jnp.ndarray,  # [E, d] f32
    c: jnp.ndarray,  # [B, S, d]
    bank_flat: jnp.ndarray,  # [N] int32
    order: jnp.ndarray,  # [N] int32
    dest: jnp.ndarray,  # [N] int32
    cap: int,
    n_banks: int,
    out_f: int,
) -> jnp.ndarray:
    """a = ReLU(W_enc_e (c − μ_e) + b_enc_e), grouped via capacity padding."""
    B, S, d = c.shape
    n = B * S
    dt = W_enc.dtype
    x = c.reshape(n, d).astype(dt) - mu[bank_flat].astype(dt)
    xp = jnp.zeros((n_banks * cap, d), dt).at[dest].set(x[order])
    pre = (
        jnp.einsum("ecd,efd->ecf", xp.reshape(n_banks, cap, d), W_enc)
        + b_enc[:, None, :]
    )
    a = jax.nn.relu(pre).reshape(n_banks * cap, out_f)
    out = jnp.zeros((n, out_f), dt).at[order].set(a[dest])
    return out.reshape(B, S, out_f)


@partial(jax.jit, static_argnames=("cap", "n_banks", "add_mu"))
def decode_routed(
    D: jnp.ndarray,  # [E, d, f]
    mu: jnp.ndarray,  # [E, d] f32
    z: jnp.ndarray,  # [B, S, f]
    bank_flat: jnp.ndarray,
    order: jnp.ndarray,
    dest: jnp.ndarray,
    cap: int,
    n_banks: int,
    add_mu: bool,
) -> jnp.ndarray:
    """ĉ = [μ_e +] D_e z, grouped via the same plan."""
    B, S, f = z.shape
    n = B * S
    dt = D.dtype
    d = D.shape[1]
    zp = jnp.zeros((n_banks * cap, f), dt).at[dest].set(z.reshape(n, f).astype(dt)[order])
    rec = jnp.einsum("ecf,edf->ecd", zp.reshape(n_banks, cap, f), D).reshape(
        n_banks * cap, d
    )
    out = jnp.zeros((n, d), dt).at[order].set(rec[dest])
    if add_mu:
        out = out + mu[bank_flat].astype(dt)
    return out.reshape(B, S, d)


def encode_global(W_enc, b_enc, mu, x):
    """GlobalSAE.encode mirror: [B, d] → [B, F]."""
    dt = W_enc.dtype
    tilde = x.astype(dt) - mu.astype(dt)
    return jax.nn.relu(tilde @ W_enc.T + b_enc)


def decode_global(D, mu, z, add_mu: bool):
    """GlobalSAE.decode mirror: [B, F] → [B, d]."""
    rec = z.astype(D.dtype) @ D.T
    if add_mu:
        rec = rec + mu.astype(D.dtype)
    return rec


# --------------------------------------------------------------------------- #
# Selection (batchtopk.py mirrors; fp32 scores, exact K)                       #
# --------------------------------------------------------------------------- #


def global_topk_mask(scores_flat: jnp.ndarray, k_total: int):
    """Exactly k_total over the flattened set. Returns (mask [n] bool, θ_b)."""
    s = scores_flat.astype(jnp.float32)
    vals, idx = jax.lax.top_k(s, k_total)
    mask = jnp.zeros(s.shape[0], dtype=bool).at[idx].set(True)
    return mask, vals[k_total - 1]


def per_token_topk_mask(scores: jnp.ndarray, k: int):
    """Exactly k per row. Returns (mask [B, n] bool, per-row k-th values)."""
    s = scores.astype(jnp.float32)
    vals, idx = jax.lax.top_k(s, k)
    mask = jnp.zeros_like(s, dtype=bool)
    mask = mask.at[jnp.arange(s.shape[0])[:, None], idx].set(True)
    return mask, vals[:, -1]


def thresholded_inference(preacts: jnp.ndarray, theta_star: float) -> jnp.ndarray:
    """z = a · 1[a > θ*], strict (spec §6)."""
    a = preacts.astype(jnp.float32)
    return a * (a > theta_star)


def aux_select_mask(scores2d: jnp.ndarray, dead2d: jnp.ndarray, k_aux: int):
    """Torch `_aux_select` mirror: top min(k_aux, n) of dead-masked scores;
    selected iff its masked score > -inf."""
    masked = jnp.where(dead2d, scores2d.astype(jnp.float32), _NEG_INF)
    k = min(k_aux, masked.shape[1])
    vals, idx = jax.lax.top_k(masked, k)
    rows = jnp.arange(masked.shape[0])[:, None]
    mask = jnp.zeros_like(dead2d)
    return mask.at[rows, idx].set(vals > _NEG_INF)


# --------------------------------------------------------------------------- #
# Losses (losses.py mirrors; fp32 reductions)                                  #
# --------------------------------------------------------------------------- #


def local_nmse(c_rows, chat_rows, mu_rows):
    c = c_rows.astype(jnp.float32)
    num = jnp.sum((c - chat_rows.astype(jnp.float32)) ** 2)
    den = jnp.maximum(jnp.sum((c - mu_rows.astype(jnp.float32)) ** 2), _DEN_EPS)
    return num / den


def aggregate_nmse(c_route, chat_route):
    c = c_route.astype(jnp.float32)
    num = jnp.sum((c - chat_route.astype(jnp.float32)) ** 2)
    cbar = jnp.mean(c, axis=0, keepdims=True)
    den = jnp.maximum(jnp.sum((c - cbar) ** 2), _DEN_EPS)
    return num / den


def aux_loss(r_rows, rhat_rows):
    r = r_rows.astype(jnp.float32)
    den = jnp.sum(r**2)
    num = jnp.sum((r - rhat_rows.astype(jnp.float32)) ** 2)
    return jnp.where(den < _DEN_EPS, 0.0, num / jnp.maximum(den, _DEN_EPS))


# --------------------------------------------------------------------------- #
# SpecAdam (optim.py mirror; identical step order and constants)               #
# --------------------------------------------------------------------------- #


def adam_init(params: dict) -> dict:
    return {
        "m": jax.tree.map(lambda p: jnp.zeros(p.shape, jnp.float32), params),
        "v": jax.tree.map(lambda p: jnp.zeros(p.shape, jnp.float32), params),
        "t": 0,
    }


def _project_decoder_grad(g32, p32, axis):
    norm = jnp.maximum(jnp.linalg.norm(p32, axis=axis, keepdims=True), 1e-12)
    dhat = p32 / norm
    parallel = jnp.sum(g32 * dhat, axis=axis, keepdims=True)
    return g32 - parallel * dhat


@partial(jax.jit, static_argnames=("decoder_axis", "betas", "eps", "clip"))
def _adam_step(
    params: dict,
    grads: dict,
    m: dict,
    v: dict,
    lr: jnp.ndarray,
    c1: jnp.ndarray,  # 1 - b1**t
    c2: jnp.ndarray,  # 1 - b2**t
    decoder_axis: int,
    betas: tuple = (0.9, 0.999),
    eps: float = 1e-8,
    clip: float = 1.0,
):
    b1, b2 = betas
    g32 = jax.tree.map(lambda g: g.astype(jnp.float32), grads)

    # 1. Global grad-norm clip over ALL params (fp32 accumulation).
    sq = sum(jnp.sum(g**2) for g in jax.tree.leaves(g32))
    total_norm = jnp.sqrt(sq)
    clip_coef = jnp.minimum(1.0, clip / (total_norm + 1e-6))

    new_params, new_m, new_v = {}, {}, {}
    for k in params:
        g = g32[k] * clip_coef
        p32 = params[k].astype(jnp.float32)
        # 2. Tangent-space projection for decoder columns.
        if k == "D":
            g = _project_decoder_grad(g, p32, decoder_axis)
        # 3. Adam in fp32.
        mk = m[k] * b1 + g * (1.0 - b1)
        vk = v[k] * b2 + g * g * (1.0 - b2)
        p32 = p32 - lr * (mk / c1) / (jnp.sqrt(vk / c2) + eps)
        # 4. Renormalize decoder columns.
        if k == "D":
            p32 = p32 / jnp.maximum(
                jnp.linalg.norm(p32, axis=decoder_axis, keepdims=True), 1e-12
            )
        new_params[k] = p32.astype(params[k].dtype)
        new_m[k], new_v[k] = mk, vk
    return new_params, new_m, new_v, total_norm


class SpecAdamJax:
    """Stateful wrapper matching torch SpecAdam semantics (see optim.py)."""

    def __init__(self, params: dict, decoder_axis: int, betas=(0.9, 0.999), clip=1.0):
        self.state = adam_init(params)
        self.decoder_axis = decoder_axis
        self.betas = (float(betas[0]), float(betas[1]))
        self.clip = float(clip)

    def step(self, params: dict, grads: dict, lr: float) -> tuple[dict, float]:
        self.state["t"] += 1
        t = self.state["t"]
        c1 = 1.0 - self.betas[0] ** t
        c2 = 1.0 - self.betas[1] ** t
        new_params, self.state["m"], self.state["v"], norm = _adam_step(
            params,
            grads,
            self.state["m"],
            self.state["v"],
            jnp.float32(lr),
            jnp.float32(c1),
            jnp.float32(c2),
            self.decoder_axis,
            self.betas,
            1e-8,
            self.clip,
        )
        return new_params, float(norm)
