"""Expert-level SAE mass (spec §14) computed from FeatureDump artifacts.

For token t and bank e:

    m_count[t, e] = Σ_j 1[z_{t,e,j} > 0]        (count mass)
    m_act[t, e]   = Σ_j z_{t,e,j}               (coefficient mass — primary)
    m_energy[t, e] = ‖D_e z_{t,e}‖²             (reconstructed-vector energy)

and for modality M: q_M(e) = Σ_{t∈M} m_act[t,e] / Σ_{t∈M} Σ_{e'} m_act[t,e'].

Bank attribution follows the FeatureDump contract (docs/INTERFACES.md): the
bank of a sparse activation is `experts[act_token, act_slot]`. For condition C
the dump producer is responsible for writing the *bank* ids used by the SAE
into `experts` slots (the contract states bank = experts[token, slot]).

Global conditions (B/F) have a single shared dictionary and therefore no bank
structure: q_M(e) is undefined, and every bank-mass function returns None for
them. Spec analyses §14–§16.3 apply to A/C/D/E only; §16.4 probes over router
gates and feature identity apply to all conditions (see analysis.probes).

m_energy requires decoder columns, which the analysis stage does not have; it
is therefore read from the dump's optional per-token per-slot `energy` field
([N, 6] float32, ‖D_e z_{t,e}‖² at slot order), written by
`moe_sae.eval.reconstruction.dump_features`. Dumps without the field yield
m_energy = None and the energy robustness check is skipped.

Decisions:
- meta.json fields `condition`, `f`, `f_dict`, `num_banks`, `split`,
  `theta_star` are parsed when present; `condition` and `f` remain
  caller-overridable for metaless dumps (run_analysis --condition/--f).
- `FeatureDump.f_dict` is normalized to the TOTAL dictionary width F_total
  (64·f banked, 6f/64f global). Producers that write the per-dict-bank width
  (sae.conditions semantics: f_dict == f for banked conditions) are detected
  via `f` and normalized at load; a banked f_dict matching neither f nor 64·f
  is rejected as inconsistent meta.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from safetensors.torch import load_file

from moe_sae.capture.writer import DOMAIN_NAMES
from moe_sae.model_loading import NUM_ROUTED_EXPERTS, TOP_K_ROUTED

BANKED_CONDITIONS = ("A", "C", "D", "E")
GLOBAL_CONDITIONS = ("B", "F")

TOKEN_TYPE_TEXT = 0
TOKEN_TYPE_VISUAL = 1

_PER_TOKEN_FIELDS = ("experts", "gates", "token_type", "example_id", "domain", "l0")
_ACT_FIELDS = ("act_token", "act_slot", "act_latent", "act_value")


@dataclass
class FeatureDump:
    """A θ*-inference FeatureDump loaded into numpy (all shards concatenated).

    `act_token` is re-indexed to the concatenated token order (shard-local
    token indices are offset by the cumulative token count of prior shards).
    """

    condition: str  # A..F
    f: int | None  # per-bank width (per-bank-equivalent for B/F); None if unknown
    experts: np.ndarray  # [N, 6] int64
    gates: np.ndarray  # [N, 6] float32
    token_type: np.ndarray  # [N] uint8 — 0 text, 1 visual
    example_id: np.ndarray  # [N] int64
    domain: np.ndarray  # [N] uint8
    l0: np.ndarray | None  # [N] int64
    act_token: np.ndarray  # [M] int64
    act_slot: np.ndarray  # [M] int64
    act_latent: np.ndarray  # [M] int64
    act_value: np.ndarray  # [M] float64
    energy: np.ndarray | None  # [N, 6] float64 or None
    act_bank: np.ndarray | None = None  # [M] int64 — assigned bank (≠ experts
    # slot lookup for condition C, whose banks are per-example permutations)
    f_dict: int | None = None  # TOTAL dictionary width F_total (from meta.json,
    # normalized — see module Decisions); None if unknown
    split: str | None = None  # producer split tag from meta.json (train/dev/eval)
    num_banks: int | None = None  # dictionary bank count from meta.json (64 or 1)
    theta_star: float | list | None = None  # frozen θ* from meta.json (scalar,
    # or condition D's per-bank list under A4 — null entries = undefined banks)

    def act_banks(self) -> np.ndarray:
        """Bank of each activation: the dump's act_bank field when present
        (exact for all conditions incl. C), else experts[token, slot]
        (exact for A/D/E only)."""
        if self.act_bank is not None:
            return self.act_bank
        return self.experts[self.act_token, self.act_slot]

    @property
    def n_tokens(self) -> int:
        return int(self.experts.shape[0])

    @property
    def is_banked(self) -> bool:
        return self.condition in BANKED_CONDITIONS


def _read_meta(root: Path) -> dict:
    # dump layout is dumps/<run>/<split>/; meta may sit at the split dir or the
    # run dir. Absent meta requires an explicit condition from the caller.
    for cand in (root / "meta.json", root.parent / "meta.json"):
        if cand.exists():
            return json.loads(cand.read_text())
    return {}


def load_dump(
    path: str | Path,
    condition: str | None = None,
    f: int | None = None,
) -> FeatureDump:
    """Load all shards of a FeatureDump split into one FeatureDump.

    `condition`/`f` override the dump's meta.json; the condition must be
    resolvable from one of the two or loading aborts (mass semantics depend
    on it — a wrong default would silently produce garbage q vectors).
    meta.json's `f_dict`/`num_banks`/`split`/`theta_star` are parsed when
    present (see module Decisions for the f_dict width normalization).
    """
    root = Path(path)
    meta = _read_meta(root)
    condition = condition if condition is not None else meta.get("condition")
    if condition not in BANKED_CONDITIONS + GLOBAL_CONDITIONS:
        raise ValueError(
            f"cannot determine SAE condition for dump at {root}: got {condition!r}; "
            "meta.json is missing (or lacks a 'condition' field) — write it, or pass "
            "condition= explicitly (--condition on scripts/run_analysis.py)"
        )
    f = f if f is not None else meta.get("f")
    f = int(f) if f is not None else None
    f_dict = meta.get("f_dict")
    if f_dict is not None:
        f_dict = int(f_dict)
        if condition in BANKED_CONDITIONS:
            total = NUM_ROUTED_EXPERTS * f if f is not None else None
            if f is None:
                # only the total is unambiguous without f; assume it and derive f
                f = f_dict // NUM_ROUTED_EXPERTS
            elif f_dict == f and f_dict != total:
                f_dict = total  # per-dict-bank width (sae.conditions) → total
            elif f_dict != total:
                raise ValueError(
                    f"meta.json at {root} is inconsistent: 'f_dict'={f_dict} is "
                    f"neither 'f'={f} (per-dict-bank width) nor 64·f={total} "
                    "(total width) for a banked condition — fix meta.json or "
                    "pass f= explicitly (--f on scripts/run_analysis.py)"
                )

    paths = sorted(root.glob("shard_*.safetensors"))
    if not paths:
        raise FileNotFoundError(f"no shard_*.safetensors under {root}")

    per_token: dict[str, list[np.ndarray]] = {k: [] for k in _PER_TOKEN_FIELDS}
    acts: dict[str, list[np.ndarray]] = {k: [] for k in _ACT_FIELDS}
    energy: list[np.ndarray] = []
    act_bank: list[np.ndarray] = []
    has_l0 = True
    has_energy = True
    has_act_bank = True
    token_offset = 0
    for p in paths:
        shard = {k: v.numpy() for k, v in load_file(str(p)).items()}
        n = shard["experts"].shape[0]
        for k in _PER_TOKEN_FIELDS:
            if k == "l0" and "l0" not in shard:
                has_l0 = False
                continue
            per_token[k].append(shard[k])
        if "energy" in shard:
            energy.append(shard["energy"])
        else:
            has_energy = False
        if "act_bank" in shard:
            act_bank.append(shard["act_bank"])
        else:
            has_act_bank = False
        for k in _ACT_FIELDS:
            a = shard[k]
            if k == "act_token":
                a = a.astype(np.int64) + token_offset
            acts[k].append(a)
        token_offset += n

    def cat(chunks: list[np.ndarray], dtype) -> np.ndarray:
        return np.concatenate(chunks, axis=0).astype(dtype)

    return FeatureDump(
        condition=condition,
        f=f,
        experts=cat(per_token["experts"], np.int64),
        gates=cat(per_token["gates"], np.float32),
        token_type=cat(per_token["token_type"], np.uint8),
        example_id=cat(per_token["example_id"], np.int64),
        domain=cat(per_token["domain"], np.uint8),
        l0=cat(per_token["l0"], np.int64) if has_l0 else None,
        act_token=cat(acts["act_token"], np.int64),
        act_slot=cat(acts["act_slot"], np.int64),
        act_latent=cat(acts["act_latent"], np.int64),
        act_value=cat(acts["act_value"], np.float64),
        energy=cat(energy, np.float64) if has_energy and energy else None,
        act_bank=cat(act_bank, np.int64) if has_act_bank and act_bank else None,
        f_dict=f_dict,
        split=meta.get("split"),
        num_banks=int(meta["num_banks"]) if meta.get("num_banks") is not None else None,
        theta_star=(
            None
            if meta.get("theta_star") is None
            else meta["theta_star"]
            if isinstance(meta["theta_star"], list)  # A4: D's per-bank θ*
            else float(meta["theta_star"])
        ),
    )


# --------------------------------------------------------------------------- #
# Per-token per-bank masses (spec §14)                                         #
# --------------------------------------------------------------------------- #


@dataclass
class BankMasses:
    """Per-token per-bank masses, dense [N, 64] (float32 to bound memory;
    all downstream reductions accumulate in float64)."""

    m_count: np.ndarray  # [N, 64] float32
    m_act: np.ndarray  # [N, 64] float32
    m_energy: np.ndarray | None  # [N, 64] float32 or None


def token_bank_masses(dump: FeatureDump) -> BankMasses | None:
    """Spec §14 masses. Returns None for global conditions (B/F): a single
    shared dictionary has no bank structure, so q_M(e) is undefined."""
    if not dump.is_banked:
        return None
    n = dump.n_tokens
    e = NUM_ROUTED_EXPERTS
    banks = dump.act_banks()  # [M] — assigned bank (permutation-aware for C)
    fired = dump.act_value > 0  # θ*-thresholded dumps are already > 0; guard anyway

    m_count = np.zeros((n, e), dtype=np.float32)
    np.add.at(m_count, (dump.act_token[fired], banks[fired]), np.float32(1.0))

    m_act = np.zeros((n, e), dtype=np.float32)
    np.add.at(m_act, (dump.act_token, banks), dump.act_value.astype(np.float32))

    m_energy = None
    if dump.energy is not None:
        # Per-(token, slot) bank attribution: default to the host expert, then
        # overwrite from activations (slots with zero activations carry zero
        # energy, so their attribution is irrelevant; slots with activations
        # get the true assigned bank — permutation-aware for condition C).
        slot_bank = dump.experts.copy()  # [N, 6]
        slot_bank[dump.act_token, dump.act_slot] = banks
        m_energy = np.zeros((n, e), dtype=np.float32)
        rows = np.repeat(np.arange(n, dtype=np.int64), TOP_K_ROUTED)
        np.add.at(
            m_energy,
            (rows, slot_bank.reshape(-1)),
            dump.energy.reshape(-1).astype(np.float32),
        )
    return BankMasses(m_count=m_count, m_act=m_act, m_energy=m_energy)


def q_from_mass(m: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray | None:
    """q(e) = Σ_{t∈mask} m[t,e] / Σ_{t∈mask,e'} m[t,e'] (float64). None if the
    selected tokens carry zero total mass (q undefined)."""
    sub = m if mask is None else m[mask]
    per_bank = sub.sum(axis=0, dtype=np.float64)
    total = per_bank.sum()
    if total <= 0:
        return None
    return per_bank / total


def modality_masks(dump: FeatureDump) -> dict[str, np.ndarray]:
    return {
        "image": dump.token_type == TOKEN_TYPE_VISUAL,
        "text": dump.token_type == TOKEN_TYPE_TEXT,
    }


def domain_masks(dump: FeatureDump) -> dict[str, np.ndarray]:
    """Token masks per dataset domain present in the dump."""
    out: dict[str, np.ndarray] = {}
    for v in sorted(np.unique(dump.domain).tolist()):
        name = DOMAIN_NAMES.get(int(v), f"domain_{int(v)}")
        out[name] = dump.domain == v
    return out


def q_by_modality(
    masses: BankMasses, dump: FeatureDump, kind: str = "act"
) -> dict[str, np.ndarray | None]:
    m = _mass_of_kind(masses, kind)
    return {mod: q_from_mass(m, mask) for mod, mask in modality_masks(dump).items()}


def q_by_domain(
    masses: BankMasses, dump: FeatureDump, kind: str = "act"
) -> dict[str, dict[str, np.ndarray | None]]:
    """q_M(e) restricted to each dataset domain, per modality."""
    m = _mass_of_kind(masses, kind)
    mods = modality_masks(dump)
    return {
        dom: {mod: q_from_mass(m, dmask & mmask) for mod, mmask in mods.items()}
        for dom, dmask in domain_masks(dump).items()
    }


def _mass_of_kind(masses: BankMasses, kind: str) -> np.ndarray:
    m = {"count": masses.m_count, "act": masses.m_act, "energy": masses.m_energy}[kind]
    if m is None:
        raise ValueError(f"mass kind {kind!r} unavailable in this dump")
    return m


# --------------------------------------------------------------------------- #
# Example-level aggregation (spec §16.1: example-level bootstrap)              #
# --------------------------------------------------------------------------- #


@dataclass
class ExampleMasses:
    """Per-example per-bank mass sums, split by modality. The resampling unit
    for every bootstrap/permutation in §16 is the *example* (spec §3, §16.1);
    both modality halves of an example move together."""

    example_ids: np.ndarray  # [n_ex] int64 (sorted unique)
    image: np.ndarray  # [n_ex, 64] float64 — Σ m over the example's visual tokens
    text: np.ndarray  # [n_ex, 64] float64 — Σ m over the example's text tokens


def per_example_masses(
    m: np.ndarray, dump: FeatureDump, token_mask: np.ndarray | None = None
) -> ExampleMasses:
    ids, inverse = np.unique(dump.example_id, return_inverse=True)
    n_ex, e = ids.shape[0], m.shape[1]
    image = np.zeros((n_ex, e), dtype=np.float64)
    text = np.zeros((n_ex, e), dtype=np.float64)
    vis = dump.token_type == TOKEN_TYPE_VISUAL
    if token_mask is not None:
        keep = token_mask
    else:
        keep = np.ones(dump.n_tokens, dtype=bool)
    vsel = keep & vis
    tsel = keep & ~vis
    np.add.at(image, inverse[vsel], m[vsel].astype(np.float64))
    np.add.at(text, inverse[tsel], m[tsel].astype(np.float64))
    return ExampleMasses(example_ids=ids, image=image, text=text)
