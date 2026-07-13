"""Feature-subtraction intervention (spec §17, changelog #2): delta math against
a naive reference, J ∩ fired(t) logic, the no-reconstruction-error property at
bit level, enriched-latent selection, and matched-control quality."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from moe_sae.capture.hooks import instrumented_capture
from moe_sae.eval.fidelity import add_to_moe_output, forward_logits
from moe_sae.eval.intervention import (
    compute_ablation_deltas,
    latent_stats_from_dump,
    load_ablation_set,
    matched_control_sets,
    removed_norm_report,
    run_intervention,
    save_ablation_set,
    select_enriched_latents,
)
from moe_sae.eval.reconstruction import ThresholdedEncoding
from moe_sae.vendor.deepseek_vl2.configuration_deepseek import DeepseekV2Config
from moe_sae.vendor.deepseek_vl2.modeling_deepseek import DeepseekV2MoE

D_MODEL = 64
N_EXPERTS = 16
TOP_K = 4
VOCAB = 32
F = 8
THETA = 0.1  # calibrated to the tiny fixture's contribution scale (~0.06)


def make_moe(dtype=torch.float32, seed=0) -> DeepseekV2MoE:
    torch.manual_seed(seed)
    cfg = DeepseekV2Config(
        hidden_size=D_MODEL,
        moe_intermediate_size=32,
        intermediate_size=128,
        n_routed_experts=N_EXPERTS,
        num_experts_per_tok=TOP_K,
        n_shared_experts=2,
        topk_method="greedy",
        scoring_func="softmax",
        n_group=1,
        topk_group=1,
        norm_topk_prob=False,
        first_k_dense_replace=0,
        num_hidden_layers=2,
    )
    return DeepseekV2MoE(cfg).to(dtype).eval()


class TinyVLM(torch.nn.Module):
    def __init__(self, moe, seed=1):
        super().__init__()
        torch.manual_seed(seed)
        self.embed = torch.nn.Embedding(VOCAB, D_MODEL)
        self.moe = moe
        self.head = torch.nn.Linear(D_MODEL, VOCAB)

    def forward(self, input_ids, **kwargs):
        h = self.embed(input_ids)
        h = h + self.moe(h)
        return SimpleNamespace(logits=self.head(h))


@dataclass
class StubEvalExample:
    input_ids: torch.Tensor
    answer_mask: torch.Tensor
    visual_mask: torch.Tensor
    example_id: int
    domain: str
    answer_text: str = ""
    images: None = None
    images_seq_mask: None = None
    images_spatial_crop: None = None


def make_example(t=12, n_answer=3, example_id=0, domain="coco", seed=0):
    g = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(0, VOCAB, (t,), generator=g)
    answer_mask = torch.zeros(t, dtype=torch.bool)
    answer_mask[-n_answer:] = True
    return StubEvalExample(
        input_ids, answer_mask, torch.zeros(t, dtype=torch.bool), example_id, domain
    )


class FakeRoutedSAE:
    """Router-tied fake (protocol-compatible) sized for the tiny model."""

    condition = "A"

    def __init__(self, num_banks=N_EXPERTS, f=F, d=D_MODEL, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.num_banks, self.f, self.d = num_banks, f, d
        self.W = torch.randn(num_banks, f, d, generator=g) * 2.0
        Dm = torch.randn(num_banks, d, f, generator=g)
        self.D = Dm / Dm.norm(dim=1, keepdim=True)
        self.mu = torch.zeros(num_banks, d)

    def encode_thresholded(self, c, experts, example_id, theta_star):
        c = c.to(torch.float32)
        n, k, d = c.shape
        toks, slots, banks, lats, vals = [], [], [], [], []
        recon = torch.zeros(n, k, d)
        for t in range(n):
            for s in range(k):
                e = int(experts[t, s])
                a = torch.relu(self.W[e] @ (c[t, s] - self.mu[e]))
                z = a * (a > theta_star)
                recon[t, s] = self.mu[e] + self.D[e] @ z
                for j in z.nonzero(as_tuple=True)[0].tolist():
                    toks.append(t)
                    slots.append(s)
                    banks.append(e)
                    lats.append(j)
                    vals.append(float(z[j]))
        return ThresholdedEncoding(
            act_token=torch.tensor(toks, dtype=torch.int64),
            act_slot=torch.tensor(slots, dtype=torch.int64),
            act_bank=torch.tensor(banks, dtype=torch.int64),
            act_latent=torch.tensor(lats, dtype=torch.int64),
            act_value=torch.tensor(vals, dtype=torch.float32),
            recon_slots=recon,
            mu_slots=self.mu[experts.to(torch.int64)],
        )

    def decoder_cols(self, bank, latent):
        return self.D[bank, :, latent]


# --------------------------------------------------------------------------- #
# Subtraction math                                                             #
# --------------------------------------------------------------------------- #


def manual_encoding(n_tokens=6):
    """Hand-built sparse encoding with repeated latents across tokens."""
    return ThresholdedEncoding(
        act_token=torch.tensor([0, 0, 1, 2, 2, 2, 4], dtype=torch.int64),
        act_slot=torch.tensor([0, 1, 0, 0, 1, 2, 0], dtype=torch.int64),
        act_bank=torch.tensor([3, 5, 3, 3, 5, 7, 5], dtype=torch.int64),
        act_latent=torch.tensor([1, 2, 1, 4, 2, 0, 6], dtype=torch.int64),
        act_value=torch.tensor([0.7, 1.2, 0.3, 2.0, 0.5, 0.9, 1.1]),
        recon_slots=torch.zeros(n_tokens, 3, 4),
        mu_slots=torch.zeros(n_tokens, 3, 4),
    )


def test_compute_ablation_deltas_matches_naive():
    sae = FakeRoutedSAE(num_banks=8, f=F, d=4)
    enc = manual_encoding()
    J = [(3, 1), (5, 2), (7, 0)]
    deltas, positions = compute_ablation_deltas(enc, J, sae.f, sae.decoder_cols, 6, 4)

    j_set = set(J)
    expected = torch.zeros(6, 4)
    expected_pos = torch.zeros(6, dtype=torch.bool)
    for i in range(enc.act_token.shape[0]):
        b, l = int(enc.act_bank[i]), int(enc.act_latent[i])
        if (b, l) in j_set:
            t = int(enc.act_token[i])
            expected[t] += float(enc.act_value[i]) * sae.D[b, :, l]
            expected_pos[t] = True
    assert torch.allclose(deltas, expected, atol=1e-6)
    assert torch.equal(positions, expected_pos)
    # token 1 fired (3,1) which IS in J
    assert positions[1].item() is True
    # tokens with empty J_t: exactly-zero delta rows, positions False
    assert positions[3].item() is False and (deltas[3] == 0).all()
    assert positions[5].item() is False and (deltas[5] == 0).all()


def test_j_intersection_with_fired_only():
    sae = FakeRoutedSAE(num_banks=8, f=F, d=4)
    enc = manual_encoding()
    # J contains only latents that never fire in enc
    J = [(0, 0), (3, 2), (5, 5)]
    deltas, positions = compute_ablation_deltas(enc, J, sae.f, sae.decoder_cols, 6, 4)
    assert (deltas == 0).all() and not positions.any()

    # empty J
    deltas, positions = compute_ablation_deltas(enc, [], sae.f, sae.decoder_cols, 6, 4)
    assert (deltas == 0).all() and not positions.any()

    # same (bank, latent) fired at several tokens → all those tokens ablated
    J = [(3, 1)]
    deltas, positions = compute_ablation_deltas(enc, J, sae.f, sae.decoder_cols, 6, 4)
    assert positions.tolist() == [True, True, False, False, False, False]


# --------------------------------------------------------------------------- #
# No-reconstruction-error property (bit level) + end-to-end                    #
# --------------------------------------------------------------------------- #


def test_untouched_positions_bit_identical():
    moe = make_moe()
    model = TinyVLM(moe).eval()
    ex = make_example(seed=2)
    sae = FakeRoutedSAE()

    with torch.no_grad():
        with instrumented_capture(moe, store_dtype=torch.float32) as sink:
            forward_logits(model, ex, "cpu")
            ch = sink.drain()[0]
    enc = sae.encode_thresholded(
            ch.per_slot_contrib,
            ch.topk_idx,
            torch.zeros(ch.per_slot_contrib.shape[0], dtype=torch.int64),
            THETA,
        )
    assert enc.act_token.numel() > 0, "fixture produced no activations; lower THETA"

    # J = the single most active latent → some tokens have empty J_t
    top = int(enc.act_value.argmax())
    J = [(int(enc.act_bank[top]), int(enc.act_latent[top]))]
    t_len = ex.input_ids.shape[0]
    deltas, positions = compute_ablation_deltas(
        enc, J, sae.f, sae.decoder_cols, t_len, D_MODEL
    )
    assert positions.any() and not positions.all()

    spy = {}

    def spy_hook(module, args, output):
        spy["out"] = output.clone()

    h = moe.register_forward_hook(spy_hook)
    with torch.no_grad():
        forward_logits(model, ex, "cpu")
    clean_moe_out = spy["out"][0]
    with add_to_moe_output(moe, -deltas, positions):
        h2 = moe.register_forward_hook(spy_hook)
        with torch.no_grad():
            forward_logits(model, ex, "cpu")
        h2.remove()
    h.remove()
    ablated_moe_out = spy["out"][0]

    # positions with empty J_t are BIT-IDENTICAL: no reconstruction error is
    # injected anywhere (spec §17, changelog #2)
    assert torch.equal(ablated_moe_out[~positions], clean_moe_out[~positions])
    # ablated positions: exactly the named features' contributions removed
    assert torch.equal(
        ablated_moe_out[positions], clean_moe_out[positions] - deltas[positions]
    )


def test_run_intervention_end_to_end():
    moe = make_moe()
    model = TinyVLM(moe).eval()
    sae = FakeRoutedSAE()
    examples = [make_example(seed=i, example_id=i, domain=d)
                for i, d in enumerate(["coco", "textvqa"])]

    # empty J → clean and ablated forwards identical, deltas exactly 0
    res = run_intervention(model, sae, THETA, [], examples, moe=moe)
    assert res["overall"]["ce_delta"] == 0.0
    assert res["overall"]["acc_delta"] == 0.0
    assert all(r["n_ablated_positions"] == 0 for r in res["examples"])

    # full J: compare against a manual subtraction forward
    J = [(b, l) for b in range(N_EXPERTS) for l in range(F)]
    res = run_intervention(model, sae, THETA, J, examples, moe=moe)
    ex = examples[0]
    with torch.no_grad():
        with instrumented_capture(moe, store_dtype=torch.float32) as sink:
            forward_logits(model, ex, "cpu")
            ch = sink.drain()[0]
        enc = sae.encode_thresholded(
            ch.per_slot_contrib,
            ch.topk_idx,
            torch.zeros(ch.per_slot_contrib.shape[0], dtype=torch.int64),
            THETA,
        )
        deltas, positions = compute_ablation_deltas(
            enc, J, sae.f, sae.decoder_cols, ex.input_ids.shape[0], D_MODEL
        )
        assert positions.any()
        with add_to_moe_output(moe, -deltas, positions):
            logits_manual = forward_logits(model, ex, "cpu")
    from moe_sae.eval.fidelity import answer_ce_and_acc

    ce_manual, _ = answer_ce_and_acc(logits_manual, ex.input_ids, ex.answer_mask)
    assert res["examples"][0]["ce_ablated"] == ce_manual
    assert res["examples"][0]["n_ablated_positions"] == int(positions.sum())
    assert res["n_ablation_latents"] == len(J)
    assert set(res["per_domain"]) == {"coco", "textvqa"}


# --------------------------------------------------------------------------- #
# Enriched-set selection + matched controls (dev-dump consumers)               #
# --------------------------------------------------------------------------- #


def write_dump_shard(dump_dir, idx, act, n_tokens, token_type, experts=None):
    """act: list of (token, slot, bank, latent, value). `experts` plants the
    per-token host routing ([n, 6]) — the A13 load-floor source; defaults to
    all-zeros (every token routed to bank 0 only)."""
    dump_dir.mkdir(parents=True, exist_ok=True)
    tok, slot, bank, lat, val = (
        zip(*act) if act else ((), (), (), (), ())
    )
    n = n_tokens
    save_file(
        {
            "act_token": torch.tensor(tok, dtype=torch.int64),
            "act_slot": torch.tensor(slot, dtype=torch.uint8),
            "act_bank": torch.tensor(bank, dtype=torch.uint8),
            "act_latent": torch.tensor(lat, dtype=torch.int32),
            "act_value": torch.tensor(val, dtype=torch.float32),
            "token_type": torch.tensor(token_type, dtype=torch.uint8),
            "experts": (
                experts.to(torch.uint8)
                if experts is not None
                else torch.zeros(n, 6, dtype=torch.uint8)
            ),
            "gates": torch.zeros(n, 6, dtype=torch.float32),
            "example_id": torch.arange(n, dtype=torch.int32),
            "domain": torch.zeros(n, dtype=torch.uint8),
            "l0": torch.zeros(n, dtype=torch.int32),
            "energy": torch.zeros(n, 6, dtype=torch.float32),
        },
        str(dump_dir / f"shard_{idx:05d}.safetensors"),
    )


def make_stats(bank_counts=None):
    return SimpleNamespace(
        p_gate_visual=torch.full((64,), 1.0 / 64.0),
        bank_counts=bank_counts
        if bank_counts is not None
        else torch.full((64,), 10**6, dtype=torch.int64),
    )


def test_latent_stats_from_dump(tmp_path):
    # 3 visual tokens (type 1), 2 text tokens (type 0)
    acts = [
        (0, 0, 2, 1, 1.0),  # visual
        (1, 0, 2, 1, 2.0),  # visual, same latent
        (2, 1, 3, 0, 0.5),  # visual
        (3, 0, 0, 3, 4.0),  # text
        (4, 0, 2, 1, 1.5),  # text, same latent as visual one
    ]
    write_dump_shard(tmp_path, 0, acts, 5, [1, 1, 1, 0, 0])
    ls = latent_stats_from_dump(tmp_path, f=4)
    assert ls["n_tokens_visual"] == 3 and ls["n_tokens_text"] == 2
    idx = {int(i): r for r, i in enumerate(ls["ids"].tolist())}
    r = idx[2 * 4 + 1]  # bank 2, latent 1
    assert int(ls["fires_visual"][r]) == 2 and int(ls["fires_text"][r]) == 1
    assert float(ls["mass"][r]) == pytest.approx(4.5)
    assert float(ls["bank_mass_visual"][2]) == pytest.approx(3.0)
    assert float(ls["bank_mass_text"][2]) == pytest.approx(1.5)
    assert float(ls["bank_mass_text"][0]) == pytest.approx(4.0)


def _enrichment_acts():
    """Banks 2 and 3 carry visual mass; bank 0 is text-dominated."""
    acts = []
    token_type = []
    n = 40
    for t in range(n):
        if t < 20:  # visual tokens
            token_type.append(1)
            acts.append((t, 0, 2, 1, 3.0))  # visual-only latent (bank 2)
            acts.append((t, 1, 3, 2, 2.0))  # visual-only latent (bank 3)
            if t < 2:
                acts.append((t, 2, 2, 0, 0.1))  # weak: fires on text too
        else:  # text tokens
            token_type.append(0)
            acts.append((t, 0, 0, 3, 3.0))  # text-only latent (bank 0)
            acts.append((t, 1, 2, 0, 0.1))  # bank-2 latent firing mostly on text
            acts.append((t, 2, 2, 3, 1.0))  # bank-2 latent firing ONLY on text
    return acts, token_type, n


def test_select_enriched_latents(tmp_path):
    acts, token_type, n = _enrichment_acts()
    write_dump_shard(tmp_path, 0, acts, n, token_type)

    J = select_enriched_latents(
        tmp_path, make_stats(), f=4, top_n_experts=2, quantile=0.5, load_floor=0
    )
    assert (2, 1) in J and (3, 2) in J  # visual-enriched latents in enriched banks
    assert (0, 3) not in J  # text bank not among enriched banks
    assert (2, 3) not in J  # never fired on visual → excluded
    assert (2, 0) not in J  # text-dominated: below the enrichment quantile
    # deterministic
    assert J == select_enriched_latents(
        tmp_path, make_stats(), f=4, top_n_experts=2, quantile=0.5, load_floor=0
    )

    # load floor: exclude bank 3 → its latents cannot be selected (explicit
    # counts + absolute floor: the A13 override/audit path)
    counts = torch.full((64,), 10**6, dtype=torch.int64)
    counts[3] = 0
    J2 = select_enriched_latents(
        tmp_path, make_stats(), f=4, top_n_experts=2, quantile=0.5,
        load_floor=500_000, bank_counts=counts,
    )
    assert all(b != 3 for b, _ in J2)


def test_select_enriched_latents_bank_counts_override(tmp_path):
    # An explicit bank_counts array is what reaches the floor comparison
    # (absolute-override/audit path).
    import numpy as np

    acts, token_type, n = _enrichment_acts()
    write_dump_shard(tmp_path, 0, acts, n, token_type)

    counts = np.full(64, 10**6, dtype=np.int64)
    counts[3] = 0  # override excludes bank 3
    stats = make_stats()  # stats-pass loads would admit EVERY bank
    J = select_enriched_latents(
        tmp_path, stats, f=4, top_n_experts=2, quantile=0.5,
        load_floor=500_000, bank_counts=counts,
    )
    assert J and all(b != 3 for b, _ in J)


def test_select_enriched_latents_split_proportional_floor(tmp_path):
    """A13: the DEFAULT floor is 10% of the split-uniform expectation
    (split_tokens×6)/64, applied to the dump's OWN host-routing counts — a
    bank starved in the analysis split is excluded even when training loads
    or the stats pass would admit it."""
    from moe_sae.eval.intervention import (
        dump_split_bank_counts,
        split_proportional_floor,
    )

    acts, token_type, n = _enrichment_acts()
    # host routing: bank 3 receives ZERO routed tokens in this split; the
    # other slots cycle over banks {0, 1, 2, 4, 5, 6}.
    cycle = [0, 1, 2, 4, 5, 6]
    experts = torch.tensor(
        [[cycle[(t + s) % 6] for s in range(6)] for t in range(n)], dtype=torch.uint8
    )
    write_dump_shard(tmp_path, 0, acts, n, token_type, experts=experts)
    stats = make_stats()  # healthy stats-pass loads everywhere

    counts, n_tok = dump_split_bank_counts(tmp_path)
    assert n_tok == n and int(counts.sum()) == 6 * n
    assert int(counts[3]) == 0 and int(counts[2]) == n
    assert split_proportional_floor(n) == pytest.approx(0.10 * n * 6 / 64)

    J = select_enriched_latents(tmp_path, stats, f=4, top_n_experts=2, quantile=0.5)
    assert (2, 1) in J
    assert all(b != 3 for b, _ in J), "split-starved bank must be below the A13 floor"

    # absolute override (kept per A13): floor 0 admits every bank again
    J2 = select_enriched_latents(
        tmp_path, stats, f=4, top_n_experts=2, quantile=0.5, load_floor=0.0
    )
    assert (3, 2) in J2


def test_save_load_ablation_set(tmp_path):
    J = [(2, 1), (3, 2)]
    save_ablation_set(J, tmp_path / "j.json")
    assert load_ablation_set(tmp_path / "j.json") == J


def test_matched_control_sets(tmp_path):
    # 30 latents in bank 1 with identical stats to the 3 J latents in bank 2:
    # matching should be exact-quality and within tolerance everywhere.
    acts = []
    token_type = [1] * 10 + [0] * 10
    for t in range(10):  # each latent fires on every visual token with value 1.0
        for l in range(3):
            acts.append((t, 0, 2, l, 1.0))
        for l in range(30):
            acts.append((t, 1, 1, l, 1.0))
    write_dump_shard(tmp_path, 0, acts, 20, token_type)

    J = [(2, 0), (2, 1), (2, 2)]
    controls, report = matched_control_sets(
        J, tmp_path, make_stats(), f=32, n_controls=4, seed=11
    )
    assert len(controls) == 4
    j_set = set(J)
    for control, srep in zip(controls, report["sets"]):
        assert len(control) == len(J)
        assert len(set(control)) == len(J)  # no duplicates within a set
        assert not (set(control) & j_set)  # disjoint from J
        assert srep["n_out_of_tolerance"] == 0
        assert srep["total_mass_ratio"] == pytest.approx(1.0)
        assert srep["total_fires_ratio"] == pytest.approx(1.0)
    # determinism under the same seed
    controls2, _ = matched_control_sets(
        J, tmp_path, make_stats(), f=32, n_controls=4, seed=11
    )
    assert controls == controls2


# --------------------------------------------------------------------------- #
# Post-hoc removed-vector-norm measurement (spec §17 fifth criterion)          #
# --------------------------------------------------------------------------- #


def _fire_all_encoding(pairs, n_tok, d):
    """Every (bank, latent) in `pairs` fires at every token with value 1.0."""
    rows = [(t, s, b, l) for t in range(n_tok) for s, (b, l) in enumerate(pairs)]
    tok, slot, bank, lat = zip(*rows)
    return ThresholdedEncoding(
        act_token=torch.tensor(tok, dtype=torch.int64),
        act_slot=torch.tensor(slot, dtype=torch.int64),
        act_bank=torch.tensor(bank, dtype=torch.int64),
        act_latent=torch.tensor(lat, dtype=torch.int64),
        act_value=torch.ones(len(rows), dtype=torch.float32),
        recon_slots=torch.zeros(n_tok, len(pairs), d),
        mu_slots=torch.zeros(n_tok, len(pairs), d),
    )


def test_removed_norm_report_equal_mass_different_norms():
    # Regression: two latent sets with EQUAL total coefficient mass (the
    # per-latent proxy) but different per-token removed-vector norms —
    # aligned vs orthogonal decoder columns. The report must expose the
    # difference and flag the mismatched set.
    d, f = 4, 8
    e1 = torch.zeros(d)
    e1[0] = 1.0
    e2 = torch.zeros(d)
    e2[1] = 1.0
    D = torch.zeros(3, d, f)
    D[0, :, 0] = e1  # bank 0: two ALIGNED unit columns (like an enriched bank)
    D[0, :, 1] = e1
    D[1, :, 0] = e1  # bank 1: two ORTHOGONAL unit columns
    D[1, :, 1] = e2
    D[2, :, 0] = e2  # bank 2: aligned — a genuinely norm-matched control
    D[2, :, 1] = e2

    def dcols(bank, latent):
        return D[bank.to(torch.int64), :, latent.to(torch.int64)]

    J = [(0, 0), (0, 1)]  # per-token removed vector 2·e1 → norm 2
    ctrl_orth = [(1, 0), (1, 1)]  # same mass (2.0), removed norm √2
    ctrl_good = [(2, 0), (2, 1)]  # same mass, same norm 2
    enc = _fire_all_encoding(J + ctrl_orth + ctrl_good, n_tok=3, d=d)

    report = removed_norm_report(J, [ctrl_orth, ctrl_good], [enc], f, dcols, d)
    assert report["J"]["n_positions"] == 3
    assert report["J"]["mean"] == pytest.approx(2.0)
    assert report["J"]["median"] == pytest.approx(2.0)
    assert report["J"]["p90"] == pytest.approx(2.0)

    orth = report["sets"][0]
    assert orth["mean"] == pytest.approx(2**0.5)
    assert orth["mean_ratio_vs_J"] == pytest.approx(2**0.5 / 2.0)
    assert orth["flag_removed_norm"] is True  # ~29% below J > 20% tolerance

    good = report["sets"][1]
    assert good["mean_ratio_vs_J"] == pytest.approx(1.0)
    assert good["flag_removed_norm"] is False

    # a wider tolerance unflags the orthogonal set
    loose = removed_norm_report(
        J, [ctrl_orth], [enc], f, dcols, d, norm_tolerance=0.5
    )
    assert loose["sets"][0]["flag_removed_norm"] is False

    # a control that never fires cannot be compared → flagged, 0 positions
    never = removed_norm_report(J, [[(1, 5)]], [enc], f, dcols, d)
    assert never["sets"][0]["n_positions"] == 0
    assert never["sets"][0]["mean"] is None
    assert never["sets"][0]["flag_removed_norm"] is True


def test_matched_control_sets_embeds_removed_norms(tmp_path):
    # Same dump fixture as test_matched_control_sets: equal per-latent mass,
    # so construction-time matching is perfect — but J's decoder columns are
    # aligned while the control bank's are orthogonal, so the measured
    # per-token removed norms differ (3 vs √3) and every set is flagged.
    acts = []
    token_type = [1] * 10 + [0] * 10
    for t in range(10):
        for l in range(3):
            acts.append((t, 0, 2, l, 1.0))
        for l in range(30):
            acts.append((t, 1, 1, l, 1.0))
    write_dump_shard(tmp_path, 0, acts, 20, token_type)

    d, f = 32, 32
    D = torch.zeros(3, d, f)
    for l in range(3):
        D[2, 0, l] = 1.0  # J bank: all columns aligned to e0
    for l in range(30):
        D[1, 1 + l, l] = 1.0  # control bank: mutually orthogonal columns

    def dcols(bank, latent):
        return D[bank.to(torch.int64), :, latent.to(torch.int64)]

    J = [(2, 0), (2, 1), (2, 2)]
    pairs = J + [(1, l) for l in range(30)]
    enc = _fire_all_encoding(pairs, n_tok=2, d=d)

    controls, report = matched_control_sets(
        J, tmp_path, make_stats(), f=f, n_controls=3, seed=11,
        decoder_cols=dcols, encoded_samples=[enc], d=d,
    )
    # per-latent proxy matching still reports parity...
    assert all(sr["total_mass_ratio"] == pytest.approx(1.0) for sr in report["sets"])
    # ...but the post-hoc token-level measurement exposes the norm mismatch
    rn = report["removed_norms"]
    assert rn["norm_tolerance"] == 0.2
    assert rn["J"]["mean"] == pytest.approx(3.0)
    assert len(rn["sets"]) == 3
    for s in rn["sets"]:
        assert s["mean"] == pytest.approx(3**0.5)
        assert s["mean_ratio_vs_J"] == pytest.approx(3**0.5 / 3.0)
        assert s["flag_removed_norm"] is True

    # without decoder access the report simply omits the measurement
    _, report2 = matched_control_sets(J, tmp_path, make_stats(), f=f, n_controls=2)
    assert "removed_norms" not in report2
