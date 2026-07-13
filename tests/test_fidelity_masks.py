"""Downstream-fidelity mechanics (spec §13, changelog #1): replaced-position
mask logic, the routed-swap hook identity (bit-level), and CE math."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from moe_sae.capture.hooks import instrumented_capture
from moe_sae.eval.fidelity import (
    add_to_moe_output,
    answer_ce_and_acc,
    forward_logits,
    greedy_generate_with_replacement,
    normalize_answer,
    prompt_length,
    replacement_mask,
    run_fidelity,
)
from moe_sae.eval.reconstruction import ThresholdedEncoding
from moe_sae.vendor.deepseek_vl2.configuration_deepseek import DeepseekV2Config
from moe_sae.vendor.deepseek_vl2.modeling_deepseek import DeepseekV2MoE

D_MODEL = 64
N_EXPERTS = 16
TOP_K = 4
VOCAB = 32


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
    """Minimal stand-in for the language stack: embed → MoE block → head."""

    def __init__(self, moe, seed=1):
        super().__init__()
        torch.manual_seed(seed)
        self.embed = torch.nn.Embedding(VOCAB, D_MODEL)
        self.moe = moe
        self.head = torch.nn.Linear(D_MODEL, VOCAB)

    def forward(self, input_ids, **kwargs):
        h = self.embed(input_ids)
        h = h + self.moe(h)  # residual added outside the MoE, as in the model
        return SimpleNamespace(logits=self.head(h))


@dataclass
class StubEvalExample:
    """Duck-typed moe_sae.data.evalsets.EvalExample (contract)."""

    input_ids: torch.Tensor
    answer_mask: torch.Tensor
    visual_mask: torch.Tensor
    example_id: int
    domain: str
    answer_text: str = ""
    images: None = None
    images_seq_mask: None = None
    images_spatial_crop: None = None


def make_example(t=12, n_answer=3, example_id=0, domain="coco", seed=0, answer_text=""):
    g = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(0, VOCAB, (t,), generator=g)
    answer_mask = torch.zeros(t, dtype=torch.bool)
    answer_mask[-n_answer:] = True
    visual_mask = torch.zeros(t, dtype=torch.bool)
    visual_mask[1:5] = True
    return StubEvalExample(
        input_ids, answer_mask, visual_mask, example_id, domain, answer_text
    )


class StubTokenizer:
    """Decode-capable stand-in (A8 accuracy is generated-text exact match)."""

    eos_token_id = None

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(i)) for i in ids)


class PerfectSAE:
    """Identity-ish fake: reconstruction == true contribution (Δ == 0)."""

    condition = "A"
    num_banks = N_EXPERTS
    f = 4

    def encode_thresholded(self, c, experts, example_id, theta_star):
        c = c.to(torch.float32)
        empty = torch.zeros(0, dtype=torch.int64)
        return ThresholdedEncoding(
            act_token=empty,
            act_slot=empty.clone(),
            act_bank=empty.clone(),
            act_latent=empty.clone(),
            act_value=torch.zeros(0, dtype=torch.float32),
            recon_slots=c.clone(),
            mu_slots=torch.zeros_like(c),
        )

    def decoder_cols(self, bank, latent):
        return torch.zeros(bank.shape[0], D_MODEL)


class MeanSAE(PerfectSAE):
    """Fake reconstructing μ = 0 everywhere: Δ = −c_route at replaced positions."""

    def encode_thresholded(self, c, experts, example_id, theta_star):
        enc = super().encode_thresholded(c, experts, example_id, theta_star)
        enc.recon_slots = torch.zeros_like(enc.recon_slots)
        return enc


# --------------------------------------------------------------------------- #
# Mask logic                                                                   #
# --------------------------------------------------------------------------- #


def test_replacement_mask_matched_and_full():
    answer = torch.tensor([False, False, True, False, True, True])
    matched = replacement_mask(answer, "matched")
    # answer spans untouched, BOS untouched, everything else replaced
    assert matched.tolist() == [False, True, False, True, False, False]
    full = replacement_mask(answer, "full")
    assert full.all() and full.shape == answer.shape
    # input mask is not mutated
    assert answer[0].item() is False


# --------------------------------------------------------------------------- #
# Hook swap identity (bit-level)                                               #
# --------------------------------------------------------------------------- #


def test_add_to_moe_output_swap_identity():
    moe = make_moe()
    t = 10
    x = torch.randn(1, t, D_MODEL)

    spy = {}

    def spy_hook(module, args, output):
        spy["out"] = output.clone()

    h = moe.register_forward_hook(spy_hook)
    with torch.no_grad():
        moe(x)
    clean = spy["out"][0]
    h.remove()

    delta = torch.randn(t, D_MODEL)
    mask = torch.zeros(t, dtype=torch.bool)
    mask[[1, 4, 7]] = True
    with add_to_moe_output(moe, delta, mask):
        h = moe.register_forward_hook(spy_hook)  # after: sees modified output
        with torch.no_grad():
            out_final = moe(x)
        h.remove()
    modified = spy["out"][0]

    # untouched rows are bit-identical; touched rows equal clean + delta bitwise
    assert torch.equal(modified[~mask], clean[~mask])
    assert torch.equal(modified[mask], clean[mask] + delta[mask])
    # and the hook's return value is what the module's caller receives
    assert torch.equal(out_final[0], modified)
    # hook removed on context exit
    with torch.no_grad():
        again = moe(x)
    assert torch.equal(again[0], clean)


def test_swap_replaces_routed_part_only():
    """output + (ĉ_route − c_route) == ĉ_route + shared: verified by capturing
    c_route with the instrumented path and reconstructing the identity."""
    moe = make_moe()
    t = 8
    x = torch.randn(1, t, D_MODEL)
    with instrumented_capture(moe, store_dtype=torch.float32) as sink:
        with torch.no_grad():
            out_clean = moe(x)
        ch = sink.drain()[0]
    c_route = ch.per_slot_contrib.sum(dim=1)  # [t, d]
    shared = out_clean[0] - c_route  # shared-expert part (residual is outside)

    recon_route = torch.randn(t, D_MODEL)  # arbitrary "SAE reconstruction"
    mask = torch.ones(t, dtype=torch.bool)
    with add_to_moe_output(moe, recon_route - c_route, mask):
        with torch.no_grad():
            out_swapped = moe(x)
    assert torch.allclose(out_swapped[0], recon_route + shared, atol=1e-5)


# --------------------------------------------------------------------------- #
# CE math                                                                      #
# --------------------------------------------------------------------------- #


def test_answer_ce_and_acc_against_manual():
    g = torch.Generator().manual_seed(3)
    t, v = 9, VOCAB
    logits = torch.randn(t, v, generator=g)
    input_ids = torch.randint(0, v, (t,), generator=g)
    answer_mask = torch.zeros(t, dtype=torch.bool)
    answer_mask[[4, 6, 8]] = True

    ce, acc = answer_ce_and_acc(logits, input_ids, answer_mask)

    ces, hits = [], []
    for pos in (4, 6, 8):
        logp = torch.log_softmax(logits[pos - 1].to(torch.float64), dim=-1)
        ces.append(-float(logp[input_ids[pos]]))
        hits.append(int(logits[pos - 1].argmax()) == int(input_ids[pos]))
    assert ce == pytest.approx(sum(ces) / 3, rel=1e-5)
    assert acc == pytest.approx(sum(hits) / 3)


# --------------------------------------------------------------------------- #
# End-to-end run_fidelity on the tiny model                                    #
# --------------------------------------------------------------------------- #


def test_run_fidelity_perfect_sae_is_noop():
    moe = make_moe()
    model = TinyVLM(moe).eval()
    examples = [make_example(seed=i, example_id=i, domain=d)
                for i, d in enumerate(["coco", "coco", "textvqa"])]
    for mode in ("matched", "full"):
        res = run_fidelity(
            model, StubTokenizer(), PerfectSAE(), 0.0, examples, mode=mode, moe=moe
        )
        assert res["overall"]["n"] == 3
        assert res["overall"]["ce_delta"] == 0.0
        # A8: accuracy is generation-based; a zero delta leaves generation
        # identical to the clean run, so the delta is exactly zero.
        assert res["overall"]["acc_delta"] == 0.0
        for rec in res["examples"]:
            assert rec["generated_clean"] == rec["generated_replaced"]
        assert set(res["per_domain"]) == {"coco", "textvqa"}
        assert res["per_domain"]["coco"]["n"] == 2


def test_run_fidelity_matches_manual_replacement():
    moe = make_moe()
    model = TinyVLM(moe).eval()
    ex = make_example(seed=5, example_id=7, domain="chartqa")
    sae = MeanSAE()
    tok = StubTokenizer()

    res = run_fidelity(model, tok, sae, 0.0, [ex], mode="matched", moe=moe)
    rec = res["examples"][0]

    # manual: capture c_route, delta = 0 − c_route at matched positions
    with torch.no_grad():
        with instrumented_capture(moe, store_dtype=torch.float32) as sink:
            logits_clean = forward_logits(model, ex, "cpu")
            ch = sink.drain()[0]
        delta = -ch.per_slot_contrib.sum(dim=1)
        mask = replacement_mask(ex.answer_mask, "matched")
        with add_to_moe_output(moe, delta, mask):
            logits_rep = forward_logits(model, ex, "cpu")

    ce_c, _ = answer_ce_and_acc(logits_clean, ex.input_ids, ex.answer_mask)
    ce_r, _ = answer_ce_and_acc(logits_rep, ex.input_ids, ex.answer_mask)
    # A8: CE stays teacher-forced
    assert rec["ce_clean"] == ce_c
    assert rec["ce_replaced"] == ce_r
    assert rec["n_replaced"] == int(mask.sum())
    # removing the routed contribution must actually change the answer CE
    assert rec["ce_delta"] != 0.0

    # A8: accuracy is greedy-generation exact match — recompute manually
    limit = int(ex.answer_mask.sum()) + 8
    with torch.no_grad():
        gen_clean = greedy_generate_with_replacement(
            model, ex, torch.zeros_like(delta), torch.zeros_like(mask),
            moe, "cpu", limit,
        )
        gen_rep = greedy_generate_with_replacement(
            model, ex, delta, mask, moe, "cpu", limit
        )
    want = normalize_answer(ex.answer_text)
    assert rec["generated_clean"] == normalize_answer(tok.decode(gen_clean))
    assert rec["generated_replaced"] == normalize_answer(tok.decode(gen_rep))
    assert rec["acc_clean"] == float(rec["generated_clean"] == want)
    assert rec["acc_replaced"] == float(rec["generated_replaced"] == want)


def test_run_fidelity_generation_accuracy_exact_match():
    """A8: an example whose answer_text equals the model's clean greedy
    continuation scores acc_clean == 1.0; a full c_route removal perturbs the
    forward yet accuracy stays a produced, in-range metric."""
    moe = make_moe()
    model = TinyVLM(moe).eval()
    ex = make_example(seed=11, example_id=3, domain="coco")
    limit = int(ex.answer_mask.sum()) + 8
    d = D_MODEL
    with torch.no_grad():
        gen = greedy_generate_with_replacement(
            model, ex, torch.zeros(ex.input_ids.shape[0], d),
            torch.zeros(ex.input_ids.shape[0], dtype=torch.bool),
            moe, "cpu", limit,
        )
    ex.answer_text = StubTokenizer().decode(gen)  # plant the "right" answer

    res = run_fidelity(model, StubTokenizer(), MeanSAE(), 0.0, [ex], mode="matched", moe=moe)
    rec = res["examples"][0]
    assert rec["acc_clean"] == 1.0  # clean generation reproduces answer_text
    assert rec["acc_replaced"] in (0.0, 1.0)  # exact-match metric produced
    assert res["overall"]["acc_delta"] == rec["acc_replaced"] - 1.0


def test_generation_replacement_masks_prefill_only(monkeypatch):
    """A8 mechanics: during generation the replacement mask covers ONLY the
    frozen prompt positions — generated tokens are never replaced (mask False,
    delta rows zero at every step)."""
    import moe_sae.eval.fidelity as fid

    moe = make_moe()
    model = TinyVLM(moe).eval()
    ex = make_example(t=12, n_answer=3, seed=3)
    mask = replacement_mask(ex.answer_mask, "matched")
    g = torch.Generator().manual_seed(0)
    delta = torch.randn(12, D_MODEL, generator=g)
    plen = prompt_length(ex.answer_mask)
    assert plen == 9

    seen = []
    real = fid.add_to_moe_output

    def spy(moe_, delta_, positions_):
        seen.append((delta_.clone(), positions_.clone()))
        return real(moe_, delta_, positions_)

    monkeypatch.setattr(fid, "add_to_moe_output", spy)
    with torch.no_grad():
        gen = fid.greedy_generate_with_replacement(
            model, ex, delta, mask, moe, "cpu", max_new_tokens=4
        )
    assert len(gen) == 4 and len(seen) == 4
    for step, (d_, p_) in enumerate(seen):
        assert p_.shape[0] == plen + step
        assert torch.equal(p_[:plen], mask[:plen])  # prefill mask respected
        assert not p_[plen:].any()  # generated positions never replaced
        assert torch.equal(d_[:plen], delta[:plen])
        assert (d_[plen:] == 0).all()


def test_generation_eos_and_unhooked_equivalence():
    """Greedy generation with a zero delta equals naive unhooked greedy
    decoding; eos_token_id terminates generation early."""
    moe = make_moe()
    model = TinyVLM(moe).eval()
    ex = make_example(t=10, n_answer=2, seed=6)
    plen = prompt_length(ex.answer_mask)
    zero_d = torch.zeros(10, D_MODEL)
    no_mask = torch.zeros(10, dtype=torch.bool)
    with torch.no_grad():
        gen = greedy_generate_with_replacement(
            model, ex, zero_d, no_mask, moe, "cpu", max_new_tokens=5
        )
        # naive reference: full re-forward argmax without any hook
        ids = ex.input_ids[:plen].clone()
        want = []
        for _ in range(5):
            logits = model(input_ids=ids.unsqueeze(0)).logits[0]
            nxt = int(logits[-1].argmax())
            want.append(nxt)
            ids = torch.cat([ids, torch.tensor([nxt])])
    assert gen == want
    with torch.no_grad():
        gen_eos = greedy_generate_with_replacement(
            model, ex, zero_d, no_mask, moe, "cpu", max_new_tokens=5,
            eos_token_id=want[0],
        )
    assert gen_eos == [want[0]]  # stopped at EOS


def test_run_fidelity_matched_vs_full_differ():
    moe = make_moe()
    model = TinyVLM(moe).eval()
    ex = make_example(seed=9, example_id=1, domain="ai2d")
    sae = MeanSAE()
    tok = StubTokenizer()
    r_matched = run_fidelity(model, tok, sae, 0.0, [ex], mode="matched", moe=moe)
    r_full = run_fidelity(model, tok, sae, 0.0, [ex], mode="full", moe=moe)
    # full mode additionally perturbs answer/BOS positions
    assert r_matched["examples"][0]["n_replaced"] < r_full["examples"][0]["n_replaced"]
    assert r_matched["examples"][0]["ce_replaced"] != r_full["examples"][0]["ce_replaced"]
