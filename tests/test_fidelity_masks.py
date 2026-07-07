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


def make_example(t=12, n_answer=3, example_id=0, domain="coco", seed=0):
    g = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(0, VOCAB, (t,), generator=g)
    answer_mask = torch.zeros(t, dtype=torch.bool)
    answer_mask[-n_answer:] = True
    visual_mask = torch.zeros(t, dtype=torch.bool)
    visual_mask[1:5] = True
    return StubEvalExample(input_ids, answer_mask, visual_mask, example_id, domain)


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
        res = run_fidelity(model, None, PerfectSAE(), 0.0, examples, mode=mode, moe=moe)
        assert res["overall"]["n"] == 3
        assert res["overall"]["ce_delta"] == 0.0
        assert res["overall"]["acc_delta"] == 0.0
        assert set(res["per_domain"]) == {"coco", "textvqa"}
        assert res["per_domain"]["coco"]["n"] == 2


def test_run_fidelity_matches_manual_replacement():
    moe = make_moe()
    model = TinyVLM(moe).eval()
    ex = make_example(seed=5, example_id=7, domain="chartqa")
    sae = MeanSAE()

    res = run_fidelity(model, None, sae, 0.0, [ex], mode="matched", moe=moe)
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

    ce_c, acc_c = answer_ce_and_acc(logits_clean, ex.input_ids, ex.answer_mask)
    ce_r, acc_r = answer_ce_and_acc(logits_rep, ex.input_ids, ex.answer_mask)
    assert rec["ce_clean"] == ce_c
    assert rec["ce_replaced"] == ce_r
    assert rec["acc_clean"] == acc_c and rec["acc_replaced"] == acc_r
    assert rec["n_replaced"] == int(mask.sum())
    # removing the routed contribution must actually change the answer CE
    assert rec["ce_delta"] != 0.0


def test_run_fidelity_matched_vs_full_differ():
    moe = make_moe()
    model = TinyVLM(moe).eval()
    ex = make_example(seed=9, example_id=1, domain="ai2d")
    sae = MeanSAE()
    r_matched = run_fidelity(model, None, sae, 0.0, [ex], mode="matched", moe=moe)
    r_full = run_fidelity(model, None, sae, 0.0, [ex], mode="full", moe=moe)
    # full mode additionally perturbs answer/BOS positions
    assert r_matched["examples"][0]["n_replaced"] < r_full["examples"][0]["n_replaced"]
    assert r_matched["examples"][0]["ce_replaced"] != r_full["examples"][0]["ce_replaced"]
