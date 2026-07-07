"""Streamed FP32 reference forward (spec §3 gate machinery): the layer-streamed
forward must reproduce a direct full-model FP32 forward exactly, on tiny random
checkpoints (never the real model — 16 GB host constraint)."""

from contextlib import contextmanager

import pytest
import torch

from moe_sae.capture.hooks import recompute_contributions, reference_capture
from moe_sae.capture.streamed import (
    CheckpointStreamer,
    build_skeleton,
    dematerialize,
    fp32_prefix_forward,
    materialize,
)
from moe_sae.vendor.deepseek_vl2.configuration_deepseek import DeepseekV2Config
from moe_sae.vendor.deepseek_vl2.modeling_deepseek import DeepseekV2ForCausalLM
from moe_sae.vendor.deepseek_vl2.modeling_deepseek_vl_v2 import (
    DeepseekVLV2Config,
    DeepseekVLV2ForCausalLM,
)

VOCAB = 128
HIDDEN = 32


def tiny_lang_kwargs(use_mla: bool) -> dict:
    """3 layers: dense layer 0 (first_k_dense_replace=1), MoE layers 1-2 with
    8 routed experts / top-3 / 2 shared — the frozen model's shape in miniature."""
    return dict(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=2,
        n_routed_experts=8,
        num_experts_per_tok=3,
        n_shared_experts=2,
        first_k_dense_replace=1,
        moe_layer_freq=1,
        topk_method="greedy",
        scoring_func="softmax",
        n_group=1,
        topk_group=1,
        norm_topk_prob=False,
        max_position_embeddings=64,
        kv_lora_rank=16,
        q_lora_rank=None,
        qk_rope_head_dim=8,
        qk_nope_head_dim=8,
        v_head_dim=16,
        use_mla=use_mla,
        bos_token_id=0,
        eos_token_id=1,
    )


def make_tiny_lm(use_mla: bool, seed: int = 0) -> DeepseekV2ForCausalLM:
    torch.manual_seed(seed)
    cfg = DeepseekV2Config(**tiny_lang_kwargs(use_mla))
    cfg._attn_implementation = "eager"
    return DeepseekV2ForCausalLM(cfg).float().eval()


def make_tiny_vl(seed: int = 0) -> DeepseekVLV2ForCausalLM:
    """Tiny full VL model: 1-block ViT (2x2 patches), downsample-2 projector
    (1x1 image grid -> 5 image tokens per (1,1)-crop image), tiny language stack."""
    torch.manual_seed(seed)
    cfg = DeepseekVLV2Config(
        tile_tag="2D",
        global_view_pos="head",
        candidate_resolutions=((32, 32),),
        vision_config=dict(
            model_name="tiny_vit",
            image_size=32,
            patch_size=16,
            width=24,
            layers=1,
            heads=2,
            mlp_ratio=2,
            global_pool="map",
            ignore_head=True,
            class_token=False,
            num_classes=0,
            use_checkpoint=False,
        ),
        projector_config=dict(
            projector_type="downsample_mlp_gelu",
            input_dim=24,
            n_embed=HIDDEN,
            depth=2,
            mlp_ratio=1,
            downsample_ratio=2,
        ),
        language_config=tiny_lang_kwargs(use_mla=True),
    )
    cfg._attn_implementation = "eager"
    cfg.language_config._attn_implementation = "eager"
    return DeepseekVLV2ForCausalLM(cfg).float().eval()


def rel_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / b.norm())


def multimodal_inputs(seed: int = 1):
    """One image, (1,1) spatial crop -> 2 tiles, 5 image-token positions."""
    torch.manual_seed(seed)
    n_image_tokens = 5  # h*(w+1) + separator + th*h*(tw*w+1) with h=w=1
    T = 12
    ids = torch.randint(0, VOCAB, (1, T))
    mask = torch.zeros(1, T, dtype=torch.bool)
    mask[0, 3 : 3 + n_image_tokens] = True
    images = torch.randn(1, 2, 3, 32, 32)
    crop = torch.tensor([[[1, 1]]], dtype=torch.long)
    return ids, images, mask, crop


# --------------------------------------------------------------------------- #
# CheckpointStreamer                                                           #
# --------------------------------------------------------------------------- #


def test_streamer_single_file_and_prefix(tmp_path):
    model = make_tiny_lm(use_mla=True)
    model.save_pretrained(tmp_path / "lm")
    streamer = CheckpointStreamer(tmp_path / "lm")

    state = streamer.load_submodule_state("model.layers.1", torch.float32)
    want = {
        k[len("model.layers.1."):]: v
        for k, v in model.state_dict().items()
        if k.startswith("model.layers.1.")
    }
    assert set(state) == set(want)
    for k in want:
        assert torch.equal(state[k], want[k].float()), k

    with pytest.raises(KeyError):
        streamer.load_submodule_state("model.layers.99")
    with pytest.raises(KeyError):
        streamer.load_tensor("nonexistent.weight")


def test_streamer_sharded_index_and_upcast(tmp_path):
    model = make_tiny_lm(use_mla=True).to(torch.bfloat16)
    model.save_pretrained(tmp_path / "bf16", max_shard_size="20KB")
    assert (tmp_path / "bf16" / "model.safetensors.index.json").exists(), (
        "test intended to exercise the sharded-index path"
    )
    streamer = CheckpointStreamer(tmp_path / "bf16")
    state = streamer.load_submodule_state("model.layers.0", torch.float32)
    ref = model.model.layers[0].state_dict()
    assert set(state) == set(ref)
    for k, t in state.items():
        assert t.dtype == torch.float32, k
        assert torch.equal(t, ref[k].to(torch.float32)), k


# --------------------------------------------------------------------------- #
# materialize / dematerialize round trip                                       #
# --------------------------------------------------------------------------- #


def test_materialize_dematerialize_cycle(tmp_path):
    model = make_tiny_lm(use_mla=True)
    model.save_pretrained(tmp_path / "lm")
    streamer = CheckpointStreamer(tmp_path / "lm")
    skeleton = build_skeleton(tmp_path / "lm", model_cls=DeepseekV2ForCausalLM)

    layer = skeleton.model.layers[1]
    assert next(layer.parameters()).is_meta
    state = streamer.load_submodule_state("model.layers.1", torch.float32)
    materialize(layer, state, "cpu", torch.float32)
    assert not any(p.is_meta for p in layer.parameters())
    assert not any(b.is_meta for b in layer.buffers())
    assert all(p.dtype == torch.float32 for p in layer.parameters())

    dematerialize(layer)
    assert all(p.is_meta for p in layer.parameters())
    # non-persistent buffers (rotary caches) survive for re-materialization
    materialize(layer, state, "cpu", torch.float32)
    assert not any(b.is_meta for b in layer.buffers())


# --------------------------------------------------------------------------- #
# FP32 streamed forward == direct full-model forward                           #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("use_mla", [True, False])
def test_prefix_forward_matches_direct(tmp_path, use_mla):
    """Dense layer 0 + MoE layer 1, both attention variants (the real tiny
    checkpoint uses MHA; MLA covers the other DeepseekV2 configuration)."""
    model = make_tiny_lm(use_mla=use_mla)
    model.save_pretrained(tmp_path / "lm")
    torch.manual_seed(1)
    ids = torch.randint(0, VOCAB, (1, 21))

    with torch.no_grad():
        ref = model(
            input_ids=ids, output_hidden_states=True, use_cache=False, return_dict=True
        ).hidden_states

    streamer = CheckpointStreamer(tmp_path / "lm")
    skeleton = build_skeleton(tmp_path / "lm", model_cls=DeepseekV2ForCausalLM)
    out = fp32_prefix_forward(
        streamer, skeleton, {"input_ids": ids}, n_layers=2, device="cpu"
    )
    # hidden_states[2] is the input to layer 2 == output of layer 1 (target)
    assert out.shape == ref[2].shape
    assert rel_diff(out, ref[2]) < 1e-6


def test_prefix_forward_rematerializes(tmp_path):
    """A second call re-materializes every dematerialized submodule."""
    model = make_tiny_lm(use_mla=True)
    model.save_pretrained(tmp_path / "lm")
    streamer = CheckpointStreamer(tmp_path / "lm")
    skeleton = build_skeleton(tmp_path / "lm", model_cls=DeepseekV2ForCausalLM)

    for seed in (1, 2):
        torch.manual_seed(seed)
        ids = torch.randint(0, VOCAB, (1, 9 + seed))
        with torch.no_grad():
            ref = model(
                input_ids=ids, output_hidden_states=True, use_cache=False,
                return_dict=True,
            ).hidden_states[2]
        out = fp32_prefix_forward(
            streamer, skeleton, {"input_ids": ids}, n_layers=2, device="cpu"
        )
        assert rel_diff(out, ref) < 1e-6


def test_prefix_forward_example_list(tmp_path):
    """List input (per-layer IO amortization) matches per-example calls."""
    model = make_tiny_lm(use_mla=True)
    model.save_pretrained(tmp_path / "lm")
    streamer = CheckpointStreamer(tmp_path / "lm")
    skeleton = build_skeleton(tmp_path / "lm", model_cls=DeepseekV2ForCausalLM)

    torch.manual_seed(3)
    batches = [{"input_ids": torch.randint(0, VOCAB, (1, t))} for t in (7, 13, 21)]
    outs = fp32_prefix_forward(streamer, skeleton, batches, n_layers=2, device="cpu")
    assert isinstance(outs, list) and len(outs) == 3
    for b, o in zip(batches, outs):
        single = fp32_prefix_forward(streamer, skeleton, b, n_layers=2, device="cpu")
        assert torch.equal(o, single)


def test_on_target_moe_capture(tmp_path):
    """The context factory wraps the target layer's MoE: the reference capture
    plus independent recompute must agree with the model's own aggregate."""
    model = make_tiny_lm(use_mla=True)
    model.save_pretrained(tmp_path / "lm")
    streamer = CheckpointStreamer(tmp_path / "lm")
    skeleton = build_skeleton(tmp_path / "lm", model_cls=DeepseekV2ForCausalLM)

    records = []

    @contextmanager
    def cap(moe):
        with reference_capture(moe) as sink:
            yield sink
        x, topk_idx, topk_weight, agg = sink.drain()[0]
        per_slot = recompute_contributions(x, topk_idx, topk_weight, moe)
        records.append((per_slot, agg))

    torch.manual_seed(4)
    batches = [{"input_ids": torch.randint(0, VOCAB, (1, t))} for t in (11, 17)]
    fp32_prefix_forward(
        streamer, skeleton, batches, n_layers=2, device="cpu", on_target_moe=cap
    )
    assert len(records) == 2  # one context entry per example, in order
    for i, (per_slot, agg) in enumerate(records):
        assert per_slot.shape == (batches[i]["input_ids"].shape[1], 3, HIDDEN)
        per_token_rel = (per_slot.sum(dim=1) - agg).norm(dim=-1) / agg.norm(
            dim=-1
        ).clamp_min(1e-30)
        assert per_token_rel.max() < 1e-6


# --------------------------------------------------------------------------- #
# Full VL model: text-only and multimodal paths                                #
# --------------------------------------------------------------------------- #


def test_vl_text_only_matches_direct(tmp_path):
    model = make_tiny_vl()
    model.save_pretrained(tmp_path / "vl")
    torch.manual_seed(5)
    ids = torch.randint(0, VOCAB, (1, 15))

    with torch.no_grad():
        ref = model(
            input_ids=ids, images=None, output_hidden_states=True, use_cache=False,
            return_dict=True,
        ).hidden_states[2]

    streamer = CheckpointStreamer(tmp_path / "vl")
    skeleton = build_skeleton(tmp_path / "vl", model_cls=DeepseekVLV2ForCausalLM)
    out = fp32_prefix_forward(
        streamer, skeleton, {"input_ids": ids, "images": None}, n_layers=2,
        device="cpu",
    )
    assert rel_diff(out, ref) < 1e-6


def test_vl_multimodal_matches_direct(tmp_path):
    """Vision tower + projector + tile scatter (prepare_inputs_embeds) streamed
    in FP32 must match the direct multimodal forward."""
    model = make_tiny_vl()
    model.save_pretrained(tmp_path / "vl")
    ids, images, mask, crop = multimodal_inputs()

    with torch.no_grad():
        ref = model(
            input_ids=ids,
            images=images,
            images_seq_mask=mask,
            images_spatial_crop=crop,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        ).hidden_states[2]

    streamer = CheckpointStreamer(tmp_path / "vl")
    skeleton = build_skeleton(tmp_path / "vl", model_cls=DeepseekVLV2ForCausalLM)
    batch = {
        "input_ids": ids,
        "images": images,
        "images_seq_mask": mask,
        "images_spatial_crop": crop,
    }
    out = fp32_prefix_forward(streamer, skeleton, batch, n_layers=2, device="cpu")
    assert rel_diff(out, ref) < 1e-6

    # vision material was freed again after the embedding phase
    assert next(skeleton.vision.parameters()).is_meta
    assert skeleton.image_newline.is_meta


# --------------------------------------------------------------------------- #
# run_validation end-to-end on the tiny checkpoint                             #
# --------------------------------------------------------------------------- #


def test_run_validation_end_to_end_tiny(tmp_path, monkeypatch):
    """Both phases of the gate against a tiny random VL checkpoint: FP32 gate
    must PASS (rel err ~1e-7 « 1e-5), the production cross-check must hold,
    and the BF16 report must be populated. The frozen-model-only pieces
    (spec §2 assertions, BF16 load, target-layer index) are monkeypatched to
    their tiny-model equivalents; the target is the first MoE layer (index 1)."""
    import json

    import moe_sae.capture.validate as V

    model = make_tiny_vl()
    model.save_pretrained(tmp_path / "ckpt")

    inputs = tmp_path / "inputs"
    inputs.mkdir()
    ids, images, mask, crop = multimodal_inputs(seed=6)
    torch.save(
        {
            "input_ids": ids.squeeze(0),
            "images": images,
            "images_seq_mask": mask,
            "images_spatial_crop": crop,
            "example_id": 1,
            "domain": "coco",
        },
        inputs / "examples_0000.pt",
    )
    torch.manual_seed(7)
    torch.save(
        {
            "input_ids": torch.randint(0, VOCAB, (17,)),
            "images": None,
            "images_seq_mask": None,
            "images_spatial_crop": None,
            "example_id": 2,
            "domain": "textvqa",
        },
        inputs / "examples_0001.pt",
    )
    torch.save(
        {
            "input_ids": torch.randint(0, VOCAB, (9,)),
            "images": None,
            "images_seq_mask": None,
            "images_spatial_crop": None,
            "example_id": 3,
            "domain": "chartqa",
        },
        inputs / "examples_0002.pt",
    )

    bf16_model = make_tiny_vl().to(torch.bfloat16).eval()  # same seed -> same weights
    monkeypatch.setattr(V, "TARGET_LAYER_ZERO_INDEXED", 1)
    monkeypatch.setattr(V, "assert_frozen_architecture", lambda m: None)
    monkeypatch.setattr(V, "load_model", lambda model_id, device: (bf16_model, None))
    monkeypatch.setattr(
        V, "get_target_moe", lambda m: m.language.model.layers[1].mlp
    )

    report = V.run_validation(
        inputs_dir=inputs,
        out_path=tmp_path / "report.json",
        n_tokens=20,
        device="cpu",
        model_id=str(tmp_path / "ckpt"),
        chunk_examples=2,
    )

    # token budget: first chunk (12 + 17 = 29 >= 20 tokens) suffices
    assert report["n_examples"] == 2
    assert report["fp32"]["n_tokens"] == 29
    assert report["gate"] == "PASS"
    assert report["fp32"]["status"] == "PASS"
    assert report["fp32"]["max_rel"] < 1e-5
    assert report["production_crosscheck"]["pass"]
    assert report["bf16"]["n_tokens"] == 29
    assert report["bf16"]["max"] is not None and report["bf16"]["max"] > 0
    assert set(report["bf16"]["correlations"]) == {
        "max_gate",
        "token_position",
        "visual_vs_text",
    }
    assert {"coco", "textvqa"} <= set(report["per_domain"])
    assert "chartqa" not in report["per_domain"]  # beyond the token budget

    on_disk = json.loads((tmp_path / "report.json").read_text())
    assert on_disk["gate"] == "PASS"
