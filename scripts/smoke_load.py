"""Smoke test: load frozen model, run spec §2 assertions, probe layer-5 routing."""

import logging

import torch

from moe_sae.model_loading import get_target_moe, load_model, load_processor

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main() -> None:
    model, info = load_model()
    print(f"Loaded: {info}")

    processor = load_processor()
    tokenizer = processor.tokenizer

    prompt = "The chart shows quarterly revenue for a small company."
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    print(f"input_ids shape: {tuple(ids.shape)}")

    moe = get_target_moe(model)
    captured: dict = {}

    def gate_hook(module, inputs, output):
        topk_idx, topk_weight, aux_loss = output
        captured["topk_idx"] = topk_idx.detach().cpu()
        captured["topk_weight"] = topk_weight.detach().cpu()

    handle = moe.gate.register_forward_hook(gate_hook)
    with torch.no_grad():
        embeds = model.prepare_inputs_embeds(input_ids=ids, images=None)
        out = model.language(
            inputs_embeds=embeds,
            attention_mask=torch.ones_like(ids),
            use_cache=False,
        )
    handle.remove()

    logits = out.logits
    print(f"logits shape: {tuple(logits.shape)}, finite: {torch.isfinite(logits).all().item()}")
    print(f"layer-5 gate topk_idx shape: {tuple(captured['topk_idx'].shape)}")
    print(f"first token's experts: {captured['topk_idx'][0].tolist()}")
    print(f"first token's gates:   {[round(g, 4) for g in captured['topk_weight'][0].tolist()]}")
    print(f"gate dtype: {captured['topk_weight'].dtype}")
    next_id = logits[0, -1].argmax().item()
    print(f"greedy next token: {next_id!r} -> {tokenizer.decode([next_id])!r}")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
