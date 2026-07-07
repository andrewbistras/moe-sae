# Vendored DeepSeek-VL2 modeling code — patch log

Source: https://github.com/deepseek-ai/DeepSeek-VL2 @ main (shallow clone, 2026-07-06),
directory `deepseek_vl2/models/`. Vendored because the model architecture
(`deepseek_vl_v2`) is not native to `transformers` and the upstream package pins
CUDA-only optional dependencies (flash-attn / xformers) that do not exist on macOS.

Frozen-spec discipline (spec v0.1.1 §2): patches may only change the *execution
backend*, never the math. Every patch is listed here with its reason. No patch
touches the MoE routing, expert MLPs, gating, normalization, rotary embeddings,
or any weight-bearing computation.

| # | File | Change | Reason |
|---|------|--------|--------|
| 1 | `modeling_deepseek_vl_v2.py` | Removed unused `from attrdict import AttrDict` import. | `attrdict` is unmaintained and fails to import on Python ≥ 3.10 (`collections.Mapping` removal). The symbol is never referenced in the file. |
| 2 | `siglip_vit.py` `Attention.forward` | Replaced `xformers.ops.memory_efficient_attention` / `flash_attn_qkvpacked_func` / CUDA-only `torch.backends.cuda.sdp_kernel` context with portable `F.scaled_dot_product_attention`. | xformers and flash-attn are unavailable on macOS; `torch.backends.cuda.sdp_kernel` asserts CUDA. SDPA computes the identical non-causal attention (same 1/√d scaling); only the reduction order of the underlying kernel differs, which is implementation-defined numerical behavior per spec §3. |

Unpatched files: `configuration_deepseek.py`, `modeling_deepseek.py`,
`processing_deepseek_vl_v2.py`, `conversation.py`, `__init__.py` — byte-identical
to upstream. `modeling_deepseek.py`'s flash-attn import is already guarded by
`is_flash_attn_2_available()` and the model defaults to the `eager` attention
class when flash-attn is absent.
