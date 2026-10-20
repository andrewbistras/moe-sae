# Experiment-2 MaxText auxiliary return

This directory contains the first reviewable MaxText fork delta for Experiment
2. It is anchored to parent commit
`48c7da3e28e8f57fe36ddf09d00eec53072a2ea0`.

The patch adds `return_selected_outputs: bool = False` to `RoutedMoE.__call__`.
The default call still returns exactly:

```python
(aggregate, load_balance_loss, bias_updates)
```

The opt-in call returns:

```python
(aggregate, load_balance_loss, bias_updates, selected_outputs)
```

`selected_outputs` has native route-slot order and shape
`[batch, sequence, top_k, hidden]`. It is captured after the selected experts'
output projections (and Gemma's learned per-expert output scale) and before the
existing FP32 router-weighted reduction. The reduction expression itself is
unchanged.

The auxiliary path deliberately fails closed unless the call is Gemma 4,
sparse, non-ring, dropless, non-hash, non-serving-quantized, non-vLLM, and
TP1. These are the path properties qualified by the current EP16/FSDP1/TP1
harvester. Default calls retain all existing MaxText paths.

Verify a pristine checkout before applying:

```bash
python patches/maxtext/verify_selected_outputs_patch.py \
  --maxtext-root /path/to/maxtext
git -C /path/to/maxtext apply \
  /path/to/moe-sae/patches/maxtext/0001-gemma4-return-selected-outputs.patch
```

This tranche does not yet add the moe-sae component adapter, expose router IDs
or weights, capture the shared branch, or claim TPU numerical/sharding parity.
Those require the next adapter and live EP16 validation tranches.
