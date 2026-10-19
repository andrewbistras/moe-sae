# Frozen configurations

The paper's 24 training runs are defined by four configuration files in the
repository's `configs/` directory. None has changed since the runs: the A
configs were last modified in `8e2bfa2` and the R/B/F configs in `301e0dc`
(both 2026-07-29).

| Runs in the paper | Panel | Configuration | Layout name in the file |
|---|---|---|---|
| A (NR-SAE), seeds 0–2 | SAE | [`gemma4_experiment5_routed_ladder_sae_v3.yaml`](../../configs/gemma4_experiment5_routed_ladder_sae_v3.yaml) | `A512ReLUInclusiveTopK` |
| A (NR-SAE), seeds 0–2 | transcoder | [`gemma4_experiment5_routed_ladder_transcoder_v3.yaml`](../../configs/gemma4_experiment5_routed_ladder_transcoder_v3.yaml) | `A512ReLUInclusiveTopK` |
| R, B, F, seeds 0–2 | SAE | [`gemma4_component_closure_sae_v1.yaml`](../../configs/gemma4_component_closure_sae_v1.yaml) | `R`, `B`, `F` |
| R, B, F, seeds 0–2 | transcoder | [`gemma4_component_closure_transcoder_v1.yaml`](../../configs/gemma4_component_closure_transcoder_v1.yaml) | `R`, `B`, `F` |

The A runs come from the experiment that preceded the controls; the paper
calls them the six aligned runs carried over. The R/B/F configs import them
read-only and pin each A config by its file SHA-256
(`existing_a_reference.source_config_file_sha256`). Both pins match the files
here. To check all four files, run this from the repository root:

```sh
sha256sum -c release/configs/SHA256SUMS
```

## Paper notation

| Paper | Configuration |
|---|---|
| A, expert identity, 128 × 512 | `bank_identity: true_expert_id`, `bank_count: 128`, `bank_width: 512` |
| R, random identity, 128 × 512 | `bank_identity: per_sequence_randomized_expert_id` |
| B, rank bank, 8 × 512 | `bank_identity: router_rank`, `bank_width: 512` |
| F, wide rank bank, 8 × 8,192 | `bank_identity: router_rank`, `bank_width: 8192` |
| four features per bank, 32 per token | `topk_per_component: 4`, `selected_slots_per_token: 32` |
| ReLU, zero-valued entries eligible | `encoder_activation: relu`, `selection_policy: inclusive_zero_topk` |
| 611 updates of 16,384 tokens; warm-up of 61 | `planned_updates`, `effective_batch_tokens`, `warmup_updates` |
| final batch masks its last 10,624 tokens | `tail_mask_tokens: 10624` |
| fixed evaluation capture, 250,000 tokens | `eval_target_manifest_sha256: 31454e5c…`, `eval_tokens: 250000` |
| Reconstruction (SAE) / Prediction (transcoder) | `panel: sae` / `panel: transcoder` |

## Not included

The `*.template.yaml` files in `configs/` hold deployment placeholders:
accelerator slices, regions and storage paths. They describe where the runs
executed, not what was trained, and are not part of the scientific
configuration.

The protocols behind these files are
[`docs/EXPERIMENT_5_LADDER_V3.md`](../../docs/EXPERIMENT_5_LADDER_V3.md) for A
and [`docs/A512_COMPONENT_CLOSURE_PROTOCOL.md`](../../docs/A512_COMPONENT_CLOSURE_PROTOCOL.md)
for the controls and the final evaluation.
