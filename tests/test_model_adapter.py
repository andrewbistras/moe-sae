from __future__ import annotations

import os
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from moe_sae.model import (
    CaptureBatch,
    Gemma4PrefixAdapter,
    MaxTextUnavailableError,
    build_checkpoint_conversion_command,
    checkpoint_inventory_sha256,
    checkpoint_conversion_spec,
    maxtext_checkout_environment,
    require_maxtext,
    validate_capture_batch,
    verify_maxtext_checkout,
)


class FakeParam:
    def __init__(self, value):
        self.value = value


class IdentityNorm:
    def __call__(self, value):
        return value


class FakeAttention:
    def __call__(self, query, key_value, positions, **kwargs):
        del key_value, positions, kwargs
        return query, None


class FakeGate:
    def __init__(self):
        self.calls = 0

    def __call__(self, gate_inputs):
        self.calls += 1
        shape = gate_inputs.shape[:-1] + (128,)
        logits = np.broadcast_to(np.arange(128, dtype=np.float32), shape)
        return logits, None


class FakeRoutedMoE:
    def __init__(self):
        self.gate = FakeGate()
        self.rngs = object()
        self.calls = 0

    def __call__(self, routed_inputs, *, gate_inputs, out_sharding):
        del out_sharding
        self.calls += 1
        self.gate(gate_inputs)  # the production routed kernel computes routing once
        return routed_inputs + 2.0, None, None

    def get_topk(self, gate_logits, pre_bias_logits, rngs):
        del pre_bias_logits, rngs
        ids = np.argsort(gate_logits, axis=-1)[..., -8:][..., ::-1]
        selected = np.take_along_axis(gate_logits, ids, axis=-1)
        selected = np.exp(selected - selected.max(axis=-1, keepdims=True))
        weights = selected / selected.sum(axis=-1, keepdims=True)
        return weights.astype(np.float32), ids.astype(np.int32)


class FakeSharedExperts:
    def __call__(self, inputs, *, intermediate_sharding, out_sharding):
        del intermediate_sharding, out_sharding
        return inputs + 5.0


class FakeMLP:
    def __init__(self):
        self.pre_feedforward_layernorm_2 = IdentityNorm()
        self.gate_norm = IdentityNorm()
        self.pre_forward_scale_2 = FakeParam(np.ones((2816,), dtype=np.float32))
        self.post_feedforward_layernorm_1 = IdentityNorm()
        self.post_feedforward_layernorm_2 = lambda value: value * 3.0
        self.moe_block = SimpleNamespace(
            routed_moe=FakeRoutedMoE(),
            shared_experts=FakeSharedExperts(),
        )


class FakeLayer:
    def __init__(self, index, calls):
        self.layer_idx = index
        self.attention_type = "local_sliding" if (index + 1) % 6 else "global"
        self.calls = calls
        self.pre_self_attention_norm = IdentityNorm()
        self.self_attention = FakeAttention()
        self.post_self_attention_norm = IdentityNorm()
        self.pre_ffw_norm = IdentityNorm()
        self.post_ffw_norm = IdentityNorm()
        self.layer_scalar = FakeParam(np.ones((1,), dtype=np.float32))
        self.config = SimpleNamespace(dtype=np.float32)
        self.mlp = FakeMLP()

    def __call__(
        self, hidden, segment_ids, positions, deterministic, model_mode, **kwargs
    ):
        del segment_ids, positions, deterministic, model_mode, kwargs
        self.calls.append(self.layer_idx)
        if self.layer_idx == 14:
            residual = hidden * 2.0
            routed, _, _ = self.mlp.moe_block.routed_moe(
                residual,
                gate_inputs=residual * (2816**-0.5),
                out_sharding=None,
            )
            routed = self.mlp.post_feedforward_layernorm_2(routed)
            shared = self.mlp.moe_block.shared_experts(
                residual,
                intermediate_sharding=None,
                out_sharding=None,
            )
            combined = self.post_ffw_norm(routed + shared)
            return (combined + residual) * self.layer_scalar.value, None
        return hidden + 1.0, None


class FakeDecoder:
    def __init__(self, calls):
        self.layers = [FakeLayer(index, calls) for index in range(30)]

    def _apply_embedding(
        self, token_embedder, token_ids, positions, deterministic, model_mode
    ):
        del token_embedder, positions, deterministic, model_mode
        return np.repeat(token_ids[..., None].astype(np.float32), 2816, axis=-1)


class FakeModel:
    def __init__(self):
        self.calls = []
        self.token_embedder = object()
        self.decoder = FakeDecoder(self.calls)
        self.config = SimpleNamespace(
            model_name="gemma4-26b",
            dtype="bfloat16",
            weight_dtype="bfloat16",
            num_decoder_layers=30,
            emb_dim=2816,
            num_query_heads=16,
            num_kv_heads=8,
            global_num_kv_heads=2,
            head_dim=256,
            global_head_dim=512,
            mlp_dim=2112,
            moe_mlp_dim=704,
            num_experts=128,
            num_experts_per_tok=8,
            shared_experts=1,
            scan_layers=False,
            pure_nnx_decoder=True,
            use_multimodal=False,
            norm_topk_prob=True,
            use_random_routing=False,
            share_kv_projections=True,
            use_post_attn_norm=True,
            use_post_ffw_norm=True,
            float32_gate_logits=False,
            sliding_window_size=1024,
            dropout_rate=0.0,
            load_balance_loss_weight=0.0,
            capacity_factor=-1.0,
        )


def test_prefix_adapter_executes_exact_prefix_and_captures_shapes():
    model = FakeModel()
    adapter = Gemma4PrefixAdapter(model)
    tokens = np.array([[2, 3, 4], [5, 6, 7]], dtype=np.int32)
    positions = np.tile(np.arange(3, dtype=np.int32), (2, 1))
    segments = np.ones_like(tokens)
    sequence_ids = np.array([100, 101], dtype=np.int64)

    batch = adapter(tokens, positions, segments, sequence_ids)

    # Only indices 0..13 execute their full layer call.  Index 14 takes the
    # attention/routed-only path, and layers 15..29 are never touched.
    assert model.calls == list(range(14))
    assert model.decoder.layers[14].mlp.moe_block.routed_moe.calls == 1
    assert model.decoder.layers[14].mlp.moe_block.routed_moe.gate.calls == 2
    assert batch.activation.shape == (6, 2816)
    assert batch.expert_ids.shape == (6, 8)
    assert batch.router_weights.shape == (6, 8)
    assert batch.sequence_id.tolist() == [100, 100, 100, 101, 101, 101]
    assert batch.position.tolist() == [0, 1, 2, 0, 1, 2]
    assert batch.expert_ids.dtype == np.uint8
    assert np.all(batch.expert_ids == np.arange(127, 119, -1, dtype=np.uint8))
    assert np.allclose(batch.router_weights.sum(axis=1), 1.0)

    # Embed token + 14 complete layers; target attention adds the residual;
    # routed MoE adds 2 and its post norm multiplies by 3.
    expected_first = ((2.0 + 14.0) * 2.0 + 2.0) * 3.0
    assert batch.activation[0, 0] == expected_first
    summary = validate_capture_batch(batch)
    assert summary.rows == 6 and summary.max_expert_id == 127


def test_static_valid_indices_compact_before_host_storage():
    model = FakeModel()
    adapter = Gemma4PrefixAdapter(model)
    tokens = np.arange(8, dtype=np.int32).reshape(2, 4)
    positions = np.tile(np.arange(4, dtype=np.int32), (2, 1))
    batch = adapter(
        tokens,
        positions,
        np.ones_like(tokens),
        np.array([[10] * 4, [20] * 4], dtype=np.int64),
        valid_token_indices=np.array([1, 2, 4, 7], dtype=np.int32),
    )
    assert batch.rows == 4
    assert batch.position.tolist() == [1, 2, 0, 3]
    assert batch.sequence_id.tolist() == [10, 10, 20, 20]


def test_instrumentation_parity_recomposes_complete_target_block():
    model = FakeModel()
    adapter = Gemma4PrefixAdapter(model)
    tokens = np.array([[2, 3]], dtype=np.int32)
    positions = np.array([[0, 1]], dtype=np.int32)
    segments = np.ones_like(tokens)

    result = adapter.instrumentation_parity(tokens, positions, segments, segments)

    assert bool(np.asarray(result))
    # The unmodified target block executes, but blocks after it never do.
    assert model.calls == list(range(14)) + [14]


def test_model_architecture_drift_fails_before_forward():
    model = FakeModel()
    model.config.num_experts = 64
    with pytest.raises(ValueError, match="num_experts must be 128"):
        Gemma4PrefixAdapter(model)

    model = FakeModel()
    model.decoder.layers = model.decoder.layers[:15]
    with pytest.raises(ValueError, match="expected 30 unscanned decoder layers"):
        Gemma4PrefixAdapter(model)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (
            lambda batch: CaptureBatch(
                batch.activation,
                batch.expert_ids[:, :7],
                batch.router_weights,
                batch.sequence_id,
                batch.position,
            ),
            "expert_ids",
        ),
        (
            lambda batch: CaptureBatch(
                batch.activation,
                np.zeros_like(batch.expert_ids),
                batch.router_weights,
                batch.sequence_id,
                batch.position,
            ),
            "unique",
        ),
        (
            lambda batch: CaptureBatch(
                batch.activation,
                batch.expert_ids,
                batch.router_weights * 0.5,
                batch.sequence_id,
                batch.position,
            ),
            "sum to one",
        ),
        (
            lambda batch: CaptureBatch(
                batch.activation,
                batch.expert_ids.astype(np.int16) + 125,
                batch.router_weights,
                batch.sequence_id,
                batch.position,
            ),
            r"\[0, 127\]",
        ),
    ],
)
def test_capture_invariants_fail_closed(mutate, message):
    ids = np.tile(np.arange(8, dtype=np.uint8), (2, 1))
    good = CaptureBatch(
        activation=np.ones((2, 2816), dtype=np.float32),
        expert_ids=ids,
        router_weights=np.full((2, 8), 1 / 8, dtype=np.float32),
        sequence_id=np.array([1, 2], dtype=np.int64),
        position=np.array([0, 1], dtype=np.uint16),
    )
    with pytest.raises((ValueError, TypeError), match=message):
        validate_capture_batch(mutate(good))


def test_missing_maxtext_dependency_is_lazy_and_actionable(monkeypatch):
    real_import = __import__("importlib").import_module

    def reject_maxtext(name, *args, **kwargs):
        if name == "maxtext":
            raise ModuleNotFoundError("no maxtext")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("moe_sae.model.import_module", reject_maxtext)
    with pytest.raises(MaxTextUnavailableError, match="pinned MaxText checkout"):
        require_maxtext()


def test_checkpoint_command_is_pinned_and_never_contains_token(tmp_path):
    base = tmp_path / "src" / "maxtext" / "configs"
    base.mkdir(parents=True)
    (base / "base.yml").write_text("# fake")
    spec = checkpoint_conversion_spec(tmp_path, "gs://bucket/gemma4/converted")
    argv = build_checkpoint_conversion_command(spec)
    joined = " ".join(argv)
    assert "maxtext.checkpoint_conversion.to_maxtext" in joined
    assert "model_name=gemma4-26b" in argv
    assert "use_multimodal=false" in argv
    assert "scan_layers=false" in argv
    assert "pure_nnx_decoder=true" in argv
    assert not any(item.startswith("load_balance_loss_weight=") for item in argv)
    assert "--save_dtype=bfloat16" in argv
    assert "HF_TOKEN" not in joined and "hf_access_token" not in joined


def test_maxtext_checkout_must_be_clean_and_controls_python_imports(
    tmp_path, monkeypatch
):
    base = tmp_path / "src" / "maxtext" / "configs"
    base.mkdir(parents=True)
    (base / "base.yml").write_text("# pinned\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.name", "Test"], check=True
    )
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-qm", "fixture"], check=True
    )
    head = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    monkeypatch.setattr("moe_sae.model.MAXTEXT_COMMIT", head)

    assert verify_maxtext_checkout(tmp_path) == head
    environment = maxtext_checkout_environment(
        tmp_path,
        {"HF_TOKEN": "kept-out-of-argv", "PYTHONPATH": "/wrong/source"},
    )
    assert environment["HF_TOKEN"] == "kept-out-of-argv"
    assert environment["PYTHONPATH"].split(os.pathsep) == [
        str(tmp_path / "src"),
        str(tmp_path),
    ]
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert environment["PYTHONSAFEPATH"] == "1"

    (tmp_path / "src" / "maxtext" / "dirty.py").write_text("x = 1\n")
    with pytest.raises(RuntimeError, match="must be clean"):
        verify_maxtext_checkout(tmp_path)


def test_local_checkpoint_inventory_digest_is_content_addressed(tmp_path):
    checkpoint = tmp_path / "0" / "items"
    checkpoint.mkdir(parents=True)
    (checkpoint / "a").write_bytes(b"alpha")
    (checkpoint / "nested").mkdir()
    (checkpoint / "nested" / "b").write_bytes(b"beta")

    first = checkpoint_inventory_sha256(str(checkpoint))
    assert len(first) == 64
    assert checkpoint_inventory_sha256(str(checkpoint)) == first

    (checkpoint / "nested" / "b").write_bytes(b"changed")
    assert checkpoint_inventory_sha256(str(checkpoint)) != first
