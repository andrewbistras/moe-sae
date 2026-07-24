from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import moe_sae.contribution_model as contribution_model_module
from moe_sae.contribution_config import (
    MAXTEXT_FORK_COMMIT,
    MAXTEXT_ROUTED_MOE_PATCHED_SHA256,
)
from moe_sae.contribution_model import (
    Gemma4ContributionPrefixAdapter,
    canonicalize_selected_experts,
)
from moe_sae.model import MaxTextUnavailableError
from moe_sae.contribution_sae import canonical_router_order

_NATIVE_IDS = np.array([9, 2, 5, 1, 7, 3, 11, 0], dtype=np.int32)
_NATIVE_WEIGHTS = np.array(
    [0.05, 0.20, 0.10, 0.20, 0.15, 0.10, 0.15, 0.05],
    dtype=np.float32,
)


def _bf16(value):
    return jnp.asarray(value, dtype=jnp.bfloat16)


class FakeParam:
    def __init__(self, value):
        self.value = value


class FakeNorm:
    def __init__(self, scale: float):
        self.scale = scale

    def __call__(self, value):
        return _bf16(value * self.scale)


class FakeAttention:
    def __call__(self, query, key_value, positions, **kwargs):
        del key_value, positions, kwargs
        return _bf16(query + 0.25), None


class FakeGate:
    def __init__(self):
        self.calls = 0

    def __call__(self, gate_inputs):
        self.calls += 1
        shape = gate_inputs.shape[:-1] + (128,)
        return jnp.zeros(shape, dtype=jnp.bfloat16), None


class FakeRoutedMoE:
    def __init__(self, config):
        self.config = config
        self.gate = FakeGate()
        self.rngs = object()
        self.is_hash_routing = False
        self.quant = None
        self.tensor_parallelism = 1
        self.calls: list[bool] = []
        self.corrupt_selected = False
        self.bad_aux_arity = False
        self.bad_selected_shape = False
        self.last_selected = None
        self.last_ids = None
        self.last_weights = None

    def get_tensor_parallelism_size(self):
        return self.tensor_parallelism

    def get_topk(self, gate_logits, pre_bias_logits, rngs):
        del pre_bias_logits, rngs
        leading = gate_logits.shape[:-1]
        ids = jnp.broadcast_to(jnp.asarray(_NATIVE_IDS), leading + (8,))
        weights = jnp.broadcast_to(
            _bf16(_NATIVE_WEIGHTS),
            leading + (8,),
        )
        return weights, ids

    def __call__(
        self,
        routed_inputs,
        *,
        gate_inputs,
        out_sharding,
        return_selected_outputs=False,
    ):
        del out_sharding
        self.calls.append(return_selected_outputs)
        gate_logits, pre_bias_logits = self.gate(gate_inputs)
        weights, ids = self.get_topk(
            gate_logits,
            pre_bias_logits,
            self.rngs,
        )
        selected = _bf16(
            routed_inputs[..., None, :] + ids.astype(jnp.float32)[..., None] * 0.03125
        )
        flat_selected = selected.astype(jnp.float32).reshape((-1, 8, 2816))
        flat_weights = weights.astype(jnp.float32).reshape((-1, 8))
        aggregate = jnp.einsum(
            "BKE,BK->BE",
            flat_selected,
            flat_weights,
            precision=jax.lax.Precision.DEFAULT,
        )
        aggregate = _bf16(aggregate.reshape(routed_inputs.shape))
        self.last_selected = selected
        self.last_ids = ids
        self.last_weights = weights
        if not return_selected_outputs:
            return aggregate, None, None
        if self.bad_aux_arity:
            return aggregate, None, None
        returned = selected + _bf16(1.0) if self.corrupt_selected else selected
        if self.bad_selected_shape:
            returned = returned[..., :7, :]
        return aggregate, None, None, returned


class UnpatchedRoutedMoE(FakeRoutedMoE):
    def __call__(self, routed_inputs, *, gate_inputs, out_sharding):
        return super().__call__(
            routed_inputs,
            gate_inputs=gate_inputs,
            out_sharding=out_sharding,
        )


class FakeSharedExperts:
    def __init__(self):
        self.last_output = None

    def __call__(self, inputs, *, intermediate_sharding, out_sharding):
        del intermediate_sharding, out_sharding
        output = _bf16(inputs * 1.375 + 0.5)
        self.last_output = output
        return output


class FakeMLP:
    def __init__(self, config):
        self.pre_feedforward_layernorm_2 = FakeNorm(0.75)
        self.gate_norm = FakeNorm(1.125)
        self.pre_forward_scale_2 = FakeParam(np.ones((2816,), dtype=np.float32))
        self.post_feedforward_layernorm_1 = FakeNorm(0.625)
        self.post_feedforward_layernorm_2 = FakeNorm(1.25)
        self.moe_block = SimpleNamespace(
            routed_moe=FakeRoutedMoE(config),
            shared_experts=FakeSharedExperts(),
        )


class FakeLayer:
    def __init__(self, index, calls, config):
        self.index = index
        self.layer_idx = 0
        self.attention_type = "global" if (index + 1) % 6 == 0 else "local_sliding"
        self.calls = calls
        self.config = SimpleNamespace(dtype=jnp.bfloat16)
        self.pre_self_attention_norm = FakeNorm(0.875)
        self.self_attention = FakeAttention()
        self.post_self_attention_norm = FakeNorm(0.9375)
        self.pre_ffw_norm = FakeNorm(0.8125)
        self.post_ffw_norm = FakeNorm(1.0625)
        self.layer_scalar = FakeParam(np.array([0.875], dtype=np.float32))
        self.mlp = FakeMLP(config)

    def __call__(
        self,
        hidden,
        segment_ids,
        positions,
        deterministic,
        model_mode,
        **kwargs,
    ):
        del deterministic, model_mode, kwargs
        self.calls.append(self.index)
        if self.index != 14:
            return _bf16(hidden + (self.index + 1) / 64.0), None

        normalized = self.pre_self_attention_norm(hidden)
        attention = self.self_attention(
            normalized,
            normalized,
            positions,
            decoder_segment_ids=segment_ids,
        )[0]
        attention = self.post_self_attention_norm(attention)
        residual = attention + hidden
        routed_inputs = self.mlp.pre_feedforward_layernorm_2(residual)
        gate_inputs = (
            self.mlp.gate_norm(residual)
            * (2816**-0.5)
            * self.mlp.pre_forward_scale_2.value
        )
        aggregate, _, _ = self.mlp.moe_block.routed_moe(
            routed_inputs,
            gate_inputs=gate_inputs,
            out_sharding=None,
        )
        routed = self.mlp.post_feedforward_layernorm_2(aggregate)
        shared_raw = self.mlp.moe_block.shared_experts(
            self.pre_ffw_norm(residual),
            intermediate_sharding=None,
            out_sharding=None,
        )
        shared = self.mlp.post_feedforward_layernorm_1(shared_raw)
        outer = self.post_ffw_norm(routed + shared)
        return (outer + residual) * _bf16(self.layer_scalar.value), None


class FakeDecoder:
    def __init__(self, calls, config):
        self.layers = [FakeLayer(index, calls, config) for index in range(30)]
        for index, layer in enumerate(self.layers):
            setattr(self, f"layers_{index}", layer)

    def _apply_embedding(
        self,
        token_embedder,
        token_ids,
        positions,
        deterministic,
        model_mode,
    ):
        del token_embedder, positions, deterministic, model_mode
        return _bf16(jnp.repeat(token_ids[..., None], 2816, axis=-1))


class FakeModel:
    def __init__(self):
        self.calls: list[int] = []
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
            decoder_block="gemma4",
            attention="dot_product",
            sparse_matmul=True,
            use_ring_of_experts=False,
            ragged_buffer_factor=0.0,
            float32_weight_sum=True,
            matmul_precision="default",
        )
        self.token_embedder = object()
        self.decoder = FakeDecoder(self.calls, self.config)

    @property
    def routed(self):
        return self.decoder.layers[14].mlp.moe_block.routed_moe

    @property
    def shared(self):
        return self.decoder.layers[14].mlp.moe_block.shared_experts


def _inputs(batch=2, sequence=3):
    tokens = np.arange(1, batch * sequence + 1, dtype=np.int32).reshape(
        batch,
        sequence,
    )
    positions = np.tile(np.arange(sequence, dtype=np.int32), (batch, 1))
    segments = np.ones_like(tokens)
    sequence_ids = np.arange(100, 100 + batch, dtype=np.int64)
    return tokens, positions, segments, sequence_ids


def test_canonical_order_reorders_ids_weights_and_outputs_together():
    ids = np.broadcast_to(_NATIVE_IDS, (2, 8)).copy()
    weights = np.broadcast_to(_NATIVE_WEIGHTS, (2, 8)).copy()
    output = np.broadcast_to(
        _NATIVE_IDS[None, :, None],
        (2, 8, 3),
    ).astype(np.float32)

    ordered_output, ordered_ids, ordered_weights, source_slots = (
        canonicalize_selected_experts(output, ids, weights)
    )
    oracle_ids, oracle_weights, oracle_slots = canonical_router_order(
        ids,
        np.asarray(_bf16(weights)).astype(np.float32),
    )

    np.testing.assert_array_equal(np.asarray(ordered_ids), oracle_ids)
    np.testing.assert_array_equal(np.asarray(ordered_weights), oracle_weights)
    np.testing.assert_array_equal(np.asarray(source_slots), oracle_slots)
    np.testing.assert_array_equal(
        np.asarray(ordered_output)[..., 0],
        oracle_ids,
    )
    assert str(ordered_output.dtype) == "bfloat16"
    assert str(ordered_weights.dtype) == "bfloat16"
    assert ordered_ids.dtype == jnp.uint8


def test_adapter_captures_canonical_raw_components_and_compacts_rows():
    model = FakeModel()
    adapter = Gemma4ContributionPrefixAdapter(model)
    tokens, positions, segments, sequence_ids = _inputs()
    keep = np.array([1, 4, 5], dtype=np.int32)

    batch = adapter(
        tokens,
        positions,
        segments,
        sequence_ids,
        valid_token_indices=keep,
    )

    assert model.calls == list(range(14))
    assert model.routed.calls == [True]
    assert batch.expert_output.shape == (3, 8, 2816)
    assert batch.expert_ids.shape == (3, 8)
    assert batch.router_weights.shape == (3, 8)
    assert batch.shared_output.shape == (3, 2816)
    assert batch.sequence_id.tolist() == [100, 101, 101]
    assert batch.position.tolist() == [1, 1, 2]
    assert batch.position.dtype == jnp.uint16
    assert batch.expert_ids.dtype == jnp.uint8
    assert str(batch.expert_output.dtype) == "bfloat16"
    assert str(batch.router_weights.dtype) == "bfloat16"
    assert str(batch.shared_output.dtype) == "bfloat16"

    native_ids = np.asarray(model.routed.last_ids).reshape((-1, 8))
    native_weights = (
        np.asarray(model.routed.last_weights).astype(np.float32).reshape((-1, 8))
    )
    _, _, source_slots = canonical_router_order(native_ids, native_weights)
    native_output = np.asarray(model.routed.last_selected).reshape((-1, 8, 2816))
    expected_output = np.take_along_axis(
        native_output,
        source_slots[..., None],
        axis=1,
    )[keep]
    np.testing.assert_array_equal(np.asarray(batch.expert_output), expected_output)

    expected_shared = np.asarray(model.shared.last_output).reshape((-1, 2816))[keep]
    np.testing.assert_array_equal(np.asarray(batch.shared_output), expected_shared)


def test_instrumentation_proves_aux_default_and_complete_block_parity():
    model = FakeModel()
    adapter = Gemma4ContributionPrefixAdapter(model)
    inputs = _inputs(batch=1, sequence=2)

    result = adapter.instrumentation_parity(*inputs)

    assert bool(np.asarray(result))
    assert model.calls == list(range(14)) + [14]
    # The full block and explicit unchanged comparison use the default API;
    # the component seam is the sole auxiliary call.
    assert model.routed.calls == [False, True, False]


def test_instrumentation_rejects_corrupted_selected_output():
    model = FakeModel()
    model.routed.corrupt_selected = True
    adapter = Gemma4ContributionPrefixAdapter(model)

    assert not bool(np.asarray(adapter.instrumentation_parity(*_inputs(1, 1))))


def test_instrumentation_evidence_exposes_the_two_hashed_outputs():
    model = FakeModel()
    adapter = Gemma4ContributionPrefixAdapter(model)

    evidence = adapter.instrumentation_evidence(*_inputs(1, 1))

    assert bool(np.asarray(evidence.equal))
    np.testing.assert_array_equal(
        np.asarray(evidence.unmodified_full_block),
        np.asarray(evidence.instrumented_full_block),
    )
    assert evidence.unmodified_full_block.shape == (1, 1, 2816)


def test_capture_path_is_jittable_and_keeps_the_wire_shapes_static():
    model = FakeModel()
    adapter = Gemma4ContributionPrefixAdapter(model)
    tokens, positions, segments, sequence_ids = _inputs(1, 2)
    sequence_ids = sequence_ids.astype(np.int32)
    capture = jax.jit(
        lambda t, p, s, q: adapter(
            t,
            p,
            s,
            q,
            valid_token_indices=jnp.array([1], dtype=jnp.int32),
        )
    )

    result = capture(tokens, positions, segments, sequence_ids)

    assert result.expert_output.shape == (1, 8, 2816)
    assert result.shared_output.shape == (1, 2816)
    assert result.expert_ids.shape == (1, 8)


def test_unpatched_routed_api_fails_at_adapter_construction():
    model = FakeModel()
    model.decoder.layers[14].mlp.moe_block.routed_moe = UnpatchedRoutedMoE(model.config)

    with pytest.raises(TypeError, match="not the pinned Experiment-2 patch"):
        Gemma4ContributionPrefixAdapter(model)


def test_from_maxtext_binds_the_exact_clean_fork_and_import_source(
    tmp_path, monkeypatch
):
    root = tmp_path / "maxtext"
    imported_path = root / "src" / "maxtext" / "__init__.py"

    def fake_run(argv, **_kwargs):
        stdout = MAXTEXT_FORK_COMMIT + "\n" if "rev-parse" in argv else ""
        return SimpleNamespace(stdout=stdout)

    def fake_import(name):
        if name == "maxtext":
            return SimpleNamespace(__file__=str(imported_path))
        if name == "jax":
            return object()
        raise AssertionError(name)

    monkeypatch.setattr(contribution_model_module.subprocess, "run", fake_run)
    monkeypatch.setattr(contribution_model_module, "import_module", fake_import)

    adapter = Gemma4ContributionPrefixAdapter.from_maxtext(
        FakeModel(),
        maxtext_root=root,
    )

    assert isinstance(adapter, Gemma4ContributionPrefixAdapter)


def test_from_maxtext_rejects_checkout_or_import_identity_drift(tmp_path, monkeypatch):
    root = tmp_path / "maxtext"

    def wrong_head(argv, **_kwargs):
        stdout = "0" * 40 + "\n" if "rev-parse" in argv else ""
        return SimpleNamespace(stdout=stdout)

    monkeypatch.setattr(contribution_model_module.subprocess, "run", wrong_head)
    with pytest.raises(MaxTextUnavailableError, match="pinned fork commit"):
        Gemma4ContributionPrefixAdapter.from_maxtext(
            FakeModel(),
            maxtext_root=root,
        )


def test_from_installed_maxtext_binds_pep610_commit_and_source(tmp_path, monkeypatch):
    import moe_sae.runtime as runtime_module

    package = tmp_path / "maxtext"
    source = package / "layers" / "moe.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"patched source")
    monkeypatch.setattr(
        contribution_model_module,
        "MAXTEXT_ROUTED_MOE_PATCHED_SHA256",
        __import__("hashlib").sha256(b"patched source").hexdigest(),
    )
    monkeypatch.setattr(
        runtime_module,
        "installed_maxtext_commit",
        lambda: MAXTEXT_FORK_COMMIT,
    )

    def fake_import(name):
        if name == "maxtext":
            return SimpleNamespace(__file__=str(package / "__init__.py"))
        if name == "jax":
            return object()
        raise AssertionError(name)

    monkeypatch.setattr(contribution_model_module, "import_module", fake_import)

    adapter = Gemma4ContributionPrefixAdapter.from_installed_maxtext(FakeModel())

    assert isinstance(adapter, Gemma4ContributionPrefixAdapter)
    assert len(MAXTEXT_ROUTED_MOE_PATCHED_SHA256) == 64

    root = tmp_path / "checkout"

    def clean_head(argv, **_kwargs):
        stdout = MAXTEXT_FORK_COMMIT + "\n" if "rev-parse" in argv else ""
        return SimpleNamespace(stdout=stdout)

    monkeypatch.setattr(contribution_model_module.subprocess, "run", clean_head)
    monkeypatch.setattr(
        contribution_model_module,
        "import_module",
        lambda name: (
            SimpleNamespace(__file__="/different/maxtext/__init__.py")
            if name == "maxtext"
            else object()
        ),
    )
    with pytest.raises(MaxTextUnavailableError, match="does not come from"):
        Gemma4ContributionPrefixAdapter.from_maxtext(
            FakeModel(),
            maxtext_root=root,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda model: setattr(model.config, "decoder_block", "deepseek"), "Gemma 4"),
        (lambda model: setattr(model.config, "attention", "vllm_rpa"), "fused-vLLM"),
        (lambda model: setattr(model.config, "sparse_matmul", False), "sparse_matmul"),
        (
            lambda model: setattr(model.config, "use_ring_of_experts", True),
            "ring of experts",
        ),
        (
            lambda model: setattr(model.config, "ragged_buffer_factor", 1.1),
            "ragged_buffer_factor",
        ),
        (
            lambda model: setattr(model.config, "float32_weight_sum", False),
            "float32_weight_sum",
        ),
        (
            lambda model: setattr(model.config, "matmul_precision", "high"),
            "matmul_precision=default",
        ),
        (
            lambda model: setattr(model.routed, "tensor_parallelism", 2),
            "tensor parallelism 1",
        ),
        (
            lambda model: setattr(model.routed, "is_hash_routing", True),
            "hash routing",
        ),
        (
            lambda model: setattr(
                model.routed,
                "quant",
                SimpleNamespace(quant_mode="SERVE"),
            ),
            "serving quantization",
        ),
    ],
)
def test_unqualified_maxtext_paths_fail_before_forward(mutation, message):
    model = FakeModel()
    mutation(model)

    with pytest.raises(ValueError, match=message):
        Gemma4ContributionPrefixAdapter(model)


def test_auxiliary_arity_and_shape_drift_fail_closed():
    model = FakeModel()
    model.routed.bad_aux_arity = True
    adapter = Gemma4ContributionPrefixAdapter(model)
    with pytest.raises(TypeError, match="must return exactly"):
        adapter(*_inputs(1, 1))

    model = FakeModel()
    model.routed.bad_selected_shape = True
    adapter = Gemma4ContributionPrefixAdapter(model)
    with pytest.raises(ValueError, match="selected expert outputs"):
        adapter(*_inputs(1, 1))
