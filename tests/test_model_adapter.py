from __future__ import annotations

import hashlib
import os
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from moe_sae.model import (
    CaptureBatch,
    GEMMA4_REPOSITORY_FILES,
    Gemma4PrefixAdapter,
    HuggingFaceRepoFile,
    HuggingFaceSnapshot,
    MaxTextUnavailableError,
    assert_hf_snapshot_unchanged,
    build_checkpoint_conversion_command,
    checkpoint_conversion_spec,
    checkpoint_inventory_sha256,
    checkpoint_validation_command,
    expected_hf_repository_inventory_sha256,
    expected_hf_snapshot_inventory_sha256,
    materialize_hf_snapshot,
    maxtext_checkout_environment,
    require_maxtext,
    validate_capture_batch,
    verify_hf_snapshot,
    verify_maxtext_checkout,
)


_MODEL_ID = "google/gemma-4-26B-A4B"
_MODEL_REVISION = "24548b62aa021d562695c04aaf7758a1ea47990b"
_FIXTURE_MODEL = b"immutable model weights"
_FIXTURE_TOKENIZER = b'{"version": "fixture"}\n'


def _git_blob_oid(payload: bytes) -> str:
    digest = hashlib.sha1(  # noqa: S324 - the fixture checks Git blob identity
        usedforsecurity=False
    )
    digest.update(f"blob {len(payload)}\0".encode("ascii"))
    digest.update(payload)
    return digest.hexdigest()


def _lfs_pointer_oid(payload_sha256: str, size_bytes: int) -> str:
    pointer = (
        "version https://git-lfs.github.com/spec/v1\n"
        f"oid sha256:{payload_sha256}\n"
        f"size {size_bytes}\n"
    ).encode("ascii")
    return _git_blob_oid(pointer)


_FIXTURE_MODEL_SHA256 = hashlib.sha256(_FIXTURE_MODEL).hexdigest()
_FIXTURE_REPO_FILES = (
    HuggingFaceRepoFile(
        path="model.safetensors",
        size_bytes=len(_FIXTURE_MODEL),
        git_blob_oid=_lfs_pointer_oid(
            _FIXTURE_MODEL_SHA256, len(_FIXTURE_MODEL)
        ),
        content_sha256=_FIXTURE_MODEL_SHA256,
        lfs_sha256=_FIXTURE_MODEL_SHA256,
    ),
    HuggingFaceRepoFile(
        path="tokenizer.json",
        size_bytes=len(_FIXTURE_TOKENIZER),
        git_blob_oid=_git_blob_oid(_FIXTURE_TOKENIZER),
        content_sha256=hashlib.sha256(_FIXTURE_TOKENIZER).hexdigest(),
    ),
)


def _snapshot_fixture(tmp_path):
    blobs = tmp_path / "models--google--gemma-4-26B-A4B" / "blobs"
    snapshot = blobs.parent / "snapshots" / _MODEL_REVISION
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    blob = blobs / ("a" * 64)
    blob.write_bytes(_FIXTURE_MODEL)
    (snapshot / "model.safetensors").symlink_to(blob)
    (snapshot / "tokenizer.json").write_text(
        _FIXTURE_TOKENIZER.decode("utf-8"), encoding="utf-8"
    )
    return verify_hf_snapshot(
        snapshot,
        model_id=_MODEL_ID,
        revision=_MODEL_REVISION,
        expected_repo_files=_FIXTURE_REPO_FILES,
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
    fixture = _snapshot_fixture(tmp_path / "hub")
    snapshot = HuggingFaceSnapshot(
        model_id=fixture.model_id,
        revision=fixture.revision,
        path=fixture.path,
        inventory_sha256=expected_hf_snapshot_inventory_sha256(),
        repository_inventory_sha256=expected_hf_repository_inventory_sha256(),
    )
    spec = checkpoint_conversion_spec(
        tmp_path,
        "gs://bucket/gemma4/converted",
        hf_snapshot=snapshot,
    )
    argv = build_checkpoint_conversion_command(spec)
    validation_argv = checkpoint_validation_command(
        spec, "gs://bucket/gemma4/converted/0/items"
    )
    joined = " ".join(argv)
    assert "maxtext.checkpoint_conversion.to_maxtext" in joined
    assert "model_name=gemma4-26b" in argv
    assert "use_multimodal=false" in argv
    assert "scan_layers=false" in argv
    assert "pure_nnx_decoder=true" in argv
    assert not any(item.startswith("load_balance_loss_weight=") for item in argv)
    assert "--save_dtype=bfloat16" in argv
    assert "HF_TOKEN" not in joined and "hf_access_token" not in joined
    assert f"--hf_model_path={snapshot.path}" in argv
    assert f"--hf_model_path={snapshot.path}" in validation_argv
    assert f"tokenizer_path={snapshot.path}" in validation_argv
    assert "hardware=cpu" in validation_argv
    assert "skip_jax_distributed_system=true" in validation_argv
    assert "attention=dot_product" in validation_argv
    assert "dtype=float32" in validation_argv
    assert spec.hf_model_id == _MODEL_ID
    assert spec.hf_revision == _MODEL_REVISION
    assert spec.hf_snapshot_inventory_sha256 == snapshot.inventory_sha256
    assert spec.hf_repository_inventory_sha256 == (
        snapshot.repository_inventory_sha256
    )


def test_checkpoint_spec_rejects_an_unpinned_snapshot_inventory(tmp_path):
    base = tmp_path / "src" / "maxtext" / "configs"
    base.mkdir(parents=True)
    (base / "base.yml").write_text("# fake")
    snapshot = _snapshot_fixture(tmp_path / "hub")

    with pytest.raises(ValueError, match="repository inventory"):
        checkpoint_conversion_spec(
            tmp_path,
            "gs://bucket/gemma4/converted",
            hf_snapshot=snapshot,
        )


def test_materialize_hf_snapshot_pins_exact_commit_and_hashes_content(
    tmp_path, monkeypatch
):
    expected = _snapshot_fixture(tmp_path)
    calls = []

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        return str(expected.path)

    real_import = __import__("importlib").import_module

    def import_fake_hub(name, *args, **kwargs):
        if name == "huggingface_hub":
            return SimpleNamespace(snapshot_download=snapshot_download)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("moe_sae.model.import_module", import_fake_hub)
    result = materialize_hf_snapshot(
        _MODEL_ID,
        _MODEL_REVISION,
        cache_dir=tmp_path / "cache",
        token="secret-that-must-not-be-recorded",
        expected_repo_files=_FIXTURE_REPO_FILES,
    )

    assert result == expected
    assert calls == [
        {
            "repo_id": _MODEL_ID,
            "repo_type": "model",
            "revision": _MODEL_REVISION,
            "token": "secret-that-must-not-be-recorded",
            "local_files_only": False,
            "cache_dir": str(tmp_path / "cache"),
        }
    ]
    assert "secret" not in repr(result)


def test_pinned_gemma_repository_inventory_is_complete_and_offline_digestible():
    assert len(GEMMA4_REPOSITORY_FILES) == 10
    assert sum(item.lfs_sha256 is not None for item in GEMMA4_REPOSITORY_FILES) == 3
    assert expected_hf_repository_inventory_sha256() == (
        "7e1512d2391671f81faeb2593775811436deab90ba1e3a3867d916213dea0f47"
    )
    assert expected_hf_snapshot_inventory_sha256() == (
        "8660c5dee92018c74f36bf9b46686844c6e2f71890fe53d0cf6e008a3b1a7836"
    )


def test_hf_snapshot_rejects_missing_extra_and_wrong_ordinary_blob(tmp_path):
    snapshot = _snapshot_fixture(tmp_path)
    tokenizer = snapshot.path / "tokenizer.json"
    tokenizer.unlink()
    with pytest.raises(RuntimeError, match="file set differs"):
        verify_hf_snapshot(
            snapshot.path,
            model_id=_MODEL_ID,
            revision=_MODEL_REVISION,
            expected_repo_files=_FIXTURE_REPO_FILES,
        )

    tokenizer.write_bytes(_FIXTURE_TOKENIZER)
    (snapshot.path / "unexpected.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="file set differs"):
        verify_hf_snapshot(
            snapshot.path,
            model_id=_MODEL_ID,
            revision=_MODEL_REVISION,
            expected_repo_files=_FIXTURE_REPO_FILES,
        )

    (snapshot.path / "unexpected.json").unlink()
    tokenizer.write_text('{"version":"same-size?"}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="content changed"):
        verify_hf_snapshot(
            snapshot.path,
            model_id=_MODEL_ID,
            revision=_MODEL_REVISION,
            expected_repo_files=_FIXTURE_REPO_FILES,
        )


@pytest.mark.parametrize("revision", ["main", "A" * 40, "a" * 39, "g" * 40])
def test_hf_snapshot_rejects_non_commit_revisions(tmp_path, revision):
    with pytest.raises(ValueError, match="exact lowercase 40-character commit SHA"):
        materialize_hf_snapshot(_MODEL_ID, revision, cache_dir=tmp_path)


def test_hf_snapshot_inventory_detects_cache_mutation(tmp_path):
    snapshot = _snapshot_fixture(tmp_path)
    assert (
        assert_hf_snapshot_unchanged(
            snapshot, expected_repo_files=_FIXTURE_REPO_FILES
        )
        == snapshot.inventory_sha256
    )

    (snapshot.path / "tokenizer.json").write_text(
        '{"version": "mutated"}\n', encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="content changed"):
        assert_hf_snapshot_unchanged(
            snapshot, expected_repo_files=_FIXTURE_REPO_FILES
        )


def test_hf_snapshot_path_must_be_commit_addressed(tmp_path):
    snapshot = tmp_path / "model"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="commit-addressed directory"):
        verify_hf_snapshot(
            snapshot,
            model_id=_MODEL_ID,
            revision=_MODEL_REVISION,
        )


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
