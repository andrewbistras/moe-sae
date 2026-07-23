from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import stat
import sys
import textwrap

import pytest

from moe_sae.r1 import maxtext_tap_patch as tap


_PROBE_SENTINEL = "MAXTEXT_TAP_PROBE="


def _git_blob(payload: bytes) -> str:
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload, usedforsecurity=False).hexdigest()


def _checkout_candidates() -> list[Path]:
    configured = os.environ.get("MAXTEXT_C03_CHECKOUT")
    if configured:
        return [Path(configured).expanduser()]
    checkout_cache = Path.home() / ".cache" / "uv" / "git-v0" / "checkouts"
    if not checkout_cache.is_dir():
        return []
    return sorted(checkout_cache.glob("*/c03f8bd3"))


@pytest.fixture(scope="session")
def pinned_maxtext_checkout() -> Path:
    configured = os.environ.get("MAXTEXT_C03_CHECKOUT")
    rejected: list[str] = []
    for candidate in _checkout_candidates():
        target = candidate.joinpath(*tap.TARGET_RELATIVE_PATH.parts)
        try:
            source_hash = tap.sha256_file(target)
        except tap.MaxTextTapPatchError as exc:
            rejected.append(f"{candidate}: {exc}")
            continue
        revision = subprocess.run(
            ["git", "-C", str(candidate), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        if (
            source_hash == tap.BASE_MOE_SHA256
            and revision.returncode == 0
            and revision.stdout.strip() == tap.MAXTEXT_BASE_COMMIT
        ):
            return candidate
        rejected.append(
            f"{candidate}: source={source_hash}, revision={revision.stdout.strip()!r}"
        )
    detail = "; ".join(rejected) if rejected else "no checkout candidate found"
    if configured:
        pytest.fail(f"MAXTEXT_C03_CHECKOUT is not the exact pinned tree: {detail}")
    pytest.skip(
        "external MaxText c03f8bd307d18049fa814fc8c104bd9ec8c292aa "
        f"checkout is absent ({detail})"
    )


@pytest.fixture(scope="session")
def copied_patched_checkout(
    pinned_maxtext_checkout: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    destination = tmp_path_factory.mktemp("maxtext-c03-tap") / "maxtext"
    shutil.copytree(
        pinned_maxtext_checkout,
        destination,
        symlinks=True,
        ignore=shutil.ignore_patterns(".git", ".venv", "__pycache__", "*.pyc"),
    )
    result = tap.apply_maxtext_tap_patch(destination)
    assert result.source_state == "patched"
    return destination


def _copy_target_only(source_root: Path, destination: Path) -> Path:
    source = source_root.joinpath(*tap.TARGET_RELATIVE_PATH.parts)
    target = destination.joinpath(*tap.TARGET_RELATIVE_PATH.parts)
    target.parent.mkdir(parents=True)
    shutil.copy2(source, target)
    return target


def test_patch_artifact_and_static_contract_are_sealed() -> None:
    patch_path = tap.default_patch_path()
    payload = patch_path.read_bytes()
    text = payload.decode("utf-8")
    assert hashlib.sha256(payload).hexdigest() == tap.PATCH_SHA256
    assert text.startswith(
        "diff --git a/src/maxtext/layers/moe.py "
        "b/src/maxtext/layers/moe.py\n"
        f"index {tap.BASE_MOE_GIT_BLOB}..{tap.PATCHED_MOE_GIT_BLOB} 100644\n"
    )
    assert "capture_router_metadata: bool = False" in text
    assert "top_k_indices.astype(jnp.int32)" in text
    assert "top_k_weights.astype(jnp.float32)" in text
    assert "out_specs=sparse_matmul_out_specs" in text
    assert "jnp.concatenate(top_k_indices_list, axis=1)" in text
    assert "jnp.concatenate(top_k_weights_list, axis=1)" in text
    assert "global_top_k_indices" in text
    assert "sow(nnx.Intermediate" not in text
    for unsupported in (
        "float32_gate_logits=False",
        "sparse_matmul=False",
        "attention=vllm_rpa",
        "hash routing",
        "random routing",
        "num_moe_emb_chunks>0",
        "ragged_buffer_factor>0",
    ):
        assert unsupported in text


def test_pinned_maxtext_router_compute_default_is_not_fp32(
    pinned_maxtext_checkout: Path,
) -> None:
    base_config = (
        pinned_maxtext_checkout / "src" / "maxtext" / "configs" / "base.yml"
    ).read_text(encoding="utf-8")
    assert "\nfloat32_gate_logits: false " in base_config


def test_exact_apply_attest_and_idempotence(
    pinned_maxtext_checkout: Path,
    tmp_path: Path,
) -> None:
    target = _copy_target_only(pinned_maxtext_checkout, tmp_path)
    base = tap.attest_maxtext_tree(tmp_path, expected_state="base")
    assert base.as_dict() == {
        "schema": "maxtext-routed-moe-prefix-tap-attestation-v1",
        "expected_upstream_commit": tap.MAXTEXT_BASE_COMMIT,
        "target_relative_path": tap.TARGET_RELATIVE_PATH.as_posix(),
        "source_state": "base",
        "source_sha256": tap.BASE_MOE_SHA256,
        "patch_sha256": tap.PATCH_SHA256,
    }
    original_mode = target.stat().st_mode & 0o777
    patched = tap.apply_maxtext_tap_patch(tmp_path)
    assert patched.source_state == "patched"
    assert tap.sha256_file(target) == tap.PATCHED_MOE_SHA256
    assert _git_blob(target.read_bytes()) == tap.PATCHED_MOE_GIT_BLOB
    assert target.stat().st_mode & 0o777 == original_mode
    compile(target.read_text(encoding="utf-8"), str(target), "exec")
    assert tap.apply_maxtext_tap_patch(tmp_path) == patched
    assert tap.attest_maxtext_tree(tmp_path, expected_state="patched") == patched
    with pytest.raises(tap.MaxTextTapPatchError, match="expected base"):
        tap.attest_maxtext_tree(tmp_path, expected_state="base")


def test_source_drift_fails_before_mutation(
    pinned_maxtext_checkout: Path,
    tmp_path: Path,
) -> None:
    target = _copy_target_only(pinned_maxtext_checkout, tmp_path)
    target.write_bytes(target.read_bytes() + b"# drift\n")
    before = target.read_bytes()
    with pytest.raises(tap.MaxTextTapPatchError, match="source drift|unexpected size"):
        tap.apply_maxtext_tap_patch(tmp_path)
    assert target.read_bytes() == before
    assert not list(target.parent.glob(".moe.py.tap-*.tmp"))


def test_patch_drift_fails_before_mutation(
    pinned_maxtext_checkout: Path,
    tmp_path: Path,
) -> None:
    target = _copy_target_only(pinned_maxtext_checkout, tmp_path / "tree")
    bad_patch = tmp_path / "drift.patch"
    corrupted = bytearray(tap.default_patch_path().read_bytes())
    corrupted[-1] ^= 1
    bad_patch.write_bytes(corrupted)
    before = target.read_bytes()
    with pytest.raises(tap.MaxTextTapPatchError, match="patch hash drift"):
        tap.apply_maxtext_tap_patch(tmp_path / "tree", patch_path=bad_patch)
    assert target.read_bytes() == before


def test_atomic_replace_failure_preserves_base_and_cleans_temp(
    pinned_maxtext_checkout: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _copy_target_only(pinned_maxtext_checkout, tmp_path)
    before = target.read_bytes()

    def fail_replace(
        source: str | bytes,
        destination: str | bytes,
        **kwargs: int,
    ) -> None:
        del source, destination, kwargs
        raise OSError("injected replace failure")

    monkeypatch.setattr(tap.os, "replace", fail_replace)
    with pytest.raises(tap.MaxTextTapPatchError, match="injected replace failure"):
        tap.apply_maxtext_tap_patch(tmp_path)
    assert target.read_bytes() == before
    assert tap.sha256_file(target) == tap.BASE_MOE_SHA256
    assert not list(target.parent.glob(".moe.py.tap-*.tmp"))


def test_symlink_target_is_rejected(
    pinned_maxtext_checkout: Path,
    tmp_path: Path,
) -> None:
    real_target = _copy_target_only(pinned_maxtext_checkout, tmp_path / "real")
    linked_target = (tmp_path / "linked").joinpath(*tap.TARGET_RELATIVE_PATH.parts)
    linked_target.parent.mkdir(parents=True)
    linked_target.symlink_to(real_target)
    with pytest.raises(tap.MaxTextTapPatchError, match="securely read"):
        tap.apply_maxtext_tap_patch(tmp_path / "linked")
    assert tap.sha256_file(real_target) == tap.BASE_MOE_SHA256


def test_symlinked_root_is_rejected(
    pinned_maxtext_checkout: Path,
    tmp_path: Path,
) -> None:
    real_root = tmp_path / "real-tree"
    target = _copy_target_only(pinned_maxtext_checkout, real_root)
    linked_root = tmp_path / "linked-tree"
    linked_root.symlink_to(real_root, target_is_directory=True)
    with pytest.raises(tap.MaxTextTapPatchError, match="without symlinks"):
        tap.apply_maxtext_tap_patch(linked_root)
    assert tap.sha256_file(target) == tap.BASE_MOE_SHA256


def test_symlinked_target_parent_is_rejected(
    pinned_maxtext_checkout: Path,
    tmp_path: Path,
) -> None:
    root = tmp_path / "tree"
    maxtext_parent = root / "src" / "maxtext"
    maxtext_parent.mkdir(parents=True)
    actual_layers = tmp_path / "actual-layers"
    source = pinned_maxtext_checkout.joinpath(*tap.TARGET_RELATIVE_PATH.parts)
    actual_layers.mkdir()
    shutil.copy2(source, actual_layers / "moe.py")
    (maxtext_parent / "layers").symlink_to(actual_layers, target_is_directory=True)
    with pytest.raises(tap.MaxTextTapPatchError, match="without symlinks"):
        tap.apply_maxtext_tap_patch(root)
    assert tap.sha256_file(actual_layers / "moe.py") == tap.BASE_MOE_SHA256


def test_cooperative_directory_lock_blocks_concurrent_application(
    pinned_maxtext_checkout: Path,
    tmp_path: Path,
) -> None:
    target = _copy_target_only(pinned_maxtext_checkout, tmp_path)
    descriptor = os.open(target.parent, os.O_RDONLY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(tap.MaxTextTapPatchError, match="cooperative"):
            tap.apply_maxtext_tap_patch(tmp_path)
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
    assert tap.sha256_file(target) == tap.BASE_MOE_SHA256


def test_parent_replacement_race_fails_without_redirecting_replace(
    pinned_maxtext_checkout: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "tree"
    target = _copy_target_only(pinned_maxtext_checkout, root)
    parent = target.parent
    detached_parent = tmp_path / "detached-layers"
    real_fsync = tap.os.fsync
    real_rename = tap.os.rename
    replaced = False

    def replacing_fsync(descriptor: int) -> None:
        nonlocal replaced
        real_fsync(descriptor)
        if not replaced and stat.S_ISREG(os.fstat(descriptor).st_mode):
            real_rename(parent, detached_parent)
            parent.mkdir()
            shutil.copy2(detached_parent / "moe.py", parent / "moe.py")
            replaced = True

    monkeypatch.setattr(tap.os, "fsync", replacing_fsync)
    with pytest.raises(tap.MaxTextTapPatchError, match="directory chain changed"):
        tap.apply_maxtext_tap_patch(root)
    assert replaced
    assert tap.sha256_file(parent / "moe.py") == tap.BASE_MOE_SHA256
    assert tap.sha256_file(detached_parent / "moe.py") == tap.BASE_MOE_SHA256
    assert not list(parent.glob(".moe.py.tap-*.tmp"))
    assert not list(detached_parent.glob(".moe.py.tap-*.tmp"))


def test_bounded_reader_rejects_oversize_file(tmp_path: Path) -> None:
    oversize = tmp_path / "oversize.bin"
    oversize.write_bytes(b"x" * (16 * 1024 * 1024 + 1))
    with pytest.raises(tap.MaxTextTapPatchError, match="read bound"):
        tap.sha256_file(oversize)


def test_sealed_target_rejects_noncanonical_mode(
    pinned_maxtext_checkout: Path,
    tmp_path: Path,
) -> None:
    target = _copy_target_only(pinned_maxtext_checkout, tmp_path)
    target.chmod(0o600)
    with pytest.raises(tap.MaxTextTapPatchError, match="mode is 0600"):
        tap.apply_maxtext_tap_patch(tmp_path)
    assert target.stat().st_mode & 0o777 == 0o600


def test_reader_rejects_hardlink_alias(
    pinned_maxtext_checkout: Path,
    tmp_path: Path,
) -> None:
    target = _copy_target_only(pinned_maxtext_checkout, tmp_path)
    alias = tmp_path / "moe-alias.py"
    os.link(target, alias)
    with pytest.raises(tap.MaxTextTapPatchError, match="exactly one hard link"):
        tap.sha256_file(target)


def test_bounded_reader_detects_path_replacement(
    pinned_maxtext_checkout: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _copy_target_only(pinned_maxtext_checkout, tmp_path / "tree")
    replacement = tmp_path / "replacement.py"
    shutil.copy2(target, replacement)
    real_read = tap.os.read
    real_replace = tap.os.replace
    replaced = False

    def replacing_read(descriptor: int, size: int) -> bytes:
        nonlocal replaced
        payload = real_read(descriptor, size)
        if payload and not replaced:
            replaced = True
            real_replace(replacement, target)
        return payload

    monkeypatch.setattr(tap.os, "read", replacing_read)
    with pytest.raises(tap.MaxTextTapPatchError, match="changed while"):
        tap.sha256_file(target)
    assert replaced


_NUMERICAL_PROBE = textwrap.dedent(
    f"""
    import json
    import ml_dtypes
    import numpy as np
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from jax.sharding import Mesh
    from maxtext.configs import pyconfig
    from maxtext.layers.initializers import nd_dense_init
    from maxtext.layers.moe import RoutedMoE
    from maxtext.utils import maxtext_utils
    from tests.utils.test_helpers import get_test_config_path
    from moe_sae.r1.router_oracle import (
        RouterGeometryV1, RouterOracleError, numpy_router_oracle,
    )

    cfg = pyconfig.initialize(
        [None, get_test_config_path()], run_name="tap_cpu_probe",
        enable_checkpointing=False, model_name="deepseek3-671b",
        decoder_block="deepseek", override_model_config=True,
        dtype="bfloat16", weight_dtype="bfloat16", base_emb_dim=8,
        base_mlp_dim=16, base_moe_mlp_dim=16, num_experts=8,
        num_experts_per_tok=2, n_routing_groups=2, topk_routing_group=1,
        routed_score_func="sigmoid", routed_bias=True,
        routed_scaling_factor=2.5, sparse_matmul=True, megablox=False,
        use_tokamax_gmm=False, use_gmm_v2=False, use_ring_of_experts=False,
        ragged_buffer_factor=-1.0, per_device_batch_size=2,
        max_target_length=4, max_prefill_predict_length=4,
        float32_gate_logits=True, attention="dot_product", mlp_bias=False,
        vocab_size=32, ici_data_parallelism=1, ici_fsdp_parallelism=1,
        ici_tensor_parallelism=1, ici_sequence_parallelism=1,
        ici_expert_parallelism=1, dcn_data_parallelism=1,
        dcn_fsdp_parallelism=1, dcn_tensor_parallelism=1,
        dcn_sequence_parallelism=1, dcn_expert_parallelism=1,
    )
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    kwargs = dict(
        config=cfg, num_experts=cfg.num_experts,
        num_experts_per_tok=cfg.num_experts_per_tok, mesh=mesh,
        kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", "mlp"), dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
    )
    tap_off = RoutedMoE(**kwargs, rngs=nnx.Rngs(params=0))
    tap_on = RoutedMoE(
        **kwargs, rngs=nnx.Rngs(params=0), capture_router_metadata=True
    )
    assert tap_off.gate.kernel[...].dtype == jnp.bfloat16
    assert tap_off.gate.bias[...].dtype == jnp.float32
    hidden = jnp.asarray(
        np.linspace(-1.5, 1.5, 64).reshape(2, 4, 8), dtype=jnp.bfloat16
    )
    geometry = RouterGeometryV1(
        hidden_dim=8, expert_count=8, group_count=2, selected_group_count=1,
        experts_per_token=2, group_score_top_k=2, routed_scaling_factor=2.5,
    )
    hidden_np = np.asarray(hidden)
    gate_np = np.asarray(tap_on.gate.kernel[...])
    boundary_hidden = hidden_np[:1, :1, :]
    rng = np.random.default_rng(20260723)
    boundary_bias = None
    rounded_boundary = None
    for _ in range(20_000):
        candidate = np.asarray(rng.normal(0.0, 0.2, 8), dtype=np.float32)
        rounded = np.asarray(candidate, dtype=ml_dtypes.bfloat16).astype(
            np.float32
        )
        if np.array_equal(candidate, rounded):
            continue
        try:
            fp32_choice = numpy_router_oracle(
                boundary_hidden, gate_np, candidate, geometry=geometry
            )
            rounded_choice = numpy_router_oracle(
                boundary_hidden, gate_np, rounded, geometry=geometry
            )
        except RouterOracleError:
            continue
        if not np.array_equal(fp32_choice.expert_ids, rounded_choice.expert_ids):
            boundary_bias = candidate
            rounded_boundary = rounded
            break
    assert boundary_bias is not None
    assert rounded_boundary is not None
    assert np.max(np.abs(boundary_bias - rounded_boundary)) > 0.0
    tap_off.gate.bias[...] = jnp.asarray(boundary_bias, dtype=jnp.float32)
    tap_on.gate.bias[...] = jnp.asarray(boundary_bias, dtype=jnp.float32)

    off_result = nnx.jit(tap_off)(hidden)
    on_result = nnx.jit(tap_on)(hidden)
    assert len(off_result) == 3
    assert len(on_result) == 5
    assert np.array_equal(np.asarray(off_result[0]), np.asarray(on_result[0]))
    expert_ids = np.asarray(on_result[3])
    router_weights = np.asarray(on_result[4])
    assert expert_ids.shape == (2, 4, 2)
    assert router_weights.shape == (2, 4, 2)
    assert expert_ids.dtype == np.int32
    assert router_weights.dtype == np.float32

    oracle = numpy_router_oracle(
        hidden_np, np.asarray(tap_on.gate.kernel[...]),
        np.asarray(tap_on.gate.bias[...]), geometry=geometry,
    )
    order = np.argsort(expert_ids, axis=-1, kind="stable")
    ordered_ids = np.take_along_axis(expert_ids, order, axis=-1)
    ordered_weights = np.take_along_axis(router_weights, order, axis=-1)
    assert np.array_equal(ordered_ids, oracle.expert_ids)
    assert np.allclose(
        ordered_weights, oracle.router_weights, rtol=1e-6, atol=1e-6
    )

    abstract_hidden = jax.ShapeDtypeStruct(hidden.shape, hidden.dtype)
    off_hlo = str(
        nnx.jit(tap_off).lower(abstract_hidden).compiler_ir(dialect="stablehlo")
    )
    on_hlo = str(
        nnx.jit(tap_on).lower(abstract_hidden).compiler_ir(dialect="stablehlo")
    )
    off_dots = off_hlo.count("stablehlo.dot_general")
    on_dots = on_hlo.count("stablehlo.dot_general")
    assert off_dots == on_dots

    def altered(**updates):
        return pyconfig.HyperParameters(
            cfg._pydantic_config.model_copy(update=updates)
        )

    default_kwargs = dict(kwargs)
    default_kwargs["config"] = altered(float32_gate_logits=False)
    default_tap_off = RoutedMoE(
        **default_kwargs, rngs=nnx.Rngs(params=0)
    )
    default_tap_off.gate.bias[...] = jnp.asarray(
        boundary_bias, dtype=jnp.float32
    )
    assert len(nnx.jit(default_tap_off)(hidden)) == 3

    chunk_kwargs = dict(kwargs)
    chunk_kwargs["config"] = altered(
        use_ring_of_experts=True, num_moe_token_chunks=2
    )
    chunked = RoutedMoE(
        **chunk_kwargs, rngs=nnx.Rngs(params=0),
        capture_router_metadata=True,
    )
    chunk_result = nnx.jit(chunked)(hidden)
    assert chunk_result[3].shape == (2, 4, 2)
    assert chunk_result[4].shape == (2, 4, 2)

    for update, marker in (
        ({{"float32_gate_logits": False}}, "float32_gate_logits=False"),
        ({{"sparse_matmul": False}}, "sparse_matmul=False"),
        ({{"use_random_routing": True}}, "random routing"),
        ({{"num_moe_emb_chunks": 1}}, "num_moe_emb_chunks>0"),
        ({{"ragged_buffer_factor": 1.5}}, "ragged_buffer_factor>0"),
    ):
        bad_kwargs = dict(kwargs)
        bad_kwargs["config"] = altered(**update)
        try:
            RoutedMoE(
                **bad_kwargs, rngs=nnx.Rngs(params=0),
                capture_router_metadata=True,
            )
        except ValueError as error:
            assert marker in str(error)
        else:
            raise AssertionError(marker)

    print({_PROBE_SENTINEL!r} + json.dumps({{
        "aggregate_bitwise_equal": True,
        "bias_rounding_max_abs_error": float(np.max(np.abs(
            boundary_bias - rounded_boundary
        ))),
        "dot_general_count": off_dots,
        "max_router_weight_abs_error": float(np.max(np.abs(
            ordered_weights - oracle.router_weights
        ))),
        "metadata_shape": list(expert_ids.shape),
        "rounded_bias_changes_ids": True,
        "chunk_metadata_shape": list(chunk_result[3].shape),
    }}, sort_keys=True))
    """
)


_MULTI_DEVICE_RING_PROBE = textwrap.dedent(
    f"""
    import json
    import numpy as np
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from jax.sharding import Mesh
    from maxtext.configs import pyconfig
    from maxtext.layers.initializers import nd_dense_init
    from maxtext.layers.moe import RoutedMoE
    from maxtext.utils import maxtext_utils
    from tests.utils.test_helpers import get_test_config_path
    from moe_sae.r1.router_oracle import RouterGeometryV1, numpy_router_oracle

    assert jax.device_count() == 8
    cfg = pyconfig.initialize(
        [None, get_test_config_path()], run_name="tap_cpu_ring_probe",
        enable_checkpointing=False, model_name="deepseek3-671b",
        decoder_block="deepseek", override_model_config=True,
        dtype="bfloat16", weight_dtype="bfloat16", base_emb_dim=8,
        base_mlp_dim=16, base_moe_mlp_dim=16, num_experts=8,
        num_experts_per_tok=2, n_routing_groups=2, topk_routing_group=1,
        routed_score_func="sigmoid", routed_bias=True,
        routed_scaling_factor=2.5, sparse_matmul=True, megablox=True,
        use_tokamax_gmm=False, use_gmm_v2=False, use_ring_of_experts=True,
        use_ragged_sort=True, num_moe_token_chunks=2,
        num_moe_emb_chunks=0, ragged_buffer_factor=-1.0,
        per_device_batch_size=4, max_target_length=4,
        max_prefill_predict_length=4, float32_gate_logits=True,
        attention="dot_product", mlp_bias=False, vocab_size=32,
        ici_data_parallelism=1, ici_fsdp_parallelism=1,
        ici_tensor_parallelism=1, ici_sequence_parallelism=1,
        ici_expert_parallelism=8, dcn_data_parallelism=1,
        dcn_fsdp_parallelism=1, dcn_tensor_parallelism=1,
        dcn_sequence_parallelism=1, dcn_expert_parallelism=1,
    )
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    kwargs = dict(
        config=cfg, num_experts=cfg.num_experts,
        num_experts_per_tok=cfg.num_experts_per_tok, mesh=mesh,
        kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed_moe", None), intermediate_dim=16,
        dtype=cfg.dtype, weight_dtype=cfg.weight_dtype,
    )
    tap_off = RoutedMoE(**kwargs, rngs=nnx.Rngs(params=0))
    tap_on = RoutedMoE(
        **kwargs, rngs=nnx.Rngs(params=0), capture_router_metadata=True
    )
    hidden = jnp.asarray(
        np.linspace(-2.0, 2.0, 32 * 4 * 8).reshape(32, 4, 8),
        dtype=jnp.bfloat16,
    )
    off_result = nnx.jit(tap_off)(hidden)
    on_result = nnx.jit(tap_on)(hidden)
    jax.block_until_ready(off_result[0])
    jax.block_until_ready(on_result[0])
    assert np.array_equal(np.asarray(off_result[0]), np.asarray(on_result[0]))

    expert_ids = np.asarray(on_result[3])
    router_weights = np.asarray(on_result[4])
    assert expert_ids.shape == (32, 4, 2)
    assert router_weights.shape == (32, 4, 2)
    assert expert_ids.dtype == np.int32
    assert router_weights.dtype == np.float32
    shards = sorted(
        on_result[3].addressable_shards,
        key=lambda shard: shard.index[0].start,
    )
    assert len(shards) == 8
    for rank, shard in enumerate(shards):
        batch_slice = shard.index[0]
        assert batch_slice == slice(rank * 4, (rank + 1) * 4)
        assert shard.data.shape == (4, 4, 2)
        assert np.array_equal(
            np.asarray(shard.data), expert_ids[batch_slice, :, :]
        )

    geometry = RouterGeometryV1(
        hidden_dim=8, expert_count=8, group_count=2, selected_group_count=1,
        experts_per_token=2, group_score_top_k=2, routed_scaling_factor=2.5,
    )
    oracle = numpy_router_oracle(
        np.asarray(hidden), np.asarray(tap_on.gate.kernel[...]),
        np.asarray(tap_on.gate.bias[...]), geometry=geometry,
    )
    order = np.argsort(expert_ids, axis=-1, kind="stable")
    ordered_ids = np.take_along_axis(expert_ids, order, axis=-1)
    ordered_weights = np.take_along_axis(router_weights, order, axis=-1)
    assert np.array_equal(ordered_ids, oracle.expert_ids)
    assert np.allclose(
        ordered_weights, oracle.router_weights, rtol=1e-6, atol=1e-6
    )

    abstract_hidden = jax.ShapeDtypeStruct(hidden.shape, hidden.dtype)
    hlo = str(
        nnx.jit(tap_on).lower(abstract_hidden).compiler_ir(dialect="stablehlo")
    )
    assert "stablehlo.all_gather" in hlo
    assert "stablehlo.reduce_scatter" in hlo

    a2a_config = pyconfig.HyperParameters(
        cfg._pydantic_config.model_copy(
            update={{"use_ring_of_experts": False, "num_moe_token_chunks": 1}}
        )
    )
    a2a_kwargs = dict(kwargs)
    a2a_kwargs["config"] = a2a_config
    a2a_off = RoutedMoE(**a2a_kwargs, rngs=nnx.Rngs(params=0))
    a2a_on = RoutedMoE(
        **a2a_kwargs, rngs=nnx.Rngs(params=0),
        capture_router_metadata=True,
    )
    a2a_off_hlo = str(
        nnx.jit(a2a_off).lower(abstract_hidden).compiler_ir(dialect="stablehlo")
    )
    a2a_on_hlo = str(
        nnx.jit(a2a_on).lower(abstract_hidden).compiler_ir(dialect="stablehlo")
    )
    a2a_collectives = a2a_on_hlo.count("@ragged_all_to_all")
    assert a2a_collectives >= 2
    assert a2a_off_hlo.count("stablehlo.dot_general") == a2a_on_hlo.count(
        "stablehlo.dot_general"
    )
    print({_PROBE_SENTINEL!r} + json.dumps({{
        "a2a_collective_count": a2a_collectives,
        "aggregate_bitwise_equal": True,
        "device_count": jax.device_count(),
        "max_router_weight_abs_error": float(np.max(np.abs(
            ordered_weights - oracle.router_weights
        ))),
        "metadata_shape": list(expert_ids.shape),
        "per_shard_shape": list(shards[0].data.shape),
        "token_chunks": cfg.num_moe_token_chunks,
    }}, sort_keys=True))
    """
)


def _probe_python() -> Path:
    configured = os.environ.get("MAXTEXT_C03_PYTHON")
    if configured:
        candidate = Path(configured).expanduser()
        if not candidate.is_file():
            pytest.fail(f"MAXTEXT_C03_PYTHON is not a file: {candidate}")
        return candidate
    if (
        importlib.util.find_spec("jax") is None
        or importlib.util.find_spec("flax") is None
    ):
        pytest.skip(
            "CPU MaxText integration dependencies are absent; install the TPU/MaxText "
            "extra or set MAXTEXT_C03_PYTHON"
        )
    return Path(sys.executable)


def test_copied_pinned_source_cpu_tap_parity_and_static_hlo(
    copied_patched_checkout: Path,
) -> None:
    python = _probe_python()
    repository_root = Path(__file__).resolve().parents[2]
    pythonpath = os.pathsep.join(
        (
            str(copied_patched_checkout / "src"),
            str(copied_patched_checkout),
            str(repository_root / "src"),
            os.environ.get("PYTHONPATH", ""),
        )
    )
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "JAX_PLATFORMS": "cpu",
            "PYTHONPATH": pythonpath,
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        }
    )
    result = subprocess.run(
        [str(python), "-c", _NUMERICAL_PROBE],
        cwd=repository_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        pytest.fail(
            "pinned MaxText CPU probe failed\n"
            f"stdout:\n{result.stdout[-8000:]}\n"
            f"stderr:\n{result.stderr[-8000:]}"
        )
    records = [
        line.removeprefix(_PROBE_SENTINEL)
        for line in result.stdout.splitlines()
        if line.startswith(_PROBE_SENTINEL)
    ]
    assert len(records) == 1
    report = json.loads(records[0])
    assert report["aggregate_bitwise_equal"] is True
    assert report["bias_rounding_max_abs_error"] > 0.0
    assert report["dot_general_count"] > 0
    assert report["max_router_weight_abs_error"] <= 1e-6
    assert report["metadata_shape"] == [2, 4, 2]
    assert report["rounded_bias_changes_ids"] is True
    assert report["chunk_metadata_shape"] == [2, 4, 2]


def test_copied_pinned_source_cpu_ep8_ring_collectives_and_chunks(
    copied_patched_checkout: Path,
) -> None:
    python = _probe_python()
    repository_root = Path(__file__).resolve().parents[2]
    pythonpath = os.pathsep.join(
        (
            str(copied_patched_checkout / "src"),
            str(copied_patched_checkout),
            str(repository_root / "src"),
            os.environ.get("PYTHONPATH", ""),
        )
    )
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "JAX_PLATFORMS": "cpu",
            "PYTHONPATH": pythonpath,
            "XLA_FLAGS": "--xla_force_host_platform_device_count=8",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        }
    )
    result = subprocess.run(
        [str(python), "-c", _MULTI_DEVICE_RING_PROBE],
        cwd=repository_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    if result.returncode != 0:
        pytest.fail(
            "pinned MaxText EP8 CPU ring probe failed\n"
            f"stdout:\n{result.stdout[-8000:]}\n"
            f"stderr:\n{result.stderr[-8000:]}"
        )
    records = [
        line.removeprefix(_PROBE_SENTINEL)
        for line in result.stdout.splitlines()
        if line.startswith(_PROBE_SENTINEL)
    ]
    assert len(records) == 1
    report = json.loads(records[0])
    assert report["aggregate_bitwise_equal"] is True
    assert report["a2a_collective_count"] >= 2
    assert report["device_count"] == 8
    assert report["max_router_weight_abs_error"] <= 1e-6
    assert report["metadata_shape"] == [32, 4, 2]
    assert report["per_shard_shape"] == [4, 4, 2]
    assert report["token_chunks"] == 2
