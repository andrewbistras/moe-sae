"""Tests for ordered, content-bound application of the MaxText patch series."""

from __future__ import annotations

import ast
import hashlib
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from moe_sae.r1 import maxtext_patch_series as series
from moe_sae.r1.maxtext_tap_patch import BASE_MOE_SHA256, sha256_file


def _checkout_candidates() -> list[Path]:
    configured = os.environ.get("MAXTEXT_C03_CHECKOUT")
    if configured:
        return [Path(configured).expanduser()]
    cache = Path.home() / ".cache" / "uv" / "git-v0" / "checkouts"
    return sorted(cache.glob("*/c03f8bd3")) if cache.is_dir() else []


@pytest.fixture(scope="session")
def pinned_checkout() -> Path:
    for candidate in _checkout_candidates():
        revision = subprocess.run(
            ["git", "-C", str(candidate), "rev-parse", "HEAD"],
            capture_output=True,
            check=False,
            text=True,
        )
        if (
            revision.returncode == 0
            and revision.stdout.strip() == series.MAXTEXT_BASE_COMMIT
            and sha256_file(
                candidate / "src" / "maxtext" / "layers" / "moe.py"
            )
            == BASE_MOE_SHA256
        ):
            return candidate
    pytest.skip("the external pinned MaxText c03f8bd checkout is absent")


@pytest.fixture(scope="session")
def patch_root() -> Path:
    configured = os.environ.get("MAXTEXT_R1_PATCH_ROOT")
    root = Path(configured) if configured else series.default_patch_root()
    if not all((root / name).is_file() for name, _digest in series.PATCH_SERIES):
        pytest.fail(f"exact R1 patch series is absent from {root}")
    return root.resolve()


def _clone(source: Path, destination: Path) -> Path:
    subprocess.run(
        [
            "git",
            "clone",
            "--local",
            "--no-hardlinks",
            str(source),
            str(destination),
        ],
        check=True,
        capture_output=True,
    )
    return destination.resolve()


def _status(root: Path) -> tuple[str, ...]:
    result = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    )
    return tuple(result.stdout.rstrip("\n").splitlines()) if result.stdout else ()


@pytest.fixture(scope="session")
def applied_checkout(
    pinned_checkout: Path,
    patch_root: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    checkout = _clone(
        pinned_checkout,
        tmp_path_factory.mktemp("native-maxtext") / "maxtext",
    )
    series.apply_pinned_maxtext_patch_series(checkout, patch_root=patch_root)
    return checkout


def _class_method(
    tree: ast.Module,
    class_name: str,
    method_name: str,
) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for member in node.body:
                if isinstance(member, ast.FunctionDef) and member.name == method_name:
                    return member
    raise AssertionError(f"{class_name}.{method_name} is absent")


def _assignment(tree: ast.Module, name: str) -> ast.AST:
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
        ):
            return node.value
    raise AssertionError(f"{name} assignment is absent")


def test_exact_series_applies_in_order_and_attests_post_hashes(
    pinned_checkout: Path,
    patch_root: Path,
    tmp_path: Path,
) -> None:
    checkout = _clone(pinned_checkout, tmp_path / "maxtext")

    attestation = series.apply_pinned_maxtext_patch_series(
        checkout, patch_root=patch_root
    )

    assert attestation.base_commit == series.MAXTEXT_BASE_COMMIT
    assert attestation.patch_sha256 == tuple(
        digest for _name, digest in series.PATCH_SERIES
    )
    assert dict(attestation.post_apply_sha256) == dict(series.POST_APPLY_SHA256)
    assert _status(checkout) == series.EXPECTED_STATUS
    with pytest.raises(series.MaxTextPatchSeriesError, match="not clean"):
        series.apply_pinned_maxtext_patch_series(checkout, patch_root=patch_root)


def test_wrong_commit_and_dirty_checkout_fail_before_mutation(
    pinned_checkout: Path,
    patch_root: Path,
    tmp_path: Path,
) -> None:
    wrong = _clone(pinned_checkout, tmp_path / "wrong")
    subprocess.run(
        ["git", "-C", str(wrong), "checkout", "--detach", "HEAD^"],
        check=True,
        capture_output=True,
    )
    with pytest.raises(series.MaxTextPatchSeriesError, match="MaxText HEAD"):
        series.apply_pinned_maxtext_patch_series(wrong, patch_root=patch_root)

    dirty = _clone(pinned_checkout, tmp_path / "dirty")
    (dirty / "operator-note.txt").write_text("not image-owned\n", encoding="utf-8")
    before = sha256_file(dirty / "src" / "maxtext" / "layers" / "moe.py")
    with pytest.raises(series.MaxTextPatchSeriesError, match="not clean"):
        series.apply_pinned_maxtext_patch_series(dirty, patch_root=patch_root)
    assert sha256_file(dirty / "src" / "maxtext" / "layers" / "moe.py") == before


def test_patch_hash_drift_fails_before_git_apply(
    pinned_checkout: Path,
    patch_root: Path,
    tmp_path: Path,
) -> None:
    checkout = _clone(pinned_checkout, tmp_path / "maxtext")
    drifted = tmp_path / "patches"
    drifted.mkdir()
    for name, _digest in series.PATCH_SERIES:
        shutil.copy2(patch_root / name, drifted / name)
    target = drifted / series.PATCH_SERIES[1][0]
    payload = bytearray(target.read_bytes())
    payload[-1] ^= 1
    target.write_bytes(payload)

    with pytest.raises(series.MaxTextPatchSeriesError, match="SHA-256 drift"):
        series.apply_pinned_maxtext_patch_series(checkout, patch_root=drifted)
    assert _status(checkout) == ()


def test_git_apply_check_is_ordered_and_precedes_mutation(
    pinned_checkout: Path,
    patch_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = _clone(pinned_checkout, tmp_path / "maxtext")
    staged = tmp_path / "patches"
    staged.mkdir()
    first_name, first_sha = series.PATCH_SERIES[0]
    shutil.copy2(patch_root / first_name, staged / first_name)
    broken_name = "0002-broken.patch"
    broken = (
        b"diff --git a/src/maxtext/layers/nnx_decoders.py "
        b"b/src/maxtext/layers/nnx_decoders.py\n"
        b"--- a/src/maxtext/layers/nnx_decoders.py\n"
        b"+++ b/src/maxtext/layers/nnx_decoders.py\n"
        b"@@ -1,1 +1,1 @@\n-not-the-base\n+never-applied\n"
    )
    (staged / broken_name).write_bytes(broken)
    monkeypatch.setattr(
        series,
        "PATCH_SERIES",
        (
            (first_name, first_sha),
            (broken_name, hashlib.sha256(broken).hexdigest()),
        ),
    )

    with pytest.raises(series.MaxTextPatchSeriesError, match="git apply --check"):
        series.apply_pinned_maxtext_patch_series(checkout, patch_root=staged)
    assert _status(checkout) == ()
    assert (
        sha256_file(checkout / "src" / "maxtext" / "layers" / "moe.py")
        == BASE_MOE_SHA256
    )


def test_post_apply_hash_drift_never_returns_attestation(
    pinned_checkout: Path,
    patch_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = _clone(pinned_checkout, tmp_path / "maxtext")
    wrong = dict(series.POST_APPLY_SHA256)
    wrong["src/maxtext/layers/nnx_decoders.py"] = "0" * 64
    monkeypatch.setattr(series, "POST_APPLY_SHA256", wrong)

    with pytest.raises(series.MaxTextPatchSeriesError, match="post-apply file"):
        series.apply_pinned_maxtext_patch_series(checkout, patch_root=patch_root)
    assert _status(checkout) == series.EXPECTED_STATUS


def test_symlinked_checkout_is_not_owned(
    pinned_checkout: Path,
    patch_root: Path,
    tmp_path: Path,
) -> None:
    checkout = _clone(pinned_checkout, tmp_path / "maxtext")
    linked = tmp_path / "linked"
    linked.symlink_to(checkout, target_is_directory=True)
    with pytest.raises(series.MaxTextPatchSeriesError, match="non-symlink"):
        series.apply_pinned_maxtext_patch_series(linked, patch_root=patch_root)


def test_native_moe_guard_is_on_moe_only_and_target_returns_three_arrays(
    applied_checkout: Path,
) -> None:
    path = applied_checkout / "src/maxtext/models/deepseek.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    dense_call = _class_method(tree, "DeepSeekDenseLayer", "__call__")
    moe_call = _class_method(tree, "DeepSeekMoELayer", "__call__")
    target = _class_method(tree, "DeepSeekMoELayer", "r1_forward_routed_target")

    assert "r1_force_native_moe" not in {
        argument.arg for argument in dense_call.args.args
    }
    assert "r1_force_native_moe" in {
        argument.arg for argument in moe_call.args.args
    }
    moe_source = ast.get_source_segment(source, moe_call)
    assert moe_source is not None
    assert (
        "self.config.use_batch_split_schedule and not r1_force_native_moe"
        in moe_source
    )

    target_source = ast.get_source_segment(source, target)
    assert target_source is not None
    assert ".routed_moe" in target_source
    assert "shared_experts" not in target_source
    assert "decoder_segment_ids" in target_source
    returns = [node for node in ast.walk(target) if isinstance(node, ast.Return)]
    assert len(returns) == 1
    assert isinstance(returns[0].value, ast.Tuple)
    assert len(returns[0].value.elts) == 3


def test_native_prefix_is_thin_scanned_and_has_exact_overlay_partition(
    applied_checkout: Path,
) -> None:
    path = applied_checkout / "src/maxtext/models/deepseek_prefix.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    assert "deepseek_batchsplit" not in source
    assert "batchsplit." not in source
    assert ".route(" not in source
    assert ".unroute(" not in source
    assert "jax.lax.scan(" in source
    assert "r1_force_native_moe=True" in source
    assert "nnx.replace_by_pure_dict" in source
    assert "target overlay template aliases the scanned source state" in source

    target_specs = _assignment(tree, "_TARGET_PARAM_SPECS")
    assert isinstance(target_specs, ast.Dict)
    assert len(target_specs.keys) == 14
    shared_paths = _assignment(tree, "_SHARED_TEMPLATE_PATHS")
    assert isinstance(shared_paths, ast.Call)
    assert isinstance(shared_paths.args[0], ast.Set)
    assert len(shared_paths.args[0].elts) == 3

    completed = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_run_completed_layers"
    )
    completed_source = ast.get_source_segment(source, completed)
    assert completed_source is not None
    assert "segment_ids" in completed_source
    assert "positions" in completed_source
    assert "nnx.state(" not in completed_source
    assert "nnx.update(" not in completed_source


def test_native_sparse_compute_preserves_batchsplit_expert_storage_axis(
    applied_checkout: Path,
) -> None:
    source = (
        applied_checkout / "src/maxtext/layers/moe.py"
    ).read_text(encoding="utf-8")
    assert (
        'expert_axis = "expert_only" if '
        'self.config.use_batch_split_schedule else "exp"'
    ) in source
    assert (
        'self._logical_to_mesh_axes((expert_axis, None, "mlp_no_fsdp"))'
        in source
    )
    assert "def r1_direct_router_metadata(" in source
    assert "top_k_indices.astype(jnp.int32)" in source
    assert "top_k_weights.astype(jnp.float32)" in source
