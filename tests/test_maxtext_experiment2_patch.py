from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PATCH_DIR = ROOT / "patches" / "maxtext"
MANIFEST_PATH = PATCH_DIR / "selected_outputs_manifest.json"
PATCH_PATH = PATCH_DIR / "0001-gemma4-return-selected-outputs.patch"
VERIFY_PATH = PATCH_DIR / "verify_selected_outputs_patch.py"


def _load_verifier():
    spec = importlib.util.spec_from_file_location(
        "verify_selected_outputs_patch", VERIFY_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manifest_pins_exact_maxtext_source_and_return_contract() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["base_commit"] == "48c7da3e28e8f57fe36ddf09d00eec53072a2ea0"
    assert manifest["fork_repository"] == (
        "https://github.com/andrewbistras/maxtext.git"
    )
    assert manifest["fork_branch"] == "gemma4-contribution-v1"
    assert manifest["fork_commit"] == (
        "73bafe53569eb17dc18542f2568dd591a8db9e28"
    )
    assert manifest["patch_sha256"] == (
        "9da1ec2a0792a569123093c8840c20d9ee09891854f2665a972d3d7340d8c42d"
    )
    assert manifest["files"] == {
        "src/maxtext/layers/moe.py": {
            "base_git_blob_sha1": "afa2799f9b542d21e1f84de3aca96a697c3161ed",
            "base_sha256": "8850cfe6dcf65fa1a8059ce0a4ab3036057cf8757f4526d97cd8267d5e24269a",
            "patched_git_blob_sha1": "1b9c6ed087ab9efa7df84ac1122c922153199a24",
            "patched_sha256": "e731a50cf30f6bd4295f7891e97c8fdcb104092abf63268bc39aa5e6c8bdc531",
        }
    }
    assert manifest["performance_optimizations"] == [
        "linear_sender_to_expert_ragged_permutation",
        "linear_inverse_permutation",
    ]
    assert manifest["default_return_arity"] == 3
    assert manifest["auxiliary_return_arity"] == 4
    assert manifest["auxiliary_shape"] == "[batch, sequence, top_k, hidden]"


def test_patch_is_source_anchored_and_fail_closed() -> None:
    verifier = _load_verifier()
    manifest = verifier.load_manifest()
    verifier.verify_patch_content(manifest)

    text = PATCH_PATH.read_text(encoding="utf-8")
    assert text.count("diff --git ") == 1
    assert "+++ b/src/maxtext/layers/moe.py" in text
    assert "return_selected_outputs=return_selected_outputs" in text
    assert (
        "+        return output, routing.lb_loss, routing.bias_updates, selected_outputs"
        in text
    )
    assert "return output, routing.lb_loss, routing.bias_updates" in text


def test_inverse_sort_restores_native_flattened_token_slot_order() -> None:
    expert_ids = np.array(
        [
            [[8, 2, 7, 1], [3, 8, 6, 0]],
            [[5, 4, 1, 7], [6, 0, 4, 2]],
        ],
        dtype=np.int32,
    )
    native_outputs = np.arange(expert_ids.size * 3).reshape(*expert_ids.shape, 3)
    sorted_selected_experts = np.argsort(expert_ids.reshape(-1), kind="stable")
    expert_grouped_outputs = native_outputs.reshape(-1, 3)[sorted_selected_experts]

    inverse = np.empty_like(sorted_selected_experts)
    inverse[sorted_selected_experts] = np.arange(sorted_selected_experts.size)
    recovered = expert_grouped_outputs[inverse].reshape(*expert_ids.shape, 3)
    np.testing.assert_array_equal(recovered, native_outputs)

    patch_text = PATCH_PATH.read_text(encoding="utf-8")
    assert "RoutedMoE._invert_permutation(sorted_selected_experts)" in patch_text
    assert "def _linear_local_sort_indices(" in patch_text
    assert ".at[indices].set(positions)" in patch_text
    assert (
        "+          jnp.argsort(sorted_selected_experts)" not in patch_text
    )
    assert (
        '(batch_logical_axis, "activation_norm_length", None, "activation_embed")'
        in patch_text
    )


def test_content_verifier_rejects_mutated_patch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verifier = _load_verifier()
    manifest = verifier.load_manifest()
    mutated = PATCH_PATH.read_text(encoding="utf-8").replace(
        "tensor parallelism != 1", "tensor parallelism unsupported"
    )
    mutated_path = tmp_path / PATCH_PATH.name
    mutated_path.write_text(mutated, encoding="utf-8")
    monkeypatch.setattr(verifier, "HERE", tmp_path)
    manifest["patch_sha256"] = verifier.sha256_file(mutated_path)
    with pytest.raises(verifier.VerificationError, match="missing required anchors"):
        verifier.verify_patch_content(manifest)


def test_verifier_cli_checks_static_content() -> None:
    completed = subprocess.run(
        [sys.executable, str(VERIFY_PATH)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert "verification passed" in completed.stdout


def test_patch_applies_to_optional_pinned_checkout() -> None:
    maxtext_root_value = os.environ.get("MOE_SAE_TEST_MAXTEXT_ROOT")
    if not maxtext_root_value:
        pytest.skip(
            "set MOE_SAE_TEST_MAXTEXT_ROOT for a real pinned-checkout apply test"
        )
    completed = subprocess.run(
        [
            sys.executable,
            str(VERIFY_PATH),
            "--maxtext-root",
            maxtext_root_value,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert "verification passed" in completed.stdout


def test_optional_published_fork_checkout_is_the_exact_one_commit_delta() -> None:
    maxtext_root_value = os.environ.get("MOE_SAE_TEST_MAXTEXT_FORK_ROOT")
    if not maxtext_root_value:
        pytest.skip(
            "set MOE_SAE_TEST_MAXTEXT_FORK_ROOT for a real fork-checkout test"
        )
    completed = subprocess.run(
        [
            sys.executable,
            str(VERIFY_PATH),
            "--maxtext-fork-root",
            maxtext_root_value,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert "verification passed" in completed.stdout
