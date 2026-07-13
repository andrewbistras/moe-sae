"""A15 draw-epoch robustness (moe_sae.analysis.draw_epoch): draw-count
alignment from capture shards to the TRAIN-split dump, stratification, and
the per-stratum §16.1/§16.4 recomputation."""

import numpy as np
import pytest
import torch

from test_eval_reconstruction import FakeRoutedSAE, build_capture

from moe_sae.analysis.draw_epoch import (
    draw_epoch_robustness,
    load_text_draw_counts,
    per_token_draw_counts,
)
from moe_sae.analysis.mass import load_dump, token_bank_masses
from moe_sae.capture.writer import iter_shards
from moe_sae.data.splits import split_mask
from moe_sae.eval.reconstruction import evaluate_reconstruction


@pytest.fixture()
def train_dump(tmp_path):
    build_capture(tmp_path / "cap", n_per_pool=12)
    sae = FakeRoutedSAE()
    evaluate_reconstruction(sae, 0.8, tmp_path / "cap", "train", tmp_path / "dump")
    return tmp_path / "cap", load_dump(tmp_path / "dump")


def planted_counts(cap_root):
    """Per text shard: draw count = 1 + (original row % 3) → ≥2 strata among
    the kept rows."""
    counts = {}
    for i, shard in enumerate(iter_shards(cap_root, "text")):
        n = shard["domain"].shape[0]
        counts[i] = 1 + (np.arange(n, dtype=np.int64) % 3)
    return counts


def test_per_token_draw_counts_alignment(train_dump):
    cap_root, dump = train_dump
    counts = planted_counts(cap_root)
    per_tok = per_token_draw_counts(dump, cap_root, counts, split="train")
    vis = dump.token_type == 1
    assert (per_tok[vis] == -1).all()  # visual: undefined
    assert (per_tok[~vis] >= 1).all()  # text: planted counts
    # exact alignment: walk the capture text shards' kept rows in order
    expected = []
    for i, shard in enumerate(iter_shards(cap_root, "text")):
        keep = split_mask(shard["domain"], shard["example_id"], "train").numpy()
        expected.extend(counts[i][np.nonzero(keep)[0]].tolist())
    np.testing.assert_array_equal(per_tok[~vis], expected)


def test_per_token_draw_counts_misalignment_raises(train_dump):
    cap_root, dump = train_dump
    counts = planted_counts(cap_root)
    bad = dump  # corrupt the example ids → alignment check must fire
    bad.example_id = bad.example_id.copy()
    bad.example_id[-1] += 1
    with pytest.raises(AssertionError, match="alignment"):
        per_token_draw_counts(bad, cap_root, counts, split="train")


def test_draw_epoch_robustness_report(train_dump):
    cap_root, dump = train_dump
    per_tok = per_token_draw_counts(dump, cap_root, planted_counts(cap_root))
    rep = draw_epoch_robustness(dump, token_bank_masses(dump), per_tok, seed=0)
    assert rep["n_strata"] >= 2
    labels = [r["stratum"] for r in rep["strata"]]
    assert len(set(labels)) == rep["n_strata"]
    for r in rep["strata"]:
        assert r["n_text_tokens"] > 0
        assert r["jsd_image_vs_stratum_text"] is None or (
            0.0 <= r["jsd_image_vs_stratum_text"] <= 1.0
        )
        assert 0.0 <= r["probe1_router_gates_acc"] <= 1.0
        assert 0.0 <= r["probe3_feature_identity_acc"] <= 1.0
        assert r["probe3_minus_probe1_acc"] == pytest.approx(
            r["probe3_feature_identity_acc"] - r["probe1_router_gates_acc"]
        )
    mono = rep["monotonicity_spearman"]
    for key in (
        "jsd_image_vs_stratum_text",
        "probe1_router_gates_acc",
        "probe3_feature_identity_acc",
        "probe3_minus_probe1_acc",
    ):
        assert key in mono
        assert mono[key] is None or -1.0 <= mono[key] <= 1.0
    assert "TRAIN-split" in rep["interpretation"]  # recorded decision


def test_load_text_draw_counts_roundtrip(tmp_path, train_dump):
    cap_root, dump = train_dump
    counts = planted_counts(cap_root)
    np.savez(
        str(tmp_path / "text_draw_counts.npz"),
        **{f"shard_{i:05d}": a for i, a in counts.items()},
    )
    loaded = load_text_draw_counts(tmp_path / "text_draw_counts.npz")
    assert set(loaded) == set(counts)
    for i in counts:
        np.testing.assert_array_equal(loaded[i], counts[i])
    per_tok = per_token_draw_counts(dump, cap_root, loaded)
    assert (per_tok[dump.token_type == 0] >= 1).all()
