"""Validation-gate math (spec §3): per-token relative error against naive
references, the zero-norm guard, gate verdict logic, the token-budget check,
and the correlation flagging used by the BF16 report. Synthetic tensors only —
no model (run_validation is exercised with the model phases monkeypatched)."""

import numpy as np
import pytest
import torch

import moe_sae.capture.validate as validate
from moe_sae.capture.validate import (
    CORR_FLAG_MESSAGE,
    GATE_REL_TOL,
    _TokenAccumulator,
    correlation_report,
    gate_verdict,
    per_token_rel_err,
    per_token_slot_rel_err,
    spearman_rho,
    summarize_rel,
    token_budget_verdict,
)


# --------------------------------------------------------------------------- #
# per-token relative error                                                     #
# --------------------------------------------------------------------------- #


def test_per_token_rel_err_matches_naive():
    torch.manual_seed(0)
    agg = torch.randn(50, 16)
    csum = agg + 1e-4 * torch.randn(50, 16)

    rel, zero = per_token_rel_err(csum, agg)
    assert rel.dtype == torch.float32
    assert not zero.any()

    # naive per-token reference over the SAME fp32 inputs, in float64
    naive = torch.tensor(
        [
            float(
                torch.linalg.vector_norm(csum[t].double() - agg[t].double())
                / torch.linalg.vector_norm(agg[t].double())
            )
            for t in range(50)
        ],
        dtype=torch.float64,
    )
    assert torch.allclose(rel.double(), naive, rtol=1e-5, atol=1e-12)


def test_per_token_rel_err_zero_norm_guard():
    torch.manual_seed(1)
    agg = torch.randn(8, 4)
    agg[3] = 0.0  # zero-norm aggregate: report separately, never divide
    csum = torch.randn(8, 4)

    rel, zero = per_token_rel_err(csum, agg)
    assert zero.tolist() == [False, False, False, True, False, False, False, False]
    assert torch.isfinite(rel).all()
    assert rel[3] == 0.0
    # unmasked rows unaffected by the guard
    naive2 = (csum[2] - agg[2]).norm() / agg[2].norm()
    assert torch.allclose(rel[2], naive2)


def test_per_token_rel_err_upcasts_bf16():
    torch.manual_seed(2)
    agg = torch.randn(10, 8)
    rel, zero = per_token_rel_err(agg.bfloat16(), agg.bfloat16())
    assert rel.dtype == torch.float32
    assert torch.isfinite(rel).all()
    assert (rel == 0).all()  # identical inputs after upcast


def test_per_token_slot_rel_err_matches_naive():
    torch.manual_seed(3)
    ref = torch.randn(20, 6, 8)
    prod = ref + 1e-4 * torch.randn(20, 6, 8)

    rel, zero = per_token_slot_rel_err(prod, ref)
    assert not zero.any()
    naive = torch.tensor(
        [
            float(
                torch.linalg.matrix_norm(prod[t].double() - ref[t].double())
                / torch.linalg.matrix_norm(ref[t].double())
            )
            for t in range(20)
        ],
        dtype=torch.float64,
    )
    assert torch.allclose(rel.double(), naive, rtol=1e-5, atol=1e-12)


def test_per_token_slot_rel_err_zero_guard():
    ref = torch.zeros(3, 2, 4)
    ref[0] = 1.0
    prod = ref.clone()
    prod[1] += 0.5  # discrepancy at a zero-norm reference token
    rel, zero = per_token_slot_rel_err(prod, ref)
    assert zero.tolist() == [False, True, True]
    assert torch.isfinite(rel).all()
    assert rel[0] == 0.0


# --------------------------------------------------------------------------- #
# gate verdict                                                                 #
# --------------------------------------------------------------------------- #


def test_gate_verdict_pass():
    rel = np.full(1000, 1e-7)
    v = gate_verdict(rel)
    assert v["status"] == "PASS"
    assert v["n_violations"] == 0
    assert v["n_tokens"] == 1000
    assert v["tol"] == GATE_REL_TOL == 1e-5  # frozen, spec §3


def test_gate_verdict_strict_inequality():
    # spec §3: "< 1e-5 for every token" — a token AT the threshold fails
    v = gate_verdict(np.array([1e-7, GATE_REL_TOL]))
    assert v["status"] == "FAIL"
    assert v["n_violations"] == 1


def test_gate_verdict_worst_offenders_ranked():
    rel = np.array([1e-7, 5e-5, 2e-5])
    v = gate_verdict(
        rel,
        example_id=np.array([10, 11, 12]),
        token_pos=np.array([0, 1, 2]),
        n_worst=2,
    )
    assert v["status"] == "FAIL"
    assert v["n_violations"] == 2
    assert len(v["worst"]) == 2
    assert v["worst"][0] == {"rel_err": 5e-5, "example_id": 11, "token": 1}
    assert v["worst"][1] == {"rel_err": 2e-5, "example_id": 12, "token": 2}


def test_gate_verdict_empty_cannot_pass():
    v = gate_verdict(np.array([]))
    assert v["status"] == "FAIL"
    assert "note" in v


# --------------------------------------------------------------------------- #
# token budget (spec §3: PASS may not stand on fewer tokens than requested)    #
# --------------------------------------------------------------------------- #


def test_token_budget_verdict():
    # sufficient: gate unchanged either way
    gate, budget = token_budget_verdict("PASS", 100, 100)
    assert gate == "PASS"
    assert budget == {"requested": 100, "actual": 100, "sufficient": True}
    gate, _ = token_budget_verdict("PASS", 100, 150)
    assert gate == "PASS"

    # insufficient: a PASS is demoted, never allowed to stand
    gate, budget = token_budget_verdict("PASS", 100, 99)
    assert gate == "INSUFFICIENT_TOKENS"
    assert budget["sufficient"] is False
    assert budget["requested"] == 100 and budget["actual"] == 99
    assert "note" in budget

    # insufficient + FAIL: FAIL is more severe and stays, shortfall recorded
    gate, budget = token_budget_verdict("FAIL", 100, 50)
    assert gate == "FAIL"
    assert budget["sufficient"] is False


def _fake_gate_acc(n: int) -> _TokenAccumulator:
    """Synthetic all-passing FP32-gate accumulator with n tokens."""
    acc = _TokenAccumulator()
    acc.add(
        rel=np.full(n, 1e-8),
        zero=np.zeros(n, dtype=bool),
        xrel=np.full(n, 1e-8),
        xzero=np.zeros(n, dtype=bool),
        example_id=np.zeros(n, dtype=np.int64),
        token_pos=np.arange(n, dtype=np.int64),
    )
    acc.fields.setdefault("domain", []).append(np.array(["coco"] * n))
    acc.fields.setdefault("zero_contrib_norm", []).append(np.array([]))
    return acc


def _write_dummy_inputs(tmp_path, n_files=2):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    for i in range(n_files):
        torch.save(
            {"input_ids": torch.zeros(4, dtype=torch.long)},
            inputs / f"examples_{i:04d}.pt",
        )
    return inputs


def test_run_validation_refuses_pass_on_insufficient_tokens(tmp_path, monkeypatch):
    # Regression: inputs dir too small — the gate phase accumulates fewer
    # tokens than requested, every one of them passing. run_validation must
    # NOT report gate PASS.
    inputs = _write_dummy_inputs(tmp_path)

    def fake_phase(files, n_tokens, device, model_id, chunk_examples):
        return _fake_gate_acc(50), list(files)

    monkeypatch.setattr(validate, "_run_fp32_gate_phase", fake_phase)
    report = validate.run_validation(
        inputs, tmp_path / "report.json", n_tokens=100, device="cpu",
        bf16_report=False,
    )
    assert report["fp32"]["status"] == "PASS"  # per-token criterion did pass
    assert report["gate"] == "INSUFFICIENT_TOKENS"  # ...but the budget did not
    assert report["gate"] != "PASS"
    assert report["token_budget"]["requested"] == 100
    assert report["token_budget"]["actual"] == 50
    assert report["token_budget"]["sufficient"] is False


def test_run_validation_passes_when_budget_met(tmp_path, monkeypatch):
    inputs = _write_dummy_inputs(tmp_path)

    def fake_phase(files, n_tokens, device, model_id, chunk_examples):
        return _fake_gate_acc(120), list(files)

    monkeypatch.setattr(validate, "_run_fp32_gate_phase", fake_phase)
    report = validate.run_validation(
        inputs, tmp_path / "report.json", n_tokens=100, device="cpu",
        bf16_report=False,
    )
    assert report["gate"] == "PASS"
    assert report["token_budget"] == {
        "requested": 100, "actual": 120, "sufficient": True,
    }


# --------------------------------------------------------------------------- #
# summary statistics                                                           #
# --------------------------------------------------------------------------- #


def test_summarize_rel():
    rng = np.random.default_rng(0)
    rel = rng.uniform(size=500)
    s = summarize_rel(rel)
    assert s["n_tokens"] == 500
    assert s["max"] == pytest.approx(float(rel.max()))
    assert s["median"] == pytest.approx(float(np.median(rel)))
    assert s["p99"] == pytest.approx(float(np.percentile(rel, 99)))

    empty = summarize_rel(np.array([]))
    assert empty == {"n_tokens": 0, "max": None, "median": None, "p99": None}


# --------------------------------------------------------------------------- #
# Spearman structure diagnostics (BF16 report)                                 #
# --------------------------------------------------------------------------- #


def test_spearman_monotonic():
    x = np.arange(100, dtype=float)
    assert spearman_rho(x, 2 * x + 1) == pytest.approx(1.0)
    assert spearman_rho(x, -(x**3)) == pytest.approx(-1.0)  # rank-based, nonlinear ok


def test_spearman_undefined_inputs():
    x = np.arange(10, dtype=float)
    assert spearman_rho(x, np.full(10, 3.0)) is None  # constant covariate
    assert spearman_rho(np.full(10, 3.0), x) is None
    assert spearman_rho(np.array([1.0]), np.array([2.0])) is None  # n < 2


def test_correlation_report_flags_structure():
    rng = np.random.default_rng(1)
    err = np.arange(200, dtype=float)
    report = correlation_report(
        err,
        {
            "strong": err + rng.normal(scale=1.0, size=200),  # near-perfect rank corr
            "none": np.full(200, 0.5),  # undefined
            "weak": rng.permutation(err),  # shuffled: |rho| ~ 0
        },
    )
    assert report["strong"]["rho"] > 0.3
    assert report["strong"]["flag"] == CORR_FLAG_MESSAGE
    assert report["none"]["rho"] is None and "flag" not in report["none"]
    assert abs(report["weak"]["rho"]) < 0.3 and "flag" not in report["weak"]
