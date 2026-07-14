from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

import moe_sae.cli as cli_module
from moe_sae.config import Gemma4Config


_CHECKPOINT_SHA256 = "a" * 64
_ENVIRONMENT_SHA256 = "b" * 64
_TRAIN_MANIFEST_SHA256 = "c" * 64
_EVAL_MANIFEST_SHA256 = "d" * 64
_CONVERSION_ATTESTATION_SHA256 = "9" * 64


class _Capture:
    def __init__(self, *, tokens: int, digest: str, sst: float | None = None):
        self.total_tokens = tokens
        self.manifest_sha256 = digest
        self.manifest = {
            "metadata": {
                "checkpoint_sha256": _CHECKPOINT_SHA256,
                "environment_sha256": _ENVIRONMENT_SHA256,
                "conversion_attestation_sha256": _CONVERSION_ATTESTATION_SHA256,
            }
        }
        self._sst = sst

    @property
    def mean(self) -> np.ndarray:
        return np.zeros(2_816, dtype=np.float64)

    def sst_about(self, mean: np.ndarray) -> float:
        assert mean.shape == (2_816,)
        assert self._sst is not None
        return self._sst


def _report_args(
    tmp_path,
    monkeypatch,
    *,
    sealed_eval_sst: float,
    recorded_sst: float,
    sse: float,
    ev: float,
    nmse: float,
):
    config = Gemma4Config()
    train_capture = _Capture(
        tokens=config.train_tokens,
        digest=_TRAIN_MANIFEST_SHA256,
    )
    eval_capture = _Capture(
        tokens=config.eval_tokens,
        digest=_EVAL_MANIFEST_SHA256,
        sst=sealed_eval_sst,
    )

    monkeypatch.setattr(cli_module, "_config", lambda args: config)
    monkeypatch.setattr(
        cli_module, "_require_production_artifact_root", lambda root: None
    )
    monkeypatch.setattr(
        cli_module,
        "_sealed_dataset",
        lambda path: eval_capture if path.name == "eval" else train_capture,
    )
    monkeypatch.setattr(cli_module, "_require_exact_capture", lambda *args: None)
    monkeypatch.setattr(
        cli_module, "_require_capture_identity", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        cli_module,
        "_load_harvester_report",
        lambda *args, **kwargs: {
            "checkpoint_sha256": _CHECKPOINT_SHA256,
            "chip_seconds": 100.0,
        },
    )
    monkeypatch.setattr(
        cli_module,
        "_coordinated_environment",
        lambda: {"sha256": _ENVIRONMENT_SHA256},
    )
    monkeypatch.setattr(
        cli_module,
        "_load_conversion_attestation",
        lambda *args: _CONVERSION_ATTESTATION_SHA256,
    )
    monkeypatch.setattr(cli_module, "_process_context", lambda: (0, 1))
    monkeypatch.setattr(cli_module, "_global_barrier", lambda label: None)

    run_ids: dict[tuple[str, int], str] = {}
    for index, (condition, seed) in enumerate(
        ((condition, seed) for condition in config.conditions for seed in config.seeds),
        start=1,
    ):
        run_instance_id = f"{index:032x}"
        run_ids[(condition, seed)] = run_instance_id
        evaluation = {
            "config_sha256": config.config_hash,
            "condition": condition,
            "seed": seed,
            "tokens": config.eval_tokens,
            "sse": sse,
            "sst": recorded_sst,
            "ev": ev,
            "nmse": nmse,
            "l0": 64.0,
            "dead_rate": 0.01,
            "topk_valid": True,
            "chip_seconds": 10.0 if condition == "A" else 100.0,
            "wall_seconds": 1.0,
            "tokens_per_second": 1_000.0,
            "evaluation_seconds_excluded": 0.5,
            "train_capture_manifest_sha256": _TRAIN_MANIFEST_SHA256,
            "eval_capture_manifest_sha256": _EVAL_MANIFEST_SHA256,
            "capture_checkpoint_sha256": _CHECKPOINT_SHA256,
            "environment": {"sha256": _ENVIRONMENT_SHA256},
            "training_environment_sha256": _ENVIRONMENT_SHA256,
            "run_instance_id": run_instance_id,
        }
        evaluation_path = cli_module._evaluation_path(tmp_path, condition, seed)
        evaluation_path.parent.mkdir(parents=True, exist_ok=True)
        evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")

        chip_seconds = evaluation["chip_seconds"]
        marginal_seconds = chip_seconds / 16.0
        capacity_seconds = 0.125 if condition == "A" else 0.0
        optimizer_seconds = marginal_seconds - capacity_seconds
        run_record = {
            "config_sha256": config.config_hash,
            "condition": condition,
            "seed": seed,
            "authoritative": True,
            "tokens_seen": config.train_tokens,
            "chip_count": 16,
            "health": {"passed": True},
            "capture_manifest_sha256": _TRAIN_MANIFEST_SHA256,
            "capture_checkpoint_sha256": _CHECKPOINT_SHA256,
            "run_instance_id": run_instance_id,
            "environment": {"sha256": _ENVIRONMENT_SHA256},
            "optimizer_wall_seconds": optimizer_seconds,
            "a_capacity_precompute_wall_seconds": capacity_seconds,
            "marginal_wall_seconds": marginal_seconds,
            "chip_seconds": chip_seconds,
            "tokens_per_second": config.train_tokens / marginal_seconds,
            "optimizer_tokens_per_second": config.train_tokens / optimizer_seconds,
            "compile_seconds": 0.0,
            "checkpoint_seconds": 0.0,
        }
        run_path = cli_module._run_root(tmp_path, condition, seed) / "run.json"
        run_path.parent.mkdir(parents=True, exist_ok=True)
        run_path.write_text(json.dumps(run_record), encoding="utf-8")

    monkeypatch.setattr(
        cli_module,
        "_load_run_identity",
        lambda path, expected: run_ids[(expected["condition"], expected["seed"])],
    )
    return SimpleNamespace(
        config="unused.yaml",
        artifact_root=str(tmp_path),
        harvester=None,
        output=str(tmp_path / "report.json"),
    )


def test_report_rejects_stored_ev_nmse_inconsistent_with_sse(tmp_path, monkeypatch):
    args = _report_args(
        tmp_path,
        monkeypatch,
        sealed_eval_sst=100.0,
        recorded_sst=100.0,
        sse=25.0,
        ev=0.90,
        nmse=0.10,
    )

    with pytest.raises(RuntimeError):
        cli_module._command_report(args)


def test_report_rejects_sst_not_derived_from_sealed_captures(tmp_path, monkeypatch):
    args = _report_args(
        tmp_path,
        monkeypatch,
        sealed_eval_sst=100.0,
        recorded_sst=80.0,
        sse=20.0,
        ev=0.75,
        nmse=0.25,
    )

    with pytest.raises(RuntimeError):
        cli_module._command_report(args)
