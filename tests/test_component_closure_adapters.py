from __future__ import annotations

from moe_sae.component_closure_worker import (
    MATURE_WORKER_MODULE,
    WORKER_MODULE,
    _replace_worker_module,
    install_worker_adapter,
)
from moe_sae.component_closure_config import PHASE_NAME
from moe_sae.component_closure_evaluation import partition_bundle_retry


def test_worker_supervisor_child_stays_behind_closure_loader() -> None:
    assert _replace_worker_module(
        ["python", "-m", MATURE_WORKER_MODULE, "train-child"]
    ) == ["python", "-m", WORKER_MODULE, "train-child"]
    from moe_sae import contribution_training_worker as worker

    original = {
        "LONG_HORIZON_PHASES": worker.LONG_HORIZON_PHASES,
        "_load_worker_config": worker._load_worker_config,
        "_finalized_horizon_config": worker._finalized_horizon_config,
        "_supervise": worker._supervise,
    }
    try:
        installed = install_worker_adapter()
        assert PHASE_NAME in installed.LONG_HORIZON_PHASES
    finally:
        for name, value in original.items():
            setattr(worker, name, value)


def test_resident_bundle_retry_skips_completed_subset() -> None:
    bundle = {
        "cells": [
            {"logical_condition": "A", "checkpoint_tokens": 2_000_000},
            {"logical_condition": "R", "checkpoint_tokens": 2_000_000},
            {"logical_condition": "B", "checkpoint_tokens": 2_000_000},
        ]
    }
    completed_keys = {("A", 2_000_000), ("B", 2_000_000)}

    def inspect(cell: dict) -> dict | None:
        key = (cell["logical_condition"], cell["checkpoint_tokens"])
        return {"result_sha256": "a" * 64} if key in completed_keys else None

    completed, pending = partition_bundle_retry(bundle, inspect)
    assert completed == (0, 2)
    assert pending == (1,)
