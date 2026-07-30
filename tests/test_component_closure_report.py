from __future__ import annotations

import copy

import pytest

from moe_sae.component_closure_config import (
    CHECKPOINT_TOKENS,
    CONDITIONS,
    PANELS,
    PROTOCOL_ID,
    SEEDS,
)
from moe_sae.component_closure_report import (
    EVIDENCE_SCHEMA_VERSION,
    ComponentClosureReportError,
    build_closure_report,
    render_markdown,
    validate_score_grid,
)
from moe_sae.contribution_artifacts import canonical_sha256

SHA = "a" * 64
REVISION = "b" * 40


def _evidence() -> dict:
    # EV values make F-B=0.50 and A-B=0.30, hence G=60%.
    ev = {"A": 0.70, "R": 0.55, "B": 0.40, "F": 0.90}
    cells = []
    for panel in PANELS:
        for seed in SEEDS:
            for checkpoint in CHECKPOINT_TOKENS:
                for condition in CONDITIONS:
                    sst = [1.0, 2.0, 3.0, 4.0]
                    cells.append(
                        {
                            "panel": panel,
                            "seed": seed,
                            "checkpoint_tokens": checkpoint,
                            "condition": condition,
                            "sequence_id": [10, 11, 12, 13],
                            "component_sse": [
                                value * (1.0 - ev[condition]) for value in sst
                            ],
                            "component_sst": sst,
                            "evaluation_result_sha256": SHA,
                        }
                    )
    costs = []
    for panel in PANELS:
        for seed in SEEDS:
            for condition in CONDITIONS:
                wall = {"A": 10.0, "R": 11.0, "B": 5.0, "F": 20.0}[condition]
                costs.append(
                    {
                        "panel": panel,
                        "seed": seed,
                        "condition": condition,
                        "accelerator_type": "v5litepod-16",
                        "chip_count": 16,
                        "marginal_wall_seconds": wall,
                        "chip_seconds": 16 * wall,
                        "chip_seconds_scope": "optimizer_loop",
                        "chip_seconds_excludes": ["compile", "evaluation"],
                        "training_source_revision": REVISION,
                    }
                )
    body = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "cells": cells,
        "costs": costs,
    }
    return {**body, "evidence_sha256": canonical_sha256(body)}


def test_report_computes_closure_contrasts_g_and_cost_compatibility() -> None:
    report = build_closure_report(_evidence(), bootstrap_replicates=200)
    row = report["panels"]["sae"]["milestones"]["10000000"]
    assert row["contrasts"]["A-R"]["estimate"] == pytest.approx(0.15)
    assert row["contrasts"]["F-B"]["estimate"] == pytest.approx(0.50)
    assert row["g"]["estimate"] == pytest.approx(60.0)
    assert row["g"]["status"] == "defined"
    assert report["panels"]["sae"]["cost_compatibility"][
        "a_over_f_all_seeds_comparable"
    ]
    assert "| 10,000,000 |" in render_markdown(report)


def test_report_rejects_noncommon_sst() -> None:
    value = _evidence()
    value["cells"][1]["component_sst"][0] = 9.0
    body = {key: item for key, item in value.items() if key != "evidence_sha256"}
    value["evidence_sha256"] = canonical_sha256(body)
    with pytest.raises(ComponentClosureReportError):
        validate_score_grid(value)
