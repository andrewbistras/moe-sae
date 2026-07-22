from __future__ import annotations

import json
from pathlib import Path

import pytest

from moe_sae.sae_paths import SAEOperationPaths
from moe_sae.sae_phase_results import (
    SAEPhaseResultError,
    phase_result_path,
    publish_phase_result,
    validate_phase_result,
)


SHA = "a" * 64


def paths(tmp_path: Path) -> SAEOperationPaths:
    return SAEOperationPaths(tmp_path / "artifacts", SHA)


def test_publish_and_revalidate_exact_result(tmp_path: Path) -> None:
    selected = paths(tmp_path)
    result = publish_phase_result(
        selected,
        "gate-A-seed-1",
        {"tokens_seen": 5_013_504, "health": {"passed": True}},
        created_at_utc="2026-07-22T12:00:00+00:00",
    )
    assert len(result) == 64
    assert validate_phase_result(selected, "gate-A-seed-1") == result
    assert phase_result_path(selected, "gate-A-seed-1").is_file()


def test_same_evidence_is_idempotent_even_with_another_timestamp(
    tmp_path: Path,
) -> None:
    selected = paths(tmp_path)
    evidence = {"passed": True, "nested": [1, "two", None]}
    first = publish_phase_result(
        selected,
        "validate-prerequisites",
        evidence,
        created_at_utc="first",
    )
    second = publish_phase_result(
        selected,
        "validate-prerequisites",
        evidence,
        created_at_utc="second",
    )
    assert second == first
    record = json.loads(
        phase_result_path(selected, "validate-prerequisites").read_text()
    )
    assert record["created_at_utc"] == "first"


def test_conflicting_evidence_cannot_replace_result(tmp_path: Path) -> None:
    selected = paths(tmp_path)
    publish_phase_result(selected, "smoke-F", {"outcome": "one"})
    with pytest.raises(SAEPhaseResultError, match="conflicting immutable"):
        publish_phase_result(selected, "smoke-F", {"outcome": "two"})


@pytest.mark.parametrize(
    "field,value",
    (
        ("evidence_sha256", "0" * 64),
        ("result_sha256", "1" * 64),
        ("record_sha256", "2" * 64),
        ("operation_sha256", "3" * 64),
    ),
)
def test_tampering_fails_closed(
    tmp_path: Path, field: str, value: str
) -> None:
    selected = paths(tmp_path)
    path = phase_result_path(selected, "smoke-A")
    publish_phase_result(selected, "smoke-A", {"passed": True})
    record = json.loads(path.read_text())
    record[field] = value
    path.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
    with pytest.raises(SAEPhaseResultError):
        validate_phase_result(selected, "smoke-A")


def test_noncanonical_bytes_fail_even_if_value_is_valid(tmp_path: Path) -> None:
    selected = paths(tmp_path)
    path = phase_result_path(selected, "smoke-B")
    publish_phase_result(selected, "smoke-B", {"passed": True})
    record = json.loads(path.read_text())
    path.write_text(json.dumps(record, sort_keys=True) + "\n")
    with pytest.raises(SAEPhaseResultError, match="canonically encoded"):
        validate_phase_result(selected, "smoke-B")


def test_rejects_non_json_evidence_and_unknown_phase(tmp_path: Path) -> None:
    selected = paths(tmp_path)
    with pytest.raises(SAEPhaseResultError, match="strict JSON"):
        publish_phase_result(selected, "smoke-F", {"value": float("nan")})
    with pytest.raises(SAEPhaseResultError, match="unknown SAE phase"):
        phase_result_path(selected, "not-a-phase")
