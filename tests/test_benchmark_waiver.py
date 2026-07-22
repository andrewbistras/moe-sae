from __future__ import annotations

import copy

import pytest

from moe_sae.benchmark_waiver import (
    BenchmarkWaiverError,
    create_performance_waiver,
    selection_gate_status,
    validate_performance_waiver,
)
from moe_sae.harvest import HarvestError, _benchmark_telemetry
from moe_sae.runtime import (
    HARVEST_MESHES,
    V4_32_BF16_PEAK_FLOPS,
    BenchmarkObservation,
)


def observation(*, dropped: int = 0, recompilations: int = 0) -> dict:
    return BenchmarkObservation(
        mesh=HARVEST_MESHES[0],
        per_device_sequence_batch=1,
        synchronized_steps=200,
        elapsed_seconds=12.0,
        device_elapsed_seconds=10.0,
        persistence_seconds=1.5,
        useful_flops=0.10 * V4_32_BF16_PEAK_FLOPS * 10.0,
        output_tokens=3_276_800,
        persisted_tokens=3_276_800,
        input_wait_seconds=0.1,
        recompilations=recompilations,
        dropped_routed_tokens=dropped,
        max_writer_backlog=1,
        instrumentation_equal=True,
    ).as_dict()


def identities() -> dict[str, str]:
    return {
        "config_sha256": "1" * 64,
        "checkpoint_sha256": "2" * 64,
        "conversion_attestation_sha256": "3" * 64,
        "storage_attestation_sha256": "4" * 64,
        "environment_sha256": "5" * 64,
    }


def waiver(selected: dict) -> dict:
    return create_performance_waiver(
        selected,
        minimum_mfu=0.50,
        identities=identities(),
        reason="Operator chose a bounded capture while kernel work continues.",
        operator_identity="operator@example.com",
        operator_authorization="Explicitly authorized activation capture.",
        source_record="/control/attempt/raw.json",
        source_record_sha256="6" * 64,
        observation_producer={
            **identities(),
            "benchmark_sweep_id": "attempt-6",
            "bootstrap_record": "/control/sweep/bootstrap.json",
            "bootstrap_record_sha256": "7" * 64,
        },
        cross_environment_evidence_authorized=False,
        authorized_at_utc="2026-07-22T01:02:03+00:00",
    )


def test_waiver_preserves_failed_gate_and_is_self_hashed():
    selected = observation()
    record = waiver(selected)
    assert selected["passes_hard_gate"] is False
    assert record["actual_passes_hard_gate"] is False
    assert record["actual_useful_mfu"] == pytest.approx(0.10)
    normalized, loaded = selection_gate_status(
        {"selected": selected, "performance_waiver": record},
        minimum_mfu=0.50,
        expected_identities=identities(),
    )
    assert normalized["passes_hard_gate"] is False
    assert loaded == record


def test_waiver_tampering_and_non_mfu_failures_are_rejected():
    selected = observation()
    record = waiver(selected)
    tampered = copy.deepcopy(record)
    tampered["reason"] = "changed"
    with pytest.raises(BenchmarkWaiverError, match="invalid, tampered"):
        validate_performance_waiver(
            tampered,
            selected,
            minimum_mfu=0.50,
            expected_identities=identities(),
        )
    with pytest.raises(BenchmarkWaiverError, match="may excuse only"):
        waiver(observation(dropped=1))
    with pytest.raises(BenchmarkWaiverError, match="may excuse only"):
        waiver(observation(recompilations=1))


def test_harvest_accepts_only_the_signed_mfu_exception():
    selected = observation()
    selected["performance_waiver"] = waiver(selected)
    telemetry = _benchmark_telemetry(
        selected,
        mesh=HARVEST_MESHES[0],
        per_device_sequence_batch=1,
    )
    assert telemetry is not None
    assert telemetry["useful_mfu"] == pytest.approx(0.10)
    assert telemetry["performance_waiver_sha256"] == selected[
        "performance_waiver"
    ]["waiver_sha256"]

    unsigned = observation()
    with pytest.raises(HarvestError, match="did not pass"):
        _benchmark_telemetry(
            unsigned,
            mesh=HARVEST_MESHES[0],
            per_device_sequence_batch=1,
        )
