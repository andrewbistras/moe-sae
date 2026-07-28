from __future__ import annotations

import copy
from dataclasses import replace

import pytest

from moe_sae.contribution_artifacts import canonical_sha256
from moe_sae.contribution_performance import (
    AUXILIARY_THROUGHPUT_ADMISSION_MESSAGE,
    MAX_LEARNING_HORIZON_ACCELERATOR_SECONDS,
    MAX_QUALIFICATION_SECONDS,
    MAX_UNGATED_ACCELERATOR_SECONDS,
    MINIMUM_AUXILIARY_THROUGHPUT_FRACTION,
    MINIMUM_PERSISTED_GOODPUT_FRACTION,
    MINIMUM_USEFUL_PREFIX_MFU,
    PERSISTED_GOODPUT_ADMISSION_MESSAGE,
    USEFUL_MFU_ADMISSION_MESSAGE,
    ContributionPerformanceError,
    ExactPerformanceIdentity,
    QualificationEvidence,
    authorization_to_dict,
    authorize_accelerator_run,
    build_qualification_receipt,
    validate_qualification_receipt,
)


def digest(label: str) -> str:
    return canonical_sha256({"label": label})


def identity(label: str = "qualified") -> ExactPerformanceIdentity:
    return ExactPerformanceIdentity.from_components(
        graph={
            "name": "gemma4-block15-selected-output-prefix",
            "revision": label,
            "static_shape": [128, 1024],
            "instrumentation": True,
        },
        hardware={
            "accelerator": "v4-32",
            "device_inventory_sha256": digest(f"{label}:devices"),
        },
        config={
            "config_sha256": digest(f"{label}:config"),
            "maxtext_fork_commit": "1" * 40,
            "writer_backlog_limit_batches": 2,
        },
    )


def passing_evidence(
    *,
    selected_identity: ExactPerformanceIdentity | None = None,
    mfu: float = MINIMUM_USEFUL_PREFIX_MFU,
) -> QualificationEvidence:
    peak = 4_400_000_000_000_000
    device_seconds = 40.0
    useful_flops = round(mfu * peak * device_seconds)
    return QualificationEvidence(
        identity=selected_identity or identity(),
        source_evidence_sha256=digest("raw-benchmark-evidence"),
        useful_flops_ledger_sha256=digest("useful-prefix-flops-v1"),
        qualification_wall_seconds=180.0,
        measured_device_seconds=device_seconds,
        persisted_window_seconds=44.0,
        useful_flops=useful_flops,
        peak_flops_per_second=peak,
        warmup_steps=5,
        synchronized_measurement_steps=200,
        measured_graph_compile_count=1,
        auxiliary_validation_compile_count=1,
        total_compile_count=2,
        recompilations_after_warmup=0,
        oom_events=0,
        nonfinite_events=0,
        capacity_overflow_events=0,
        dropped_routed_tokens=0,
        output_tokens=4_000_000,
        persisted_tokens=4_000_000,
        max_writer_backlog_batches=1,
        writer_backlog_limit_batches=2,
        writer_failures=0,
        writer_queue_drained=True,
        static_shapes=True,
        parameters_resident_before_timing=True,
        inputs_resident_before_timing=True,
        synchronized_device_timing=True,
        instrumentation_disabled_output_sha256=digest("model-output"),
        instrumentation_enabled_output_sha256=digest("model-output"),
        writer_disabled_output_tokens=4_000_000,
        writer_disabled_device_seconds=38.0,
    )


def reseal(value: dict) -> dict:
    selected = copy.deepcopy(value)
    selected.pop("receipt_sha256", None)
    selected["receipt_sha256"] = canonical_sha256(selected)
    return selected


def test_identity_hashes_exact_canonical_graph_hardware_and_config():
    first = ExactPerformanceIdentity.from_components(
        graph={"b": 2, "a": 1},
        hardware={"kind": "v4", "chips": 16},
        config={"batch": 128},
    )
    reordered = ExactPerformanceIdentity.from_components(
        graph={"a": 1, "b": 2},
        hardware={"chips": 16, "kind": "v4"},
        config={"batch": 128},
    )
    changed = ExactPerformanceIdentity.from_components(
        graph={"a": 1, "b": 3},
        hardware={"chips": 16, "kind": "v4"},
        config={"batch": 128},
    )

    assert first == reordered
    assert first.identity_sha256 == reordered.identity_sha256
    assert first.graph_sha256 != changed.graph_sha256
    assert first.identity_sha256 != changed.identity_sha256


def test_receipt_derives_mfu_throughput_parity_and_self_hash():
    evidence = passing_evidence()
    receipt = build_qualification_receipt(evidence)
    validated = validate_qualification_receipt(
        receipt,
        expected_identity=evidence.identity,
    )

    assert validated["status"] == "passed"
    assert validated["performance"]["useful_prefix_mfu"] == pytest.approx(0.03)
    assert validated["performance"]["device_output_tokens_per_second"] == pytest.approx(
        100_000.0
    )
    assert validated["performance"]["persisted_tokens_per_second"] == pytest.approx(
        4_000_000 / 44
    )
    assert validated["performance"]["auxiliary_throughput_fraction"] == pytest.approx(
        0.95
    )
    assert receipt["receipt_sha256"] == canonical_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (
            {"qualification_wall_seconds": 600.000_001},
            "ten-minute",
        ),
        (
            {"measured_graph_compile_count": 2, "total_compile_count": 3},
            "exactly one measured",
        ),
        ({"auxiliary_validation_compile_count": 2}, "not derived"),
        ({"total_compile_count": 3}, "not derived"),
        ({"recompilations_after_warmup": 1}, "recompilation"),
        ({"oom_events": 1}, "OOM"),
        ({"nonfinite_events": 1}, "non-finite"),
        ({"capacity_overflow_events": 1}, "capacity overflow"),
        ({"dropped_routed_tokens": 1}, "dropped routed"),
        ({"writer_failures": 1}, "writer failure"),
        ({"writer_queue_drained": False}, "not drained"),
        ({"max_writer_backlog_batches": 2}, "backlog"),
        ({"static_shapes": False}, "static-shape"),
        ({"parameters_resident_before_timing": False}, "not device-resident"),
        ({"inputs_resident_before_timing": False}, "not device-resident"),
        ({"synchronized_device_timing": False}, "not synchronized"),
        (
            {"instrumentation_enabled_output_sha256": digest("different")},
            "instrumentation changed",
        ),
        ({"persisted_tokens": 3_999_999}, "token count differs"),
        ({"persisted_window_seconds": 68.0}, "60%"),
        ({"writer_disabled_device_seconds": 33.0}, "85%"),
        ({"useful_flops": 1}, "3%"),
    ],
)
def test_receipt_fails_closed_on_each_performance_fault(changes, message):
    with pytest.raises(ContributionPerformanceError, match=message):
        build_qualification_receipt(replace(passing_evidence(), **changes))


def test_measured_b128_evidence_clears_only_the_frozen_experiment2_floors():
    measured = replace(
        passing_evidence(mfu=0.0365598),
        persisted_window_seconds=(
            40.0 / 0.6388945
        ),
        writer_disabled_device_seconds=40.0 * 0.9008767,
    )
    receipt = build_qualification_receipt(measured)

    assert receipt["policy"] == {
        "minimum_useful_prefix_mfu": 0.03,
        "qualification_max_seconds": 600,
        "minimum_persisted_goodput_fraction": 0.60,
        "minimum_auxiliary_throughput_fraction": 0.85,
    }
    assert receipt["performance"]["useful_prefix_mfu"] == pytest.approx(
        0.0365598
    )
    assert (
        receipt["performance"]["persisted_tokens_per_second"]
        / receipt["performance"]["device_output_tokens_per_second"]
        == pytest.approx(0.6388945)
    )
    assert receipt["performance"]["auxiliary_throughput_fraction"] == pytest.approx(
        0.9008767
    )


def test_experiment2_admission_constants_and_messages_are_exact():
    assert MINIMUM_USEFUL_PREFIX_MFU == 0.03
    assert MINIMUM_PERSISTED_GOODPUT_FRACTION == 0.60
    assert MINIMUM_AUXILIARY_THROUGHPUT_FRACTION == 0.85
    assert "Gemma Experiment-2 3%" in USEFUL_MFU_ADMISSION_MESSAGE
    assert "Gemma Experiment-2 60%" in PERSISTED_GOODPUT_ADMISSION_MESSAGE
    assert "Gemma Experiment-2 85%" in AUXILIARY_THROUGHPUT_ADMISSION_MESSAGE


def test_receipt_rejects_tampering_even_if_claimed_status_remains_passed():
    receipt = build_qualification_receipt(passing_evidence(mfu=0.11))
    receipt["performance"]["useful_prefix_mfu"] = 0.99

    with pytest.raises(ContributionPerformanceError, match="self-hash"):
        validate_qualification_receipt(receipt)


def test_receipt_rederives_metrics_after_an_attacker_reseals_it():
    receipt = build_qualification_receipt(passing_evidence(mfu=0.11))
    receipt["performance"]["useful_prefix_mfu"] = 0.99
    receipt = reseal(receipt)

    with pytest.raises(ContributionPerformanceError, match="not derived"):
        validate_qualification_receipt(receipt)


@pytest.mark.parametrize(
    "field",
    ["graph_sha256", "hardware_sha256", "config_sha256"],
)
def test_long_run_receipt_must_match_each_exact_identity_component(field):
    qualified_identity = identity("qualified")
    changed_values = {
        "graph_sha256": qualified_identity.graph_sha256,
        "hardware_sha256": qualified_identity.hardware_sha256,
        "config_sha256": qualified_identity.config_sha256,
    }
    changed_values[field] = digest(f"different:{field}")
    requested_identity = ExactPerformanceIdentity(**changed_values)
    receipt = build_qualification_receipt(
        passing_evidence(selected_identity=qualified_identity)
    )

    with pytest.raises(
        ContributionPerformanceError,
        match="exact graph/hardware/configuration",
    ):
        authorize_accelerator_run(
            identity=requested_identity,
            maximum_runtime_seconds=601,
            purpose="workload",
            qualification_receipt=receipt,
        )


def test_direct_hour_boundary_and_exact_receipt_authorization():
    selected_identity = identity()

    qualification = authorize_accelerator_run(
        identity=selected_identity,
        maximum_runtime_seconds=MAX_QUALIFICATION_SECONDS,
        purpose="qualification",
    )
    smoke = authorize_accelerator_run(
        identity=selected_identity,
        maximum_runtime_seconds=MAX_UNGATED_ACCELERATOR_SECONDS,
        purpose="workload",
    )
    receipt = build_qualification_receipt(
        passing_evidence(selected_identity=selected_identity)
    )
    production = authorize_accelerator_run(
        identity=selected_identity,
        maximum_runtime_seconds=8 * 60 * 60,
        purpose="workload",
        qualification_receipt=receipt,
    )

    assert qualification.basis == "bounded_qualification"
    assert smoke.basis == "bounded_ungated_workload"
    assert production.basis == "qualified_exact_identity"
    assert production.qualification_receipt_sha256 == receipt["receipt_sha256"]
    assert authorization_to_dict(production)["maximum_runtime_seconds"] == 28_800


def test_learning_horizon_has_explicit_twelve_hour_direct_profile():
    selected_identity = identity()

    horizon = authorize_accelerator_run(
        identity=selected_identity,
        maximum_runtime_seconds=MAX_LEARNING_HORIZON_ACCELERATOR_SECONDS,
        purpose="workload",
        workload_profile="learning_horizon_10m",
    )

    assert horizon.basis == "bounded_learning_horizon_workload"
    assert horizon.qualification_receipt_sha256 is None
    with pytest.raises(
        ContributionPerformanceError,
        match="direct workload profile limit",
    ):
        authorize_accelerator_run(
            identity=selected_identity,
            maximum_runtime_seconds=(
                MAX_LEARNING_HORIZON_ACCELERATOR_SECONDS + 1
            ),
            purpose="workload",
            workload_profile="learning_horizon_10m",
        )


def test_no_path_authorizes_an_overlong_qualification_or_unreceipted_workload():
    selected_identity = identity()
    receipt = build_qualification_receipt(
        passing_evidence(selected_identity=selected_identity)
    )

    with pytest.raises(ContributionPerformanceError, match="ten-minute"):
        authorize_accelerator_run(
            identity=selected_identity,
            maximum_runtime_seconds=601,
            purpose="qualification",
            qualification_receipt=receipt,
        )
    with pytest.raises(
        ContributionPerformanceError,
        match="requires an exact qualification receipt",
    ):
        authorize_accelerator_run(
            identity=selected_identity,
            maximum_runtime_seconds=MAX_UNGATED_ACCELERATOR_SECONDS + 1,
            purpose="workload",
        )


def test_policy_arguments_may_be_stricter_but_never_weaker():
    evidence = passing_evidence(mfu=0.11)
    strict = build_qualification_receipt(
        evidence,
        minimum_useful_mfu=0.11,
        qualification_max_seconds=300,
    )
    assert strict["policy"]["minimum_useful_prefix_mfu"] == 0.11
    assert strict["policy"]["qualification_max_seconds"] == 300

    with pytest.raises(ContributionPerformanceError, match="cannot weaken"):
        build_qualification_receipt(
            evidence,
            minimum_useful_mfu=MINIMUM_USEFUL_PREFIX_MFU - 0.01,
        )
    with pytest.raises(ContributionPerformanceError, match="exceed 3600"):
        authorize_accelerator_run(
            identity=evidence.identity,
            maximum_runtime_seconds=1,
            purpose="workload",
            ungated_max_seconds=MAX_UNGATED_ACCELERATOR_SECONDS + 1,
        )
