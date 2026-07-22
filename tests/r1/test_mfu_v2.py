from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass, replace
from fractions import Fraction
import hashlib

import pytest

import moe_sae.r1.mfu_v2 as m
from moe_sae.r1.flops import (
    DEEPSEEK_R1_PREFIX_FORMULA_SHA256,
    deepseek_r1_prefix_formula_manifest,
)
from moe_sae.r1.image_protocol import DeviceInventoryV1, HostAttestationV1
from moe_sae.r1.protocol import (
    ContentRefV1,
    ProtocolError,
    RegionalReplicaV1,
    canonical_sha256,
)


PROJECT = "neural-dragnet"
ZONE = "us-central1-b"
SERVICE_ACCOUNT = "r1-worker@neural-dragnet.iam.gserviceaccount.com"


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _ref(label: str, media_type: str, *, size: int = 4_096) -> ContentRefV1:
    digest = _sha(label)
    return ContentRefV1(
        sha256=digest,
        size_bytes=size,
        media_type=media_type,
        replicas=(
            RegionalReplicaV1(
                region="us-central1",
                uri=f"gs://r1-evidence/{digest}",
                generation=1,
                size_bytes=size,
                crc32c="AAAAAA==",
                sha256=digest,
            ),
        ),
    )


def _record_ref(record, media_type: str) -> ContentRefV1:
    digest = record.digest
    size = len(record.canonical_bytes())
    return ContentRefV1(
        sha256=digest,
        size_bytes=size,
        media_type=media_type,
        replicas=(
            RegionalReplicaV1(
                region="us-central1",
                uri=f"gs://r1-evidence/{digest}",
                generation=1,
                size_bytes=size,
                crc32c="AAAAAA==",
                sha256=digest,
            ),
        ),
    )


def _with_extra_replica(reference: ContentRefV1) -> ContentRefV1:
    return ContentRefV1(
        sha256=reference.sha256,
        size_bytes=reference.size_bytes,
        media_type=reference.media_type,
        replicas=reference.replicas
        + (
            RegionalReplicaV1(
                region="europe-west4",
                uri=f"gs://r1-evidence-eu/{reference.sha256}",
                generation=2,
                size_bytes=reference.size_bytes,
                crc32c="AAAAAA==",
                sha256=reference.sha256,
            ),
        ),
    )


def _inventory(*, salt: str = "base") -> DeviceInventoryV1:
    deployment = _sha(f"deployment-{salt}")
    image_manifest = _sha(f"image-manifest-{salt}")
    image_link = (
        f"https://www.googleapis.com/compute/v1/projects/{PROJECT}/"
        "global/images/r1-v6e-image"
    )
    hosts = []
    for process in range(16):
        instance = f"r1-v6e-slice-{process:02d}"
        disk = f"{instance}-boot"
        device_ids = tuple(range(process * 4, process * 4 + 4))
        hosts.append(
            HostAttestationV1(
                deployment_envelope_sha256=deployment,
                project=PROJECT,
                zone=ZONE,
                instance_name=instance,
                instance_id=7_000_000_000_000_000_000 + process,
                instance_self_link=(
                    f"https://www.googleapis.com/compute/v1/projects/{PROJECT}/"
                    f"zones/{ZONE}/instances/{instance}"
                ),
                machine_type="ct6e-standard-4t",
                machine_type_self_link=(
                    f"https://www.googleapis.com/compute/v1/projects/{PROJECT}/"
                    f"zones/{ZONE}/machineTypes/ct6e-standard-4t"
                ),
                accelerator_type="v6e-64",
                service_account=SERVICE_ACCOUNT,
                boot_disk_name=disk,
                boot_disk_id=8_000_000_000_000_000_000 + process,
                boot_disk_self_link=(
                    f"https://www.googleapis.com/compute/v1/projects/{PROJECT}/"
                    f"zones/{ZONE}/disks/{disk}"
                ),
                boot_source_image_id=2_012_345_678_901_234_567,
                boot_source_image_self_link=image_link,
                image_manifest_sha256=image_manifest,
                vfio_devices=(
                    "/dev/vfio/0",
                    "/dev/vfio/1",
                    "/dev/vfio/2",
                    "/dev/vfio/3",
                ),
                jax_process_index=process,
                jax_process_count=16,
                jax_global_device_ids=device_ids,
                jax_device_coordinates=tuple(
                    (device // 8, device % 8, 0) for device in device_ids
                ),
                jax_core_on_chip=(0, 0, 0, 0),
                cache_attestation_sha256=_sha(f"cache-{salt}-{process}"),
                attested_at_unix_ms=1_753_201_000_000 + process,
            )
        )
    return DeviceInventoryV1(
        deployment_envelope_sha256=deployment,
        topology="8x8",
        hosts=tuple(hosts),
    )


@dataclass(frozen=True)
class _Bundle:
    run: m.PrefixMFURunContractV2
    formula: m.PrefixFlopFormulaV2
    inventory: DeviceInventoryV1
    windows: tuple[m.PrefixMFUWindowEvidenceV2, ...]
    receipts: m.MFUSemanticReceiptsV2
    authority: _Authority


@dataclass(frozen=True)
class _Authority:
    """Mock of the externally allowlisted, format-parsing production authority."""

    identity: ContentRefV1
    content_result: object = True
    semantic_result: object = True
    content_error: Exception | None = None
    semantic_error: Exception | None = None

    def verify_content(self, _reference: ContentRefV1) -> bool:
        if self.content_error is not None:
            raise self.content_error
        return self.content_result  # type: ignore[return-value]

    def verify_semantics(self, **_evidence) -> bool:
        if self.semantic_error is not None:
            raise self.semantic_error
        return self.semantic_result  # type: ignore[return-value]


def _split_duration(total: int, count: int) -> tuple[int, ...]:
    quotient, remainder = divmod(total, count)
    return (quotient + 1,) * remainder + (quotient,) * (count - remainder)


def _bundle() -> _Bundle:
    steps = 500
    sequences_per_step = 256
    tensor = _ref("tensor", m.MEDIA_TENSOR_INVENTORY_V2)
    executable = _ref("executable", m.MEDIA_EXECUTABLE_V2)
    hlo = _ref("hlo", m.MEDIA_HLO_V2)
    formula = m.PrefixFlopFormulaV2.build(
        tensor_inventory=tensor,
        executable=executable,
        hlo=hlo,
        counted_operation_inventory=_ref("operations", m.MEDIA_OPERATION_INVENTORY_V2),
        sequences_per_step=sequences_per_step,
    )
    real_manifest = _ref("real-input", m.MEDIA_INPUT_MANIFEST_V2)
    synthetic_manifest = _ref("synthetic-input", m.MEDIA_INPUT_MANIFEST_V2)
    manifests = {"REAL": real_manifest, "SYNTHETIC": synthetic_manifest}

    slices = []
    for group in range(3):
        for data_kind in ("REAL", "SYNTHETIC"):
            histogram = m.PackedSegmentHistogramV2(
                (m.SegmentLengthBinV2(1_024, steps * sequences_per_step),)
            )
            slices.append(
                m.PackedInputSliceV2(
                    group_index=group,
                    data_kind=data_kind,
                    parent_input_manifest_sha256=manifests[data_kind].sha256,
                    batch_start=group * steps,
                    batch_end=(group + 1) * steps,
                    packed_sequence_count=steps * sequences_per_step,
                    histogram=histogram,
                    segment_ledger=_ref(
                        f"ledger-{group}-{data_kind}",
                        m.MEDIA_SEGMENT_LEDGER_V2,
                    ),
                )
            )
    by_slice = {(item.group_index, item.data_kind): item for item in slices}
    entries = []
    window_number = 0
    for group in range(3):
        for data_kind in ("REAL", "SYNTHETIC"):
            for instrumentation in ("BASELINE", "INSTRUMENTED"):
                entries.append(
                    m.MFUWindowPlanEntryV2(
                        window_id=(
                            f"g{group}-{data_kind.lower()}-{instrumentation.lower()}"
                        ),
                        group_index=group,
                        data_kind=data_kind,
                        instrumentation_kind=instrumentation,
                        step_start=5 + window_number * steps,
                        step_end=5 + (window_number + 1) * steps,
                        input_slice_sha256=by_slice[(group, data_kind)].digest,
                    )
                )
                window_number += 1
    plan = m.PrefixMFUWindowPlanV2(tuple(slices), tuple(entries))
    inventory = _inventory()
    semantic_authority = _ref("semantic-authority", m.MEDIA_SEMANTIC_AUTHORITY_V2)
    run = m.PrefixMFURunContractV2(
        attempt_id="headline-attempt-001",
        scientific_identity=_ref("science", m.MEDIA_SCIENTIFIC_IDENTITY_V2),
        implementation_identity=_ref(
            "implementation", m.MEDIA_IMPLEMENTATION_IDENTITY_V2
        ),
        deployment_identity=_ref("deployment", m.MEDIA_DEPLOYMENT_IDENTITY_V2),
        semantic_authority=semantic_authority,
        device_inventory=_record_ref(inventory, m.MEDIA_DEVICE_INVENTORY_V2),
        tensor_inventory=tensor,
        executable=executable,
        hlo=hlo,
        flop_formula=_record_ref(formula, m.MEDIA_FLOP_FORMULA_V2),
        real_input_manifest=real_manifest,
        synthetic_input_manifest=synthetic_manifest,
        window_plan=plan,
        sequences_per_step=sequences_per_step,
        warmup_steps=5,
        accelerator_type="v6e-64",
        topology="8x8",
        physical_chips=64,
        denominator_dtype="bfloat16",
        peak_flops_per_chip=m.V6E_CHIP_BF16_PEAK_FLOPS_V2,
    )

    windows = []
    for entry in entries:
        input_slice = plan.input_slice(entry.input_slice_sha256)
        useful = formula.useful_flops(input_slice.histogram)
        wall = (useful * 1_000_000_000 * 100 + 52 * m.V6E64_BF16_PEAK_FLOPS_V2 - 1) // (
            52 * m.V6E64_BF16_PEAK_FLOPS_V2
        )
        durations = _split_duration(wall, steps)
        instrumented = entry.instrumentation_kind == "INSTRUMENTED"
        real = entry.data_kind == "REAL"
        output_tokens = input_slice.histogram.valid_tokens if instrumented else 0
        persisted_tokens = output_tokens if instrumented and real else 0
        persistence_duration = (wall * 20 + 18) // 19 if persisted_tokens else 0
        dispatch = (
            input_slice.packed_sequence_count * 1_024 * m.PREFIX_SELECTED_EXPERTS_V2
        )
        windows.append(
            m.PrefixMFUWindowEvidenceV2(
                run_contract_sha256=run.digest,
                window_id=entry.window_id,
                input_slice_sha256=entry.input_slice_sha256,
                step_start=entry.step_start,
                step_end=entry.step_end,
                step_trace=_ref(f"step-{entry.window_id}", m.MEDIA_STEP_TRACE_V2),
                timing_blocks=_ref(
                    f"blocks-{entry.window_id}", m.MEDIA_TIMING_BLOCKS_V2
                ),
                profiler_trace=_ref(
                    f"profile-{entry.window_id}", m.MEDIA_PROFILER_TRACE_V2
                ),
                host_metrics=_ref(f"hosts-{entry.window_id}", m.MEDIA_HOST_METRICS_V2),
                integrity_events=_ref(
                    f"events-{entry.window_id}", m.MEDIA_INTEGRITY_EVENTS_V2
                ),
                routing_capacity_ledger=_ref(
                    f"routing-{entry.window_id}",
                    m.MEDIA_ROUTING_CAPACITY_LEDGER_V2,
                ),
                persistence_receipt=(
                    _ref(
                        f"persist-{entry.window_id}",
                        m.MEDIA_PERSISTENCE_RECEIPT_V2,
                    )
                    if persisted_tokens
                    else None
                ),
                step_duration_ns=durations,
                input_wait_duration_ns=tuple(value // 100 for value in durations),
                host_total_step_duration_ns=(wall,) * 16,
                peak_hbm_bytes_per_device=(20_000_000_000,) * 64,
                writer_queue_depth_by_step=(0,) * steps,
                device_output_tokens=output_tokens,
                device_output_duration_ns=wall if output_tokens else 0,
                persisted_tokens=persisted_tokens,
                persistence_duration_ns=persistence_duration,
                dispatched_routed_rows_by_invocation=(dispatch,) * 28,
                allocated_routed_capacity_rows_by_invocation=(
                    ((dispatch * 105 + 99) // 100,) * 28
                ),
                oom_events=0,
                recompilations=0,
                dropped_routed_tokens=0,
                capacity_overflows=0,
                nonfinite_events=0,
            )
        )
    receipts = m.MFUSemanticReceiptsV2(
        authority_identity=semantic_authority,
        predeclaration_receipt=_ref(
            "predeclaration", m.MEDIA_PREDECLARATION_RECEIPT_V2
        ),
        compile_warmup_receipt=_ref("compile-warmup", m.MEDIA_WARMUP_RECEIPT_V2),
        formula_reconciliation_receipt=_ref(
            "formula-reconciliation",
            m.MEDIA_FORMULA_RECONCILIATION_RECEIPT_V2,
        ),
        input_reduction_receipts=tuple(
            _ref(
                f"input-reduction-{item.group_index}-{item.data_kind}",
                m.MEDIA_INPUT_REDUCTION_RECEIPT_V2,
            )
            for item in plan.input_slices
        ),
        window_reduction_receipts=tuple(
            _ref(
                f"window-reduction-{entry.window_id}",
                m.MEDIA_WINDOW_REDUCTION_RECEIPT_V2,
            )
            for entry in plan.entries
        ),
    )
    return _Bundle(
        run,
        formula,
        inventory,
        tuple(windows),
        receipts,
        _Authority(semantic_authority),
    )


def _qualify(bundle: _Bundle):
    return m.qualify_prefix_mfu_v2(
        run_contract=bundle.run,
        formula=bundle.formula,
        device_inventory=bundle.inventory,
        windows=bundle.windows,
        semantic_receipts=bundle.receipts,
        evidence_authority=bundle.authority,
    )


def _with_window(bundle: _Bundle, index: int, **changes) -> _Bundle:
    windows = list(bundle.windows)
    windows[index] = replace(windows[index], **changes)
    return replace(bundle, windows=tuple(windows))


def _retime(bundle: _Bundle, index: int, wall: int) -> _Bundle:
    window = bundle.windows[index]
    changes = {
        "step_duration_ns": _split_duration(wall, 500),
        "input_wait_duration_ns": (0,) * 500,
        "host_total_step_duration_ns": (wall,) * 16,
    }
    if window.device_output_tokens:
        changes["device_output_duration_ns"] = wall
    if window.persisted_tokens:
        changes["persistence_duration_ns"] = (wall * 20 + 18) // 19
    return _with_window(bundle, index, **changes)


def test_verified_happy_path_is_canonical_and_all_metrics_are_derived() -> None:
    bundle = _bundle()
    qualification = _qualify(bundle)

    assert qualification.minimum_real_window_mfu >= Fraction(51, 100)
    assert qualification.aggregate_real_mfu >= Fraction(51, 100)
    raw = bundle.windows[0].as_dict()
    assert not ({"useful_flops", "mfu", "rate"} & set(raw))
    encoded = qualification.as_dict()
    assert "qualification_status" not in encoded["derived"]
    assert encoded["derived"]["overall_qualification_status"] == "UNVERIFIED"
    assert encoded["derived"]["semantic_authentication_status"] == (
        "MUST_BE_REVERIFIED_BY_CONSUMER"
    )
    assert encoded["derived"]["conservative_lower_bound_method"] == (
        "MINIMUM_OF_ALL_PLANNED_REAL_WINDOWS"
    )
    assert encoded["derived"]["statistical_interval_status"].startswith("NOT_DERIVED")
    rebuilt = m.PrefixMFUQualificationV2.from_dict_verified(
        encoded, evidence_authority=bundle.authority
    )
    assert rebuilt.canonical_bytes() == qualification.canonical_bytes()
    rebuilt_json = m.PrefixMFUQualificationV2.from_json_verified(
        qualification.canonical_bytes(),
        evidence_authority=bundle.authority,
    )
    assert rebuilt_json.canonical_bytes() == qualification.canonical_bytes()
    with pytest.raises(m.MFUV2ContractError, match="from_dict_verified"):
        m.PrefixMFUQualificationV2.from_dict(encoded)
    with pytest.raises(m.MFUV2ContractError, match="construction is private"):
        m.PrefixMFUQualificationV2(
            run_contract=bundle.run,
            formula=bundle.formula,
            device_inventory=bundle.inventory,
            windows=bundle.windows,
            semantic_receipts=bundle.receipts,
        )
    with pytest.raises(FrozenInstanceError):
        bundle.windows[0].step_duration_ns = (1,)  # type: ignore[misc]


def test_histogram_derives_counts_and_rejects_forged_or_oversize_segments() -> None:
    histogram = m.PackedSegmentHistogramV2((m.SegmentLengthBinV2(2, 2),))
    assert (histogram.valid_tokens, histogram.causal_pairs) == (4, 6)
    forged = histogram.as_dict()
    forged["causal_pairs"] = 9
    with pytest.raises(m.MFUV2ContractError, match="not derived"):
        m.PackedSegmentHistogramV2.from_dict(forged)
    with pytest.raises(ProtocolError, match="<= 1024"):
        m.SegmentLengthBinV2(1_025, 1)


def test_qualification_requires_every_window_in_exact_order() -> None:
    bundle = _bundle()
    with pytest.raises(m.MFUV2ContractError, match="every planned window"):
        _qualify(replace(bundle, windows=bundle.windows[:-1]))
    swapped = list(bundle.windows)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    with pytest.raises(m.MFUV2ContractError, match="exact canonical order"):
        _qualify(replace(bundle, windows=tuple(swapped)))


def test_duplicate_trace_slice_is_rejected() -> None:
    bundle = _bundle()
    tampered = _with_window(bundle, 1, step_trace=bundle.windows[0].step_trace)
    with pytest.raises(m.MFUV2ContractError, match="duplicate step_trace"):
        _qualify(tampered)


def test_each_real_window_must_clear_51_percent() -> None:
    bundle = _bundle()
    window = bundle.windows[0]
    input_slice = bundle.run.window_plan.input_slice(window.input_slice_sha256)
    useful = bundle.formula.useful_flops(input_slice.histogram)
    wall = (useful * 1_000_000_000 * 2) // m.V6E64_BF16_PEAK_FLOPS_V2
    tampered = _with_window(
        bundle,
        0,
        step_duration_ns=_split_duration(wall, 500),
        input_wait_duration_ns=(0,) * 500,
        host_total_step_duration_ns=(wall,) * 16,
    )
    with pytest.raises(m.MFUV2ContractError, match="minimum real window"):
        _qualify(tampered)


def test_derived_qualification_cannot_be_caller_overwritten() -> None:
    qualification = _qualify(_bundle())
    raw = qualification.as_dict()
    raw["derived"]["minimum_real_window_mfu"]["parts_per_billion"] = 999_999_999
    with pytest.raises(m.MFUV2ContractError, match="not canonical derivations"):
        m.PrefixMFUQualificationV2.from_dict_verified(
            raw, evidence_authority=_bundle().authority
        )


def test_unresolved_or_exceptional_content_fails_closed() -> None:
    bundle = _bundle()
    for authority in (
        replace(bundle.authority, content_result=False),
        replace(bundle.authority, content_result=None),
        replace(bundle.authority, content_error=OSError("offline")),
    ):
        with pytest.raises(m.MFUV2ContractError, match="content"):
            m.qualify_prefix_mfu_v2(
                run_contract=bundle.run,
                formula=bundle.formula,
                device_inventory=bundle.inventory,
                windows=bundle.windows,
                semantic_receipts=bundle.receipts,
                evidence_authority=authority,
            )


def test_semantic_authority_is_predeclared_and_must_attest() -> None:
    bundle = _bundle()
    for authority in (
        replace(bundle.authority, semantic_result=False),
        replace(bundle.authority, semantic_result=None),
        replace(bundle.authority, semantic_error=ValueError("bad trace")),
    ):
        with pytest.raises(m.MFUV2ContractError, match="semantic"):
            m.qualify_prefix_mfu_v2(
                run_contract=bundle.run,
                formula=bundle.formula,
                device_inventory=bundle.inventory,
                windows=bundle.windows,
                semantic_receipts=bundle.receipts,
                evidence_authority=authority,
            )

    wrong_authority = _Authority(_ref("wrong-authority", m.MEDIA_SEMANTIC_AUTHORITY_V2))
    with pytest.raises(m.MFUV2ContractError, match="not the authority frozen"):
        m.qualify_prefix_mfu_v2(
            run_contract=bundle.run,
            formula=bundle.formula,
            device_inventory=bundle.inventory,
            windows=bundle.windows,
            semantic_receipts=bundle.receipts,
            evidence_authority=wrong_authority,
        )
    with pytest.raises(m.MFUV2ContractError, match="authority"):
        m.qualify_prefix_mfu_v2(
            run_contract=bundle.run,
            formula=bundle.formula,
            device_inventory=bundle.inventory,
            windows=bundle.windows,
            semantic_receipts=bundle.receipts,
            evidence_authority=lambda _reference: True,  # type: ignore[arg-type]
        )

    wrong_receipts = replace(
        bundle.receipts, authority_identity=wrong_authority.identity
    )
    with pytest.raises(m.MFUV2ContractError, match="different evidence authority"):
        m.qualify_prefix_mfu_v2(
            run_contract=bundle.run,
            formula=bundle.formula,
            device_inventory=bundle.inventory,
            windows=bundle.windows,
            semantic_receipts=wrong_receipts,
            evidence_authority=bundle.authority,
        )


def test_semantic_receipt_grid_is_exact_and_content_bound() -> None:
    bundle = _bundle()
    with pytest.raises(m.MFUV2ContractError, match="exactly six"):
        replace(
            bundle.receipts,
            input_reduction_receipts=(bundle.receipts.input_reduction_receipts[:-1]),
        )
    duplicated = list(bundle.receipts.window_reduction_receipts)
    duplicated[1] = duplicated[0]
    with pytest.raises(m.MFUV2ContractError, match="must be distinct"):
        replace(
            bundle.receipts,
            window_reduction_receipts=tuple(duplicated),
        )


def test_formula_and_device_inventory_are_content_bound() -> None:
    bundle = _bundle()
    with pytest.raises(m.MFUV2ContractError, match="flop_formula SHA-256"):
        _qualify(
            replace(
                bundle,
                formula=replace(
                    bundle.formula,
                    counted_operation_inventory=_ref(
                        "different-ops", m.MEDIA_OPERATION_INVENTORY_V2
                    ),
                ),
            )
        )
    with pytest.raises(m.MFUV2ContractError, match="device_inventory SHA-256"):
        _qualify(replace(bundle, inventory=_inventory(salt="other")))


def test_dependency_identity_ignores_equivalent_replica_lists() -> None:
    bundle = _bundle()
    formula = replace(
        bundle.formula,
        tensor_inventory=_with_extra_replica(bundle.formula.tensor_inventory),
    )
    run = replace(
        bundle.run,
        flop_formula=_record_ref(formula, m.MEDIA_FLOP_FORMULA_V2),
    )
    run.verify_dependencies(formula, bundle.inventory)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("oom_events", 1, "integrity event"),
        ("recompilations", 1, "integrity event"),
        ("dropped_routed_tokens", 1, "integrity event"),
        ("capacity_overflows", 1, "integrity event"),
        ("nonfinite_events", 1, "integrity event"),
    ],
)
def test_integrity_events_are_hard_failures(
    field: str, value: int, message: str
) -> None:
    bundle = _bundle()
    with pytest.raises(m.MFUV2ContractError, match=message):
        _qualify(_with_window(bundle, 0, **{field: value}))


def test_output_and_routing_counts_are_recomputed_from_plan() -> None:
    bundle = _bundle()
    with pytest.raises(m.MFUV2ContractError, match="device output token count"):
        _qualify(
            _with_window(
                bundle,
                1,
                device_output_tokens=bundle.windows[1].device_output_tokens - 1,
            )
        )
    with pytest.raises(m.MFUV2ContractError, match="dispatch count"):
        dispatches = list(bundle.windows[0].dispatched_routed_rows_by_invocation)
        dispatches[7] -= 1
        _qualify(
            _with_window(
                bundle,
                0,
                dispatched_routed_rows_by_invocation=tuple(dispatches),
            )
        )


def test_persistence_and_capacity_amplification_gates_are_exact() -> None:
    bundle = _bundle()
    real_instrumented = bundle.windows[1]
    with pytest.raises(m.MFUV2ContractError, match="persisted/device"):
        _qualify(
            _with_window(
                bundle,
                1,
                persistence_duration_ns=(
                    real_instrumented.device_output_duration_ns * 2
                ),
            )
        )
    dispatch = bundle.windows[0].dispatched_routed_rows_by_invocation[0]
    capacities = list(bundle.windows[0].allocated_routed_capacity_rows_by_invocation)
    capacities[11] = dispatch * 111 // 100
    with pytest.raises(m.MFUV2ContractError, match="per-invocation capacity"):
        _qualify(
            _with_window(
                bundle,
                0,
                allocated_routed_capacity_rows_by_invocation=tuple(capacities),
            )
        )


def test_wait_latency_hbm_queue_and_host_timing_gates() -> None:
    bundle = _bundle()
    window = bundle.windows[0]
    wall = window.wall_time_ns
    mutations = (
        (
            {"input_wait_duration_ns": _split_duration((wall * 5 + 99) // 100, 500)},
            "input wait",
        ),
        ({"peak_hbm_bytes_per_device": (28_000_000_000,) * 64}, "HBM"),
        ({"writer_queue_depth_by_step": (2,) + (0,) * 499}, "writer queue"),
        (
            {"host_total_step_duration_ns": (wall,) * 15 + (wall * 106 // 100,)},
            "host",
        ),
    )
    for changes, message in mutations:
        with pytest.raises(m.MFUV2ContractError, match=message):
            _qualify(_with_window(bundle, 0, **changes))


def test_instrumentation_delta_is_paired_on_the_same_input_slice() -> None:
    bundle = _bundle()
    window = bundle.windows[3]  # synthetic instrumented
    wall = window.wall_time_ns * 106 // 100
    tampered = _with_window(
        bundle,
        3,
        step_duration_ns=_split_duration(wall, 500),
        input_wait_duration_ns=(0,) * 500,
        host_total_step_duration_ns=(wall,) * 16,
        device_output_duration_ns=wall,
    )
    with pytest.raises(m.MFUV2ContractError, match="instrumentation MFU delta"):
        _qualify(tampered)


def test_plan_rejects_density_mismatch_overlap_and_weakened_warmup() -> None:
    bundle = _bundle()
    duplicate_ledgers = list(bundle.run.window_plan.input_slices)
    duplicate_ledgers[1] = replace(
        duplicate_ledgers[1], segment_ledger=duplicate_ledgers[0].segment_ledger
    )
    with pytest.raises(m.MFUV2ContractError, match="ledgers must be unique"):
        replace(
            bundle.run.window_plan,
            input_slices=tuple(duplicate_ledgers),
        )

    slices = list(bundle.run.window_plan.input_slices)
    slices[1] = replace(
        slices[1],
        histogram=m.PackedSegmentHistogramV2((m.SegmentLengthBinV2(1_023, 1_000),)),
    )
    entries = list(bundle.run.window_plan.entries)
    entries[2] = replace(entries[2], input_slice_sha256=slices[1].digest)
    entries[3] = replace(entries[3], input_slice_sha256=slices[1].digest)
    with pytest.raises(m.MFUV2ContractError, match="density matched"):
        replace(
            bundle.run.window_plan,
            input_slices=tuple(slices),
            entries=tuple(entries),
        )

    entries = list(bundle.run.window_plan.entries)
    entries[1] = replace(
        entries[1],
        step_start=entries[0].step_start,
        step_end=entries[0].step_end,
    )
    with pytest.raises(m.MFUV2ContractError, match="overlap"):
        replace(bundle.run.window_plan, entries=tuple(entries))

    entries = list(bundle.run.window_plan.entries)
    first_interval = (entries[0].step_start, entries[0].step_end)
    second_interval = (entries[1].step_start, entries[1].step_end)
    entries[0] = replace(
        entries[0], step_start=second_interval[0], step_end=second_interval[1]
    )
    entries[1] = replace(
        entries[1], step_start=first_interval[0], step_end=first_interval[1]
    )
    with pytest.raises(m.MFUV2ContractError, match="canonical plan order"):
        replace(bundle.run.window_plan, entries=tuple(entries))
    with pytest.raises(ProtocolError, match=">= 5"):
        replace(bundle.run, warmup_steps=4)
    entries = list(bundle.run.window_plan.entries)
    entries[0] = replace(entries[0], step_start=0, step_end=500)
    pre_warmup_plan = replace(bundle.run.window_plan, entries=tuple(entries))
    with pytest.raises(m.MFUV2ContractError, match="after declared warmup"):
        replace(bundle.run, window_plan=pre_warmup_plan)


def test_formula_digest_dtype_and_hardware_denominator_cannot_be_weakened() -> None:
    bundle = _bundle()
    assert DEEPSEEK_R1_PREFIX_FORMULA_SHA256 == canonical_sha256(
        deepseek_r1_prefix_formula_manifest()
    )
    with pytest.raises(m.MFUV2ContractError, match="formula digest"):
        replace(bundle.formula, analytic_formula_sha256=_sha("forged"))
    with pytest.raises(m.MFUV2ContractError, match="counted_matmul_dtype"):
        replace(bundle.formula, counted_matmul_dtype="float32")
    with pytest.raises(m.MFUV2ContractError, match="exactly 64"):
        replace(bundle.run, physical_chips=32)
    with pytest.raises(m.MFUV2ContractError, match="v6e BF16 peak"):
        replace(bundle.run, peak_flops_per_chip=1)


def test_raw_window_rejects_caller_metrics_and_wrong_plan_binding() -> None:
    bundle = _bundle()
    raw = bundle.windows[0].as_dict()
    raw["mfu"] = {"numerator": 99, "denominator": 100}
    with pytest.raises(ProtocolError, match="unknown fields"):
        m.PrefixMFUWindowEvidenceV2.from_dict(raw)
    with pytest.raises(m.MFUV2ContractError, match="wrong run contract"):
        _qualify(_with_window(bundle, 0, run_contract_sha256=_sha("other-run")))
    with pytest.raises(m.MFUV2ContractError, match="frozen plan"):
        _qualify(
            _with_window(
                bundle,
                0,
                step_start=bundle.windows[0].step_start + 1,
                step_end=bundle.windows[0].step_end + 1,
            )
        )


def test_mfu_above_physical_peak_is_rejected() -> None:
    bundle = _bundle()
    window = bundle.windows[2]
    input_slice = bundle.run.window_plan.input_slice(window.input_slice_sha256)
    useful = bundle.formula.useful_flops(input_slice.histogram)
    wall = useful * 1_000_000_000 * 100 // (101 * m.V6E64_BF16_PEAK_FLOPS_V2)
    with pytest.raises(m.MFUV2ContractError, match="physical peak"):
        _qualify(_retime(bundle, 2, wall))


def test_p95_gate_uses_raw_integer_step_vector() -> None:
    bundle = _bundle()
    base = bundle.windows[2].wall_time_ns // 500
    low = base * 8 // 10
    high = base * 12 // 10
    durations = (low,) * 26 + (base,) * 448 + (high,) * 26
    wall = sum(durations)
    with pytest.raises(m.MFUV2ContractError, match="p95/p50"):
        _qualify(
            _with_window(
                bundle,
                2,
                step_duration_ns=durations,
                input_wait_duration_ns=(0,) * 500,
                host_total_step_duration_ns=(wall,) * 16,
            )
        )


def test_baseline_output_and_instrumented_duration_are_not_trusted() -> None:
    bundle = _bundle()
    baseline = bundle.windows[0]
    with pytest.raises(m.MFUV2ContractError, match="baseline contains"):
        _qualify(
            _with_window(
                bundle,
                0,
                device_output_tokens=1,
                device_output_duration_ns=baseline.wall_time_ns,
            )
        )
    instrumented = bundle.windows[1]
    with pytest.raises(m.MFUV2ContractError, match="differs from wall"):
        _qualify(
            _with_window(
                bundle,
                1,
                device_output_duration_ns=(instrumented.device_output_duration_ns + 1),
            )
        )


def test_real_to_synthetic_goodput_is_derived_per_paired_group() -> None:
    bundle = _bundle()
    slower_wall = bundle.windows[0].wall_time_ns * 115 // 100
    tampered = _retime(bundle, 0, slower_wall)
    tampered = _retime(tampered, 1, slower_wall)
    with pytest.raises(m.MFUV2ContractError, match="real/synthetic"):
        _qualify(tampered)
