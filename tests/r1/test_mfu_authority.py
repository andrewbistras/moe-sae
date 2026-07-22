from __future__ import annotations

import base64
from dataclasses import dataclass, replace
from fractions import Fraction
import hashlib
from typing import Any, Mapping

import pytest

import moe_sae.r1 as r1
import moe_sae.r1.mfu_authority as a
import moe_sae.r1.mfu_v2 as m
from moe_sae.r1.control_store import MemoryJSONCASStore
from moe_sae.r1.controller import (
    CampaignController,
    InvalidReceipt,
    LeaseUnavailable,
    StaleAttempt,
)
from moe_sae.r1.flops import deepseek_r1_prefix_flop_coefficients
from moe_sae.r1.image_protocol import DeviceInventoryV1, HostAttestationV1
from moe_sae.r1.protocol import (
    ContentRefV1,
    HarvestRangeV1,
    RegionalReplicaV1,
    canonical_sha256,
    parse_json,
    require_record_object,
)


PROJECT = "neural-dragnet"
ZONE = "us-central1-b"
REGION = "us-central1"
SERVICE_ACCOUNT = "r1-worker@neural-dragnet.iam.gserviceaccount.com"
BASE_START_NS = 1_753_201_100_000_000_000


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _crc(value: bytes) -> str:
    crc = 0xFFFFFFFF
    for octet in value:
        crc ^= octet
        for _ in range(8):
            crc = (crc >> 1) ^ (0x82F63B78 if crc & 1 else 0)
    return base64.b64encode(
        ((crc ^ 0xFFFFFFFF) & 0xFFFFFFFF).to_bytes(4, "big")
    ).decode("ascii")


def _stored_ref(
    store: MemoryJSONCASStore,
    *,
    label: str,
    media_type: str,
    value: Any,
    prefix: str = "gs://r1-raw",
) -> ContentRefV1:
    encoded = value.canonical_bytes() if hasattr(value, "canonical_bytes") else None
    digest = value.digest if hasattr(value, "digest") else canonical_sha256(value)
    uri = f"{prefix}/{label}/{digest}.json"
    version = store.put_immutable(uri, value)
    if encoded is not None:
        assert version.value == encoded
    return ContentRefV1(
        sha256=version.sha256,
        size_bytes=len(version.value),
        media_type=media_type,
        replicas=(
            RegionalReplicaV1(
                region=REGION,
                uri=uri,
                generation=version.generation,
                size_bytes=len(version.value),
                crc32c=_crc(version.value),
                sha256=version.sha256,
            ),
        ),
    )


def _raw(
    store: MemoryJSONCASStore,
    label: str,
    media_type: str,
    schema: str,
    **fields: Any,
) -> ContentRefV1:
    return _stored_ref(
        store,
        label=label,
        media_type=media_type,
        value={"schema": schema, **fields},
    )


def _decoded(
    content: a.ResolvedContentV2,
    *,
    schema: str,
    fields: tuple[str, ...],
) -> Mapping[str, Any]:
    parsed = parse_json(content.value)
    return require_record_object(parsed, schema=schema, fields=fields)


def _identity(store: MemoryJSONCASStore, label: str, media_type: str) -> ContentRefV1:
    return _raw(
        store,
        f"identity-{label}",
        media_type,
        "moe-sae.test.mfu-authority-identity.v2",
        identity=label,
        build_sha256=_sha(f"build-{label}"),
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


def _split_duration(total: int, count: int) -> tuple[int, ...]:
    quotient, remainder = divmod(total, count)
    return (quotient + 1,) * remainder + (quotient,) * (count - remainder)


def _content_identity(reference: ContentRefV1) -> dict[str, Any]:
    return {
        "sha256": reference.sha256,
        "size_bytes": reference.size_bytes,
        "media_type": reference.media_type,
    }


def _artifact_bundle_sha(references: tuple[ContentRefV1, ...]) -> str:
    return canonical_sha256(
        {
            "schema": "moe-sae.r1.mfu-artifact-bundle.v2",
            "artifacts": [_content_identity(item) for item in references],
        }
    )


def _device_ids(inventory: DeviceInventoryV1) -> tuple[int, ...]:
    return tuple(
        sorted(
            device_id
            for host in inventory.hosts
            for device_id in host.jax_global_device_ids
        )
    )


def _device_ids_sha(inventory: DeviceInventoryV1) -> str:
    return canonical_sha256(
        {
            "schema": "moe-sae.r1.mfu-device-ids.v2",
            "device_ids": list(_device_ids(inventory)),
        }
    )


def _coefficient_sha() -> str:
    return canonical_sha256(
        {
            "schema": "moe-sae.r1.deepseek-r1-prefix-coefficients.v2",
            "coefficients": deepseek_r1_prefix_flop_coefficients(),
        }
    )


FORMULA_SCHEMA = "moe-sae.test.raw-formula-artifact.v2"
MANIFEST_SCHEMA = "moe-sae.test.raw-input-manifest.v2"
LEDGER_SCHEMA = "moe-sae.test.raw-segment-ledger.v2"
TRACE_SCHEMA = "moe-sae.test.raw-step-trace.v2"
TIMING_SCHEMA = "moe-sae.test.raw-timing-blocks.v2"
PROFILE_SCHEMA = "moe-sae.test.raw-profiler-trace.v2"
HOST_SCHEMA = "moe-sae.test.raw-host-metrics.v2"
EVENT_SCHEMA = "moe-sae.test.raw-integrity-events.v2"
ROUTING_SCHEMA = "moe-sae.test.raw-routing-capacity.v2"
PERSIST_SCHEMA = "moe-sae.test.raw-persistence-receipt.v2"


@dataclass(frozen=True)
class _FormulaParser:
    identity: ContentRefV1

    def parse_formula(
        self,
        *,
        run_contract: m.PrefixMFURunContractV2,
        formula: m.PrefixFlopFormulaV2,
        device_inventory: DeviceInventoryV1,
        artifacts: Mapping[str, a.ResolvedContentV2],
    ) -> a.MFUFormulaReconciliationReceiptPayloadV2:
        del device_inventory
        expected_roles = {
            "tensor_inventory": ("tensor", 7_168),
            "executable": ("executable", 1_024),
            "hlo": ("hlo", 1_024),
            "operation_inventory": ("operations", 28),
        }
        if set(artifacts) != set(expected_roles):
            raise a.MFUAuthorityError("formula artifact roles are not exact")
        for role, (kind, scalar) in expected_roles.items():
            record = _decoded(
                artifacts[role],
                schema=FORMULA_SCHEMA,
                fields=("role", "scalar", "dtype"),
            )
            if record != {
                "schema": FORMULA_SCHEMA,
                "role": kind,
                "scalar": scalar,
                "dtype": "bfloat16",
            }:
                raise a.MFUAuthorityError("formula raw artifact is inconsistent")
        return a.MFUFormulaReconciliationReceiptPayloadV2(
            run_contract_sha256=run_contract.digest,
            formula_sha256=formula.digest,
            executable_sha256=run_contract.executable.sha256,
            hlo_sha256=run_contract.hlo.sha256,
            tensor_inventory_sha256=run_contract.tensor_inventory.sha256,
            operation_inventory_sha256=(formula.counted_operation_inventory.sha256),
            analytic_formula_sha256=formula.analytic_formula_sha256,
            coefficient_ledger_sha256=_coefficient_sha(),
            sequences_per_step=run_contract.sequences_per_step,
            sequence_length=formula.sequence_length,
            counted_matmul_dtype=formula.counted_matmul_dtype,
            denominator_dtype=run_contract.denominator_dtype,
        )


@dataclass(frozen=True)
class _InputParser:
    identity: ContentRefV1

    def parse_input(
        self,
        *,
        run_contract: m.PrefixMFURunContractV2,
        input_slice: m.PackedInputSliceV2,
        parent_manifest: a.ResolvedContentV2,
        segment_ledger: a.ResolvedContentV2,
    ) -> a.MFUInputReductionReceiptPayloadV2:
        manifest = _decoded(
            parent_manifest,
            schema=MANIFEST_SCHEMA,
            fields=("data_kind", "dataset_sha256"),
        )
        ledger = _decoded(
            segment_ledger,
            schema=LEDGER_SCHEMA,
            fields=(
                "parent_manifest_sha256",
                "group_index",
                "data_kind",
                "batch_start",
                "batch_end",
                "packed_sequence_count",
                "histogram",
            ),
        )
        histogram = m.PackedSegmentHistogramV2.from_dict(ledger["histogram"])
        expected = {
            "parent_manifest_sha256": parent_manifest.reference.sha256,
            "group_index": input_slice.group_index,
            "data_kind": input_slice.data_kind,
            "batch_start": input_slice.batch_start,
            "batch_end": input_slice.batch_end,
            "packed_sequence_count": input_slice.packed_sequence_count,
            "histogram": input_slice.histogram.as_dict(),
        }
        if manifest["data_kind"] != input_slice.data_kind:
            raise a.MFUAuthorityError("manifest has the wrong data kind")
        if {key: ledger[key] for key in expected} != expected:
            raise a.MFUAuthorityError("ledger does not reduce to the input slice")
        if histogram.canonical_bytes() != input_slice.histogram.canonical_bytes():
            raise a.MFUAuthorityError("ledger histogram is inconsistent")
        return a.MFUInputReductionReceiptPayloadV2(
            run_contract_sha256=run_contract.digest,
            input_slice_sha256=input_slice.digest,
            parent_input_manifest_sha256=parent_manifest.reference.sha256,
            segment_ledger_sha256=segment_ledger.reference.sha256,
            group_index=input_slice.group_index,
            data_kind=input_slice.data_kind,
            batch_start=input_slice.batch_start,
            batch_end=input_slice.batch_end,
            packed_sequence_count=input_slice.packed_sequence_count,
            histogram_sha256=histogram.digest,
            valid_tokens=histogram.valid_tokens,
            causal_pairs=histogram.causal_pairs,
        )


def _window_refs(
    evidence: m.PrefixMFUWindowEvidenceV2,
) -> tuple[ContentRefV1, ...]:
    base = (
        evidence.step_trace,
        evidence.timing_blocks,
        evidence.profiler_trace,
        evidence.host_metrics,
        evidence.integrity_events,
        evidence.routing_capacity_ledger,
    )
    return (
        (*base, evidence.persistence_receipt)
        if evidence.persistence_receipt is not None
        else base
    )


def _window_payload(
    *,
    run: m.PrefixMFURunContractV2,
    entry: m.MFUWindowPlanEntryV2,
    evidence: m.PrefixMFUWindowEvidenceV2,
    inventory: DeviceInventoryV1,
    controller_fence: int,
    measurement_execution_id: str,
    started: int,
    finished: int,
    capacity_steps: int,
    capacity_router_invocations: int,
    capacity_routed_experts: int,
    cells: int,
    matrix_sha: str,
    maximum_numerator: int,
    maximum_denominator: int,
    overloads: int,
) -> a.MFUWindowReductionReceiptPayloadV2:
    return a.MFUWindowReductionReceiptPayloadV2(
        run_contract_sha256=run.digest,
        attempt_id=run.attempt_id,
        measurement_execution_id=measurement_execution_id,
        window_id=entry.window_id,
        input_slice_sha256=entry.input_slice_sha256,
        executable_sha256=run.executable.sha256,
        hlo_sha256=run.hlo.sha256,
        device_inventory_sha256=run.device_inventory.sha256,
        controller_fence=controller_fence,
        device_ids_sha256=_device_ids_sha(inventory),
        artifact_bundle_sha256=_artifact_bundle_sha(_window_refs(evidence)),
        observation_sha256=a.window_observation_digest_v2(evidence),
        step_start=entry.step_start,
        step_end=entry.step_end,
        measurement_started_at_unix_ns=started,
        measurement_finished_at_unix_ns=finished,
        capacity_steps=capacity_steps,
        capacity_router_invocations=capacity_router_invocations,
        capacity_routed_experts=capacity_routed_experts,
        capacity_cell_count=cells,
        capacity_matrix_sha256=matrix_sha,
        maximum_capacity_numerator=maximum_numerator,
        maximum_capacity_denominator=maximum_denominator,
        overloaded_capacity_cells=overloads,
    )


@dataclass(frozen=True)
class _WindowParser:
    identity: ContentRefV1

    @staticmethod
    def _trace(content: a.ResolvedContentV2) -> Mapping[str, Any]:
        return _decoded(
            content,
            schema=TRACE_SCHEMA,
            fields=(
                "attempt_id",
                "measurement_execution_id",
                "window_id",
                "executable_sha256",
                "hlo_sha256",
                "device_inventory_sha256",
                "controller_fence",
                "step_start",
                "step_end",
                "measurement_started_at_unix_ns",
                "measurement_finished_at_unix_ns",
                "step_duration_ns",
                "input_wait_duration_ns",
                "compile_started_at_unix_ns",
                "compile_finished_at_unix_ns",
                "warmup_started_at_unix_ns",
                "warmup_finished_at_unix_ns",
                "completed_warmup_steps",
                "first_measurement_step",
                "first_measurement_started_at_unix_ns",
            ),
        )

    def parse_compile_warmup(
        self,
        *,
        run_contract: m.PrefixMFURunContractV2,
        device_inventory: DeviceInventoryV1,
        windows: tuple[m.PrefixMFUWindowEvidenceV2, ...],
        artifacts: Mapping[str, a.ResolvedContentV2],
    ) -> a.MFUCompileWarmupReceiptPayloadV2:
        if set(artifacts) != {
            "executable",
            "hlo",
            "device_inventory",
            *(f"window_{index}_step_trace" for index in range(len(windows))),
        }:
            raise a.MFUAuthorityError("compile/warmup artifact grid is incomplete")
        executable = _decoded(
            artifacts["executable"],
            schema=FORMULA_SCHEMA,
            fields=("role", "scalar", "dtype"),
        )
        hlo = _decoded(
            artifacts["hlo"],
            schema=FORMULA_SCHEMA,
            fields=("role", "scalar", "dtype"),
        )
        parsed_inventory = DeviceInventoryV1.from_dict(
            parse_json(artifacts["device_inventory"].value)
        )
        if executable["role"] != "executable" or hlo["role"] != "hlo":
            raise a.MFUAuthorityError("compile artifacts have the wrong roles")
        if parsed_inventory.canonical_bytes() != device_inventory.canonical_bytes():
            raise a.MFUAuthorityError("compiled inventory bytes do not match")
        traces = tuple(
            self._trace(artifacts[f"window_{index}_step_trace"])
            for index in range(len(windows))
        )
        for trace, window in zip(traces, windows, strict=True):
            if (
                trace["attempt_id"] != run_contract.attempt_id
                or trace["window_id"] != window.window_id
                or trace["executable_sha256"] != run_contract.executable.sha256
                or trace["hlo_sha256"] != run_contract.hlo.sha256
                or trace["device_inventory_sha256"]
                != run_contract.device_inventory.sha256
            ):
                raise a.MFUAuthorityError("trace is spliced across executions")
        first = traces[0]
        return a.MFUCompileWarmupReceiptPayloadV2(
            run_contract_sha256=run_contract.digest,
            attempt_id=run_contract.attempt_id,
            measurement_execution_id=first["measurement_execution_id"],
            executable_sha256=run_contract.executable.sha256,
            hlo_sha256=run_contract.hlo.sha256,
            device_inventory_sha256=run_contract.device_inventory.sha256,
            controller_fence=first["controller_fence"],
            compile_started_at_unix_ns=first["compile_started_at_unix_ns"],
            compile_finished_at_unix_ns=first["compile_finished_at_unix_ns"],
            warmup_started_at_unix_ns=first["warmup_started_at_unix_ns"],
            warmup_finished_at_unix_ns=first["warmup_finished_at_unix_ns"],
            completed_warmup_steps=first["completed_warmup_steps"],
            first_measurement_step=first["first_measurement_step"],
            first_measurement_started_at_unix_ns=(
                first["first_measurement_started_at_unix_ns"]
            ),
            measurement_recompilations=sum(window.recompilations for window in windows),
        )

    def parse_window(
        self,
        *,
        run_contract: m.PrefixMFURunContractV2,
        entry: m.MFUWindowPlanEntryV2,
        input_slice: m.PackedInputSliceV2,
        evidence: m.PrefixMFUWindowEvidenceV2,
        device_inventory: DeviceInventoryV1,
        artifacts: Mapping[str, a.ResolvedContentV2],
    ) -> a.MFUWindowReductionReceiptPayloadV2:
        del input_slice
        expected_count = 7 if evidence.persistence_receipt is not None else 6
        if tuple(artifacts) != tuple(
            f"artifact_{index}" for index in range(expected_count)
        ):
            raise a.MFUAuthorityError("window artifact roles are not canonical")
        trace = self._trace(artifacts["artifact_0"])
        timing = _decoded(
            artifacts["artifact_1"],
            schema=TIMING_SCHEMA,
            fields=(
                "measurement_execution_id",
                "window_id",
                "device_output_tokens",
                "device_output_duration_ns",
                "persisted_tokens",
                "persistence_duration_ns",
            ),
        )
        profile = _decoded(
            artifacts["artifact_2"],
            schema=PROFILE_SCHEMA,
            fields=(
                "measurement_execution_id",
                "window_id",
                "executable_sha256",
                "hlo_sha256",
                "device_ids",
            ),
        )
        host = _decoded(
            artifacts["artifact_3"],
            schema=HOST_SCHEMA,
            fields=(
                "measurement_execution_id",
                "window_id",
                "host_total_step_duration_ns",
                "peak_hbm_bytes_per_device",
                "writer_queue_depth_by_step",
            ),
        )
        events = _decoded(
            artifacts["artifact_4"],
            schema=EVENT_SCHEMA,
            fields=(
                "measurement_execution_id",
                "window_id",
                "oom_events",
                "recompilations",
                "dropped_routed_tokens",
                "capacity_overflows",
                "nonfinite_events",
            ),
        )
        routing = _decoded(
            artifacts["artifact_5"],
            schema=ROUTING_SCHEMA,
            fields=(
                "measurement_execution_id",
                "window_id",
                "shape",
                "default_numerator",
                "default_denominator",
                "overrides",
                "dispatched_routed_rows_by_invocation",
                "allocated_routed_capacity_rows_by_invocation",
            ),
        )
        if (
            trace["attempt_id"] != run_contract.attempt_id
            or any(
                record["measurement_execution_id"] != trace["measurement_execution_id"]
                for record in (timing, profile, host, events, routing)
            )
            or trace["window_id"] != evidence.window_id
            or trace["executable_sha256"] != run_contract.executable.sha256
            or trace["hlo_sha256"] != run_contract.hlo.sha256
            or trace["device_inventory_sha256"] != run_contract.device_inventory.sha256
            or trace["step_start"] != entry.step_start
            or trace["step_end"] != entry.step_end
            or trace["step_duration_ns"] != list(evidence.step_duration_ns)
            or trace["input_wait_duration_ns"] != list(evidence.input_wait_duration_ns)
            or profile["executable_sha256"] != run_contract.executable.sha256
            or profile["hlo_sha256"] != run_contract.hlo.sha256
            or profile["device_ids"] != list(_device_ids(device_inventory))
            or timing["window_id"] != evidence.window_id
            or profile["window_id"] != evidence.window_id
            or host["window_id"] != evidence.window_id
            or events["window_id"] != evidence.window_id
            or routing["window_id"] != evidence.window_id
            or host["host_total_step_duration_ns"]
            != list(evidence.host_total_step_duration_ns)
            or host["peak_hbm_bytes_per_device"]
            != list(evidence.peak_hbm_bytes_per_device)
            or host["writer_queue_depth_by_step"]
            != list(evidence.writer_queue_depth_by_step)
        ):
            raise a.MFUAuthorityError("raw window scalars do not match evidence")
        expected_timing = {
            field: getattr(evidence, field)
            for field in (
                "device_output_tokens",
                "device_output_duration_ns",
                "persisted_tokens",
                "persistence_duration_ns",
            )
        }
        expected_events = {
            field: getattr(evidence, field)
            for field in (
                "oom_events",
                "recompilations",
                "dropped_routed_tokens",
                "capacity_overflows",
                "nonfinite_events",
            )
        }
        if any(timing[field] != value for field, value in expected_timing.items()):
            raise a.MFUAuthorityError("timing reduction does not match evidence")
        if any(events[field] != value for field, value in expected_events.items()):
            raise a.MFUAuthorityError("event reduction does not match evidence")
        if routing["dispatched_routed_rows_by_invocation"] != list(
            evidence.dispatched_routed_rows_by_invocation
        ) or routing["allocated_routed_capacity_rows_by_invocation"] != list(
            evidence.allocated_routed_capacity_rows_by_invocation
        ):
            raise a.MFUAuthorityError("invocation capacity totals do not match")
        shape = routing["shape"]
        if (
            not isinstance(shape, list)
            or len(shape) != 3
            or any(type(item) is not int or item <= 0 for item in shape)
        ):
            raise a.MFUAuthorityError("routing shape is malformed")
        cells = shape[0] * shape[1] * shape[2]
        ratios = [(routing["default_numerator"], routing["default_denominator"])]
        overloads = 0
        seen_cells: set[tuple[int, int, int]] = set()
        for override in routing["overrides"]:
            if (
                not isinstance(override, list)
                or len(override) != 5
                or any(type(item) is not int for item in override)
            ):
                raise a.MFUAuthorityError("routing override is malformed")
            coordinate = tuple(override[:3])
            if coordinate in seen_cells or any(
                coordinate[axis] < 0 or coordinate[axis] >= shape[axis]
                for axis in range(3)
            ):
                raise a.MFUAuthorityError("routing override coordinate is invalid")
            seen_cells.add(coordinate)
            numerator, denominator = override[3:]
            if numerator <= 0 or denominator <= 0:
                raise a.MFUAuthorityError("routing ratio must be positive")
            ratios.append((numerator, denominator))
            overloads += numerator * 10 > denominator * 11
        maximum_numerator, maximum_denominator = max(
            ratios, key=lambda pair: Fraction(pair[0], pair[1])
        )
        if evidence.persistence_receipt is not None:
            persisted = _decoded(
                artifacts["artifact_6"],
                schema=PERSIST_SCHEMA,
                fields=(
                    "measurement_execution_id",
                    "window_id",
                    "tokens",
                    "duration_ns",
                    "queue_drained",
                ),
            )
            if persisted != {
                "schema": PERSIST_SCHEMA,
                "measurement_execution_id": trace["measurement_execution_id"],
                "window_id": evidence.window_id,
                "tokens": evidence.persisted_tokens,
                "duration_ns": evidence.persistence_duration_ns,
                "queue_drained": 1,
            }:
                raise a.MFUAuthorityError("persistence receipt is inconsistent")
        return _window_payload(
            run=run_contract,
            entry=entry,
            evidence=evidence,
            inventory=device_inventory,
            controller_fence=trace["controller_fence"],
            measurement_execution_id=trace["measurement_execution_id"],
            started=trace["measurement_started_at_unix_ns"],
            finished=trace["measurement_finished_at_unix_ns"],
            capacity_steps=shape[0],
            capacity_router_invocations=shape[1],
            capacity_routed_experts=shape[2],
            cells=cells,
            matrix_sha=artifacts["artifact_5"].reference.sha256,
            maximum_numerator=maximum_numerator,
            maximum_denominator=maximum_denominator,
            overloads=overloads,
        )


@dataclass(frozen=True)
class _Fixture:
    store: MemoryJSONCASStore
    controller: CampaignController
    registry: a.ControllerMFUEvidenceRegistryV2
    policy: a.TrustedMFUAuthorityPolicyV2
    resolver: a.JSONCASGenerationPinnedResolverV2
    formula_parser: _FormulaParser
    input_parser: _InputParser
    window_parser: _WindowParser
    authority: a.ProtocolOracleMFUEvidenceAuthorityV2
    run: m.PrefixMFURunContractV2
    formula: m.PrefixFlopFormulaV2
    inventory: DeviceInventoryV1
    windows: tuple[m.PrefixMFUWindowEvidenceV2, ...]
    receipts: m.MFUSemanticReceiptsV2
    formula_receipt: a.MFUFormulaReconciliationReceiptPayloadV2
    compile_receipt: a.MFUCompileWarmupReceiptPayloadV2
    input_receipts: tuple[a.MFUInputReductionReceiptPayloadV2, ...]
    window_receipts: tuple[a.MFUWindowReductionReceiptPayloadV2, ...]


def _harvest(scientific_identity_sha256: str) -> HarvestRangeV1:
    return HarvestRangeV1(
        scientific_identity_sha256=scientific_identity_sha256,
        corpus_fingerprint_sha256=_sha("corpus"),
        split="en/train",
        record_start=0,
        record_end=1,
        scientific_token_start=0,
        scientific_token_end=1_024,
    )


def _build_fixture(
    *,
    salt: str = "base",
    overload_window: int | None = None,
    capacity_shape: tuple[int, int, int] | None = None,
    trace_controller_fence: int = 1,
    lease_duration_ms: int = 3_600_000,
    splice_host_execution: bool = False,
    warmup_overlap: bool = False,
    late_predeclare: bool = False,
) -> _Fixture:
    store = MemoryJSONCASStore()
    authority_identity = _identity(
        store, f"authority-{salt}", m.MEDIA_SEMANTIC_AUTHORITY_V2
    )
    resolver_identity = _identity(
        store, f"resolver-{salt}", a.MEDIA_CONTENT_RESOLVER_IDENTITY_V2
    )
    formula_parser_identity = _identity(
        store, f"formula-parser-{salt}", a.MEDIA_FORMULA_PARSER_IDENTITY_V2
    )
    input_parser_identity = _identity(
        store, f"input-parser-{salt}", a.MEDIA_INPUT_PARSER_IDENTITY_V2
    )
    window_parser_identity = _identity(
        store, f"window-parser-{salt}", a.MEDIA_WINDOW_PARSER_IDENTITY_V2
    )
    policy = a.TrustedMFUAuthorityPolicyV2(
        policy_id=f"r1-mfu-policy-{salt}",
        authority_identity=authority_identity,
        resolver_identity=resolver_identity,
        formula_parser_identity=formula_parser_identity,
        input_parser_identity=input_parser_identity,
        window_parser_identity=window_parser_identity,
        controller_receipt_prefix="gs://trusted-control/mfu-receipts/",
        allowed_replica_regions=(REGION,),
    )

    tensor = _raw(
        store,
        f"tensor-{salt}",
        m.MEDIA_TENSOR_INVENTORY_V2,
        FORMULA_SCHEMA,
        role="tensor",
        scalar=7_168,
        dtype="bfloat16",
    )
    executable = _raw(
        store,
        f"executable-{salt}",
        m.MEDIA_EXECUTABLE_V2,
        FORMULA_SCHEMA,
        role="executable",
        scalar=1_024,
        dtype="bfloat16",
    )
    hlo = _raw(
        store,
        f"hlo-{salt}",
        m.MEDIA_HLO_V2,
        FORMULA_SCHEMA,
        role="hlo",
        scalar=1_024,
        dtype="bfloat16",
    )
    operations = _raw(
        store,
        f"operations-{salt}",
        m.MEDIA_OPERATION_INVENTORY_V2,
        FORMULA_SCHEMA,
        role="operations",
        scalar=28,
        dtype="bfloat16",
    )
    formula = m.PrefixFlopFormulaV2.build(
        tensor_inventory=tensor,
        executable=executable,
        hlo=hlo,
        counted_operation_inventory=operations,
        sequences_per_step=256,
    )
    formula_ref = _stored_ref(
        store,
        label=f"formula-{salt}",
        media_type=m.MEDIA_FLOP_FORMULA_V2,
        value=formula,
    )
    inventory = _inventory(salt=salt)
    inventory_ref = _stored_ref(
        store,
        label=f"inventory-{salt}",
        media_type=m.MEDIA_DEVICE_INVENTORY_V2,
        value=inventory,
    )
    manifests = {
        kind: _raw(
            store,
            f"manifest-{salt}-{kind.lower()}",
            m.MEDIA_INPUT_MANIFEST_V2,
            MANIFEST_SCHEMA,
            data_kind=kind,
            dataset_sha256=_sha(f"dataset-{salt}-{kind}"),
        )
        for kind in ("REAL", "SYNTHETIC")
    }
    slices = []
    steps = 500
    for group in range(3):
        for kind in ("REAL", "SYNTHETIC"):
            histogram = m.PackedSegmentHistogramV2(
                (m.SegmentLengthBinV2(1_024, steps * 256),)
            )
            ledger = _raw(
                store,
                f"ledger-{salt}-{group}-{kind.lower()}",
                m.MEDIA_SEGMENT_LEDGER_V2,
                LEDGER_SCHEMA,
                parent_manifest_sha256=manifests[kind].sha256,
                group_index=group,
                data_kind=kind,
                batch_start=group * steps,
                batch_end=(group + 1) * steps,
                packed_sequence_count=steps * 256,
                histogram=histogram.as_dict(),
            )
            slices.append(
                m.PackedInputSliceV2(
                    group_index=group,
                    data_kind=kind,
                    parent_input_manifest_sha256=manifests[kind].sha256,
                    batch_start=group * steps,
                    batch_end=(group + 1) * steps,
                    packed_sequence_count=steps * 256,
                    histogram=histogram,
                    segment_ledger=ledger,
                )
            )
    by_slice = {(item.group_index, item.data_kind): item for item in slices}
    entries = []
    window_index = 0
    for group in range(3):
        for kind in ("REAL", "SYNTHETIC"):
            for instrumentation in ("BASELINE", "INSTRUMENTED"):
                entries.append(
                    m.MFUWindowPlanEntryV2(
                        window_id=(
                            f"g{group}-{kind.lower()}-{instrumentation.lower()}"
                        ),
                        group_index=group,
                        data_kind=kind,
                        instrumentation_kind=instrumentation,
                        step_start=5 + window_index * steps,
                        step_end=5 + (window_index + 1) * steps,
                        input_slice_sha256=by_slice[(group, kind)].digest,
                    )
                )
                window_index += 1
    plan = m.PrefixMFUWindowPlanV2(tuple(slices), tuple(entries))
    science = _identity(store, f"science-{salt}", m.MEDIA_SCIENTIFIC_IDENTITY_V2)
    implementation = _identity(
        store, f"implementation-{salt}", m.MEDIA_IMPLEMENTATION_IDENTITY_V2
    )
    deployment = _identity(store, f"deployment-{salt}", m.MEDIA_DEPLOYMENT_IDENTITY_V2)
    run = m.PrefixMFURunContractV2(
        attempt_id=f"headline-attempt-{salt}",
        scientific_identity=science,
        implementation_identity=implementation,
        deployment_identity=deployment,
        semantic_authority=authority_identity,
        device_inventory=inventory_ref,
        tensor_inventory=tensor,
        executable=executable,
        hlo=hlo,
        flop_formula=formula_ref,
        real_input_manifest=manifests["REAL"],
        synthetic_input_manifest=manifests["SYNTHETIC"],
        window_plan=plan,
        sequences_per_step=256,
        warmup_steps=5,
        accelerator_type="v6e-64",
        topology="8x8",
        physical_chips=64,
        denominator_dtype="bfloat16",
        peak_flops_per_chip=m.V6E_CHIP_BF16_PEAK_FLOPS_V2,
    )

    safe_warmup_finished = BASE_START_NS - 1_000_000_000
    measurement_execution_id = f"measurement-execution-{salt}"
    raw_warmup_finished = BASE_START_NS + 1 if warmup_overlap else safe_warmup_finished
    cursor = BASE_START_NS
    windows = []
    for index, entry in enumerate(entries):
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
        allocated = (dispatch * 105 + 99) // 100
        trace = _raw(
            store,
            f"trace-{salt}-{entry.window_id}",
            m.MEDIA_STEP_TRACE_V2,
            TRACE_SCHEMA,
            attempt_id=run.attempt_id,
            measurement_execution_id=measurement_execution_id,
            window_id=entry.window_id,
            executable_sha256=run.executable.sha256,
            hlo_sha256=run.hlo.sha256,
            device_inventory_sha256=run.device_inventory.sha256,
            controller_fence=trace_controller_fence,
            step_start=entry.step_start,
            step_end=entry.step_end,
            measurement_started_at_unix_ns=cursor,
            measurement_finished_at_unix_ns=cursor + wall,
            step_duration_ns=list(durations),
            input_wait_duration_ns=[value // 100 for value in durations],
            compile_started_at_unix_ns=BASE_START_NS - 10_000_000_000,
            compile_finished_at_unix_ns=BASE_START_NS - 9_000_000_000,
            warmup_started_at_unix_ns=BASE_START_NS - 8_000_000_000,
            warmup_finished_at_unix_ns=raw_warmup_finished,
            completed_warmup_steps=5,
            first_measurement_step=entries[0].step_start,
            first_measurement_started_at_unix_ns=BASE_START_NS,
        )
        timing = _raw(
            store,
            f"timing-{salt}-{entry.window_id}",
            m.MEDIA_TIMING_BLOCKS_V2,
            TIMING_SCHEMA,
            measurement_execution_id=measurement_execution_id,
            window_id=entry.window_id,
            device_output_tokens=output_tokens,
            device_output_duration_ns=wall if output_tokens else 0,
            persisted_tokens=persisted_tokens,
            persistence_duration_ns=persistence_duration,
        )
        profile = _raw(
            store,
            f"profile-{salt}-{entry.window_id}",
            m.MEDIA_PROFILER_TRACE_V2,
            PROFILE_SCHEMA,
            measurement_execution_id=measurement_execution_id,
            window_id=entry.window_id,
            executable_sha256=run.executable.sha256,
            hlo_sha256=run.hlo.sha256,
            device_ids=list(_device_ids(inventory)),
        )
        host = _raw(
            store,
            f"host-{salt}-{entry.window_id}",
            m.MEDIA_HOST_METRICS_V2,
            HOST_SCHEMA,
            measurement_execution_id=(
                f"spliced-execution-{salt}"
                if splice_host_execution and index == 0
                else measurement_execution_id
            ),
            window_id=entry.window_id,
            host_total_step_duration_ns=[wall] * 16,
            peak_hbm_bytes_per_device=[20_000_000_000] * 64,
            writer_queue_depth_by_step=[0] * steps,
        )
        events = _raw(
            store,
            f"events-{salt}-{entry.window_id}",
            m.MEDIA_INTEGRITY_EVENTS_V2,
            EVENT_SCHEMA,
            measurement_execution_id=measurement_execution_id,
            window_id=entry.window_id,
            oom_events=0,
            recompilations=0,
            dropped_routed_tokens=0,
            capacity_overflows=0,
            nonfinite_events=0,
        )
        routing = _raw(
            store,
            f"routing-{salt}-{entry.window_id}",
            m.MEDIA_ROUTING_CAPACITY_LEDGER_V2,
            ROUTING_SCHEMA,
            measurement_execution_id=measurement_execution_id,
            window_id=entry.window_id,
            shape=list(
                capacity_shape
                if capacity_shape is not None and index == 0
                else (steps, 28, 256)
            ),
            default_numerator=105,
            default_denominator=100,
            overrides=([[0, 0, 0, 111, 100]] if overload_window == index else []),
            dispatched_routed_rows_by_invocation=[dispatch] * 28,
            allocated_routed_capacity_rows_by_invocation=[allocated] * 28,
        )
        persistence = (
            _raw(
                store,
                f"persist-{salt}-{entry.window_id}",
                m.MEDIA_PERSISTENCE_RECEIPT_V2,
                PERSIST_SCHEMA,
                measurement_execution_id=measurement_execution_id,
                window_id=entry.window_id,
                tokens=persisted_tokens,
                duration_ns=persistence_duration,
                queue_drained=1,
            )
            if persisted_tokens
            else None
        )
        windows.append(
            m.PrefixMFUWindowEvidenceV2(
                run_contract_sha256=run.digest,
                window_id=entry.window_id,
                input_slice_sha256=entry.input_slice_sha256,
                step_start=entry.step_start,
                step_end=entry.step_end,
                step_trace=trace,
                timing_blocks=timing,
                profiler_trace=profile,
                host_metrics=host,
                integrity_events=events,
                routing_capacity_ledger=routing,
                persistence_receipt=persistence,
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
                allocated_routed_capacity_rows_by_invocation=(allocated,) * 28,
                oom_events=0,
                recompilations=0,
                dropped_routed_tokens=0,
                capacity_overflows=0,
                nonfinite_events=0,
            )
        )
        cursor += wall + 1_000_000
    windows_tuple = tuple(windows)

    formula_parser = _FormulaParser(formula_parser_identity)
    input_parser = _InputParser(input_parser_identity)
    window_parser = _WindowParser(window_parser_identity)
    resolver = a.JSONCASGenerationPinnedResolverV2(store, identity=resolver_identity)

    registered_ns = (
        BASE_START_NS + 1_000_000 if late_predeclare else BASE_START_NS - 20_000_000_000
    )
    controller = CampaignController(
        store,
        base_uri="gs://trusted-control",
        campaign_id=f"mfu-campaign-{salt}",
        clock_ms=lambda: registered_ns // 1_000_000,
        content_verifier=lambda _content: True,
    )
    work = _harvest(run.scientific_identity.sha256)
    controller.initialize((work,))
    lease = controller.acquire(
        work.work_id,
        attempt_id=run.attempt_id,
        executor_id="mfu-benchmark-controller",
        deployment_identity_sha256=run.deployment_identity.sha256,
        lease_duration_ms=lease_duration_ms,
    )
    predeclaration = controller.predeclare_mfu_v2(
        work_id=work.work_id,
        run_contract=run,
        policy=policy,
        replica_region=REGION,
        measurement_not_before_unix_ns=(
            BASE_START_NS + 2_000_000 if late_predeclare else BASE_START_NS
        ),
    )
    registry = controller.mfu_evidence_registry_v2(policy=policy, replica_region=REGION)
    assert lease.fence == 1

    formula_artifacts = {
        name: resolver.resolve(reference)
        for name, reference in (
            ("tensor_inventory", run.tensor_inventory),
            ("executable", run.executable),
            ("hlo", run.hlo),
            ("operation_inventory", formula.counted_operation_inventory),
        )
    }
    formula_receipt = formula_parser.parse_formula(
        run_contract=run,
        formula=formula,
        device_inventory=inventory,
        artifacts=formula_artifacts,
    )
    formula_receipt_ref = registry.publish_parser_receipt(formula_receipt)

    first_trace = window_parser._trace(resolver.resolve(windows_tuple[0].step_trace))
    compile_receipt = a.MFUCompileWarmupReceiptPayloadV2(
        run_contract_sha256=run.digest,
        attempt_id=run.attempt_id,
        measurement_execution_id=first_trace["measurement_execution_id"],
        executable_sha256=run.executable.sha256,
        hlo_sha256=run.hlo.sha256,
        device_inventory_sha256=run.device_inventory.sha256,
        controller_fence=first_trace["controller_fence"],
        compile_started_at_unix_ns=first_trace["compile_started_at_unix_ns"],
        compile_finished_at_unix_ns=first_trace["compile_finished_at_unix_ns"],
        warmup_started_at_unix_ns=first_trace["warmup_started_at_unix_ns"],
        warmup_finished_at_unix_ns=safe_warmup_finished,
        completed_warmup_steps=5,
        first_measurement_step=entries[0].step_start,
        first_measurement_started_at_unix_ns=BASE_START_NS,
        measurement_recompilations=0,
    )
    compile_receipt_ref = registry.publish_parser_receipt(compile_receipt)

    input_receipts = []
    input_receipt_refs = []
    for input_slice in plan.input_slices:
        parent = (
            run.real_input_manifest
            if input_slice.data_kind == "REAL"
            else run.synthetic_input_manifest
        )
        receipt = input_parser.parse_input(
            run_contract=run,
            input_slice=input_slice,
            parent_manifest=resolver.resolve(parent),
            segment_ledger=resolver.resolve(input_slice.segment_ledger),
        )
        input_receipts.append(receipt)
        input_receipt_refs.append(registry.publish_parser_receipt(receipt))

    window_receipts = []
    window_receipt_refs = []
    for index, (entry, evidence) in enumerate(
        zip(plan.entries, windows_tuple, strict=True)
    ):
        artifacts = {
            f"artifact_{artifact_index}": resolver.resolve(reference)
            for artifact_index, reference in enumerate(_window_refs(evidence))
        }
        if overload_window == index:
            trace_record = window_parser._trace(artifacts["artifact_0"])
            receipt = _window_payload(
                run=run,
                entry=entry,
                evidence=evidence,
                inventory=inventory,
                controller_fence=trace_record["controller_fence"],
                measurement_execution_id=trace_record["measurement_execution_id"],
                started=trace_record["measurement_started_at_unix_ns"],
                finished=trace_record["measurement_finished_at_unix_ns"],
                capacity_steps=steps,
                capacity_router_invocations=28,
                capacity_routed_experts=256,
                cells=steps * 28 * 256,
                matrix_sha=artifacts["artifact_5"].reference.sha256,
                maximum_numerator=105,
                maximum_denominator=100,
                overloads=0,
            )
        else:
            receipt = window_parser.parse_window(
                run_contract=run,
                entry=entry,
                input_slice=plan.input_slice(entry.input_slice_sha256),
                evidence=evidence,
                device_inventory=inventory,
                artifacts=artifacts,
            )
        window_receipts.append(receipt)
        window_receipt_refs.append(registry.publish_parser_receipt(receipt))

    receipts = m.MFUSemanticReceiptsV2(
        authority_identity=authority_identity,
        predeclaration_receipt=predeclaration,
        compile_warmup_receipt=compile_receipt_ref,
        formula_reconciliation_receipt=formula_receipt_ref,
        input_reduction_receipts=tuple(input_receipt_refs),
        window_reduction_receipts=tuple(window_receipt_refs),
    )
    authority = a.ProtocolOracleMFUEvidenceAuthorityV2(
        policy=policy,
        resolver=resolver,
        formula_parser=formula_parser,
        input_parser=input_parser,
        window_parser=window_parser,
    )
    return _Fixture(
        store=store,
        controller=controller,
        registry=registry,
        policy=policy,
        resolver=resolver,
        formula_parser=formula_parser,
        input_parser=input_parser,
        window_parser=window_parser,
        authority=authority,
        run=run,
        formula=formula,
        inventory=inventory,
        windows=windows_tuple,
        receipts=receipts,
        formula_receipt=formula_receipt,
        compile_receipt=compile_receipt,
        input_receipts=tuple(input_receipts),
        window_receipts=tuple(window_receipts),
    )


def _qualify_unverified(
    fixture: _Fixture,
    *,
    windows: tuple[m.PrefixMFUWindowEvidenceV2, ...] | None = None,
    receipts: m.MFUSemanticReceiptsV2 | None = None,
) -> m.PrefixMFUQualificationV2:
    return fixture.authority.qualify_unverified(
        run_contract=fixture.run,
        formula=fixture.formula,
        device_inventory=fixture.inventory,
        windows=fixture.windows if windows is None else windows,
        semantic_receipts=fixture.receipts if receipts is None else receipts,
    )


def test_generation_pinned_resolver_requires_the_exact_generation() -> None:
    fixture = _build_fixture()
    reference = fixture.run.executable
    assert fixture.resolver.resolve(reference).reference == reference
    replica = replace(
        reference.replicas[0], generation=reference.replicas[0].generation + 1
    )
    stale = replace(reference, replicas=(replica,))
    with pytest.raises(a.MFUAuthorityError, match="exact generation"):
        fixture.resolver.resolve(stale)


def test_gcs_resolver_hashes_bytes_and_never_lists() -> None:
    store = MemoryJSONCASStore()
    identity = _identity(store, "gcs-resolver", a.MEDIA_CONTENT_RESOLVER_IDENTITY_V2)
    reference = _raw(
        store,
        "gcs-payload",
        m.MEDIA_HLO_V2,
        "moe-sae.test.gcs-payload.v2",
        value=7,
    )
    stored = store.read(reference.replicas[0].uri)
    assert stored is not None

    class _Blob:
        def __init__(self, value: bytes) -> None:
            replica = reference.replicas[0]
            self.generation = replica.generation
            self.size = replica.size_bytes
            self.crc32c = replica.crc32c
            self.value = value

        def reload(self, **kwargs: Any) -> None:
            assert kwargs["if_generation_match"] == self.generation

        def download_as_bytes(self, **kwargs: Any) -> bytes:
            assert kwargs["if_generation_match"] == self.generation
            return self.value

    class _Bucket:
        def __init__(self, blob: _Blob) -> None:
            self._blob = blob

        def blob(self, _name: str) -> _Blob:
            return self._blob

    class _Client:
        def __init__(self, blob: _Blob) -> None:
            self._blob = blob

        def bucket(self, _name: str) -> _Bucket:
            return _Bucket(self._blob)

    resolver = a.GCSGenerationPinnedContentResolverV2(
        identity=identity,
        client=_Client(_Blob(stored.value)),
    )
    assert resolver.resolve(reference).value == stored.value
    corrupt = a.GCSGenerationPinnedContentResolverV2(
        identity=identity,
        client=_Client(_Blob(stored.value[:-1] + b"0")),
    )
    with pytest.raises(a.MFUAuthorityError, match="SHA-256|CRC32C"):
        corrupt.resolve(reference)


def test_policy_parser_and_resolver_identities_fail_closed() -> None:
    fixture = _build_fixture()
    wrong_resolver = _identity(
        fixture.store,
        "untrusted-resolver",
        a.MEDIA_CONTENT_RESOLVER_IDENTITY_V2,
    )
    wrong_formula = _identity(
        fixture.store,
        "untrusted-formula-parser",
        a.MEDIA_FORMULA_PARSER_IDENTITY_V2,
    )
    wrong_input = _identity(
        fixture.store,
        "untrusted-input-parser",
        a.MEDIA_INPUT_PARSER_IDENTITY_V2,
    )
    wrong_window = _identity(
        fixture.store,
        "untrusted-window-parser",
        a.MEDIA_WINDOW_PARSER_IDENTITY_V2,
    )
    cases = (
        (
            a.JSONCASGenerationPinnedResolverV2(
                fixture.store,
                identity=replace(
                    fixture.policy.resolver_identity,
                    replicas=(
                        replace(
                            fixture.policy.resolver_identity.replicas[0],
                            generation=(
                                fixture.policy.resolver_identity.replicas[0].generation
                                + 1
                            ),
                        ),
                    ),
                ),
            ),
            fixture.formula_parser,
            fixture.input_parser,
            fixture.window_parser,
            "resolver",
        ),
        (
            a.JSONCASGenerationPinnedResolverV2(
                fixture.store,
                identity=wrong_resolver,
            ),
            fixture.formula_parser,
            fixture.input_parser,
            fixture.window_parser,
            "resolver",
        ),
        (
            fixture.resolver,
            replace(fixture.formula_parser, identity=wrong_formula),
            fixture.input_parser,
            fixture.window_parser,
            "formula parser",
        ),
        (
            fixture.resolver,
            fixture.formula_parser,
            replace(fixture.input_parser, identity=wrong_input),
            fixture.window_parser,
            "input parser",
        ),
        (
            fixture.resolver,
            fixture.formula_parser,
            fixture.input_parser,
            replace(fixture.window_parser, identity=wrong_window),
            "window parser",
        ),
    )
    for resolver, formula_parser, input_parser, window_parser, message in cases:
        with pytest.raises(a.MFUAuthorityError, match=message):
            a.ProtocolOracleMFUEvidenceAuthorityV2(
                policy=fixture.policy,
                resolver=resolver,
                formula_parser=formula_parser,
                input_parser=input_parser,
                window_parser=window_parser,
            )


def test_semantic_scalar_tamper_is_rejected_with_unchanged_raw_receipts() -> None:
    fixture = _build_fixture()
    original = fixture.windows[0]
    waits = list(original.input_wait_duration_ns)
    waits[0] += 1
    windows = (
        replace(original, input_wait_duration_ns=tuple(waits)),
        *fixture.windows[1:],
    )
    with pytest.raises(m.MFUV2ContractError, match="semantic verification"):
        _qualify_unverified(fixture, windows=windows)


@pytest.mark.parametrize(
    ("kind", "message"),
    (
        ("run", "semantic verification"),
        ("executable", "semantic verification"),
        ("hlo", "semantic verification"),
        ("inventory", "semantic verification"),
    ),
)
def test_cross_run_executable_hlo_and_inventory_receipt_splices_fail(
    kind: str, message: str
) -> None:
    fixture = _build_fixture(salt=kind)
    if kind == "run":
        bad = replace(fixture.formula_receipt, run_contract_sha256=_sha("other-run"))
        receipts = replace(
            fixture.receipts,
            formula_reconciliation_receipt=fixture.registry.publish_parser_receipt(bad),
        )
    elif kind in {"executable", "hlo"}:
        field = f"{kind}_sha256"
        bad = replace(fixture.compile_receipt, **{field: _sha(f"other-{kind}")})
        receipts = replace(
            fixture.receipts,
            compile_warmup_receipt=fixture.registry.publish_parser_receipt(bad),
        )
    else:
        bad = replace(
            fixture.window_receipts[0],
            device_inventory_sha256=_sha("other-inventory"),
        )
        refs = list(fixture.receipts.window_reduction_receipts)
        refs[0] = fixture.registry.publish_parser_receipt(bad)
        receipts = replace(fixture.receipts, window_reduction_receipts=tuple(refs))
    with pytest.raises(m.MFUV2ContractError, match=message):
        _qualify_unverified(fixture, receipts=receipts)


def test_one_per_step_router_expert_overload_cannot_hide_in_aggregate() -> None:
    fixture = _build_fixture(overload_window=0)
    window = fixture.windows[0]
    assert max(
        Fraction(capacity, dispatched)
        for dispatched, capacity in zip(
            window.dispatched_routed_rows_by_invocation,
            window.allocated_routed_capacity_rows_by_invocation,
            strict=True,
        )
    ) <= Fraction(11, 10)
    with pytest.raises(m.MFUV2ContractError, match="semantic verification"):
        _qualify_unverified(fixture)


def test_capacity_shape_permutation_cannot_preserve_only_the_cell_product() -> None:
    fixture = _build_fixture(capacity_shape=(28, 500, 256))
    with pytest.raises(m.MFUV2ContractError, match="semantic verification"):
        _qualify_unverified(fixture)


def test_trace_controller_fence_must_match_preregistered_lease() -> None:
    fixture = _build_fixture(trace_controller_fence=2)
    with pytest.raises(m.MFUV2ContractError, match="semantic verification"):
        _qualify_unverified(fixture)


def test_measurement_must_finish_inside_the_admitted_lease() -> None:
    fixture = _build_fixture(lease_duration_ms=30_000)
    with pytest.raises(m.MFUV2ContractError, match="semantic verification"):
        _qualify_unverified(fixture)


def test_same_attempt_artifacts_require_one_measurement_execution() -> None:
    with pytest.raises(a.MFUAuthorityError, match="raw window scalars"):
        _build_fixture(splice_host_execution=True)


def test_late_controller_predeclaration_is_rejected() -> None:
    fixture = _build_fixture(late_predeclare=True)
    with pytest.raises(m.MFUV2ContractError, match="semantic verification"):
        _qualify_unverified(fixture)


def test_warmup_overlapping_first_measurement_is_rejected() -> None:
    fixture = _build_fixture(warmup_overlap=True)
    with pytest.raises(m.MFUV2ContractError, match="semantic verification"):
        _qualify_unverified(fixture)


def test_report_consumer_is_disabled_and_raw_qualification_stays_unverified() -> None:
    fixture = _build_fixture()

    class _LiarAuthority:
        identity = fixture.policy.authority_identity
        policy = fixture.policy

        def verify_content(self, _reference: ContentRefV1) -> bool:
            return True

        def verify_semantics(self, **_evidence: Any) -> bool:
            return True

    with pytest.raises(a.MFUAuthorityError, match="closed factory"):
        a.MFUReportConsumerV2(
            policy=fixture.policy,
            authority=_LiarAuthority(),  # type: ignore[arg-type]
        )

    qualification = _qualify_unverified(fixture)
    assert "qualification_status" not in qualification.as_dict()
    assert qualification.as_dict()["derived"]["overall_qualification_status"] == (
        "UNVERIFIED"
    )
    assert qualification.as_dict()["derived"]["numerical_gate_status"] == "SATISFIED"
    with pytest.raises(a.MFUAuthorityError, match="closed factory"):
        a.VerifiedMFUQualificationEnvelopeV2.from_dict(qualification.as_dict())


def test_controller_oracle_reducer_stops_before_headline_pass() -> None:
    fixture = _build_fixture()
    predecl_ref = fixture.receipts.predeclaration_receipt
    assert predecl_ref.replicas[0].uri.startswith(
        fixture.policy.controller_receipt_prefix
    )
    resolved = fixture.resolver.resolve(predecl_ref)
    predecl = a.MFUPredeclarationReceiptPayloadV2.from_dict(parse_json(resolved.value))
    head = fixture.store.read(fixture.controller.head_uri)
    assert head is not None
    assert predecl.campaign_head_generation == head.generation
    assert predecl.registered_at_unix_ns < BASE_START_NS

    qualification = _qualify_unverified(fixture)
    assert qualification.minimum_real_window_mfu >= Fraction(51, 100)
    assert qualification.as_dict()["derived"]["overall_qualification_status"] == (
        "UNVERIFIED"
    )
    with pytest.raises(a.MFUAuthorityError, match="closed factory"):
        fixture.authority.qualify_and_seal(
            run_contract=fixture.run,
            formula=fixture.formula,
            device_inventory=fixture.inventory,
            windows=fixture.windows,
            semantic_receipts=fixture.receipts,
        )
    with pytest.raises(a.MFUAuthorityError, match="closed factory"):
        a.ProductionMFUEvidenceAuthorityV2(
            policy=fixture.policy,
            resolver=fixture.resolver,
            formula_parser=fixture.formula_parser,
            input_parser=fixture.input_parser,
            window_parser=fixture.window_parser,
        )
    with pytest.raises(a.MFUAuthorityError, match="closed factory"):
        fixture.registry.publish_verified_envelope(object())  # type: ignore[arg-type]


def test_same_identity_liar_parsers_cannot_cross_production_pass_gate() -> None:
    fixture = _build_fixture(salt="same-identity-liars")

    @dataclass(frozen=True)
    class _FormulaLiar:
        identity: ContentRefV1

        def parse_formula(self, **_ignored: Any) -> Any:
            return fixture.formula_receipt

    @dataclass(frozen=True)
    class _InputLiar:
        identity: ContentRefV1

        def parse_input(self, **values: Any) -> Any:
            by_digest = dict(
                zip(
                    (item.digest for item in fixture.run.window_plan.input_slices),
                    fixture.input_receipts,
                    strict=True,
                )
            )
            return by_digest[values["input_slice"].digest]

    @dataclass(frozen=True)
    class _WindowLiar:
        identity: ContentRefV1

        def parse_compile_warmup(self, **_ignored: Any) -> Any:
            return fixture.compile_receipt

        def parse_window(self, **values: Any) -> Any:
            by_id = dict(
                zip(
                    (item.window_id for item in fixture.run.window_plan.entries),
                    fixture.window_receipts,
                    strict=True,
                )
            )
            return by_id[values["entry"].window_id]

    formula_liar = _FormulaLiar(fixture.policy.formula_parser_identity)
    input_liar = _InputLiar(fixture.policy.input_parser_identity)
    window_liar = _WindowLiar(fixture.policy.window_parser_identity)
    oracle = a.ProtocolOracleMFUEvidenceAuthorityV2(
        policy=fixture.policy,
        resolver=fixture.resolver,
        formula_parser=formula_liar,
        input_parser=input_liar,
        window_parser=window_liar,
    )
    qualification = oracle.qualify_unverified(
        run_contract=fixture.run,
        formula=fixture.formula,
        device_inventory=fixture.inventory,
        windows=fixture.windows,
        semantic_receipts=fixture.receipts,
    )
    assert qualification.as_dict()["derived"]["overall_qualification_status"] == (
        "UNVERIFIED"
    )
    with pytest.raises(a.MFUAuthorityError, match="exact real parsers"):
        oracle.qualify_and_seal()
    with pytest.raises(a.MFUAuthorityError, match="exact real parsers"):
        a.ProductionMFUEvidenceAuthorityV2(
            policy=fixture.policy,
            resolver=fixture.resolver,
            formula_parser=formula_liar,
            input_parser=input_liar,
            window_parser=window_liar,
        )


def test_cross_campaign_prereg_replay_cannot_cross_production_pass_gate() -> None:
    fixture = _build_fixture(salt="cross-campaign-prereg")
    original = a.MFUPredeclarationReceiptPayloadV2.from_dict(
        parse_json(
            fixture.resolver.resolve(fixture.receipts.predeclaration_receipt).value
        )
    )
    rogue_registry = a.ControllerMFUEvidenceRegistryV2(
        fixture.store,
        policy=fixture.policy,
        campaign_id="other-campaign",
        campaign_head_uri="gs://other-control/campaigns/other/head.json",
        campaign_head_generation=999,
        campaign_head_revision=777,
        replica_region=REGION,
        clock_ns=lambda: original.registered_at_unix_ns,
    )
    replayed_ref = rogue_registry._predeclare_from_controller(
        fixture.run,
        work_id="other-work",
        controller_fence=original.controller_fence,
        executor_id="other-executor",
        lease_acquired_at_unix_ns=original.lease_acquired_at_unix_ns,
        lease_expires_at_unix_ns=original.lease_expires_at_unix_ns,
        measurement_not_before_unix_ns=original.measurement_not_before_unix_ns,
    )
    replayed = a.MFUPredeclarationReceiptPayloadV2.from_dict(
        parse_json(fixture.resolver.resolve(replayed_ref).value)
    )
    assert (replayed.campaign_id, replayed.work_id, replayed.executor_id) == (
        "other-campaign",
        "other-work",
        "other-executor",
    )

    receipts = replace(fixture.receipts, predeclaration_receipt=replayed_ref)
    # The injectable oracle deliberately exposes the still-missing controller
    # replay check, but it can emit only an UNVERIFIED protocol artifact.
    qualification = _qualify_unverified(fixture, receipts=receipts)
    assert qualification.as_dict()["derived"]["overall_qualification_status"] == (
        "UNVERIFIED"
    )
    with pytest.raises(a.MFUAuthorityError, match="controller admission event"):
        fixture.authority.qualify_and_seal(
            run_contract=fixture.run,
            formula=fixture.formula,
            device_inventory=fixture.inventory,
            windows=fixture.windows,
            semantic_receipts=receipts,
        )


def test_controller_active_lease_admission_rejects_unowned_runs() -> None:
    fixture = _build_fixture(salt="lease-admission")
    work = _harvest(fixture.run.scientific_identity.sha256)

    admitted = fixture.controller.predeclare_mfu_v2(
        work_id=work.work_id,
        run_contract=fixture.run,
        policy=fixture.policy,
        replica_region=REGION,
        measurement_not_before_unix_ns=BASE_START_NS,
    )
    payload = a.MFUPredeclarationReceiptPayloadV2.from_dict(
        parse_json(fixture.resolver.resolve(admitted).value)
    )
    assert payload.attempt_id == fixture.run.attempt_id
    assert payload.controller_fence == 1

    with pytest.raises(StaleAttempt, match="active lease"):
        fixture.controller.predeclare_mfu_v2(
            work_id=work.work_id,
            run_contract=replace(fixture.run, attempt_id="stale-attempt"),
            policy=fixture.policy,
            replica_region=REGION,
            measurement_not_before_unix_ns=BASE_START_NS,
        )

    wrong_deployment = _identity(
        fixture.store,
        "wrong-lease-deployment",
        m.MEDIA_DEPLOYMENT_IDENTITY_V2,
    )
    with pytest.raises(InvalidReceipt, match="deployment"):
        fixture.controller.predeclare_mfu_v2(
            work_id=work.work_id,
            run_contract=replace(fixture.run, deployment_identity=wrong_deployment),
            policy=fixture.policy,
            replica_region=REGION,
            measurement_not_before_unix_ns=BASE_START_NS,
        )

    now_ms = [BASE_START_NS // 1_000_000 - 2_000]
    unleased_store = MemoryJSONCASStore()
    unleased = CampaignController(
        unleased_store,
        base_uri="gs://unleased-control",
        campaign_id="unleased-mfu-campaign",
        clock_ms=lambda: now_ms[0],
    )
    unleased.initialize((work,))
    with pytest.raises(LeaseUnavailable, match="active lease"):
        unleased.predeclare_mfu_v2(
            work_id=work.work_id,
            run_contract=fixture.run,
            policy=fixture.policy,
            replica_region=REGION,
            measurement_not_before_unix_ns=BASE_START_NS,
        )
    unleased.acquire(
        work.work_id,
        attempt_id=fixture.run.attempt_id,
        executor_id="expiring-mfu-controller",
        deployment_identity_sha256=fixture.run.deployment_identity.sha256,
        lease_duration_ms=1,
    )
    now_ms[0] += 1
    with pytest.raises(LeaseUnavailable, match="active lease"):
        unleased.predeclare_mfu_v2(
            work_id=work.work_id,
            run_contract=fixture.run,
            policy=fixture.policy,
            replica_region=REGION,
            measurement_not_before_unix_ns=BASE_START_NS,
        )


def test_public_facade_keeps_v1_and_v2_authority_surfaces_side_by_side() -> None:
    assert r1.MFUQualificationV1 is not r1.PrefixMFUQualificationV2
    assert (
        r1.ProtocolOracleMFUEvidenceAuthorityV2
        is a.ProtocolOracleMFUEvidenceAuthorityV2
    )
    assert r1.ProductionMFUEvidenceAuthorityV2 is a.ProductionMFUEvidenceAuthorityV2
    assert r1.MFUReportConsumerV2 is a.MFUReportConsumerV2
