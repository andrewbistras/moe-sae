from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import json
import math

import pytest

from moe_sae.r1.protocol import (
    ActivationContractV1,
    AttemptLeaseV1,
    AttemptReceiptV1,
    ContentRefV1,
    CostLedgerV1,
    DeploymentIdentityV1,
    HarvestRangeV1,
    ImplementationIdentityV1,
    OptimizerContractV1,
    ProgressV1,
    ProtocolError,
    RegionalReplicaV1,
    SAEContractV1,
    SAEJobV1,
    ScientificIdentityV1,
    canonical_json_bytes,
    parse_json,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def _replica(
    *, region: str = "us-central1", uri_suffix: str = "a"
) -> RegionalReplicaV1:
    return RegionalReplicaV1(
        region=region,
        uri=f"gs://r1-{region}/objects/{uri_suffix}",
        generation=17,
        size_bytes=4096,
        crc32c="AAAAAA==",
        sha256=SHA_A,
    )


def _content() -> ContentRefV1:
    return ContentRefV1(
        sha256=SHA_A,
        size_bytes=4096,
        media_type="application/x-safetensors",
        replicas=(_replica(),),
    )


def _activation() -> ActivationContractV1:
    return ActivationContractV1(
        layer_one_indexed=31,
        capture_site="weighted-routed-expert-aggregate-before-shared-expert",
        hidden_dim=7168,
        expert_count=256,
        routed_experts_per_token=8,
        sequence_length=1024,
        activation_dtype="bfloat16",
        expert_id_dtype="uint16",
        router_weight_dtype="bfloat16",
        router_weight_sum=2.5,
        router_selection_semantics="deepseek3-biased-grouped-sigmoid-v1",
    )


def _sae() -> SAEContractV1:
    return SAEContractV1(
        conditions=("A", "B", "F"),
        bank_count=256,
        bank_width=512,
        baseline_width=4096,
        full_width=131072,
        top_k_features=64,
        per_token_topk=True,
        objective="globally-centered-normalized-sse",
        seeds=(1, 2, 3),
    )


def _optimizer() -> OptimizerContractV1:
    return OptimizerContractV1(
        training_tokens=50_000_000,
        updates=3052,
        effective_batch_tokens=16_384,
        microbatch_tokens=4096,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        weight_decay=0.0,
        gradient_clip_norm=1.0,
        warmup_updates=250,
        peak_learning_rate=3e-4,
        final_learning_rate=3e-5,
    )


def _science() -> ScientificIdentityV1:
    return ScientificIdentityV1(
        model_id="deepseek-ai/DeepSeek-R1-0528",
        model_revision="4236a6af538feda4548eca9ab308586007567f52",
        model_tensor_inventory_sha256=SHA_A,
        tokenizer_id="deepseek-ai/DeepSeek-R1-0528",
        tokenizer_revision="4236a6af538feda4548eca9ab308586007567f52",
        corpus_id="allenai/c4",
        corpus_revision="3.1.0",
        corpus_fingerprint_sha256=SHA_B,
        train_split="en/train",
        evaluation_split="en/validation",
        activation=_activation(),
        sae=_sae(),
        optimizer=_optimizer(),
        report_contract_version="reconstruction-efficiency-v1",
    )


def _implementation() -> ImplementationIdentityV1:
    return ImplementationIdentityV1(
        repository_commit="1" * 40,
        maxtext_commit="2" * 40,
        image_manifest_sha256=SHA_C,
        python_version="3.12.11",
        jax_version="0.8.2",
        libtpu_version="0.0.36",
        adapter_version="deepseek-r1-prefix-v1",
        converter_version="deepseek-r1-prefix-converter-v1",
    )


def _deployment(
    implementation: ImplementationIdentityV1,
) -> DeploymentIdentityV1:
    return DeploymentIdentityV1(
        implementation_identity_sha256=implementation.digest,
        deployment_envelope_sha256=SHA_C,
        device_inventory_sha256=SHA_D,
        attempt_id="attempt-0001",
    )


def test_scientific_implementation_and_deployment_identities_are_separate() -> None:
    science = _science()
    implementation = _implementation()
    deployment = _deployment(implementation)

    moved = replace(
        deployment,
        deployment_envelope_sha256=SHA_B,
        device_inventory_sha256=SHA_A,
        attempt_id="attempt-0002",
    )
    rebuilt = replace(implementation, adapter_version="deepseek-r1-prefix-v2")
    changed_science = replace(science, corpus_revision="3.1.1")

    assert moved.digest != deployment.digest
    assert rebuilt.digest != implementation.digest
    assert science.digest == _science().digest
    assert changed_science.digest != science.digest
    assert science.digest not in {implementation.digest, deployment.digest}


def test_harvest_and_sae_work_ids_are_topology_neutral() -> None:
    harvest = HarvestRangeV1(
        scientific_identity_sha256=_science().digest,
        corpus_fingerprint_sha256=SHA_B,
        split="en/train",
        record_start=12_000,
        record_end=13_000,
        scientific_token_start=9_800_000,
        scientific_token_end=10_600_000,
    )
    job = SAEJobV1(
        scientific_identity_sha256=_science().digest,
        capture_manifest_sha256=SHA_C,
        activation_order_sha256=SHA_D,
        condition="A",
        seed=1,
    )

    first_physical_deployment = replace(
        _deployment(_implementation()),
        deployment_envelope_sha256=SHA_A,
    )
    second_physical_deployment = _deployment(_implementation())

    assert harvest.work_id == HarvestRangeV1.from_json(harvest.canonical_json()).work_id
    assert job.work_id == SAEJobV1.from_json(job.canonical_json()).work_id
    assert replace(harvest, scientific_identity_sha256=SHA_A).work_id != harvest.work_id
    assert first_physical_deployment.digest != second_physical_deployment.digest
    # Neither deployment is an input to either logical work identifier.
    assert "us-central1" not in harvest.canonical_json()
    assert "8x8" not in job.canonical_json()


def test_implementation_and_deployment_wire_fields_do_not_duplicate_provenance() -> (
    None
):
    implementation = _implementation()
    deployment = _deployment(implementation)

    assert set(implementation.as_dict()) == {
        "schema",
        "repository_commit",
        "maxtext_commit",
        "image_manifest_sha256",
        "python_version",
        "jax_version",
        "libtpu_version",
        "adapter_version",
        "converter_version",
    }
    assert set(deployment.as_dict()) == {
        "schema",
        "implementation_identity_sha256",
        "deployment_envelope_sha256",
        "device_inventory_sha256",
        "attempt_id",
    }
    assert (
        ImplementationIdentityV1.from_json(implementation.canonical_json())
        == implementation
    )
    assert DeploymentIdentityV1.from_json(deployment.canonical_json()) == deployment


def test_old_ambiguous_or_duplicated_identity_fields_are_rejected() -> None:
    implementation = _implementation().as_dict()
    implementation["image_digest"] = f"sha256:{SHA_C}"
    del implementation["image_manifest_sha256"]
    with pytest.raises(ProtocolError, match="unknown fields.*missing fields"):
        ImplementationIdentityV1.from_dict(implementation)

    deployment = _deployment(_implementation()).as_dict()
    deployment["zone"] = "us-central1-b"
    with pytest.raises(ProtocolError, match="unknown fields"):
        DeploymentIdentityV1.from_dict(deployment)

    implementation = _implementation().as_dict()
    implementation["image_manifest_sha256"] = "A" * 64
    with pytest.raises(ProtocolError, match="invalid format"):
        ImplementationIdentityV1.from_dict(implementation)


@pytest.mark.parametrize(
    "field",
    (
        "implementation_identity_sha256",
        "deployment_envelope_sha256",
        "device_inventory_sha256",
    ),
)
def test_deployment_identity_requires_lowercase_sha256_links(field: str) -> None:
    payload = _deployment(_implementation()).as_dict()
    payload[field] = "A" * 64
    with pytest.raises(ProtocolError, match="invalid format"):
        DeploymentIdentityV1.from_dict(payload)


def test_nested_round_trip_is_canonical_and_order_independent_for_replicas() -> None:
    east = _replica(region="us-east5", uri_suffix="east")
    central = _replica(region="us-central1", uri_suffix="central")
    left = ContentRefV1(
        sha256=SHA_A,
        size_bytes=4096,
        media_type="application/x-safetensors",
        replicas=(east, central),
    )
    right = ContentRefV1(
        sha256=SHA_A,
        size_bytes=4096,
        media_type="application/x-safetensors",
        replicas=(central, east),
    )

    assert left == right
    assert left.digest == right.digest
    assert ContentRefV1.from_json(left.canonical_bytes()) == left
    assert left.canonical_bytes() == canonical_json_bytes(left.as_dict())


def test_content_ref_rejects_replica_that_disagrees_with_logical_content() -> None:
    wrong = replace(_replica(), sha256=SHA_B)
    with pytest.raises(ProtocolError, match="match the logical"):
        ContentRefV1(
            sha256=SHA_A,
            size_bytes=4096,
            media_type="application/octet-stream",
            replicas=(wrong,),
        )


def test_unknown_fields_are_rejected_at_nested_boundary() -> None:
    payload = _science().as_dict()
    payload["activation"]["process_count"] = 16
    with pytest.raises(ProtocolError, match="unknown fields"):
        ScientificIdentityV1.from_dict(payload)


@pytest.mark.parametrize(
    "field,value",
    [
        ("record_start", True),
        ("record_end", False),
        ("scientific_token_start", True),
    ],
)
def test_boolean_is_never_accepted_as_integer(field: str, value: bool) -> None:
    payload = HarvestRangeV1(
        scientific_identity_sha256=SHA_D,
        corpus_fingerprint_sha256=SHA_A,
        split="train",
        record_start=1,
        record_end=2,
        scientific_token_start=3,
        scientific_token_end=4,
    ).as_dict()
    payload[field] = value
    with pytest.raises(ProtocolError, match="booleans are forbidden"):
        HarvestRangeV1.from_dict(payload)


def test_nonfinite_numbers_and_duplicate_fields_are_rejected() -> None:
    with pytest.raises(ProtocolError, match="finite"):
        replace(_activation(), router_weight_sum=math.inf)
    with pytest.raises(ProtocolError, match="non-finite"):
        parse_json('{"schema":"test.v1","value":NaN}')
    with pytest.raises(ProtocolError, match="duplicate JSON field"):
        parse_json('{"schema":"test.v1","value":1,"value":2}')
    with pytest.raises(ProtocolError):
        canonical_json_bytes({"schema": "test.v1", "value": math.nan})


def test_overlong_json_integer_is_a_protocol_error() -> None:
    with pytest.raises(ProtocolError, match="invalid JSON"):
        parse_json('{"schema":"test.v1","value":' + "9" * 5_000 + "}")


def test_schema_mismatch_is_rejected() -> None:
    payload = _content().as_dict()
    payload["schema"] = "moe-sae.r1.content-ref.v2"
    with pytest.raises(ProtocolError, match="schema mismatch"):
        ContentRefV1.from_dict(payload)


def test_records_are_frozen_and_slotted() -> None:
    record = HarvestRangeV1(
        scientific_identity_sha256=SHA_D,
        corpus_fingerprint_sha256=SHA_A,
        split="train",
        record_start=0,
        record_end=1,
        scientific_token_start=0,
        scientific_token_end=10,
    )
    assert not hasattr(record, "__dict__")
    with pytest.raises(FrozenInstanceError):
        record.record_start = 2  # type: ignore[misc]


def test_lease_progress_and_success_receipt_round_trip() -> None:
    lease = AttemptLeaseV1(
        campaign_id="r1-study-001",
        work_id="sae-work-a1",
        attempt_id="attempt-0001",
        fence=3,
        executor_id="v6e64-uscentral1b-001",
        deployment_identity_sha256=SHA_D,
        acquired_at_unix_ms=1_000,
        expires_at_unix_ms=61_000,
    )
    progress = ProgressV1(
        work_id=lease.work_id,
        attempt_id=lease.attempt_id,
        fence=lease.fence,
        sequence=12,
        phase="train",
        completed_units=10_000_000,
        total_units=50_000_000,
        updated_at_unix_ms=20_000,
    )
    cost = CostLedgerV1(
        successful_chip_seconds=640.0,
        attempted_chip_seconds=704.0,
        compilation_chip_seconds=32.0,
        checkpoint_seconds=4.5,
        queue_seconds=120.0,
        egress_bytes=0,
        replayed_units=8192,
    )
    receipt = AttemptReceiptV1(
        work_id=lease.work_id,
        attempt_id=lease.attempt_id,
        fence=lease.fence,
        final_progress_sequence=progress.sequence,
        outcome="SUCCEEDED",
        outputs=(_content(),),
        cost=cost,
        termination_reason=None,
        finished_at_unix_ms=90_000,
    )

    assert AttemptLeaseV1.from_json(lease.canonical_json()) == lease
    assert ProgressV1.from_json(progress.canonical_json()) == progress
    assert AttemptReceiptV1.from_json(receipt.canonical_json()) == receipt
    assert json.loads(receipt.canonical_json())["cost"]["schema"].endswith(".v1")


def test_failed_receipt_requires_a_reason_and_success_requires_output() -> None:
    cost = CostLedgerV1(0, 1, 0, 0, 0, 0, 0)
    with pytest.raises(ProtocolError, match="termination_reason"):
        AttemptReceiptV1("work", "attempt", 1, 0, "FAILED", (), cost, None, 1)
    with pytest.raises(ProtocolError, match="at least one output"):
        AttemptReceiptV1("work", "attempt", 1, 0, "SUCCEEDED", (), cost, None, 1)
