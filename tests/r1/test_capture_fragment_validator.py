from __future__ import annotations

import pytest

from moe_sae.r1.capture_data_plane_contract import build_capture_work_plan
from moe_sae.r1.capture_fragment_validator import (
    CaptureFragmentValidationError,
    CaptureFragmentValidatorV1,
    ValidatedCaptureFragmentReceiptV1,
)
from moe_sae.r1.gcs_provisional_capture_sink import (
    GCSProvisionalCaptureSinkV1,
)
from tests.r1.test_gcs_immutable_byte_store import (
    _store,
    _tiny_packed_capture_fragment,
)


def _provisional_fixture():
    attempt_id = "validator-attempt-0001"
    store, client = _store(prefix=f"captures/attempts/{attempt_id}")
    packed = _tiny_packed_capture_fragment()
    provisional = GCSProvisionalCaptureSinkV1(
        store,
        attempt_id=attempt_id,
    ).create_provisional(packed)
    client.calls.clear()
    return store, client, packed, provisional


def test_validator_streams_exact_generation_and_keeps_manifest_gate_false() -> None:
    store, client, packed, provisional = _provisional_fixture()
    validator = CaptureFragmentValidatorV1(store)

    validated = validator.validate(packed.plan, provisional)

    validated.require_matches(packed.plan, provisional)
    assert validated.object_create_acknowledged is True
    assert validated.full_payload_readback_verified is True
    assert validated.manifest_visible is False
    assert validated.production_sealable is False
    assert validated.physical_receipt == provisional.physical_receipt
    assert validated.validator_id == validator.validator_id
    assert validator.production_identity_qualified is False
    assert [item[0] for item in client.calls] == [
        "blob_reload",
        "download",
        "blob_reload",
    ]
    assert client.downloaded_bytes == len(packed.payload)


def test_validator_rejects_wrong_plan_before_object_io() -> None:
    store, client, packed, provisional = _provisional_fixture()
    wrong_plan = next(
        build_capture_work_plan(
            work_id="other-validator-work",
            input_range_sha256="6" * 64,
            start_token_ordinal=0,
            token_count=1,
        ).iter_fragments()
    )

    with pytest.raises(CaptureFragmentValidationError, match="exact fragment plan"):
        CaptureFragmentValidatorV1(store).validate(wrong_plan, provisional)

    assert client.calls == []
    assert client.downloaded_bytes == 0
    assert wrong_plan.payload_bytes == packed.plan.payload_bytes


def test_validator_fails_closed_on_corrupt_or_replaced_generation() -> None:
    store, client, packed, provisional = _provisional_fixture()
    physical = provisional.physical_receipt
    key = ("r1-checkpoints", f"{store.prefix}/{physical.object_id}")
    client.objects[key].value = b"x" * len(packed.payload)

    with pytest.raises(CaptureFragmentValidationError, match="readback failed"):
        CaptureFragmentValidatorV1(store).validate(packed.plan, provisional)

    assert client.downloaded_bytes == len(packed.payload)

    store, client, packed, provisional = _provisional_fixture()
    client.replace_live("r1-checkpoints", f"{store.prefix}/{packed.object_id}")
    with pytest.raises(CaptureFragmentValidationError, match="readback failed"):
        CaptureFragmentValidatorV1(store).validate(packed.plan, provisional)


def test_validated_receipt_is_capability_built_and_validator_is_frozen() -> None:
    store, client, packed, provisional = _provisional_fixture()
    validator = CaptureFragmentValidatorV1(store)

    with pytest.raises(CaptureFragmentValidationError, match="only through"):
        ValidatedCaptureFragmentReceiptV1(
            plan=packed.plan,
            provisional=provisional,
            physical_receipt=provisional.physical_receipt,
            validator_id=validator.validator_id,
        )
    with pytest.raises(AttributeError, match="immutable"):
        validator._store = store  # type: ignore[misc]
    with pytest.raises(AttributeError, match="immutable"):
        del validator._validator_id  # type: ignore[misc]
    assert client.calls == []
