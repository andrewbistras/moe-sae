from __future__ import annotations

import ml_dtypes
import numpy as np
import pytest

from moe_sae.r1.capture_data_plane_contract import build_capture_work_plan
from moe_sae.r1.capture_fragment_codec import (
    AddressableCaptureShardV1,
    pack_capture_fragment,
)
from moe_sae.r1.capture_fragment_validator import CaptureFragmentValidatorV1
from moe_sae.r1.capture_manifest import (
    CaptureManifestError,
    SealedCaptureManifestV1,
    seal_capture_manifest,
)
from moe_sae.r1.gcs_provisional_capture_sink import GCSProvisionalCaptureSinkV1
from tests.r1.test_gcs_immutable_byte_store import _store


@pytest.fixture(scope="module")
def two_fragment_cohort():
    work = build_capture_work_plan(
        work_id="two-fragment-manifest-test",
        input_range_sha256="7" * 64,
        start_token_ordinal=13,
        token_count=2_049,
    )
    activation = np.zeros(
        (2, 1_024, 7_168),
        dtype=np.dtype(ml_dtypes.bfloat16),
    )
    ids = np.broadcast_to(
        np.arange(8, dtype=np.int32),
        (2, 1_024, 8),
    ).copy()
    weights = np.full(
        (2, 1_024, 8),
        np.float32(2.5 / 8),
        dtype=np.float32,
    )
    sequence_id = np.broadcast_to(
        np.asarray([1, 2], dtype=np.int64)[:, None],
        (2, 1_024),
    ).copy()
    position = np.broadcast_to(
        np.arange(1_024, dtype=np.uint16),
        (2, 1_024),
    ).copy()
    shard = AddressableCaptureShardV1(
        activation,
        ids,
        weights,
        sequence_id,
        position,
    )
    attempt_id = "manifest-attempt-0001"
    store, client = _store(prefix=f"captures/attempts/{attempt_id}")
    sink = GCSProvisionalCaptureSinkV1(store, attempt_id=attempt_id)
    validator = CaptureFragmentValidatorV1(store)
    planned = []
    for plan in work.iter_fragments():
        packed = pack_capture_fragment(plan, shard)
        provisional = sink.create_provisional(packed)
        planned.append((plan, validator.validate(plan, provisional)))
    return work, tuple(planned), store, client


def test_exact_range_manifest_seals_only_complete_canonical_receipts(
    two_fragment_cohort,
) -> None:
    work, planned, store, _client = two_fragment_cohort

    manifest = seal_capture_manifest(work, planned)

    assert manifest.fragment_count == 2
    assert manifest.token_count == 2_049
    assert manifest.payload_bytes == work.payload_bytes
    assert manifest.store_id == store.store_id
    assert manifest.region == store.region
    assert manifest.exact_range_verified is True
    assert manifest.full_payload_readback_verified is True
    assert manifest.manifest_complete is True
    assert manifest.manifest_published is False
    assert manifest.production_sealable is False
    assert tuple(item.fragment_plan for item in manifest.entries) == tuple(
        work.iter_fragments()
    )
    assert manifest.digest == seal_capture_manifest(work, planned).digest


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "reversed"])
def test_manifest_rejects_gaps_duplicates_and_noncanonical_order(
    two_fragment_cohort,
    mutation: str,
) -> None:
    work, planned, _store, _client = two_fragment_cohort
    if mutation == "missing":
        malformed = planned[:1]
    elif mutation == "duplicate":
        malformed = (planned[0], planned[0])
    else:
        malformed = tuple(reversed(planned))

    with pytest.raises(CaptureManifestError, match="canonical exact-range"):
        seal_capture_manifest(work, malformed)


def test_manifest_rejects_receipt_from_another_plan_before_sealing(
    two_fragment_cohort,
) -> None:
    work, planned, _store, _client = two_fragment_cohort
    crossed = (
        (planned[0][0], planned[1][1]),
        (planned[1][0], planned[0][1]),
    )

    with pytest.raises(CaptureManifestError, match="fully read-back fragment"):
        seal_capture_manifest(work, crossed)


def test_sealed_manifest_cannot_be_constructed_directly(
    two_fragment_cohort,
) -> None:
    work, planned, _store, _client = two_fragment_cohort
    manifest = seal_capture_manifest(work, planned)

    with pytest.raises(CaptureManifestError, match="only through"):
        SealedCaptureManifestV1(
            work_plan=work,
            entries=manifest.entries,
        )
