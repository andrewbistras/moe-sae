from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from moe_sae.harvest_control import (
    HarvestAbortRequested,
    HarvestCampaignControl,
    HarvestLeaseHeld,
    campaign_identity,
    clear_campaign_abort,
    harvest_lane_lease_held,
    load_campaign_control,
    request_campaign_abort,
)


def metadata() -> dict[str, object]:
    return {
        "config_sha256": "a" * 64,
        "checkpoint_sha256": "b" * 64,
        "dataset_fingerprint": "c" * 64,
        "split": "train",
    }


def control(tmp_path: Path, *, interval: float = 0.0) -> HarvestCampaignControl:
    return HarvestCampaignControl(
        tmp_path,
        metadata(),
        split="train",
        total_tokens=100,
        stop_after_tokens=20,
        process_count=2,
        process_index=0,
        heartbeat_interval_seconds=interval,
    )


def test_lifetime_lease_rejects_duplicate_and_releases_for_resume(
    tmp_path: Path,
) -> None:
    first = control(tmp_path)
    assert not harvest_lane_lease_held(tmp_path, 0)
    with first:
        assert harvest_lane_lease_held(tmp_path, 0)
        with pytest.raises(HarvestLeaseHeld, match="another live process"):
            with control(tmp_path):
                pass
        status = load_campaign_control(tmp_path, first.campaign_id)
        assert len(status["hosts"]) == 1
        assert status["hosts"][0]["state"] == "starting"
    assert not harvest_lane_lease_held(tmp_path, 0)

    resumed = control(tmp_path)
    with resumed:
        assert resumed.attempt_id != first.attempt_id
        resumed.complete({"durable_host_tokens": 20})
    status = load_campaign_control(tmp_path, resumed.campaign_id)
    assert status["hosts"][0]["state"] == "completed"
    assert status["hosts"][0]["progress"]["durable_host_tokens"] == 20


def test_distinct_process_lanes_hold_leases_concurrently(tmp_path: Path) -> None:
    first = control(tmp_path)
    second = HarvestCampaignControl(
        tmp_path,
        metadata(),
        split="train",
        total_tokens=100,
        stop_after_tokens=20,
        process_count=2,
        process_index=1,
        heartbeat_interval_seconds=0.0,
    )
    with first, second:
        loaded = load_campaign_control(tmp_path, first.campaign_id)
        assert {host["process_index"] for host in loaded["hosts"]} == {0, 1}


def test_canary_and_continuation_cannot_overlap_on_same_host_lane(
    tmp_path: Path,
) -> None:
    canary = control(tmp_path)
    continuation = HarvestCampaignControl(
        tmp_path,
        metadata(),
        split="train",
        total_tokens=100,
        stop_after_tokens=100,
        process_count=2,
        process_index=0,
        heartbeat_interval_seconds=0.0,
    )
    assert canary.campaign_id != continuation.campaign_id
    with canary:
        with pytest.raises(HarvestLeaseHeld, match="another live process"):
            with continuation:
                pass


def test_failed_initial_heartbeat_releases_lease(tmp_path: Path) -> None:
    broken = HarvestCampaignControl(
        tmp_path,
        metadata(),
        split="train",
        total_tokens=100,
        stop_after_tokens=20,
        process_count=2,
        process_index=0,
        clock=lambda: 0.0,
    )
    with pytest.raises(RuntimeError, match="invalid Unix time"):
        with broken:
            pass
    with control(tmp_path):
        pass


def test_status_is_self_hashed_and_tampering_fails_closed(tmp_path: Path) -> None:
    instance = control(tmp_path)
    with instance:
        instance.heartbeat("running", progress={"batches_scanned": 3}, force=True)
    value = json.loads(instance.status_path.read_text(encoding="utf-8"))
    claimed = value.pop("status_sha256")
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    assert claimed == hashlib.sha256(canonical).hexdigest()

    value["state"] = "completed"
    value["status_sha256"] = claimed
    instance.status_path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        load_campaign_control(tmp_path, instance.campaign_id)


def test_abort_marker_is_durable_idempotent_and_explicitly_cleared(
    tmp_path: Path,
) -> None:
    instance = control(tmp_path)
    path = request_campaign_abort(
        tmp_path,
        instance.campaign_id,
        reason="operator requested a safe stop",
        requested_by="test",
    )
    assert (
        request_campaign_abort(
            tmp_path,
            instance.campaign_id,
            reason="a replay does not replace the first reason",
        )
        == path
    )
    loaded = load_campaign_control(tmp_path, instance.campaign_id)
    assert loaded["abort"]["reason"] == "operator requested a safe stop"

    with pytest.raises(HarvestAbortRequested, match="safe boundary"):
        with instance:
            instance.heartbeat(
                "running", progress={"durable_host_tokens": 7}, force=True
            )
            instance.raise_if_aborted(globally_requested=True)
    final_status = load_campaign_control(tmp_path, instance.campaign_id)["hosts"][0]
    assert final_status["state"] == "aborted"
    assert final_status["progress"]["durable_host_tokens"] == 7
    assert clear_campaign_abort(tmp_path, instance.campaign_id)
    assert not clear_campaign_abort(tmp_path, instance.campaign_id)
    assert load_campaign_control(tmp_path, instance.campaign_id)["abort"] is None


def test_campaign_identity_binds_stop_and_topology() -> None:
    base = campaign_identity(
        metadata(),
        split="train",
        total_tokens=100,
        stop_after_tokens=20,
        process_count=2,
    )
    changed_stop = campaign_identity(
        metadata(),
        split="train",
        total_tokens=100,
        stop_after_tokens=100,
        process_count=2,
    )
    changed_topology = campaign_identity(
        metadata(),
        split="train",
        total_tokens=100,
        stop_after_tokens=20,
        process_count=4,
    )
    assert len(base) == 64
    assert len({base, changed_stop, changed_topology}) == 3
