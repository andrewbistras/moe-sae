from __future__ import annotations

import json
from pathlib import Path

import pytest

from moe_sae.checkpointing import (
    checkpoint_payload_uri,
    final_pointer_from_latest,
    load_milestone_pointer,
    load_pointer,
    make_pointer,
    milestone_pointer_path,
    normalize_payload_root,
    publish_milestone_pointer,
    publish_pointer,
    validate_pointer,
)


def _latest(tmp_path: Path, identity: dict[str, object]) -> dict[str, object]:
    payload = checkpoint_payload_uri(
        tmp_path / "payloads",
        identity,
        step=12,
        tokens_seen=196_608,
        attempt_id="a" * 32,
    )
    return make_pointer(
        kind="latest",
        payload_uri=payload,
        step=12,
        tokens_seen=196_608,
        attempt_id="a" * 32,
        run_identity=identity,
    )


def test_local_pointer_roundtrip_is_self_hashed_and_identity_bound(
    tmp_path: Path,
) -> None:
    identity = {"condition": "A", "seed": 1, "run_instance_id": "b" * 32}
    latest = _latest(tmp_path, identity)
    published = publish_pointer(tmp_path / "control", latest)

    assert published == tmp_path / "control" / "latest.json"
    restored = load_pointer(
        tmp_path / "control",
        kind="latest",
        payload_root=tmp_path / "payloads",
        run_identity=identity,
    )
    assert restored == latest
    assert json.loads(published.read_text(encoding="utf-8")) == latest


def test_pointer_tampering_and_validly_hashed_namespace_escape_fail(
    tmp_path: Path,
) -> None:
    identity = {"condition": "F", "seed": 2}
    latest = _latest(tmp_path, identity)
    tampered = dict(latest)
    tampered["tokens_seen"] = 1
    with pytest.raises(RuntimeError, match="self-hash mismatch"):
        validate_pointer(
            tampered,
            expected_kind="latest",
            payload_root=tmp_path / "payloads",
            run_identity=identity,
        )

    escaped = make_pointer(
        kind="latest",
        payload_uri=str((tmp_path / "other" / Path(str(latest["payload_uri"])).name)),
        step=int(latest["step"]),
        tokens_seen=int(latest["tokens_seen"]),
        attempt_id=str(latest["attempt_id"]),
        run_identity=identity,
    )
    with pytest.raises(RuntimeError, match="escapes"):
        validate_pointer(
            escaped,
            expected_kind="latest",
            payload_root=tmp_path / "payloads",
            run_identity=identity,
        )


def test_final_pointer_reuses_latest_payload_but_has_its_own_hash(
    tmp_path: Path,
) -> None:
    identity = {"condition": "B", "seed": 3}
    latest = _latest(tmp_path, identity)
    final = final_pointer_from_latest(latest)

    assert final["kind"] == "final"
    assert final["payload_uri"] == latest["payload_uri"]
    assert final["pointer_sha256"] != latest["pointer_sha256"]
    validate_pointer(
        final,
        expected_kind="final",
        payload_root=tmp_path / "payloads",
        run_identity=identity,
    )


def test_milestone_pointer_is_create_only_and_identity_bound(
    tmp_path: Path,
) -> None:
    identity = {"condition": "A", "seed": 0}
    latest = _latest(tmp_path, identity)

    path = publish_milestone_pointer(tmp_path / "control", latest)
    assert path == milestone_pointer_path(
        tmp_path / "control",
        int(latest["tokens_seen"]),
    )
    assert (
        load_milestone_pointer(
            tmp_path / "control",
            checkpoint_target_tokens=int(latest["tokens_seen"]),
            payload_root=tmp_path / "payloads",
            run_identity=identity,
            expected_tokens_seen=int(latest["tokens_seen"]),
        )
        == latest
    )
    assert publish_milestone_pointer(tmp_path / "control", latest) == path

    different = _latest(tmp_path, {**identity, "seed": 1})
    with pytest.raises(RuntimeError, match="different content"):
        publish_milestone_pointer(tmp_path / "control", different)


def test_gcs_payload_root_is_canonical_and_rejects_dot_segments() -> None:
    assert normalize_payload_root("gs://bucket/a//b/") == "gs://bucket/a/b"
    with pytest.raises(ValueError, match="dot segments"):
        normalize_payload_root("gs://bucket/a/../b")
    with pytest.raises(ValueError, match="local or gs"):
        normalize_payload_root("s3://bucket/root")
