from __future__ import annotations

import io
import json
from pathlib import Path
import tarfile

import pytest

from moe_sae.regional_control_stage import (
    RegionalControlStageError,
    apply_control_stage_bundle,
    build_control_stage,
    write_control_stage_bundle,
)


def test_bundle_applies_exact_named_files_and_replays(tmp_path: Path) -> None:
    source = tmp_path / "central"
    source.mkdir()
    (source / "captures" / "train").mkdir(parents=True)
    (source / "captures" / "train" / "manifest.json").write_text("target")
    (source / "contracts").mkdir()
    (source / "contracts" / "sae.yaml").write_text("config")
    bundle = tmp_path / "stage.tar"

    plan = write_control_stage_bundle(
        source,
        (
            "captures/train/manifest.json",
            "contracts/sae.yaml",
        ),
        bundle,
    )
    destination = tmp_path / "regional"
    destination.mkdir()

    first = apply_control_stage_bundle(bundle, destination)
    second = apply_control_stage_bundle(bundle, destination)

    assert first == second == plan
    assert (destination / "captures/train/manifest.json").read_text() == "target"
    assert (destination / "contracts/sae.yaml").read_text() == "config"
    assert not (destination / "anything-else").exists()


def test_apply_refuses_to_replace_different_existing_file(tmp_path: Path) -> None:
    source = tmp_path / "central"
    source.mkdir()
    (source / "manifest.json").write_text("sealed")
    bundle = tmp_path / "stage.tar"
    write_control_stage_bundle(source, ("manifest.json",), bundle)
    destination = tmp_path / "regional"
    destination.mkdir()
    (destination / "manifest.json").write_text("different")

    with pytest.raises(RegionalControlStageError, match="differs"):
        apply_control_stage_bundle(bundle, destination)


def test_build_refuses_symlink_and_traversal(tmp_path: Path) -> None:
    source = tmp_path / "central"
    source.mkdir()
    external = tmp_path / "secret"
    external.write_text("no")
    (source / "link").symlink_to(external)

    with pytest.raises(RegionalControlStageError, match="escapes|regular"):
        build_control_stage(source, ("link",))
    with pytest.raises(RegionalControlStageError, match="normalized relative"):
        build_control_stage(source, ("../secret",))
    source_link = tmp_path / "central-link"
    source_link.symlink_to(source, target_is_directory=True)
    with pytest.raises(RegionalControlStageError, match="root must not be a symlink"):
        build_control_stage(source_link, ("link",))


def test_bundle_rejects_unplanned_tar_member(tmp_path: Path) -> None:
    source = tmp_path / "central"
    source.mkdir()
    (source / "manifest.json").write_text("sealed")
    bundle = tmp_path / "stage.tar"
    write_control_stage_bundle(source, ("manifest.json",), bundle)
    tampered = tmp_path / "tampered.tar"
    with tarfile.open(bundle, "r") as original, tarfile.open(tampered, "w") as output:
        for member in original.getmembers():
            handle = original.extractfile(member)
            assert handle is not None
            output.addfile(member, io.BytesIO(handle.read()))
        payload = b"surprise"
        member = tarfile.TarInfo("extra")
        member.size = len(payload)
        output.addfile(member, io.BytesIO(payload))
    destination = tmp_path / "regional"
    destination.mkdir()

    with pytest.raises(RegionalControlStageError, match="inventory"):
        apply_control_stage_bundle(tampered, destination)


def test_stage_plan_is_deterministic_and_deduplicates_payloads(
    tmp_path: Path,
) -> None:
    source = tmp_path / "central"
    source.mkdir()
    (source / "a").write_text("same")
    (source / "b").write_text("same")

    first, payloads = build_control_stage(source, ("b", "a"))
    second, _ = build_control_stage(source, ("a", "b"))

    assert first == second
    assert len(payloads) == 1
    assert first["unique_payload_bytes"] == 4
    assert json.loads(json.dumps(first)) == first
