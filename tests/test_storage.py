from __future__ import annotations

import copy
import multiprocessing as mp
from pathlib import Path
import shutil
from typing import Any

import pytest

import moe_sae.storage as storage_module
from moe_sae.object_store import LocalObjectStore
from moe_sae.storage import (
    CoordinationHooks,
    StoragePlanesQualificationConfig,
    StorageQualificationError,
    build_storage_attestation,
    qualify_orbax_checkpoint_storage,
    qualify_storage_planes,
    validate_storage_attestation,
)


def _distributed_qualifier(
    process_index: int,
    process_count: int,
    control_directory: str,
    local_parent: str,
    object_root: str,
    barrier: Any,
    gathered_values: Any,
    results: Any,
    shared_local: bool = False,
) -> None:
    gather_round = 0

    def allgather(value: Any) -> tuple[Any, ...]:
        nonlocal gather_round
        current = gather_round
        gathered_values[(current, process_index)] = value
        barrier.wait()
        values = tuple(
            gathered_values[(current, index)] for index in range(process_count)
        )
        barrier.wait()
        if process_index == 0:
            for index in range(process_count):
                del gathered_values[(current, index)]
        barrier.wait()
        gather_round += 1
        return values

    hooks = CoordinationHooks(
        process_index=lambda: process_index,
        process_count=lambda: process_count,
        barrier=lambda _label: barrier.wait(),
        allgather=allgather,
    )
    config = StoragePlanesQualificationConfig(
        minimum_control_free_bytes=1,
        minimum_local_buffer_free_bytes=1,
        gcs_probe_bytes_per_process=1024 * 1024,
        gcs_probe_block_bytes=64 * 1024,
        minimum_aggregate_upload_bytes_per_second=0,
        minimum_aggregate_download_bytes_per_second=0,
        cleanup=False,
    )
    try:
        result = qualify_storage_planes(
            control_directory,
            (
                Path(local_parent)
                if shared_local
                else Path(local_parent) / f"host-{process_index}"
            ),
            "gs://qualification-test/prefix",
            LocalObjectStore(object_root),
            config=config,
            coordination=hooks,
        )
    except BaseException as exc:  # pragma: no cover - surfaced in parent
        results.put((process_index, "error", f"{type(exc).__name__}: {exc}"))
    else:
        results.put((process_index, "result", result))


def test_two_process_split_storage_qualification(tmp_path: Path) -> None:
    """Exercise real POSIX collectives plus cyclic immutable-object reads."""

    context = mp.get_context("spawn")
    process_count = 2
    barrier = context.Barrier(process_count, timeout=30)
    manager = context.Manager()
    gathered_values = manager.dict()
    results = context.Queue()
    control = tmp_path / "control"
    local = tmp_path / "local"
    objects = tmp_path / "objects"
    processes = [
        context.Process(
            target=_distributed_qualifier,
            args=(
                index,
                process_count,
                str(control),
                str(local),
                str(objects),
                barrier,
                gathered_values,
                results,
            ),
        )
        for index in range(process_count)
    ]

    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=45)
        assert [process.exitcode for process in processes] == [0, 0]
        received = [results.get(timeout=5) for _ in range(process_count)]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        manager.shutdown()

    assert not [item for item in received if item[1] == "error"]
    qualified = sorted(
        (item for item in received if item[1] == "result"),
        key=lambda item: item[0],
    )
    assert len(qualified) == process_count
    first = qualified[0][2]
    assert first == qualified[1][2]
    assert first.passed
    assert first.require_pass() is first
    assert first.control.flock_correct
    assert first.control.atomic_rename_visible
    assert first.control.disjoint_files_visible
    assert first.local_buffer.directories == (
        str((local / "host-0").resolve()),
        str((local / "host-1").resolve()),
    )
    assert first.local_buffer.host_isolation_proven
    assert (
        first.local_buffer.expected_probe_sha256_by_process
        == first.local_buffer.observed_probe_sha256_by_process
    )
    assert first.gcs.upload is not None
    assert first.gcs.download is not None
    assert first.gcs.upload.total_bytes == 2 * 1024 * 1024
    assert first.gcs.download.total_bytes == 2 * 1024 * 1024
    assert len(first.gcs.objects) == 2
    assert all(entry["generation"] > 0 for entry in first.gcs.objects)
    assert all(len(entry["crc32c"]) == 8 for entry in first.gcs.objects)
    assert list(Path(first.control.work_directory).glob("throughput-*")) == []


def test_two_process_qualification_rejects_shared_local_buffer(tmp_path: Path) -> None:
    context = mp.get_context("spawn")
    process_count = 2
    barrier = context.Barrier(process_count, timeout=30)
    manager = context.Manager()
    gathered_values = manager.dict()
    results = context.Queue()
    processes = [
        context.Process(
            target=_distributed_qualifier,
            args=(
                index,
                process_count,
                str(tmp_path / "control"),
                str(tmp_path / "shared-local"),
                str(tmp_path / "objects"),
                barrier,
                gathered_values,
                results,
                True,
            ),
        )
        for index in range(process_count)
    ]

    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=45)
        assert [process.exitcode for process in processes] == [0, 0]
        received = [results.get(timeout=5) for _ in range(process_count)]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        manager.shutdown()

    assert not [item for item in received if item[1] == "error"]
    qualified = [item[2] for item in received if item[1] == "result"]
    assert len(qualified) == process_count
    assert qualified[0] == qualified[1]
    assert not qualified[0].passed
    assert not qualified[0].local_buffer.host_isolation_proven
    assert qualified[0].gcs.upload is None
    assert qualified[0].failures == ("local-buffer host isolation failed",)


def test_local_buffer_capacity_failure_skips_object_probe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    disk_usage = shutil._ntuple_diskusage(total=1000, used=900, free=100)
    monkeypatch.setattr(storage_module.shutil, "disk_usage", lambda _path: disk_usage)
    config = StoragePlanesQualificationConfig(
        minimum_control_free_bytes=1,
        minimum_local_buffer_free_bytes=101,
        gcs_probe_bytes_per_process=1,
        gcs_probe_block_bytes=1,
        minimum_aggregate_upload_bytes_per_second=0,
        minimum_aggregate_download_bytes_per_second=0,
    )

    result = qualify_storage_planes(
        tmp_path / "control",
        tmp_path / "local",
        "gs://qualification-test/prefix",
        LocalObjectStore(tmp_path / "objects"),
        config=config,
    )

    assert not result.passed
    assert result.control.free_bytes_by_process == (100,)
    assert result.local_buffer.free_bytes_by_process == (100,)
    assert result.gcs.upload is None
    assert result.gcs.download is None
    assert result.failures == (
        "local-buffer free space below threshold: minimum=100 required=101",
    )
    with pytest.raises(StorageQualificationError, match="local-buffer free space"):
        result.require_pass()


def test_orbax_checkpoint_storage_round_trip(tmp_path: Path) -> None:
    pytest.importorskip("orbax.checkpoint")

    result = qualify_orbax_checkpoint_storage(
        tmp_path / "checkpoint-payloads",
        "local-probe",
    )

    assert result.require_pass() is result
    assert result.finalized
    assert result.metadata_round_trip
    assert result.expected_local_sha256_by_process == (
        result.restored_local_sha256_by_process
    )
    assert result.max_save_seconds > 0
    assert result.max_restore_seconds > 0
    assert Path(result.payload_uri).is_dir()


def test_gcs_throughput_thresholds_are_hard_gates(tmp_path: Path) -> None:
    config = StoragePlanesQualificationConfig(
        minimum_control_free_bytes=1,
        minimum_local_buffer_free_bytes=1,
        gcs_probe_bytes_per_process=64 * 1024,
        gcs_probe_block_bytes=4096,
        minimum_aggregate_upload_bytes_per_second=1e30,
        minimum_aggregate_download_bytes_per_second=1e30,
    )
    result = qualify_storage_planes(
        tmp_path / "control",
        tmp_path / "local",
        "gs://qualification-test/prefix",
        LocalObjectStore(tmp_path / "objects"),
        config=config,
    )

    assert result.gcs.upload is not None
    assert result.gcs.download is not None
    assert any("GCS upload throughput below threshold" in item for item in result.failures)
    assert any("GCS download throughput below threshold" in item for item in result.failures)


def test_v3_attestation_is_self_hashed_and_bound(tmp_path: Path) -> None:
    config = StoragePlanesQualificationConfig(
        minimum_control_free_bytes=1,
        minimum_local_buffer_free_bytes=1,
        gcs_probe_bytes_per_process=4096,
        gcs_probe_block_bytes=1024,
        minimum_aggregate_upload_bytes_per_second=0,
        minimum_aggregate_download_bytes_per_second=0,
    )
    control = tmp_path / "control"
    local = tmp_path / "local"
    result = qualify_storage_planes(
        control,
        local,
        "gs://qualification-test/prefix",
        LocalObjectStore(tmp_path / "objects"),
        config=config,
    )
    payload = build_storage_attestation(
        result,
        config,
        config_sha256="a" * 64,
        environment_sha256="b" * 64,
        global_device_count=16,
        metadata={"cluster": "v4-32"},
    )

    digest = validate_storage_attestation(
        payload,
        expected_config_sha256="a" * 64,
        expected_environment_sha256="b" * 64,
        expected_control_directory=control,
        expected_local_directories=[local],
        expected_object_prefix="gs://qualification-test/prefix/",
    )
    assert digest == payload["attestation_sha256"]
    tampered = copy.deepcopy(payload)
    tampered["contract"]["gcs_probe_bytes_per_process"] += 1
    with pytest.raises(StorageQualificationError, match="self-hash mismatch"):
        validate_storage_attestation(tampered)

    isolation_tampered = copy.deepcopy(payload)
    isolation_tampered["qualification"]["local_buffer"][
        "host_isolation_proven"
    ] = False
    isolation_tampered["attestation_sha256"] = storage_module.storage_attestation_digest(
        isolation_tampered
    )
    with pytest.raises(StorageQualificationError, match="host-local buffer isolation"):
        validate_storage_attestation(isolation_tampered)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("minimum_control_free_bytes", -1),
        ("minimum_local_buffer_free_bytes", -1),
        ("gcs_probe_bytes_per_process", 0),
        ("gcs_probe_block_bytes", 0),
        ("minimum_aggregate_upload_bytes_per_second", float("nan")),
        ("minimum_aggregate_download_bytes_per_second", -1.0),
    ],
)
def test_storage_qualification_config_rejects_invalid_values(
    field: str, value: int | float
) -> None:
    with pytest.raises(ValueError):
        StoragePlanesQualificationConfig(**{field: value})
