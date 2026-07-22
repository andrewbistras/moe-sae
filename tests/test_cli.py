import argparse
import dataclasses
import json
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import moe_sae.cli as cli_module

from moe_sae.cli import (
    _balanced_padding_ids,
    _checkpoint_identity_expected,
    _load_harvester_report,
    _precompute_a_capacity,
    _sae_config,
    build_parser,
    main,
)
from moe_sae.config import Gemma4Config, MODEL_REVISION, load_config
from moe_sae.data import (
    ALLENAI_C4_REPO_ID,
    ALLENAI_C4_REVISION,
    ALLENAI_C4_SOURCE_FILES,
    GEMMA4_TOKENIZER_INVENTORY_SHA256,
    GEMMA4_TOKENIZER_SOURCE_FILES,
    PACKED_DATASET_SCHEMA,
    PACKED_RECORD_FORMAT,
)
from moe_sae.harvest import CampaignSummary
from moe_sae.model import (
    HuggingFaceSnapshot,
    expected_hf_repository_inventory_sha256,
    expected_hf_snapshot_inventory_sha256,
)
from moe_sae.runtime import (
    HARVEST_MESHES,
    V4_32_BF16_PEAK_FLOPS,
    BenchmarkObservation,
)
from moe_sae.sae_operator import SAEControl, SAEOperationSpec
from moe_sae.sae_paths import SAEOperationPaths
from moe_sae.storage import (
    MINIMUM_ACTIVATION_GCS_DOWNLOAD_BYTES_PER_SECOND,
    STORAGE_ATTESTATION_SCHEMA,
    StoragePlanesQualificationConfig,
    storage_attestation_digest,
)


def test_coordinator_json_wait_retries_stale_malformed_and_wrong_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = {
        "schema_version": "test-result-v1",
        "sweep_id": "correct-sweep",
        "attempt_id": "correct-attempt",
    }
    observations: list[str | BaseException] = [
        FileNotFoundError("not visible yet"),
        "{malformed",
        json.dumps({"schema_version": "stale-result-v0"}),
        json.dumps(
            {
                "schema_version": "test-result-v1",
                "sweep_id": "correct-sweep",
                "attempt_id": "wrong-attempt",
            }
        ),
        json.dumps(payload),
    ]

    def read_text(_path: Path, **_kwargs) -> str:
        observation = observations.pop(0)
        if isinstance(observation, BaseException):
            raise observation
        return observation

    sleeps: list[float] = []
    monkeypatch.setattr(Path, "read_text", read_text)
    monkeypatch.setattr(cli_module.time, "sleep", sleeps.append)
    monkeypatch.setattr(
        cli_module, "_COORDINATOR_JSON_VISIBILITY_TIMEOUT_SECONDS", 1.0
    )

    cli_module._wait_for_coordinator_json(tmp_path / "result.json", payload)

    assert not observations
    assert len(sleeps) == 4


@pytest.mark.parametrize(
    ("observed", "last_observation"),
    [
        (FileNotFoundError("missing"), "file is absent"),
        ("{malformed", "malformed JSON"),
        (
            json.dumps({"schema_version": "stale-result-v0"}),
            "does not match expected identity",
        ),
        (
            json.dumps(
                {
                    "schema_version": "test-result-v1",
                    "attempt_id": "wrong-attempt",
                }
            ),
            "does not match expected identity",
        ),
    ],
)
def test_coordinator_json_wait_fails_closed_at_timeout(
    observed: str | BaseException,
    last_observation: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = {
        "schema_version": "test-result-v1",
        "attempt_id": "correct-attempt",
    }

    def read_text(_path: Path, **_kwargs) -> str:
        if isinstance(observed, BaseException):
            raise observed
        return observed

    monkeypatch.setattr(Path, "read_text", read_text)
    monkeypatch.setattr(
        cli_module, "_COORDINATOR_JSON_VISIBILITY_TIMEOUT_SECONDS", 0.0
    )

    with pytest.raises(RuntimeError, match=last_observation):
        cli_module._wait_for_coordinator_json(tmp_path / "result.json", payload)


def test_coordinator_write_json_validates_after_global_barrier(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events: list[str] = []
    path = tmp_path / "result.json"
    payload = {"schema_version": "test-result-v1", "attempt_id": "a"}

    monkeypatch.setattr(cli_module, "_process_context", lambda: (0, 4))
    monkeypatch.setattr(
        cli_module, "_global_barrier", lambda label: events.append(label)
    )

    assert cli_module._coordinator_write_json(
        path, payload, barrier="publication-visible"
    )
    assert events == ["publication-visible"]
    assert json.loads(path.read_text(encoding="utf-8")) == payload


def test_noncoordinator_validates_publication_without_writing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "result.json"
    payload = {"schema_version": "test-result-v1", "attempt_id": "a"}
    path.write_text(json.dumps(payload), encoding="utf-8")
    barriers: list[str] = []

    monkeypatch.setattr(cli_module, "_process_context", lambda: (3, 4))
    monkeypatch.setattr(
        cli_module,
        "atomic_write_json",
        lambda *_args, **_kwargs: pytest.fail("noncoordinator attempted a write"),
    )
    monkeypatch.setattr(cli_module, "_global_barrier", barriers.append)

    assert not cli_module._coordinator_write_json(
        path, payload, barrier="publication-visible"
    )
    assert barriers == ["publication-visible"]


def test_coordinator_publication_failure_is_decided_before_barrier(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    decisions: list[tuple[str | None, str]] = []

    def decide(error: str | None, *, purpose: str) -> None:
        decisions.append((error, purpose))
        if error:
            raise RuntimeError("collective publication failure")

    monkeypatch.setattr(cli_module, "_process_context", lambda: (0, 4))
    monkeypatch.setattr(
        cli_module,
        "atomic_write_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("NFS down")),
    )
    monkeypatch.setattr(cli_module, "_collective_raise_if_any", decide)
    monkeypatch.setattr(
        cli_module,
        "_global_barrier",
        lambda *_args: pytest.fail("barrier entered after coordinator failure"),
    )

    with pytest.raises(RuntimeError, match="collective publication failure"):
        cli_module._coordinator_write_json(
            tmp_path / "result.json", {"ok": True}, barrier="must-not-run"
        )
    assert decisions and "NFS down" in str(decisions[0][0])


def test_immutable_publication_never_streams_into_final_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "identity.json"
    payload = {"schema_version": "identity-v1", "digest": "a" * 64}
    monkeypatch.setattr(
        cli_module.os,
        "link",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("link failed")),
    )

    with pytest.raises(RuntimeError, match="link failed"):
        cli_module._coordinator_publish_immutable_json(
            path, payload, barrier="must-not-run"
        )

    assert not path.exists()
    evidence = list(tmp_path.glob(".identity.json.*.unpublished"))
    assert len(evidence) == 1
    assert json.loads(evidence[0].read_text(encoding="utf-8")) == payload


def test_immutable_publication_accepts_exact_retry_and_rejects_conflict(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "evaluation.json"
    payload = {"schema_version": "evaluation-v1", "ev": 0.5}
    barriers: list[str] = []
    monkeypatch.setattr(cli_module, "_process_context", lambda: (0, 1))
    monkeypatch.setattr(cli_module, "_global_barrier", barriers.append)

    assert cli_module._coordinator_publish_immutable_json(
        path, payload, barrier="first-publication"
    )
    original = path.read_bytes()
    assert cli_module._coordinator_publish_immutable_json(
        path, payload, barrier="idempotent-retry"
    )
    assert path.read_bytes() == original
    assert barriers == ["first-publication", "idempotent-retry"]

    with pytest.raises(RuntimeError, match="conflicts with retry"):
        cli_module._coordinator_publish_immutable_json(
            path,
            {**payload, "ev": 0.6},
            barrier="conflicting-retry",
        )
    assert path.read_bytes() == original
    assert barriers == ["first-publication", "idempotent-retry"]


def test_attestation_timestamp_is_coordinator_broadcast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[bytes | None] = []
    monkeypatch.setattr(cli_module, "_process_context", lambda: (3, 4))

    def broadcast(source: bytes | None, *, purpose: str) -> bytes:
        observed.append(source)
        assert purpose == "attestation timestamp"
        return b"2026-07-22T03:04:05+00:00"

    monkeypatch.setattr(cli_module, "_broadcast_coordinator_bytes", broadcast)
    assert (
        cli_module._coordinator_created_at_utc()
        == "2026-07-22T03:04:05+00:00"
    )
    assert observed == [None]


def test_sae_storage_attestation_path_is_consumer_environment_versioned(
    tmp_path: Path,
) -> None:
    digest = "7" * 64
    assert cli_module._sae_storage_attestation_path(tmp_path, digest) == (
        tmp_path / "preflight" / "sae-storage" / f"{digest}.json"
    )
    with pytest.raises(RuntimeError, match="environment identity"):
        cli_module._sae_storage_attestation_path(tmp_path, "not-a-digest")


def test_sae_resident_cache_runtime_requires_exact_linger_yes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

    def run(command, **kwargs):
        calls.append((tuple(command), kwargs))
        return SimpleNamespace(returncode=0, stdout="yes\n", stderr="")

    monkeypatch.setattr(cli_module.os, "getuid", lambda: 1234)
    monkeypatch.setattr(
        cli_module.pwd,
        "getpwuid",
        lambda uid: SimpleNamespace(pw_name="sae-runner") if uid == 1234 else None,
    )
    monkeypatch.setattr(cli_module.subprocess, "run", run)

    cache_root = Path("/dev/shm/moe-sae-sae-cache")
    assert cli_module._require_lingering_sae_cache_runtime(cache_root) == cache_root
    assert calls == [
        (
            (
                "loginctl",
                "show-user",
                "sae-runner",
                "-p",
                "Linger",
                "--value",
            ),
            {
                "check": False,
                "capture_output": True,
                "text": True,
                "timeout": cli_module._LOGINCTL_LINGER_TIMEOUT_SECONDS,
            },
        )
    ]


def test_sae_resident_cache_runtime_rejects_non_shm_path_before_loginctl(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        cli_module.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("loginctl ran for an invalid path"),
    )

    with pytest.raises(RuntimeError, match="strictly below the direct /dev/shm"):
        cli_module._require_lingering_sae_cache_runtime(tmp_path / "cache")


def test_sae_resident_cache_runtime_rejects_disabled_lingering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli_module.os, "getuid", lambda: 1234)
    monkeypatch.setattr(
        cli_module.pwd,
        "getpwuid",
        lambda _uid: SimpleNamespace(pw_name="sae-runner"),
    )
    monkeypatch.setattr(
        cli_module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout="no\n", stderr=""
        ),
    )

    with pytest.raises(
        RuntimeError,
        match=r"output exactly 'yes\\n'.*enable-linger sae-runner",
    ):
        cli_module._require_lingering_sae_cache_runtime(
            "/dev/shm/moe-sae-sae-cache"
        )


def test_sae_resident_cache_runtime_rejects_malformed_yes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli_module.os, "getuid", lambda: 1234)
    monkeypatch.setattr(
        cli_module.pwd,
        "getpwuid",
        lambda _uid: SimpleNamespace(pw_name="sae-runner"),
    )
    monkeypatch.setattr(
        cli_module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout="yes \n", stderr=""
        ),
    )

    with pytest.raises(RuntimeError, match=r"output exactly 'yes\\n'"):
        cli_module._require_lingering_sae_cache_runtime(
            "/dev/shm/moe-sae-sae-cache"
        )


def test_sae_storage_linger_gate_precedes_attestation_reuse(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events: list[str] = []
    config = Gemma4Config()

    monkeypatch.setattr(cli_module, "_config", lambda _args: config)
    monkeypatch.setattr(cli_module, "_artifact_root", lambda *_args: tmp_path)
    monkeypatch.setattr(
        cli_module, "_require_production_artifact_root", lambda _root: None
    )
    monkeypatch.setattr(
        cli_module,
        "require_v4_32_runtime",
        lambda: events.append("topology"),
    )
    monkeypatch.setattr(
        cli_module,
        "_require_lingering_sae_cache_runtime",
        lambda path: events.append("linger") or Path(path),
    )
    monkeypatch.setattr(
        cli_module,
        "_collective_raise_if_any",
        lambda error, *, purpose: events.append(
            f"collective:{purpose}:{error or 'pass'}"
        ),
    )
    monkeypatch.setattr(
        cli_module,
        "_coordinated_environment",
        lambda: events.append("environment") or {"sha256": "e" * 64},
    )
    monkeypatch.setattr(
        cli_module,
        "_coordinator_read_json_if_exists",
        lambda *_args, **_kwargs: events.append("attestation") or {},
    )
    monkeypatch.setattr(
        cli_module,
        "_load_sae_storage_bundle",
        lambda *_args: events.append("reuse") or {"sae": {}},
    )
    monkeypatch.setattr(cli_module, "_process_context", lambda: (1, 4))

    args = SimpleNamespace(
        config=str(cli_module.DEFAULT_CONFIG),
        artifact_root=None,
        cache_root="/dev/shm/moe-sae-sae-cache",
    )
    assert cli_module._command_qualify_sae_storage(args) == 0
    assert events == [
        "topology",
        "linger",
        "collective:SAE resident-cache runtime:pass",
        "environment",
        "attestation",
        "reuse",
        "collective:reuse immutable SAE storage attestation:pass",
    ]


def test_coordinator_json_wait_allows_only_declared_coordinator_owned_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "storage.json"
    expected = {
        "schema_version": "storage-v1",
        "run_id": "same-run",
        "created_at_utc": "local-controller-time",
        "attestation_sha256": "a" * 64,
    }
    path.write_text(
        json.dumps(
            {
                **expected,
                "created_at_utc": "coordinator-time",
                "attestation_sha256": "b" * 64,
            }
        ),
        encoding="utf-8",
    )

    cli_module._wait_for_coordinator_json(
        path,
        expected,
        coordinator_owned_fields=("created_at_utc", "attestation_sha256"),
    )

    changed_identity = dict(expected, run_id="wrong-run")
    monkeypatch.setattr(
        cli_module, "_COORDINATOR_JSON_VISIBILITY_TIMEOUT_SECONDS", 0.0
    )
    with pytest.raises(RuntimeError, match="does not match expected identity"):
        cli_module._wait_for_coordinator_json(
            path,
            changed_identity,
            coordinator_owned_fields=(
                "created_at_utc",
                "attestation_sha256",
            ),
        )


def campaign(split: str, tokens: int, chip_seconds: float) -> dict:
    wall = chip_seconds / 16.0
    return CampaignSummary(
        schema_version="gemma4-harvest-campaign-v2",
        config_sha256=Gemma4Config().config_hash,
        checkpoint_sha256="a" * 64,
        conversion_attestation_sha256="9" * 64,
        dataset_fingerprint="5" * 64,
        environment_sha256="b" * 64,
        capture_manifest_sha256=("c" if split == "train" else "d") * 64,
        split=split,
        phase="production",
        stop_after_tokens=tokens,
        total_tokens=tokens,
        process_count=1,
        global_chips=16,
        mesh_name="ep16-fsdp1-tp1",
        per_device_sequence_batch=1,
        captured_tokens=tokens,
        new_tokens=tokens,
        resumed_tokens=0,
        phase_wall_seconds=wall,
        wall_seconds=wall,
        phase_device_seconds=wall,
        device_seconds=wall,
        phase_compile_seconds=0.0,
        compile_seconds=0.0,
        publication_seconds=0.0,
        publication_chip_seconds=0.0,
        phase_chip_seconds=chip_seconds,
        chip_seconds=chip_seconds,
        real_goodput=tokens / wall,
        synthetic_goodput=10.0,
        input_wait_fraction=0.0,
        persisted_goodput=tokens / wall,
        device_output_goodput=tokens / wall,
        max_writer_backlog=0,
        useful_flops=1.0,
        useful_mfu=0.57,
        benchmark_useful_mfu=0.57,
        performance_waiver_sha256=None,
        dropped_routed_tokens=0,
        recompilations=0,
        instrumentation_equal=True,
        canary_passed=True if split == "train" else None,
        manifest_path="manifest.json",
        host_result_paths=("host.json",),
    ).as_dict()


def test_single_cli_exposes_exact_subcommands():
    parser = build_parser()
    action = next(action for action in parser._actions if action.dest == "command")
    assert set(action.choices) == {
        "stage-c4",
        "qualify-storage",
        "qualify-sae-storage",
        "preflight",
        "convert-checkpoint",
        "benchmark-harvest",
        "authorize-benchmark-waiver",
        "harvest",
        "activation-farm",
        "activation-status",
        "activation-abort",
        "sae-smoke",
        "sae-begin",
        "sae-farm",
        "sae-status",
        "sae-abort",
        "sae-phase",
        "train",
        "eval",
        "report",
        "run",
    }


@pytest.mark.parametrize(
    ("argv", "handler"),
    [
        (
            ["sae-smoke", "--condition", "F", "--cache-root", "/tmp/smoke"],
            cli_module._command_sae_smoke,
        ),
        (
            ["sae-begin", "--cache-root", "/tmp/smoke"],
            cli_module._command_sae_begin,
        ),
        (
            ["sae-farm", "--operation-sha256", "a" * 64],
            cli_module._command_sae_farm,
        ),
        (
            ["sae-status", "--operation-sha256", "a" * 64],
            cli_module._command_sae_status,
        ),
        (
            ["sae-abort", "--operation-sha256", "a" * 64, "--reason", "stop"],
            cli_module._command_sae_abort,
        ),
        (
            [
                "sae-phase",
                "--operation-sha256",
                "a" * 64,
                "--phase",
                "full-A-seed-1",
                "--attempt-id",
                "b" * 32,
                "--attempt-number",
                "1",
            ],
            cli_module._command_sae_phase,
        ),
    ],
)
def test_sae_control_commands_dispatch_to_exact_handlers(argv, handler):
    assert build_parser().parse_args(argv).handler is handler


def test_smoke_process_ranges_are_exact_disjoint_full_batch():
    ranges = [
        cli_module._smoke_process_token_range(16_384, index, 4)
        for index in range(4)
    ]
    assert ranges == [(0, 4_096), (4_096, 8_192), (8_192, 12_288), (12_288, 16_384)]
    with pytest.raises(RuntimeError, match="exactly four"):
        cli_module._smoke_process_token_range(16_384, 0, 2)


def test_smoke_distributed_config_uses_real_production_batch_contract():
    config = load_config(Path("configs/gemma4_26b_a4b.yaml"))
    training_config = cli_module._training_config(
        config, "F", 1, a_capacity=None
    )

    distributed = cli_module._smoke_distributed_config(training_config)

    assert training_config.microbatches_per_update == (
        config.effective_batch_tokens // config.microbatch_tokens
    )
    assert distributed.microbatches_per_update == 4


def test_sae_phase_requires_inherited_fd_for_exact_live_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lane_id = "a" * 24
    attempt_id = "b" * 32
    lanes = tmp_path / "lanes"
    lanes.mkdir()
    lock_path = lanes / f"lane-{lane_id}.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)

    class Control:
        root = tmp_path

        @staticmethod
        def load_state():
            return {
                "phases": {
                    "full-A-seed-1": {
                        "status": "running",
                        "attempt_id": attempt_id,
                        "attempts": 2,
                    }
                }
            }

        @staticmethod
        def status():
            return {
                "lanes": [
                    {
                        "lane_id": lane_id,
                        "lease_held": True,
                        "state": "running",
                        "phase": "full-A-seed-1",
                        "attempt_id": attempt_id,
                    }
                ]
            }

    args = SimpleNamespace(
        phase="full-A-seed-1", attempt_id=attempt_id, attempt_number=2
    )
    monkeypatch.setenv("MOE_SAE_SAE_LEASE_FD", str(descriptor))
    try:
        result = cli_module._validate_inherited_sae_phase(args, Control())
        assert result["lease_fd"] == descriptor
        assert result["lane"]["lane_id"] == lane_id
        with pytest.raises(RuntimeError, match="exact active"):
            cli_module._validate_inherited_sae_phase(
                SimpleNamespace(
                    phase=args.phase,
                    attempt_id=args.attempt_id,
                    attempt_number=3,
                ),
                Control(),
            )
    finally:
        os.close(descriptor)


def test_smoke_cache_root_is_inferred_only_from_bound_attempt_shape(tmp_path: Path):
    campaign = "c" * 64
    attempt = "d" * 32
    smoke_root = tmp_path / "host-private-smoke"
    cache = (
        smoke_root
        / "sae-smoke"
        / "campaigns"
        / campaign
        / "conditions"
        / "00-F"
        / "attempts"
        / attempt
        / "cache"
    )
    record = (
        tmp_path
        / "sae-smoke"
        / "campaigns"
        / campaign
        / "conditions"
        / "00-F"
        / "attempts"
        / attempt
        / "attempt.json"
    )
    record.parent.mkdir(parents=True)
    record.write_text(json.dumps({"identity": {"cache_path": str(cache)}}))
    (record.parent / "result.json").write_text("{}")

    failed_attempt = "a" * 32
    failed_record = record.parents[1] / failed_attempt / "attempt.json"
    failed_record.parent.mkdir()
    failed_record.write_text(
        json.dumps({"identity": {"cache_path": str(tmp_path / "wrong-cache")}})
    )
    (failed_record.parent / "failure.json").write_text("{}")
    assert cli_module._infer_smoke_cache_root(tmp_path, campaign) == smoke_root

    record.write_text(json.dumps({"identity": {"cache_path": str(tmp_path)}}))
    with pytest.raises(RuntimeError, match="unexpected identity"):
        cli_module._infer_smoke_cache_root(tmp_path, campaign)


def test_passed_smoke_cleanup_removes_only_exact_attempt_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_root = tmp_path / "cache-root"
    attempt_cache = (
        cache_root
        / "sae-smoke"
        / "campaigns"
        / ("a" * 64)
        / "conditions"
        / "00-F"
        / "attempts"
        / ("b" * 32)
        / "cache"
    )
    attempt_cache.mkdir(parents=True)
    (attempt_cache / "activation.safetensors").write_bytes(b"diagnostic")
    unrelated = cache_root / "keep-me"
    unrelated.write_text("untouched", encoding="utf-8")
    monkeypatch.setattr(cli_module, "_collective_raise_if_any", lambda *a, **k: None)
    monkeypatch.setattr(cli_module, "_global_barrier", lambda *a, **k: None)

    cli_module._cleanup_passed_smoke_cache(attempt_cache, cache_root)

    assert not attempt_cache.exists()
    assert unrelated.read_text(encoding="utf-8") == "untouched"
    with pytest.raises(RuntimeError, match="refusing"):
        cli_module._cleanup_passed_smoke_cache(cache_root / "keep-me", cache_root)


def test_storage_roots_and_orbax_probe_are_fail_closed():
    bulk, checkpoint = cli_module._validate_gcs_storage_roots(
        "gs://bucket/experiment", "gs://bucket/experiment/checkpoints/"
    )
    assert bulk == "gs://bucket/experiment"
    assert checkpoint == "gs://bucket/experiment/checkpoints"
    with pytest.raises(RuntimeError, match="must be below"):
        cli_module._validate_gcs_storage_roots(
            "gs://bucket/experiment", "gs://bucket/other"
        )
    run_id = "qualification-run"
    digest = "a" * 64
    record = {
        "schema_version": "gemma4-orbax-storage-qualification-v1",
        "payload_uri": (
            "gs://bucket/experiment/checkpoints/qualification/"
            "orbax-qualification-run"
        ),
        "process_count": 1,
        "global_device_count": 16,
        "expected_local_sha256_by_process": [digest],
        "restored_local_sha256_by_process": [digest],
        "save_seconds_by_process": [1.0],
        "restore_seconds_by_process": [2.0],
        "max_save_seconds": 1.0,
        "max_restore_seconds": 2.0,
        "finalized": True,
        "metadata_round_trip": True,
        "failures": [],
    }
    cli_module._validate_orbax_qualification(
        record,
        checkpoint_root=checkpoint,
        run_id=run_id,
        process_count=1,
        global_device_count=16,
    )
    record["finalized"] = False
    with pytest.raises(RuntimeError, match="Orbax qualification is invalid"):
        cli_module._validate_orbax_qualification(
            record,
            checkpoint_root=checkpoint,
            run_id=run_id,
            process_count=1,
            global_device_count=16,
        )


def test_activation_storage_attestation_cannot_authorize_sae_consumers() -> None:
    payload = {
        "scope": "activation-capture",
        "metadata": {
            "sae_cache_qualified": False,
            "configured_sae_capture_cache_max_bytes": 100 * 1024**3,
        },
    }
    with pytest.raises(RuntimeError, match="activation storage cannot authorize"):
        cli_module._require_sae_cache_qualification(payload)


def test_activation_storage_attestation_seals_future_download_gate(
    tmp_path,
) -> None:
    config = Gemma4Config()
    root = tmp_path.resolve()
    local = (tmp_path / "local").resolve()
    digest = "a" * 64
    run_id = "qualification-run"
    bulk_root = "gs://bucket/experiment"
    checkpoint_root = f"{bulk_root}/checkpoints"
    contract = dataclasses.asdict(
        StoragePlanesQualificationConfig(
            minimum_control_free_bytes=config.min_control_free_bytes,
            minimum_local_buffer_free_bytes=(
                cli_module.MINIMUM_ACTIVATION_LOCAL_FREE_BYTES
            ),
            minimum_aggregate_local_write_bytes_per_second=(
                cli_module.MINIMUM_ACTIVATION_LOCAL_WRITE_BYTES_PER_SECOND
            ),
            gcs_probe_bytes_per_process=config.gcs_probe_bytes_per_process,
            gcs_probe_block_bytes=config.gcs_probe_block_bytes,
            minimum_aggregate_upload_bytes_per_second=(
                config.min_gcs_upload_bytes_per_second
            ),
            minimum_aggregate_download_bytes_per_second=(
                MINIMUM_ACTIVATION_GCS_DOWNLOAD_BYTES_PER_SECOND
            ),
        )
    )
    payload = {
        "schema_version": STORAGE_ATTESTATION_SCHEMA,
        "scope": "activation-capture",
        "created_at_utc": "2026-07-21T00:00:00+00:00",
        "config_sha256": config.config_hash,
        "environment_sha256": "b" * 64,
        "topology": {"process_count": 1, "global_device_count": 16},
        "contract": contract,
        "qualification": {
            "run_id": run_id,
            "process_count": 1,
            "failures": [],
            "control": {
                "directory": str(root),
                "negative_lookup_creation_visible": True,
            },
            "local_buffer": {
                "directories": [str(local)],
                "host_isolation_proven": True,
                "expected_probe_sha256_by_process": [digest],
                "observed_probe_sha256_by_process": [digest],
                "write": {
                    "total_bytes": config.gcs_probe_bytes_per_process,
                    "elapsed_seconds": 1.0,
                    "bytes_per_second": float(config.gcs_probe_bytes_per_process),
                    "per_process_bytes": [config.gcs_probe_bytes_per_process],
                    "per_process_seconds": [1.0],
                    "errors": [],
                },
                "write_cleanup_complete": True,
            },
            "gcs": {"prefix": bulk_root},
        },
        "metadata": {
            "bulk_gcs_root": bulk_root,
            "sae_checkpoint_root": checkpoint_root,
            "sae_cache_qualified": False,
            "configured_sae_capture_cache_max_bytes": config.capture_cache_max_bytes,
            "configured_sae_min_gcs_download_bytes_per_second": (
                config.min_gcs_download_bytes_per_second
            ),
            "orbax_checkpoint_qualification": {
                "schema_version": "gemma4-orbax-storage-qualification-v1",
                "payload_uri": (
                    f"{checkpoint_root}/qualification/orbax-{run_id}"
                ),
                "process_count": 1,
                "global_device_count": 16,
                "expected_local_sha256_by_process": [digest],
                "restored_local_sha256_by_process": [digest],
                "save_seconds_by_process": [1.0],
                "restore_seconds_by_process": [2.0],
                "max_save_seconds": 1.0,
                "max_restore_seconds": 2.0,
                "finalized": True,
                "metadata_round_trip": True,
                "failures": [],
            },
        },
    }
    payload["attestation_sha256"] = storage_attestation_digest(payload)

    assert (
        cli_module._validate_storage_attestation_payload(
            payload,
            root=root,
            config=config,
            environment_sha256="b" * 64,
            process_count=1,
            global_device_count=16,
        )
        == payload["attestation_sha256"]
    )
    payload["metadata"].pop(
        "configured_sae_min_gcs_download_bytes_per_second"
    )
    payload["attestation_sha256"] = storage_attestation_digest(payload)
    with pytest.raises(RuntimeError, match="deployment layout is incomplete"):
        cli_module._validate_storage_attestation_payload(
            payload,
            root=root,
            config=config,
            environment_sha256="b" * 64,
            process_count=1,
            global_device_count=16,
        )


def test_checkpoint_claim_is_verified_against_actual_inventory(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "state").write_bytes(b"one")
    actual = cli_module.checkpoint_inventory_sha256(str(checkpoint))
    args = SimpleNamespace(checkpoint=str(checkpoint), checkpoint_sha256=actual)
    assert cli_module._verify_checkpoint_inventory(args) == actual

    (checkpoint / "state").write_bytes(b"two")
    changed = SimpleNamespace(checkpoint=str(checkpoint), checkpoint_sha256=actual)
    with pytest.raises(RuntimeError, match="does not match"):
        cli_module._verify_checkpoint_inventory(changed)


def test_checkpoint_save_and_restore_share_one_exact_identity_builder():
    identity = _checkpoint_identity_expected(
        Gemma4Config(),
        capture_manifest_sha256="a" * 64,
        capture_checkpoint_sha256="b" * 64,
        producer_environment_sha256="c" * 64,
        consumer_environment_sha256="d" * 64,
        condition="A",
        seed=1,
        run_instance_id="f" * 32,
        activation_storage_attestation_sha256="e" * 64,
        sae_storage_attestation_sha256="f" * 64,
        capture_compatibility_sha256="1" * 64,
        canary_waiver_sha256="2" * 64,
        sealed_recovery_waiver_sha256="3" * 64,
        checkpoint_payload_root="gs://bucket/checkpoints",
        operation_sha256="4" * 64,
    )

    assert set(identity) == {
        "config_sha256",
        "capture_manifest_sha256",
        "capture_checkpoint_sha256",
        "capture_producer_environment_sha256",
        "sae_consumer_environment_sha256",
        "condition",
        "seed",
        "run_instance_id",
        "activation_storage_attestation_sha256",
        "sae_storage_attestation_sha256",
        "capture_compatibility_sha256",
        "canary_waiver_sha256",
        "sealed_recovery_waiver_sha256",
        "checkpoint_payload_root",
        "operation_sha256",
    }
    assert "tokens_seen" not in identity
    assert "steps" not in identity


def _begin_test_sae_operation(
    root: Path,
    config: Gemma4Config,
) -> tuple[SAEOperationPaths, dict, dict, dict]:
    environment = {
        "sha256": "1" * 64,
        "repository_content_sha256": "2" * 64,
    }
    bundle = {
        "train_manifest": {"manifest_sha256": "3" * 64},
        "eval_manifest": {"manifest_sha256": "4" * 64},
        "activation": {"attestation_sha256": "5" * 64},
        "sae": {"attestation_sha256": "6" * 64},
        "compatibility": {"compatibility_sha256": "7" * 64},
        "layout": {"sae_checkpoint_root": "gs://bucket/checkpoints"},
        "benchmark": {},
    }
    canary_gate = {
        "status": "passed",
        "waiver_sha256": None,
        "recovery_waiver_sha256": None,
    }
    spec = SAEOperationSpec(
        config_sha256=config.config_hash,
        repository_content_sha256=environment["repository_content_sha256"],
        consumer_environment_sha256=environment["sha256"],
        train_capture_manifest_sha256=bundle["train_manifest"][
            "manifest_sha256"
        ],
        eval_capture_manifest_sha256=bundle["eval_manifest"][
            "manifest_sha256"
        ],
        activation_storage_attestation_sha256=bundle["activation"][
            "attestation_sha256"
        ],
        sae_storage_attestation_sha256=bundle["sae"]["attestation_sha256"],
        capture_compatibility_sha256=bundle["compatibility"][
            "compatibility_sha256"
        ],
        smoke_campaign_sha256="8" * 64,
        smoke_suite_sha256="9" * 64,
        checkpoint_payload_root=bundle["layout"]["sae_checkpoint_root"],
    )
    SAEControl(root).begin(spec)
    return SAEOperationPaths(root, spec.sha256), environment, bundle, canary_gate


def test_authoritative_paths_require_and_validate_existing_operation(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    config = Gemma4Config()
    paths, environment, bundle, canary_gate = _begin_test_sae_operation(
        tmp_path, config
    )
    args = SimpleNamespace(operation_sha256=paths.operation_sha256)
    monkeypatch.setattr(
        cli_module,
        "_validated_operation_smoke_hashes",
        lambda *_args, **_kwargs: ("8" * 64, "9" * 64),
    )

    assert cli_module._operation_paths_for_command(
        args, tmp_path, authoritative=True
    ) == paths
    state = cli_module._validate_sae_operation(
        paths,
        config=config,
        environment=environment,
        storage_bundle=bundle,
        canary_gate=canary_gate,
    )
    assert state["operation_sha256"] == paths.operation_sha256

    monkeypatch.setattr(
        cli_module,
        "_validated_operation_smoke_hashes",
        lambda *_args, **_kwargs: ("a" * 64, "b" * 64),
    )
    with pytest.raises(RuntimeError, match="smoke_campaign_sha256"):
        cli_module._validate_sae_operation(
            paths,
            config=config,
            environment=environment,
            storage_bundle=bundle,
            canary_gate=canary_gate,
        )
    monkeypatch.setattr(
        cli_module,
        "_validated_operation_smoke_hashes",
        lambda *_args, **_kwargs: ("8" * 64, "9" * 64),
    )

    with pytest.raises(RuntimeError, match="state is absent"):
        cli_module._validate_sae_operation(
            SAEOperationPaths(tmp_path, "f" * 64),
            config=config,
            environment=environment,
            storage_bundle=bundle,
            canary_gate=canary_gate,
        )

    drifted_environment = dict(environment, repository_content_sha256="e" * 64)
    with pytest.raises(RuntimeError, match="repository_content_sha256"):
        cli_module._validate_sae_operation(
            paths,
            config=config,
            environment=drifted_environment,
            storage_bundle=bundle,
            canary_gate=canary_gate,
        )


def test_reference_paths_cannot_claim_an_authoritative_operation(tmp_path):
    missing = SimpleNamespace(operation_sha256=None)
    with pytest.raises(RuntimeError, match="require --operation-sha256"):
        cli_module._operation_paths_for_command(
            missing, tmp_path, authoritative=True
        )
    assert (
        cli_module._operation_paths_for_command(
            missing, tmp_path, authoritative=False
        )
        is None
    )
    with pytest.raises(RuntimeError, match="reserved for authoritative"):
        cli_module._operation_paths_for_command(
            SimpleNamespace(operation_sha256="a" * 64),
            tmp_path,
            authoritative=False,
        )


def test_cross_operation_artifacts_and_legacy_poison_are_not_selected(tmp_path):
    first = SAEOperationPaths(tmp_path, "a" * 64)
    second = SAEOperationPaths(tmp_path, "b" * 64)
    legacy = cli_module._run_root(tmp_path, "A", 1)
    first_root = first.run_root("A", 1)
    second_root = second.run_root("A", 1)
    for path in (legacy, first_root):
        path.mkdir(parents=True)
        (path / "run.json").write_text('{"poison": true}', encoding="utf-8")

    assert second_root not in {legacy, first_root}
    assert not (second_root / "run.json").exists()
    assert cli_module._training_metrics_journal_path(
        second_root,
        "c" * 32,
        operation_paths=second,
        condition="A",
        seed=1,
    ) == second.metrics_path("A", 1)
    assert second.evaluation_path("A", 1) != cli_module._evaluation_path(
        tmp_path, "A", 1
    )
    assert second.report_path != tmp_path / "report.json"


def test_operation_sha256_cli_validation_and_authoritative_requirement():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["report"])
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "train",
                "--condition",
                "A",
                "--seed",
                "1",
                "--operation-sha256",
                "not-a-digest",
            ]
        )


def test_conversion_attestation_is_self_hashed_and_parity_gated(tmp_path):
    config = Gemma4Config()
    path = cli_module._conversion_attestation_path(tmp_path)
    path.parent.mkdir(parents=True)
    payload = {
        "schema_version": "gemma4-checkpoint-conversion-v2",
        "config_sha256": config.config_hash,
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "hf_source_mode": "immutable-local-snapshot",
        "hf_repository_inventory_sha256": (
            expected_hf_repository_inventory_sha256()
        ),
        "hf_snapshot_inventory_sha256": expected_hf_snapshot_inventory_sha256(),
        "maxtext_commit": config.maxtext_commit,
        "maxtext_checkout_clean": True,
        "maxtext_import_mode": "verified-checkout-source",
        "checkpoint_sha256": "a" * 64,
        "max_kl_div_threshold": 0.03,
        "parity_passed": True,
    }
    payload["attestation_sha256"] = cli_module._conversion_attestation_digest(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert (
        cli_module._load_conversion_attestation(tmp_path, config, "a" * 64)
        == payload["attestation_sha256"]
    )

    payload["parity_passed"] = False
    payload["attestation_sha256"] = cli_module._conversion_attestation_digest(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="required parity"):
        cli_module._load_conversion_attestation(tmp_path, config, "a" * 64)

    payload["parity_passed"] = True
    payload["hf_snapshot_inventory_sha256"] = "f" * 64
    payload["attestation_sha256"] = cli_module._conversion_attestation_digest(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="required parity"):
        cli_module._load_conversion_attestation(tmp_path, config, "a" * 64)


def test_convert_checkpoint_uses_one_pinned_snapshot_and_seals_v2_attestation(
    tmp_path, monkeypatch
):
    config = Gemma4Config()
    root = tmp_path / "artifacts" / "gemma4_v1" / "production"
    maxtext_root = tmp_path / "maxtext"
    base = maxtext_root / "src" / "maxtext" / "configs" / "base.yml"
    base.parent.mkdir(parents=True)
    base.write_text("# fixture\n", encoding="utf-8")
    snapshot_path = (
        tmp_path
        / "hub"
        / "models--google--gemma-4-26B-A4B"
        / "snapshots"
        / config.model_revision
    )
    snapshot_path.mkdir(parents=True)
    snapshot = HuggingFaceSnapshot(
        model_id=config.model_id,
        revision=config.model_revision,
        path=snapshot_path,
        inventory_sha256=expected_hf_snapshot_inventory_sha256(),
        repository_inventory_sha256=expected_hf_repository_inventory_sha256(),
    )
    materialize_calls = []
    commands = []

    def materialize(*args, **kwargs):
        materialize_calls.append((args, kwargs))
        return snapshot

    def run(command, **kwargs):
        commands.append((tuple(command), kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli_module, "_artifact_root", lambda *_args: root)
    monkeypatch.setattr(cli_module, "_require_production_artifact_root", lambda _root: None)
    monkeypatch.setattr(cli_module, "verify_maxtext_checkout", lambda _root: config.maxtext_commit)
    monkeypatch.setattr(cli_module, "materialize_hf_snapshot", materialize)
    monkeypatch.setattr(cli_module, "_require_unused_conversion_output", lambda _output: None)
    monkeypatch.setattr(
        cli_module,
        "maxtext_checkout_environment",
        lambda _root: {
            "HF_TOKEN": "must-not-leak",
            "HUGGING_FACE_HUB_TOKEN": "must-not-leak",
            "HF_API_TOKEN": "must-not-leak",
            "GOOGLE_CLOUD_PROJECT": "kept",
        },
    )
    monkeypatch.setattr(cli_module.subprocess, "run", run)
    monkeypatch.setattr(
        cli_module, "assert_hf_snapshot_unchanged", lambda value: value.inventory_sha256
    )
    monkeypatch.setattr(cli_module, "checkpoint_inventory_sha256", lambda _path: "c" * 64)
    monkeypatch.setattr(cli_module, "_coordinated_environment", lambda: {"sha256": "d" * 64})

    args = SimpleNamespace(
        config=str(cli_module.DEFAULT_CONFIG),
        artifact_root=None,
        maxtext_root=str(maxtext_root),
        hf_cache_dir=str(tmp_path / "hf-cache"),
        output="gs://bucket/model",
        checkpoint_items=None,
        dry_run=False,
    )
    assert cli_module._command_convert_checkpoint(args) == 0

    assert materialize_calls == [
        (
            (config.model_id, config.model_revision),
            {
                "cache_dir": args.hf_cache_dir,
                "token": None,
                "local_files_only": False,
            },
        )
    ]
    assert len(commands) == 2
    for command, options in commands:
        joined = " ".join(command)
        assert str(snapshot_path) in joined
        assert config.model_id not in joined
        assert options["env"] == {"GOOGLE_CLOUD_PROJECT": "kept"}

    payload = json.loads(
        cli_module._conversion_attestation_path(root).read_text(encoding="utf-8")
    )
    assert payload["schema_version"] == "gemma4-checkpoint-conversion-v2"
    assert payload["model_revision"] == config.model_revision
    assert payload["hf_source_mode"] == "immutable-local-snapshot"
    assert payload["hf_repository_inventory_sha256"] == (
        expected_hf_repository_inventory_sha256()
    )
    assert payload["hf_snapshot_inventory_sha256"] == (
        expected_hf_snapshot_inventory_sha256()
    )
    assert payload["checkpoint_sha256"] == "c" * 64
    assert (
        cli_module._load_conversion_attestation(root, config, "c" * 64)
        == payload["attestation_sha256"]
    )


def test_convert_checkpoint_dry_run_never_downloads_model(tmp_path, monkeypatch, capsys):
    config = Gemma4Config()
    monkeypatch.setattr(cli_module, "_artifact_root", lambda *_args: tmp_path)
    monkeypatch.setattr(cli_module, "_require_production_artifact_root", lambda _root: None)
    monkeypatch.setattr(cli_module, "verify_maxtext_checkout", lambda _root: config.maxtext_commit)
    monkeypatch.setattr(
        cli_module,
        "materialize_hf_snapshot",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("dry-run downloaded the model")
        ),
    )
    args = SimpleNamespace(
        config=str(cli_module.DEFAULT_CONFIG),
        artifact_root=None,
        maxtext_root=str(tmp_path / "maxtext"),
        hf_cache_dir=str(tmp_path / "hf-cache"),
        output="gs://bucket/model",
        checkpoint_items=None,
        dry_run=True,
    )

    assert cli_module._command_convert_checkpoint(args) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["model_revision"] == config.model_revision
    assert plan["max_kl_div_threshold"] == 0.03


def test_convert_checkpoint_resume_validation_reuses_canonical_checkpoint(
    tmp_path, monkeypatch
):
    config = Gemma4Config()
    root = tmp_path / "artifacts" / "gemma4_v1" / "production"
    maxtext_root = tmp_path / "maxtext"
    base = maxtext_root / "src" / "maxtext" / "configs" / "base.yml"
    base.parent.mkdir(parents=True)
    base.write_text("# fixture\n", encoding="utf-8")
    snapshot_path = (
        tmp_path
        / "hub"
        / "models--google--gemma-4-26B-A4B"
        / "snapshots"
        / config.model_revision
    )
    snapshot_path.mkdir(parents=True)
    snapshot = HuggingFaceSnapshot(
        model_id=config.model_id,
        revision=config.model_revision,
        path=snapshot_path,
        inventory_sha256=expected_hf_snapshot_inventory_sha256(),
        repository_inventory_sha256=expected_hf_repository_inventory_sha256(),
    )
    materialize_calls = []
    commands = []
    inventory_calls = []

    def materialize(*args, **kwargs):
        materialize_calls.append((args, kwargs))
        return snapshot

    def run(command, **kwargs):
        commands.append((tuple(command), kwargs))
        return SimpleNamespace(returncode=0)

    def inventory(path):
        inventory_calls.append(path)
        return "c" * 64

    monkeypatch.setattr(cli_module, "_artifact_root", lambda *_args: root)
    monkeypatch.setattr(cli_module, "_require_production_artifact_root", lambda _root: None)
    monkeypatch.setattr(cli_module, "verify_maxtext_checkout", lambda _root: config.maxtext_commit)
    monkeypatch.setattr(cli_module, "materialize_hf_snapshot", materialize)
    monkeypatch.setattr(
        cli_module,
        "_require_unused_conversion_output",
        lambda _output: (_ for _ in ()).throw(
            AssertionError("resume attempted to require a fresh output")
        ),
    )
    monkeypatch.setattr(
        cli_module,
        "maxtext_checkout_environment",
        lambda _root: {"HF_TOKEN": "must-not-leak", "GOOGLE_CLOUD_PROJECT": "kept"},
    )
    monkeypatch.setattr(cli_module.subprocess, "run", run)
    monkeypatch.setattr(
        cli_module, "assert_hf_snapshot_unchanged", lambda value: value.inventory_sha256
    )
    monkeypatch.setattr(cli_module, "checkpoint_inventory_sha256", inventory)
    monkeypatch.setattr(cli_module, "_coordinated_environment", lambda: {"sha256": "d" * 64})

    output = "gs://bucket/model/attempt-1"
    items = f"{output}/0/items"
    args = SimpleNamespace(
        config=str(cli_module.DEFAULT_CONFIG),
        artifact_root=None,
        maxtext_root=str(maxtext_root),
        hf_cache_dir=str(tmp_path / "hf-cache"),
        output=output,
        checkpoint_items=None,
        resume_validation=True,
        dry_run=False,
    )
    assert cli_module._command_convert_checkpoint(args) == 0

    assert materialize_calls == [
        (
            (config.model_id, config.model_revision),
            {
                "cache_dir": args.hf_cache_dir,
                "token": None,
                "local_files_only": True,
            },
        )
    ]
    assert inventory_calls == [items, items]
    assert len(commands) == 1
    command, options = commands[0]
    assert "tests.utils.forward_pass_logit_checker" in command
    assert "megablox=false" in command
    assert not any("checkpoint_conversion.to_maxtext" in item for item in command)
    assert options["env"] == {"GOOGLE_CLOUD_PROJECT": "kept"}

    payload = json.loads(
        cli_module._conversion_attestation_path(root).read_text(encoding="utf-8")
    )
    assert payload["checkpoint"] == items
    assert payload["checkpoint_sha256"] == "c" * 64
    assert "conversion_seconds" not in payload
    assert payload["conversion_recovery"] == {
        "mode": "resume-validation",
        "converter_rerun": False,
        "conversion_timing_available": False,
        "prevalidation_checkpoint_sha256": "c" * 64,
    }


def test_convert_checkpoint_resume_validation_rejects_dry_run():
    args = SimpleNamespace(
        config=str(cli_module.DEFAULT_CONFIG),
        artifact_root=None,
        maxtext_root="unused",
        hf_cache_dir="unused",
        output="gs://bucket/model/attempt-1",
        checkpoint_items=None,
        resume_validation=True,
        dry_run=True,
    )
    with pytest.raises(ValueError, match="cannot be combined"):
        cli_module._command_convert_checkpoint(args)


def test_convert_checkpoint_resume_validation_rejects_noncanonical_items(
    tmp_path, monkeypatch
):
    config = Gemma4Config()
    root = tmp_path / "artifacts" / "gemma4_v1" / "production"
    monkeypatch.setattr(cli_module, "_artifact_root", lambda *_args: root)
    monkeypatch.setattr(
        cli_module, "_require_production_artifact_root", lambda _root: None
    )
    monkeypatch.setattr(
        cli_module,
        "verify_maxtext_checkout",
        lambda _root: config.maxtext_commit,
    )
    args = SimpleNamespace(
        config=str(cli_module.DEFAULT_CONFIG),
        artifact_root=None,
        maxtext_root="maxtext",
        hf_cache_dir="hf-cache",
        output="gs://bucket/model/attempt-1",
        checkpoint_items="gs://bucket/model/different/0/items",
        resume_validation=True,
        dry_run=False,
    )

    with pytest.raises(ValueError, match="canonical checkpoint path"):
        cli_module._command_convert_checkpoint(args)


def test_convert_checkpoint_resume_validation_rejects_checkpoint_mutation(
    tmp_path, monkeypatch
):
    config = Gemma4Config()
    root = tmp_path / "artifacts" / "gemma4_v1" / "production"
    maxtext_root = tmp_path / "maxtext"
    base = maxtext_root / "src" / "maxtext" / "configs" / "base.yml"
    base.parent.mkdir(parents=True)
    base.write_text("# fixture\n", encoding="utf-8")
    snapshot_path = (
        tmp_path
        / "hub"
        / "models--google--gemma-4-26B-A4B"
        / "snapshots"
        / config.model_revision
    )
    snapshot_path.mkdir(parents=True)
    snapshot = HuggingFaceSnapshot(
        model_id=config.model_id,
        revision=config.model_revision,
        path=snapshot_path,
        inventory_sha256=expected_hf_snapshot_inventory_sha256(),
        repository_inventory_sha256=expected_hf_repository_inventory_sha256(),
    )
    inventories = iter(("a" * 64, "b" * 64))
    monkeypatch.setattr(cli_module, "_artifact_root", lambda *_args: root)
    monkeypatch.setattr(
        cli_module, "_require_production_artifact_root", lambda _root: None
    )
    monkeypatch.setattr(
        cli_module,
        "verify_maxtext_checkout",
        lambda _root: config.maxtext_commit,
    )
    monkeypatch.setattr(
        cli_module, "materialize_hf_snapshot", lambda *args, **kwargs: snapshot
    )
    monkeypatch.setattr(cli_module, "maxtext_checkout_environment", lambda _root: {})
    monkeypatch.setattr(cli_module.subprocess, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        cli_module,
        "assert_hf_snapshot_unchanged",
        lambda value: value.inventory_sha256,
    )
    monkeypatch.setattr(
        cli_module,
        "checkpoint_inventory_sha256",
        lambda _path: next(inventories),
    )
    args = SimpleNamespace(
        config=str(cli_module.DEFAULT_CONFIG),
        artifact_root=None,
        maxtext_root=str(maxtext_root),
        hf_cache_dir=str(tmp_path / "hf-cache"),
        output="gs://bucket/model/attempt-1",
        checkpoint_items=None,
        resume_validation=True,
        dry_run=False,
    )

    with pytest.raises(RuntimeError, match="changed during resumed parity"):
        cli_module._command_convert_checkpoint(args)
    assert not cli_module._conversion_attestation_path(root).exists()


def test_conversion_output_must_be_a_fresh_local_path(tmp_path):
    output = tmp_path / "attempt-1"
    cli_module._require_unused_conversion_output(str(output))
    output.mkdir()
    with pytest.raises(RuntimeError, match="fresh attempt-specific"):
        cli_module._require_unused_conversion_output(str(output))


def test_convert_checkpoint_refuses_to_overwrite_a_sealed_attestation(
    tmp_path, monkeypatch
):
    config = Gemma4Config()
    root = tmp_path / "artifacts" / "gemma4_v1" / "production"
    attestation = cli_module._conversion_attestation_path(root)
    attestation.parent.mkdir(parents=True)
    attestation.write_text("already sealed\n", encoding="utf-8")
    monkeypatch.setattr(cli_module, "_artifact_root", lambda *_args: root)
    monkeypatch.setattr(cli_module, "_require_production_artifact_root", lambda _root: None)
    monkeypatch.setattr(cli_module, "verify_maxtext_checkout", lambda _root: config.maxtext_commit)
    monkeypatch.setattr(
        cli_module,
        "_require_unused_conversion_output",
        lambda _output: (_ for _ in ()).throw(
            AssertionError("output probe ran after sealed-attestation gate")
        ),
    )
    args = SimpleNamespace(
        config=str(cli_module.DEFAULT_CONFIG),
        artifact_root=None,
        maxtext_root=str(tmp_path / "maxtext"),
        hf_cache_dir=str(tmp_path / "hf-cache"),
        output="gs://bucket/model/new-attempt",
        checkpoint_items=None,
        dry_run=False,
    )
    with pytest.raises(RuntimeError, match="already sealed"):
        cli_module._command_convert_checkpoint(args)


def test_conversion_output_must_be_an_empty_gcs_prefix(monkeypatch):
    class FakeBlob:
        def __init__(self, exists):
            self._exists = exists

        def exists(self):
            return self._exists

    class FakeBucket:
        def __init__(self, exact_exists):
            self._exact_exists = exact_exists

        def blob(self, _name):
            return FakeBlob(self._exact_exists)

    class FakeClient:
        def __init__(self, *, exact_exists=False, descendant_exists=False):
            self.exact_exists = exact_exists
            self.descendant_exists = descendant_exists

        def bucket(self, _name):
            return FakeBucket(self.exact_exists)

        def list_blobs(self, _bucket, *, prefix, max_results):
            assert prefix == "model/attempt-1/"
            assert max_results == 1
            return iter([object()] if self.descendant_exists else [])

    class FakeStore:
        client = FakeClient()

    monkeypatch.setattr(cli_module, "GCSObjectStore", FakeStore)
    cli_module._require_unused_conversion_output("gs://bucket/model/attempt-1")

    FakeStore.client = FakeClient(descendant_exists=True)
    with pytest.raises(RuntimeError, match="fresh attempt-specific"):
        cli_module._require_unused_conversion_output(
            "gs://bucket/model/attempt-1"
        )


def test_authoritative_cost_telemetry_is_recomputed_from_components():
    config = Gemma4Config()
    run_instance_id = "a" * 32
    steps = config.planned_updates
    record = {
        "authoritative": True,
        "condition": "A",
        "seed": 1,
        "run_instance_id": run_instance_id,
        "tokens_seen": config.train_tokens,
        "steps": steps,
        "chip_count": 16,
        "optimizer_wall_seconds": 8.0,
        "compile_seconds": 2.0,
        "checkpoint_seconds": 1.0,
        "a_capacity_precompute_wall_seconds": 2.0,
        "marginal_wall_seconds": 10.0,
        "chip_seconds": 160.0,
        "tokens_per_second": config.train_tokens / 10.0,
        "optimizer_tokens_per_second": config.train_tokens / 8.0,
        "metrics_journal": {
            "schema_version": "gemma4-sae-training-journal-snapshot-v1",
            "path": f"/tmp/{run_instance_id}.jsonl",
            "binding": {
                "schema_version": "gemma4-sae-metrics-journal-v1",
                "identity": {
                    "run_instance_id": run_instance_id,
                    "condition": "A",
                    "seed": 1,
                },
                "identity_sha256": "1" * 64,
                "file_sha256": "2" * 64,
                "journal_sha256": "3" * 64,
                "row_count": steps,
                "last_step": steps,
                "last_tokens_seen": config.train_tokens,
            },
            "total_tokens": config.train_tokens,
            "effective_batch_tokens": config.effective_batch_tokens,
            "expected_l0": config.topk_features,
            "complete_prefix_topk_valid": True,
            "complete_prefix_exact_l0": True,
            "last_cumulative_optimizer_wall_seconds": 7.9,
        },
    }
    cli_module._validate_authoritative_run_timing(
        record, config, "A", expected_tokens=config.train_tokens
    )

    record["chip_seconds"] = 159.0
    with pytest.raises(RuntimeError, match="internally inconsistent"):
        cli_module._validate_authoritative_run_timing(record, config, "A")


def test_report_journal_snapshot_is_recomputed_from_bound_bytes(tmp_path):
    from moe_sae.metrics_journal import (
        JournalIdentity,
        MetricsJournal,
        training_journal_snapshot,
    )

    config = Gemma4Config()
    run_instance_id = "4" * 32
    identity = JournalIdentity(run_instance_id, "B", 1)
    journal = MetricsJournal(tmp_path / "metrics.jsonl", identity)
    journal.append(
        {
            "step": 1,
            "tokens_seen": config.effective_batch_tokens,
            "valid_tokens": config.effective_batch_tokens,
            "l0": float(config.topk_features),
            "topk_valid": True,
            "optimizer_wall_seconds": 1.0,
        }
    )
    sealed = training_journal_snapshot(
        journal,
        total_tokens=config.train_tokens,
        effective_batch_tokens=config.effective_batch_tokens,
        expected_l0=config.topk_features,
    )
    record = {
        "run_instance_id": run_instance_id,
        "condition": "B",
        "seed": 1,
        "metrics_journal": sealed,
    }

    assert cli_module._coordinated_training_journal_snapshot(record, config) == sealed
    journal.append(
        {
            "step": 2,
            "tokens_seen": 2 * config.effective_batch_tokens,
            "valid_tokens": config.effective_batch_tokens,
            "l0": float(config.topk_features),
            "topk_valid": True,
            "optimizer_wall_seconds": 2.0,
        }
    )
    assert cli_module._coordinated_training_journal_snapshot(record, config) != sealed


def test_production_artifact_root_cannot_escape_v1_namespace(tmp_path):
    with pytest.raises(RuntimeError, match="artifact root"):
        cli_module._require_production_artifact_root(tmp_path)


def test_selected_benchmark_rejects_environment_drift(tmp_path, monkeypatch):
    config = Gemma4Config()
    path = tmp_path / "selected.json"
    path.write_text(json.dumps(_selected_benchmark(config)), encoding="utf-8")
    monkeypatch.setattr(
        cli_module, "_coordinated_environment", lambda: {"sha256": "f" * 64}
    )
    with pytest.raises(RuntimeError, match="environment"):
        cli_module._load_selected_benchmark(path, config, "a" * 64, "9" * 64)


def _write_staged_manifest(tmp_path, split: str, scientific_tokens: int):
    file_path = tmp_path / f"{split}.array_record"
    file_path.write_bytes(f"packed-{split}".encode())
    manifest_path = tmp_path / f"{split}.manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": PACKED_DATASET_SCHEMA,
                "dataset": {
                    "name": ALLENAI_C4_REPO_ID,
                    "config": "en",
                    "revision": ALLENAI_C4_REVISION,
                    "split": split,
                },
                "source": {
                    "provider": "huggingface_hub",
                    "repo_type": "dataset",
                    "repo_id": ALLENAI_C4_REPO_ID,
                    "revision": ALLENAI_C4_REVISION,
                    "files": [
                        item.as_dict()
                        for item in ALLENAI_C4_SOURCE_FILES[split]
                    ],
                },
                "tokenizer": {
                    "id": "google/gemma-4-26B-A4B",
                    "revision": MODEL_REVISION,
                    "repository_inventory_sha256": (
                        GEMMA4_TOKENIZER_INVENTORY_SHA256
                    ),
                    "files": [
                        item.as_dict() for item in GEMMA4_TOKENIZER_SOURCE_FILES
                    ],
                    "padding_id": 0,
                    "eos_id": 1,
                    "bos_id": 2,
                },
                "sequence_length": 1024,
                "record_format": {
                    "name": PACKED_RECORD_FORMAT,
                    "arrayrecord_options": "group_size:1",
                    "token_ids": "int32[1024]",
                    "positions": "uint16[1024]",
                    "segment_ids": "uint16[1024]",
                },
                "packing": {
                    "strategy": "ordered_next_fit_offline",
                    "exclude_from_scientific_count": ["padding", "bos", "eos"],
                    "special_token_ids": {"padding": 0, "eos": 1, "bos": 2},
                },
                "consumer": {
                    "process_count": 4,
                    "host_sharding": "record_index_mod_process_count",
                    "per_device_sequence_batches": [1, 2, 4],
                },
                "target_scientific_tokens": scientific_tokens,
                "source_examples": 1,
                "scientific_tokens": scientific_tokens,
                "record_count": 64,
                "lane_scientific_tokens": [
                    scientific_tokens // 4
                    + (1 if index < scientific_tokens % 4 else 0)
                    for index in range(4)
                ],
                "files": [
                    {
                        "uri": file_path.name,
                        "sha256": hashlib.sha256(file_path.read_bytes()).hexdigest(),
                        "size_bytes": file_path.stat().st_size,
                        "first_record_index": 0,
                        "record_count": 64,
                        "scientific_tokens": scientific_tokens,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return file_path, manifest_path


def test_local_preflight_is_useful_without_tpu(capsys, tmp_path):
    config = Gemma4Config()
    train, train_manifest = _write_staged_manifest(
        tmp_path, "train", config.train_tokens
    )
    heldout, eval_manifest = _write_staged_manifest(
        tmp_path, "validation", config.eval_tokens
    )
    status = main(
        [
            "preflight",
            "--local",
            "--checkpoint",
            "explicit-checkpoint",
            "--grain-train",
            str(train),
            "--grain-eval",
            str(heldout),
            "--dataset-train-manifest",
            str(train_manifest),
            "--dataset-eval-manifest",
            str(eval_manifest),
            "--json",
        ]
    )
    assert status == 0
    assert '"secret_scan"' in capsys.readouterr().out


def test_cli_sae_config_matches_parameter_and_candidate_contract():
    config = Gemma4Config()
    a = _sae_config(config, "A")
    b = _sae_config(config, "B")
    f = _sae_config(config, "F")
    assert a.parameter_count == f.parameter_count == 738_328_576
    assert b.parameter_count == 46_145_536
    assert a.candidates_per_token == b.candidates_per_token == 8_192
    assert f.candidates_per_token == 131_072


def test_padding_routes_are_unique_and_balanced():
    config = Gemma4Config()
    ids = _balanced_padding_ids(4_096, config)
    assert ids.shape == (4_096, 8)
    assert np.all(np.diff(np.sort(ids, axis=1), axis=1) > 0)
    counts = np.bincount(ids.reshape(-1), minlength=128)
    assert counts.min() == counts.max() == 256


def test_capacity_is_scanned_from_fixed_order_with_margin():
    config = Gemma4Config(
        train_tokens=10,
        eval_tokens=4,
        shard_tokens=10,
        effective_batch_tokens=8,
        microbatch_tokens=4,
        warmup_updates=1,
        checkpoint_every_tokens=8,
        canary_tokens=4,
        health_gate_tokens=4,
    )
    ids = np.stack(
        [
            (np.arange(8, dtype=np.int32) + row) % config.num_routed_experts
            for row in range(10)
        ]
    )

    class Dataset:
        def iter_batches(self):
            yield SimpleNamespace(expert_ids=ids[:3])
            yield SimpleNamespace(expert_ids=ids[3:])

    capacity = _precompute_a_capacity(Dataset(), config)
    assert capacity > 0
    assert capacity % 8 == 0


def test_production_capacity_cache_is_content_addressed_by_consumer_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import jax
    import moe_sae.distributed as distributed_module

    config = Gemma4Config()

    class Dataset:
        manifest_sha256 = "c" * 64

    environment = {"repository_content_sha256": "1" * 64}
    writes: list[tuple[Path, dict]] = []
    monkeypatch.setattr(jax, "process_count", lambda: 4)
    monkeypatch.setattr(
        cli_module, "_coordinated_environment", lambda: dict(environment)
    )
    monkeypatch.setattr(
        cli_module, "_coordinator_read_json_if_exists", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        distributed_module,
        "precompute_distributed_a_capacity",
        lambda *args, **kwargs: 24,
    )
    monkeypatch.setattr(
        cli_module, "_collective_raise_if_any", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(cli_module, "_global_max_seconds", lambda _value: 2.5)

    def publish(path: Path, payload: dict, **_kwargs) -> bool:
        writes.append((path, payload))
        return True

    monkeypatch.setattr(cli_module, "_coordinator_write_json", publish)
    training = SimpleNamespace(seed=1)
    assert cli_module._production_a_capacity(
        tmp_path, Dataset(), config, training
    ) == (24, 2.5)
    first_path, first_payload = writes[-1]
    assert first_payload["schema_version"] == "gemma4-a-capacity-cache-v2"
    assert first_path.name == (
        "a_capacity-train-seed-1-"
        f"{first_payload['identity_sha256']}.json"
    )

    environment["repository_content_sha256"] = "2" * 64
    assert cli_module._production_a_capacity(
        tmp_path, Dataset(), config, training
    ) == (24, 2.5)
    assert writes[-1][0] != first_path


def test_harvester_report_uses_measured_train_and_eval_capture_cost(tmp_path):
    config = Gemma4Config()
    benchmark = tmp_path / "benchmark"
    harvest = tmp_path / "harvest"
    benchmark.mkdir()
    harvest.mkdir()
    selected_envelope = _selected_benchmark(config, useful_mfu=0.57)
    (benchmark / "selected.json").write_text(
        json.dumps(selected_envelope), encoding="utf-8"
    )
    (harvest / "train.json").write_text(
        json.dumps(campaign("train", config.train_tokens, 100.0)),
        encoding="utf-8",
    )
    (harvest / "eval.json").write_text(
        json.dumps(campaign("eval", config.eval_tokens, 5.0)),
        encoding="utf-8",
    )

    class Capture:
        def __init__(self, digest):
            self.manifest_sha256 = digest
            self.manifest = {
                "metadata": {
                    "dataset_fingerprint": "5" * 64,
                    "environment_sha256": "b" * 64,
                    "conversion_attestation_sha256": "9" * 64,
                    "storage_attestation_sha256": "8" * 64,
                }
            }

    result = _load_harvester_report(
        tmp_path,
        config,
        None,
        expected_captures={
            "train": Capture("c" * 64),
            "eval": Capture("d" * 64),
        },
    )
    assert result["chip_seconds"] == 105.0
    assert result["useful_mfu"] == 0.57
    assert set(result["campaigns"]) == {"train", "eval"}
    bad_eval = campaign("eval", config.eval_tokens, 5.0)
    bad_eval["capture_manifest_sha256"] = "e" * 64
    (harvest / "eval.json").write_text(json.dumps(bad_eval), encoding="utf-8")
    with pytest.raises(RuntimeError, match="not bound"):
        _load_harvester_report(
            tmp_path,
            config,
            None,
            expected_captures={
                "train": Capture("c" * 64),
                "eval": Capture("d" * 64),
            },
        )


def test_completed_campaign_is_bound_to_selected_benchmark(tmp_path):
    config = Gemma4Config()
    path = tmp_path / "train.json"
    path.write_text(
        json.dumps(campaign("train", config.train_tokens, 100.0)), encoding="utf-8"
    )
    benchmark = {
        "persisted_tokens_per_second": 10.0,
        "useful_mfu": 0.57,
        "dropped_routed_tokens": 0,
        "recompilations": 0,
        "instrumentation_equal": True,
    }
    assert cli_module._campaign_is_complete(
        path,
        split="train",
        tokens=config.train_tokens,
        mesh_name="ep16-fsdp1-tp1",
        sequence_batch=1,
        config_sha256=config.config_hash,
        checkpoint_sha256="a" * 64,
        conversion_attestation_sha256="9" * 64,
        dataset_fingerprint="5" * 64,
        environment_sha256="b" * 64,
        capture_manifest_sha256="c" * 64,
        benchmark=benchmark,
    )
    benchmark["useful_mfu"] = 0.58
    with pytest.raises(RuntimeError, match="incompatible"):
        cli_module._campaign_is_complete(
            path,
            split="train",
            tokens=config.train_tokens,
            mesh_name="ep16-fsdp1-tp1",
            sequence_batch=1,
            config_sha256=config.config_hash,
            checkpoint_sha256="a" * 64,
            conversion_attestation_sha256="9" * 64,
            dataset_fingerprint="5" * 64,
            environment_sha256="b" * 64,
            capture_manifest_sha256="c" * 64,
            benchmark=benchmark,
        )


def _run_args(tmp_path):
    return SimpleNamespace(
        config=str(cli_module.DEFAULT_CONFIG),
        artifact_root=str(tmp_path),
        checkpoint="orbax-checkpoint",
        checkpoint_sha256="a" * 64,
        grain_train="train.array_record@*",
        grain_eval="eval.array_record@*",
        dataset_train_manifest="train.manifest.json",
        dataset_eval_manifest="eval.manifest.json",
        operation_sha256="0" * 64,
    )


def _selected_benchmark(config, *, useful_mfu=0.55):
    selected = BenchmarkObservation(
        mesh=HARVEST_MESHES[0],
        per_device_sequence_batch=1,
        synchronized_steps=200,
        elapsed_seconds=10.0,
        device_elapsed_seconds=8.0,
        persistence_seconds=2.0,
        useful_flops=useful_mfu * V4_32_BF16_PEAK_FLOPS * 8.0,
        output_tokens=100,
        persisted_tokens=100,
        input_wait_seconds=0.0,
        recompilations=0,
        dropped_routed_tokens=0,
        max_writer_backlog=1,
        instrumentation_equal=True,
    ).as_dict()
    return {
        "schema_version": config.schema_version,
        "config_sha256": config.config_hash,
        "checkpoint_sha256": "a" * 64,
        "conversion_attestation_sha256": "9" * 64,
        "storage_attestation_sha256": "8" * 64,
        "environment": cli_module._coordinated_environment(),
        "selected": selected,
    }


def test_dataset_reuse_skips_only_semantic_arrayrecord_recount(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = Gemma4Config()
    observed = {}
    provenance = SimpleNamespace(fingerprint_sha256="5" * 64)

    def verify(*args, **kwargs):
        observed.update(kwargs)
        return provenance

    monkeypatch.setattr(cli_module, "verify_staged_dataset_manifest", verify)
    monkeypatch.setattr(cli_module, "_coordinated_sha256", lambda *args, **kwargs: None)
    args = argparse.Namespace(local=False, reuse_verified_dataset=True)

    result = cli_module._verified_dataset_provenance(
        args,
        config,
        split="train",
        grain_files="gs://bucket/train/*.array_record",
        manifest_location="gs://bucket/train/manifest.json",
    )

    assert result is provenance
    assert observed["verify_file_hashes"] is True
    assert observed["verify_arrayrecord_count"] is False


def _sae_storage_attestation() -> dict:
    return {
        "attestation_sha256": "8" * 64,
        "scope": "sae-training",
        "metadata": {"sae_cache_qualified": True},
    }


def _consumer_storage_bundle(tmp_path: Path, config: Gemma4Config) -> dict:
    return {
        "activation": {"attestation_sha256": "8" * 64},
        "sae": {"attestation_sha256": "7" * 64},
        "compatibility": {"compatibility_sha256": "6" * 64},
        "producer_environment_sha256": "b" * 64,
        "conversion_attestation_sha256": "9" * 64,
        "train_manifest": {},
        "eval_manifest": {},
        "benchmark": _selected_benchmark(config),
        "layout": {
            "capture_cache_root": tmp_path / "cache",
            "capture_cache_max_bytes": 100 * 1024**3,
            "bulk_gcs_root": "gs://test-bucket/bulk",
            "sae_checkpoint_root": "gs://test-bucket/checkpoints",
        },
    }


def test_run_fresh_state_orders_canary_health_production_and_eval(
    tmp_path, monkeypatch
):
    config = Gemma4Config()
    events: list[tuple] = []
    progress: dict[tuple[str, int], dict] = {}
    evaluations: set[tuple[str, int]] = set()

    class Dataset:
        def __init__(self, split):
            self.total_tokens = (
                config.train_tokens if split == "train" else config.eval_tokens
            )
            self.manifest = {
                "metadata": {
                    "checkpoint_sha256": "a" * 64,
                    "dataset_fingerprint": "5" * 64,
                }
            }
            self.manifest_sha256 = f"{split}-manifest"

    benchmark_root = tmp_path / "benchmark"
    benchmark_root.mkdir(parents=True)
    (benchmark_root / "selected.json").write_text(
        json.dumps(_selected_benchmark(config)), encoding="utf-8"
    )
    preflight = tmp_path / "preflight"
    preflight.mkdir()
    sae_path = cli_module._sae_storage_attestation_path(
        tmp_path, cli_module._coordinated_environment()["sha256"]
    )
    sae_path.parent.mkdir(parents=True, exist_ok=True)
    sae_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(cli_module, "run_preflight", lambda **kwargs: ())
    monkeypatch.setattr(
        cli_module,
        "_verified_dataset_provenance",
        lambda *args, **kwargs: SimpleNamespace(fingerprint_sha256="5" * 64),
    )
    monkeypatch.setattr(
        cli_module, "_require_production_artifact_root", lambda root: None
    )
    monkeypatch.setattr(
        cli_module,
        "_verify_checkpoint_inventory",
        lambda args: args.checkpoint_sha256,
    )
    monkeypatch.setattr(
        cli_module, "_load_conversion_attestation", lambda *args: "9" * 64
    )
    monkeypatch.setattr(
        cli_module,
        "_load_storage_attestation_payload",
        lambda *args, **kwargs: _sae_storage_attestation(),
    )
    monkeypatch.setattr(
        cli_module,
        "_read_storage_attestation_payload",
        lambda *args, **kwargs: _sae_storage_attestation(),
    )
    monkeypatch.setattr(
        cli_module,
        "_storage_layout",
        lambda payload: {
            "bulk_gcs_root": "gs://test-bucket/bulk",
            "sae_checkpoint_root": "gs://test-bucket/checkpoints",
        },
    )
    monkeypatch.setattr(cli_module, "_release_accelerator_state", lambda label: None)
    monkeypatch.setattr(
        cli_module,
        "_sealed_dataset",
        lambda path, **kwargs: Dataset("eval" if path.name == "eval" else "train"),
    )
    monkeypatch.setattr(
        cli_module,
        "_load_sae_storage_bundle",
        lambda *args, **kwargs: _consumer_storage_bundle(tmp_path, config),
    )
    monkeypatch.setattr(
        cli_module,
        "_consumer_canary_gate",
        lambda *args, **kwargs: {
            "status": "passed",
            "waiver_sha256": None,
            "recovery_waiver_sha256": None,
        },
    )
    monkeypatch.setattr(cli_module, "_require_exact_capture", lambda *args: None)
    monkeypatch.setattr(
        cli_module, "_require_capture_identity", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        cli_module, "_validate_sae_operation", lambda *args, **kwargs: {}
    )

    def benchmark(args):
        events.append(("benchmark",))
        destination = tmp_path / "benchmark"
        destination.mkdir(parents=True)
        (destination / "selected.json").write_text(
            json.dumps(_selected_benchmark(config)), encoding="utf-8"
        )
        return 0

    def harvest(args):
        events.append(("harvest", args.split, args.canary))
        if not args.canary:
            capture = tmp_path / "captures" / args.split
            capture.mkdir(parents=True, exist_ok=True)
            (capture / "manifest.json").write_text("{}", encoding="utf-8")
        return 0

    def train(args):
        events.append(("train", args.condition, args.seed, args.stop_after_tokens))
        progress[(args.condition, args.seed)] = {
            "tokens_seen": (
                config.train_tokens
                if args.stop_after_tokens is None
                else config.health_gate_tokens
            )
        }
        return 0

    def resume_run(root, cfg, capture, condition, seed, **kwargs):
        del root, cfg, capture
        return progress.get((condition, seed))

    def existing_eval(
        root,
        cfg,
        train,
        heldout,
        condition,
        seed,
        run_record,
        **kwargs,
    ):
        del kwargs
        del root, cfg, train, heldout, run_record
        return {} if (condition, seed) in evaluations else None

    def evaluate(args, cfg, condition, seed):
        del args, cfg
        events.append(("eval", condition, seed))
        evaluations.add((condition, seed))
        return {}

    monkeypatch.setattr(cli_module, "_command_benchmark_harvest", benchmark)
    monkeypatch.setattr(cli_module, "_command_harvest", harvest)
    monkeypatch.setattr(cli_module, "_command_train", train)
    monkeypatch.setattr(cli_module, "_load_resume_run", resume_run)
    monkeypatch.setattr(cli_module, "_existing_evaluation", existing_eval)
    monkeypatch.setattr(cli_module, "_command_eval_one", evaluate)
    monkeypatch.setattr(cli_module, "_command_report", lambda args: 0)

    assert cli_module._command_run(_run_args(tmp_path)) == 0
    assert not [event for event in events if event[0] in {"benchmark", "harvest"}]
    training = [event for event in events if event[0] == "train"]
    assert training[:3] == [
        ("train", "A", 1, config.health_gate_tokens),
        ("train", "B", 1, config.health_gate_tokens),
        ("train", "F", 1, config.health_gate_tokens),
    ]
    assert len(training) == 12
    assert len([event for event in events if event[0] == "eval"]) == 9


def test_run_resume_skips_every_matching_sealed_stage(tmp_path, monkeypatch):
    config = Gemma4Config()
    benchmark = tmp_path / "benchmark"
    benchmark.mkdir(parents=True)
    (benchmark / "selected.json").write_text(
        json.dumps(_selected_benchmark(config)), encoding="utf-8"
    )
    preflight = tmp_path / "preflight"
    preflight.mkdir()
    sae_path = cli_module._sae_storage_attestation_path(
        tmp_path, cli_module._coordinated_environment()["sha256"]
    )
    sae_path.parent.mkdir(parents=True, exist_ok=True)
    sae_path.write_text("{}", encoding="utf-8")
    for split in ("train", "eval"):
        capture = tmp_path / "captures" / split
        capture.mkdir(parents=True)
        (capture / "manifest.json").write_text("{}", encoding="utf-8")

    class Dataset:
        total_tokens = config.train_tokens
        manifest = {
            "metadata": {
                "checkpoint_sha256": "a" * 64,
                "dataset_fingerprint": "5" * 64,
            }
        }
        manifest_sha256 = "manifest"

    monkeypatch.setattr(cli_module, "run_preflight", lambda **kwargs: ())
    monkeypatch.setattr(
        cli_module,
        "_verified_dataset_provenance",
        lambda *args, **kwargs: SimpleNamespace(fingerprint_sha256="5" * 64),
    )
    monkeypatch.setattr(
        cli_module, "_require_production_artifact_root", lambda root: None
    )
    monkeypatch.setattr(
        cli_module,
        "_verify_checkpoint_inventory",
        lambda args: args.checkpoint_sha256,
    )
    monkeypatch.setattr(
        cli_module, "_load_conversion_attestation", lambda *args: "9" * 64
    )
    monkeypatch.setattr(
        cli_module,
        "_load_storage_attestation_payload",
        lambda *args, **kwargs: _sae_storage_attestation(),
    )
    monkeypatch.setattr(
        cli_module,
        "_read_storage_attestation_payload",
        lambda *args, **kwargs: _sae_storage_attestation(),
    )
    monkeypatch.setattr(
        cli_module,
        "_storage_layout",
        lambda payload: {
            "bulk_gcs_root": "gs://test-bucket/bulk",
            "sae_checkpoint_root": "gs://test-bucket/checkpoints",
        },
    )
    monkeypatch.setattr(cli_module, "_sealed_dataset", lambda path, **kwargs: Dataset())
    monkeypatch.setattr(
        cli_module,
        "_load_sae_storage_bundle",
        lambda *args, **kwargs: _consumer_storage_bundle(tmp_path, config),
    )
    monkeypatch.setattr(
        cli_module,
        "_consumer_canary_gate",
        lambda *args, **kwargs: {
            "status": "passed",
            "waiver_sha256": None,
            "recovery_waiver_sha256": None,
        },
    )
    monkeypatch.setattr(cli_module, "_require_exact_capture", lambda *args: None)
    monkeypatch.setattr(
        cli_module, "_require_capture_identity", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        cli_module, "_validate_sae_operation", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(
        cli_module, "_campaign_is_complete", lambda *args, **kwargs: True
    )
    monkeypatch.setattr(
        cli_module,
        "_load_resume_run",
        lambda *args, **kwargs: {
            "tokens_seen": config.train_tokens,
            "chip_seconds": 1.0,
        },
    )
    monkeypatch.setattr(
        cli_module, "_existing_evaluation", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(
        cli_module,
        "_command_benchmark_harvest",
        lambda args: (_ for _ in ()).throw(AssertionError("benchmark reran")),
    )
    monkeypatch.setattr(
        cli_module,
        "_command_harvest",
        lambda args: (_ for _ in ()).throw(AssertionError("harvest reran")),
    )
    monkeypatch.setattr(
        cli_module,
        "_command_train",
        lambda args: (_ for _ in ()).throw(AssertionError("training reran")),
    )
    monkeypatch.setattr(
        cli_module,
        "_command_eval_one",
        lambda *args: (_ for _ in ()).throw(AssertionError("evaluation reran")),
    )
    monkeypatch.setattr(cli_module, "_command_report", lambda args: 0)

    assert cli_module._command_run(_run_args(tmp_path)) == 0
