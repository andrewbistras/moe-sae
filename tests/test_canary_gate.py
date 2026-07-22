from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from moe_sae.canary_gate import CanaryGateError, validate_canary_gate
from moe_sae.harvest import CampaignSummary


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _self_hash(value: dict[str, Any]) -> dict[str, Any]:
    unsigned = {name: item for name, item in value.items() if name != "waiver_sha256"}
    return {**unsigned, "waiver_sha256": hashlib.sha256(_canonical(unsigned)).hexdigest()}


def _storage_payload(label: str = "qualified-storage") -> dict[str, Any]:
    payload = {
        "schema_version": "gemma4-storage-planes-qualification-v6",
        "scope": "activation-capture",
        "evidence": label,
    }
    return {
        **payload,
        "attestation_sha256": hashlib.sha256(_canonical(payload)).hexdigest(),
    }


@dataclass
class _GateFixture:
    root: Path
    train_manifest: dict[str, Any]
    production_campaign: dict[str, Any]
    selected_benchmark: dict[str, Any]
    source: dict[str, Any]
    waiver: dict[str, Any]

    @property
    def waiver_path(self) -> Path:
        return self.root / "harvest" / "train-canary-waiver.json"

    @property
    def storage_path(self) -> Path:
        return self.root / "preflight" / "storage.json"

    def write_waiver(self) -> None:
        self.waiver = _self_hash(self.waiver)
        _write_json(self.waiver_path, self.waiver)

    def validate(self) -> dict[str, Any]:
        return validate_canary_gate(
            self.root,
            train_manifest=self.train_manifest,
            production_campaign=self.production_campaign,
            selected_benchmark=self.selected_benchmark,
        )


def _real_campaign_summary(identities: dict[str, str], benchmark_waiver: str) -> dict[str, Any]:
    return CampaignSummary(
        schema_version="gemma4-harvest-campaign-v2",
        config_sha256=identities["config_sha256"],
        checkpoint_sha256=identities["checkpoint_sha256"],
        conversion_attestation_sha256=identities[
            "conversion_attestation_sha256"
        ],
        dataset_fingerprint=identities["dataset_fingerprint"],
        environment_sha256=identities["environment_sha256"],
        capture_manifest_sha256=None,
        split="train",
        phase="canary",
        stop_after_tokens=2_000_000,
        total_tokens=50_000_000,
        process_count=4,
        global_chips=16,
        mesh_name="EP16/FSDP1/TP1",
        per_device_sequence_batch=1,
        captured_tokens=2_000_000,
        new_tokens=2_000_000,
        resumed_tokens=0,
        phase_wall_seconds=100.0,
        wall_seconds=100.0,
        phase_device_seconds=90.0,
        device_seconds=90.0,
        phase_compile_seconds=10.0,
        compile_seconds=10.0,
        publication_seconds=5.0,
        publication_chip_seconds=80.0,
        phase_chip_seconds=1_600.0,
        chip_seconds=1_600.0,
        real_goodput=80.0,
        synthetic_goodput=100.0,
        input_wait_fraction=0.10,
        persisted_goodput=80.0,
        device_output_goodput=100.0,
        max_writer_backlog=1,
        useful_flops=1.0,
        useful_mfu=0.02,
        benchmark_useful_mfu=0.02,
        performance_waiver_sha256=benchmark_waiver,
        dropped_routed_tokens=0,
        recompilations=0,
        instrumentation_equal=True,
        canary_passed=False,
        manifest_path=None,
        host_result_paths=tuple(f"host-{index}.json" for index in range(4)),
    ).as_dict()


def _gate_fixture(tmp_path: Path) -> _GateFixture:
    root = tmp_path.resolve()
    identities = {
        name: _digest(name)
        for name in (
            "config_sha256",
            "checkpoint_sha256",
            "conversion_attestation_sha256",
            "dataset_fingerprint",
            "environment_sha256",
        )
    }
    benchmark_waiver = _digest("benchmark-waiver")
    selected_benchmark = {
        "performance_waiver": {"waiver_sha256": benchmark_waiver}
    }
    selected_path = root / "benchmark" / "selected.json"
    selected_file_sha256 = _write_json(selected_path, selected_benchmark)

    storage_payload = _storage_payload()
    storage_path = root / "preflight" / "storage.json"
    storage_file_sha256 = _write_json(storage_path, storage_payload)
    storage_attestation_sha256 = storage_payload["attestation_sha256"]

    objects = [
        {
            "uri": f"gs://capture/shard-{index:02d}.safetensors",
            "generation": index + 1,
            "sha256": _digest(f"object-{index}"),
        }
        for index in range(16)
    ]
    manifest_sha256 = _digest("sealed-train-manifest")
    train_manifest = {
        "schema_version": "gemma4-capture-manifest-v1",
        "manifest_sha256": manifest_sha256,
        "sealed": True,
        "total_tokens": 50_000_000,
        "metadata": {
            **identities,
            "storage_attestation_sha256": storage_attestation_sha256,
        },
        "shards": [{"object": item} for item in objects],
    }
    manifest_path = root / "captures" / "train" / "manifest.json"
    manifest_file_sha256 = _write_json(manifest_path, train_manifest)

    source = _real_campaign_summary(identities, benchmark_waiver)
    assert "storage_attestation_sha256" not in source
    source_path = (
        root
        / "captures"
        / "train"
        / "telemetry"
        / "campaign-stop-000002000000.json"
    )
    source_file_sha256 = _write_json(source_path, source)

    host_paths = [
        root / "captures" / "train" / "hosts" / f"host-p{index:05d}.json"
        for index in range(4)
    ]
    for index, path in enumerate(host_paths):
        _write_json(path, {"production_host": index})
    production_telemetry_path = (
        root
        / "captures"
        / "train"
        / "telemetry"
        / "campaign-stop-000050000000.json"
    )
    production_telemetry_sha256 = _write_json(
        production_telemetry_path, {"phase": "production"}
    )

    observed_fields = (
        "captured_tokens",
        "synthetic_goodput",
        "real_goodput",
        "input_wait_fraction",
        "device_output_goodput",
        "persisted_goodput",
        "max_writer_backlog",
        "dropped_routed_tokens",
        "recompilations",
        "instrumentation_equal",
    )
    waiver = _self_hash(
        {
            "schema_version": "gemma4-canary-performance-waiver-v1",
            "authorization_scope": "canary-performance-only",
            "preserves_canary_passed_false": True,
            "thresholds": {
                "real_to_synthetic_goodput": 0.9,
                "max_input_wait_fraction": 0.05,
                "persisted_to_device_goodput": 0.9,
                "max_writer_backlog_exclusive": 2,
            },
            "source_record": str(source_path),
            "source_record_sha256": source_file_sha256,
            "source_summary_sha256": hashlib.sha256(_canonical(source)).hexdigest(),
            "observed": {name: source[name] for name in observed_fields},
            "source_canary_passed": False,
            "source_config_sha256": identities["config_sha256"],
            "source_checkpoint_sha256": identities["checkpoint_sha256"],
            "source_conversion_attestation_sha256": identities[
                "conversion_attestation_sha256"
            ],
            "source_dataset_fingerprint": identities["dataset_fingerprint"],
            "source_environment_sha256": identities["environment_sha256"],
            "storage_attestation_sha256": storage_attestation_sha256,
            "allowed_failed_performance_checks": [
                "real_to_synthetic_goodput",
                "input_wait_fraction",
                "persisted_to_device_goodput",
            ],
            "bound_file_sha256": {
                str(selected_path): selected_file_sha256,
                str(storage_path): storage_file_sha256,
                **{
                    str(path): _digest(f"canary-host-{index}")
                    for index, path in enumerate(host_paths)
                },
            },
            "selected_benchmark_record_sha256": selected_file_sha256,
            "verified_capture_objects": objects,
            "source_benchmark_performance_waiver_sha256": benchmark_waiver,
        }
    )
    waiver_path = root / "harvest" / "train-canary-waiver.json"
    waiver_file_sha256 = _write_json(waiver_path, waiver)

    recovery = _self_hash(
        {
            "schema_version": "gemma4-sealed-capture-recovery-waiver-v1",
            "authorization_scope": "sealed-summary-and-eval-recovery-only",
            "preserves_canary_passed_false": True,
            "source_canary_waiver_file_sha256": waiver_file_sha256,
            "source_canary_waiver_sha256": waiver["waiver_sha256"],
            "production_host_ledger_sha256": {
                str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in host_paths
            },
            "production_telemetry_sha256": {
                str(production_telemetry_path): production_telemetry_sha256
            },
            "manifest_sha256": manifest_sha256,
            "manifest_path": str(manifest_path),
            "manifest_file_sha256": manifest_file_sha256,
            "manifest_total_tokens": 50_000_000,
            "manifest_shard_count": len(objects),
        }
    )
    _write_json(root / "harvest" / "train-sealed-recovery-waiver.json", recovery)

    return _GateFixture(
        root=root,
        train_manifest=train_manifest,
        production_campaign={
            "canary_passed": False,
            "capture_manifest_sha256": manifest_sha256,
        },
        selected_benchmark=selected_benchmark,
        source=source,
        waiver=waiver,
    )


def test_real_campaign_schema_without_storage_field_passes_via_bound_chain(
    tmp_path: Path,
) -> None:
    fixture = _gate_fixture(tmp_path)

    assert "storage_attestation_sha256" not in fixture.source
    assert CampaignSummary.from_mapping(fixture.source).as_dict() == fixture.source
    result = fixture.validate()

    assert result["status"] == "waived-unmet"
    assert result["canary_passed"] is False
    assert result["headline_eligible"] is False


def test_manifest_storage_identity_mismatch_fails(tmp_path: Path) -> None:
    fixture = _gate_fixture(tmp_path)
    fixture.train_manifest["metadata"]["storage_attestation_sha256"] = _digest(
        "other-storage"
    )

    with pytest.raises(
        CanaryGateError,
        match="waiver, sealed capture, and bound storage attestation differ",
    ):
        fixture.validate()


def test_waiver_storage_identity_mismatch_fails(tmp_path: Path) -> None:
    fixture = _gate_fixture(tmp_path)
    fixture.waiver["storage_attestation_sha256"] = _digest("other-storage")
    fixture.write_waiver()

    with pytest.raises(
        CanaryGateError,
        match="waiver, sealed capture, and bound storage attestation differ",
    ):
        fixture.validate()


def test_bound_storage_payload_identity_mismatch_fails(tmp_path: Path) -> None:
    fixture = _gate_fixture(tmp_path)
    changed_payload = _storage_payload("different-qualified-storage")
    changed_file_sha256 = _write_json(fixture.storage_path, changed_payload)
    fixture.waiver["bound_file_sha256"][str(fixture.storage_path)] = (
        changed_file_sha256
    )
    fixture.write_waiver()

    with pytest.raises(
        CanaryGateError,
        match="waiver, sealed capture, and bound storage attestation differ",
    ):
        fixture.validate()


def test_bound_storage_payload_must_retain_its_self_hash(tmp_path: Path) -> None:
    fixture = _gate_fixture(tmp_path)
    changed_payload = json.loads(fixture.storage_path.read_text(encoding="utf-8"))
    changed_payload["evidence"] = "tampered-without-new-attestation"
    changed_file_sha256 = _write_json(fixture.storage_path, changed_payload)
    fixture.waiver["bound_file_sha256"][str(fixture.storage_path)] = (
        changed_file_sha256
    )
    fixture.write_waiver()

    with pytest.raises(
        CanaryGateError,
        match="waiver, sealed capture, and bound storage attestation differ",
    ):
        fixture.validate()


def test_storage_attestation_must_be_explicitly_bound(tmp_path: Path) -> None:
    fixture = _gate_fixture(tmp_path)
    del fixture.waiver["bound_file_sha256"][str(fixture.storage_path)]
    fixture.write_waiver()

    with pytest.raises(
        CanaryGateError,
        match="does not bind the exact storage attestation file",
    ):
        fixture.validate()
