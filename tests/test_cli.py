import json
import hashlib
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
from moe_sae.config import Gemma4Config
from moe_sae.data import PACKED_DATASET_SCHEMA, PACKED_RECORD_FORMAT
from moe_sae.harvest import CampaignSummary


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
        "preflight",
        "convert-checkpoint",
        "benchmark-harvest",
        "harvest",
        "train",
        "eval",
        "report",
        "run",
    }


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
        environment_sha256="c" * 64,
        condition="A",
        seed=1,
        run_instance_id="d" * 32,
        storage_attestation_sha256="e" * 64,
        checkpoint_payload_root="gs://bucket/checkpoints",
    )

    assert set(identity) == {
        "config_sha256",
        "capture_manifest_sha256",
        "capture_checkpoint_sha256",
        "environment_sha256",
        "condition",
        "seed",
        "run_instance_id",
        "storage_attestation_sha256",
        "checkpoint_payload_root",
    }
    assert "tokens_seen" not in identity
    assert "steps" not in identity


def test_conversion_attestation_is_self_hashed_and_parity_gated(tmp_path):
    config = Gemma4Config()
    path = cli_module._conversion_attestation_path(tmp_path)
    path.parent.mkdir(parents=True)
    payload = {
        "schema_version": "gemma4-checkpoint-conversion-v1",
        "config_sha256": config.config_hash,
        "model_id": config.model_id,
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


def test_authoritative_cost_telemetry_is_recomputed_from_components():
    config = Gemma4Config()
    record = {
        "authoritative": True,
        "tokens_seen": config.train_tokens,
        "chip_count": 16,
        "optimizer_wall_seconds": 8.0,
        "compile_seconds": 2.0,
        "checkpoint_seconds": 1.0,
        "a_capacity_precompute_wall_seconds": 2.0,
        "marginal_wall_seconds": 10.0,
        "chip_seconds": 160.0,
        "tokens_per_second": config.train_tokens / 10.0,
        "optimizer_tokens_per_second": config.train_tokens / 8.0,
    }
    cli_module._validate_authoritative_run_timing(
        record, config, "A", expected_tokens=config.train_tokens
    )

    record["chip_seconds"] = 159.0
    with pytest.raises(RuntimeError, match="internally inconsistent"):
        cli_module._validate_authoritative_run_timing(record, config, "A")


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
                    "name": "c4",
                    "config": "en",
                    "version": "3.1.0",
                    "split": split,
                },
                "source": {
                    "provider": "tensorflow_datasets",
                    "full_name": "c4/en/3.1.0",
                    "dataset_info_sha256": "b" * 64,
                },
                "tokenizer": {
                    "id": "google/gemma-4-26B-A4B",
                    "revision": "a" * 40,
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


def test_harvester_report_uses_measured_train_and_eval_capture_cost(tmp_path):
    config = Gemma4Config()
    benchmark = tmp_path / "benchmark"
    harvest = tmp_path / "harvest"
    benchmark.mkdir()
    harvest.mkdir()
    (benchmark / "selected.json").write_text(
        json.dumps(
            {
                "schema_version": config.schema_version,
                "config_sha256": config.config_hash,
                "checkpoint_sha256": "a" * 64,
                "conversion_attestation_sha256": "9" * 64,
                "storage_attestation_sha256": "8" * 64,
                "selected": {
                    "mesh": {"name": "ep16-fsdp1-tp1"},
                    "per_device_sequence_batch": 1,
                    "useful_mfu": 0.57,
                    "persisted_tokens_per_second": 10,
                    "dropped_routed_tokens": 0,
                    "recompilations": 0,
                    "instrumentation_equal": True,
                    "passes_hard_gate": True,
                },
            }
        ),
        encoding="utf-8",
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
    )


def _selected_benchmark(config):
    return {
        "schema_version": config.schema_version,
        "config_sha256": config.config_hash,
        "checkpoint_sha256": "a" * 64,
        "conversion_attestation_sha256": "9" * 64,
        "storage_attestation_sha256": "8" * 64,
        "environment": cli_module._coordinated_environment(),
        "selected": {
            "mesh": {"name": "ep16-fsdp1-tp1"},
            "per_device_sequence_batch": 1,
            "passes_hard_gate": True,
            "useful_mfu": 0.55,
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
            self.manifest = {"metadata": {"checkpoint_sha256": "a" * 64}}
            self.manifest_sha256 = f"{split}-manifest"

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
        lambda *args, **kwargs: {"attestation_sha256": "8" * 64},
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
        lambda path: Dataset("eval" if path.name == "eval" else "train"),
    )
    monkeypatch.setattr(cli_module, "_require_exact_capture", lambda *args: None)
    monkeypatch.setattr(
        cli_module, "_require_capture_identity", lambda *args, **kwargs: None
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

    def existing_eval(root, cfg, train, heldout, condition, seed, run_record):
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
    assert events[:4] == [
        ("benchmark",),
        ("harvest", "train", True),
        ("harvest", "train", False),
        ("harvest", "eval", False),
    ]
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
    for split in ("train", "eval"):
        capture = tmp_path / "captures" / split
        capture.mkdir(parents=True)
        (capture / "manifest.json").write_text("{}", encoding="utf-8")

    class Dataset:
        total_tokens = config.train_tokens
        manifest = {"metadata": {"checkpoint_sha256": "a" * 64}}
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
        lambda *args, **kwargs: {"attestation_sha256": "8" * 64},
    )
    monkeypatch.setattr(
        cli_module,
        "_storage_layout",
        lambda payload: {
            "bulk_gcs_root": "gs://test-bucket/bulk",
            "sae_checkpoint_root": "gs://test-bucket/checkpoints",
        },
    )
    monkeypatch.setattr(cli_module, "_sealed_dataset", lambda path: Dataset())
    monkeypatch.setattr(cli_module, "_require_exact_capture", lambda *args: None)
    monkeypatch.setattr(
        cli_module, "_require_capture_identity", lambda *args, **kwargs: None
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
    monkeypatch.setattr(cli_module, "_existing_evaluation", lambda *args: {})
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
