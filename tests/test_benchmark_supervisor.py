import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

import moe_sae.cli as cli
import moe_sae.benchmark_worker as benchmark_worker
from moe_sae.config import Gemma4Config
from moe_sae.runtime import BenchmarkObservation, V4_32_BF16_PEAK_FLOPS


def _args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        config=str(cli.DEFAULT_CONFIG),
        artifact_root=str(tmp_path),
        checkpoint="checkpoint",
        checkpoint_sha256="a" * 64,
        output=str(tmp_path / "benchmark"),
        steps=200,
        grain_train="train.array_record",
        grain_eval="eval.array_record",
        dataset_train_manifest="train.json",
        dataset_eval_manifest="eval.json",
    )


def _passing(mesh, batch: int) -> BenchmarkObservation:
    return BenchmarkObservation(
        mesh=mesh,
        per_device_sequence_batch=batch,
        synchronized_steps=200,
        elapsed_seconds=1.0,
        useful_flops=0.60 * V4_32_BF16_PEAK_FLOPS,
        output_tokens=1_000,
        persisted_tokens=1_000,
        recompilations=0,
        dropped_routed_tokens=0,
        max_writer_backlog=0,
        instrumentation_equal=True,
    )


def test_public_cli_import_is_jax_cold_and_keeps_eight_commands():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import moe_sae.cli; print(int('jax' in sys.modules))",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "0"
    parser = cli.build_parser()
    action = next(item for item in parser._actions if item.dest == "command")
    assert set(action.choices) == {
        "preflight",
        "convert-checkpoint",
        "benchmark-harvest",
        "harvest",
        "train",
        "eval",
        "report",
        "run",
    }


def test_run_launches_missing_benchmark_before_tpu_preflight(tmp_path, monkeypatch):
    args = _args(tmp_path)
    config = Gemma4Config()
    events = []
    monkeypatch.setattr(cli, "_config", lambda args: config)
    monkeypatch.setattr(cli, "_artifact_root", lambda config, override: tmp_path)
    monkeypatch.setattr(cli, "_require_production_artifact_root", lambda root: None)
    monkeypatch.setattr(
        cli,
        "_command_benchmark_harvest",
        lambda args: events.append("benchmark") or 0,
    )

    def preflight(**kwargs):
        events.append("preflight")
        raise RuntimeError("stop after ordering assertion")

    monkeypatch.setattr(cli, "run_preflight", preflight)
    with pytest.raises(RuntimeError, match="ordering assertion"):
        cli._command_run(args)
    assert events == ["benchmark", "preflight"]


def test_worker_timeout_terminates_the_fresh_process_group(monkeypatch):
    monkeypatch.setattr(cli, "_BENCHMARK_KILL_GRACE_SECONDS", 0.1)
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
    )
    returncode, timed_out = cli._wait_for_worker(
        process, deadline=time.time() + 0.05
    )
    assert timed_out is True
    assert returncode != 0
    assert process.poll() is not None


def test_failed_candidate_is_published_then_next_fresh_worker_runs(
    tmp_path, monkeypatch
):
    config = Gemma4Config()
    args = _args(tmp_path)
    output = tmp_path / "benchmark"
    output.mkdir()
    sweep_id = "1" * 32
    bootstrap = tmp_path / "bootstrap.json"
    bootstrap.write_text("{}", encoding="utf-8")
    supervisor = {
        "sweep_id": sweep_id,
        "process_index": 0,
        "process_count": 1,
        "bootstrap_path": bootstrap,
    }
    commands = []

    class Process:
        pid = 12345

        def __init__(self, returncode):
            self.returncode = returncode

        def wait(self, timeout=None):
            return self.returncode

    def popen(command, **kwargs):
        del kwargs
        commands.append(tuple(command))
        if len(commands) == 1:
            return Process(3)
        values = {
            command[index]: command[index + 1]
            for index in range(len(command) - 1)
            if command[index].startswith("--")
        }
        attempt_root = Path(values["--attempt-root"])
        observation = _passing(
            cli._mesh_by_name(values["--mesh"]),
            int(values["--per-device-sequence-batch"]),
        )
        attempt_root.mkdir(parents=True, exist_ok=True)
        (attempt_root / "raw.json").write_text(
            json.dumps(
                {
                    "schema_version": "gemma4-benchmark-worker-result-v1",
                    "sweep_id": values["--sweep-id"],
                    "attempt_id": values["--attempt-id"],
                    "observation": observation.as_dict(),
                }
            ),
            encoding="utf-8",
        )
        return Process(0)

    monkeypatch.setattr(cli.subprocess, "Popen", popen)
    first = cli._run_supervised_benchmark_candidate(
        args,
        config,
        tmp_path,
        output,
        supervisor,
        mesh=cli.HARVEST_MESHES[0],
        batch=1,
    )
    second = cli._run_supervised_benchmark_candidate(
        args,
        config,
        tmp_path,
        output,
        supervisor,
        mesh=cli.HARVEST_MESHES[0],
        batch=2,
    )
    assert first.failure is not None
    assert second.failure is None
    assert second.passes_hard_gate()
    assert len(commands) == 2
    assert commands[0] != commands[1]
    assert (output / "ep16-fsdp1-tp1-b1.json").is_file()
    assert (output / "ep16-fsdp1-tp1-b2.json").is_file()


@pytest.mark.parametrize("raw_mode", ["missing", "stale"])
def test_zero_exit_without_matching_raw_result_is_failed(
    tmp_path, monkeypatch, raw_mode
):
    config = Gemma4Config()
    args = _args(tmp_path)
    output = tmp_path / "benchmark"
    output.mkdir()
    sweep_id = "2" * 32
    bootstrap = tmp_path / "bootstrap.json"
    bootstrap.write_text("{}", encoding="utf-8")
    supervisor = {
        "sweep_id": sweep_id,
        "process_index": 0,
        "process_count": 1,
        "bootstrap_path": bootstrap,
    }

    class Process:
        pid = 12346

        def wait(self, timeout=None):
            return 0

    def popen(command, **kwargs):
        del kwargs
        values = {
            command[index]: command[index + 1]
            for index in range(len(command) - 1)
            if command[index].startswith("--")
        }
        if raw_mode == "stale":
            attempt_root = Path(values["--attempt-root"])
            attempt_root.mkdir(parents=True, exist_ok=True)
            observation = _passing(cli.HARVEST_MESHES[0], 1)
            (attempt_root / "raw.json").write_text(
                json.dumps(
                    {
                        "sweep_id": values["--sweep-id"],
                        "attempt_id": "f" * 32,
                        "observation": observation.as_dict(),
                    }
                ),
                encoding="utf-8",
            )
        return Process()

    monkeypatch.setattr(cli.subprocess, "Popen", popen)
    observation = cli._run_supervised_benchmark_candidate(
        args,
        config,
        tmp_path,
        output,
        supervisor,
        mesh=cli.HARVEST_MESHES[0],
        batch=1,
    )
    assert observation.failure is not None
    assert observation.elapsed_seconds > 0.0


def test_every_fresh_candidate_reverifies_checkpoint_inventory(tmp_path, monkeypatch):
    config = Gemma4Config()
    bootstrap_path = tmp_path / "bootstrap.json"
    bootstrap_path.write_text(
        json.dumps(
            {
                "schema_version": "gemma4-benchmark-bootstrap-v1",
                "sweep_id": "3" * 32,
                "config_sha256": config.config_hash,
                "checkpoint": "checkpoint",
                "checkpoint_sha256": "a" * 64,
                "conversion_attestation_sha256": "9" * 64,
                "process_count": 1,
                "environment": {"sha256": "e" * 64},
            }
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        config="unused",
        artifact_root=str(tmp_path),
        checkpoint="checkpoint",
        checkpoint_sha256="a" * 64,
        output=str(tmp_path),
        bootstrap_record=str(bootstrap_path),
        sweep_id="3" * 32,
        attempt_id="4" * 32,
        attempt_root=str(tmp_path / "attempt"),
        mesh="ep16-fsdp1-tp1",
        per_device_sequence_batch=1,
        steps=200,
        expected_process_index=0,
        expected_process_count=1,
    )
    verified = []
    monkeypatch.setattr(benchmark_worker, "require_v4_32_runtime", lambda: None)
    monkeypatch.setattr(
        benchmark_worker, "_paths", lambda args: (config, tmp_path, tmp_path)
    )
    monkeypatch.setattr(benchmark_worker, "_process_context", lambda: (0, 1))
    monkeypatch.setattr(
        benchmark_worker,
        "_verify_checkpoint_inventory",
        lambda args: verified.append(args.checkpoint) or args.checkpoint_sha256,
    )
    monkeypatch.setattr(
        benchmark_worker,
        "_coordinated_environment",
        lambda: {"sha256": "e" * 64},
    )
    monkeypatch.setattr(
        benchmark_worker,
        "_load_conversion_attestation",
        lambda *args: "9" * 64,
    )
    monkeypatch.setattr(
        "moe_sae.harvest.benchmark_production_candidate",
        lambda **kwargs: _passing(cli.HARVEST_MESHES[0], 1),
    )
    monkeypatch.setattr(
        benchmark_worker, "_aggregate_benchmark_observation", lambda value: value
    )
    monkeypatch.setattr(
        benchmark_worker, "_coordinator_write_json", lambda *args, **kwargs: True
    )

    assert benchmark_worker._candidate(args) == 0
    assert verified == ["checkpoint"]
