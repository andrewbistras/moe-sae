from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from moe_sae.r1.prefix_benchmark_runner import (
    PrefixBenchmarkRunnerError,
    _atomic_json,
    _decode_device_memory_words,
    _local_device_memory_words,
    _mesh_argument,
    _physical_mesh_evidence,
    _seed_argument,
    _validate_output,
    parse_args,
    run,
)


def _device_memory_gather(
    *,
    peak_bytes: int = 20_000_000_000,
    bytes_limit: int | None = 32_000_000_000,
    include_peak: bool = True,
) -> np.ndarray:
    gathered = np.zeros((16, 4, 12), dtype=np.uint32)
    for process_index in range(16):
        devices = []
        for local_index in range(4):
            device_id = process_index * 4 + local_index
            stats = {
                "bytes_in_use": peak_bytes - 1,
            }
            if include_peak:
                stats["peak_bytes_in_use"] = peak_bytes
            if bytes_limit is not None:
                stats["bytes_limit"] = bytes_limit
            devices.append(
                SimpleNamespace(
                    id=device_id,
                    process_index=process_index,
                    memory_stats=lambda stats=stats: stats,
                )
            )
        gathered[process_index] = _local_device_memory_words(
            devices,
            process_index=process_index,
        )
    return gathered


def test_physical_mesh_evidence_records_all_logical_assignments() -> None:
    devices = np.empty((1, 8, 8, 1), dtype=object)
    for device_id, logical_index in enumerate(np.ndindex(devices.shape)):
        devices[logical_index] = SimpleNamespace(
            id=device_id,
            process_index=device_id // 4,
            coords=(device_id // 8, device_id % 8, 0),
        )
    mesh = SimpleNamespace(
        axis_names=("data", "fsdp", "expert", "context"),
        devices=devices,
    )
    config = SimpleNamespace(
        ici_fsdp_parallelism=8,
        ici_expert_parallelism=8,
        allow_split_physical_axes=False,
    )

    evidence = _physical_mesh_evidence(mesh, config)

    assert evidence["logical_shape"] == [1, 8, 8, 1]
    assert evidence["allow_split_physical_axes"] is False
    assert len(evidence["devices"]) == 64
    assert evidence["devices"][9] == {
        "logical_index": [0, 1, 1, 0],
        "device_id": 9,
        "process_index": 2,
        "physical_coordinates": [1, 1, 0],
    }


def test_physical_mesh_evidence_rejects_duplicate_device_identity() -> None:
    device = SimpleNamespace(id=0, process_index=0, coords=(0, 0, 0))
    mesh = SimpleNamespace(
        axis_names=("data", "fsdp", "expert", "context"),
        devices=np.full((1, 8, 8, 1), device, dtype=object),
    )
    config = SimpleNamespace(
        ici_fsdp_parallelism=8,
        ici_expert_parallelism=8,
        allow_split_physical_axes=False,
    )
    with pytest.raises(PrefixBenchmarkRunnerError, match="one-to-one"):
        _physical_mesh_evidence(mesh, config)


def test_device_memory_evidence_is_exact_bounded_and_uses_limit_fallback() -> None:
    evidence = _decode_device_memory_words(
        _device_memory_gather(bytes_limit=None)
    )
    assert evidence["source"] == "jax.Device.memory_stats"
    assert evidence["bounded_gather"] == {
        "process_count": 16,
        "records_per_process": 4,
        "uint32_words_per_record": 12,
        "total_bytes": 3_072,
    }
    assert len(evidence["devices"]) == 64
    assert evidence["devices"][63]["device_id"] == 63
    assert evidence["devices"][63]["bytes_limit_key_present"] is False
    assert evidence["devices"][63]["bytes_limit"] is None
    assert evidence["passes_15_percent_headroom"] is True


def test_device_memory_evidence_rejects_duplicate_device_identity() -> None:
    gathered = _device_memory_gather()
    gathered[15, 3, 0] = gathered[0, 0, 0]
    with pytest.raises(PrefixBenchmarkRunnerError, match="duplicate device"):
        _decode_device_memory_words(gathered)


@pytest.mark.parametrize(
    ("gathered", "expected_pass"),
    [
        (_device_memory_gather(include_peak=False), False),
        (_device_memory_gather(peak_bytes=0), False),
        (_device_memory_gather(peak_bytes=27_200_000_000), True),
        (_device_memory_gather(peak_bytes=27_200_000_001), False),
        (
            _device_memory_gather(
                peak_bytes=20_400_000_000,
                bytes_limit=24_000_000_000,
            ),
            True,
        ),
        (
            _device_memory_gather(
                peak_bytes=20_400_000_001,
                bytes_limit=24_000_000_000,
            ),
            False,
        ),
        (_device_memory_gather(bytes_limit=0), False),
    ],
)
def test_device_memory_evidence_requires_positive_peak_and_exact_headroom(
    gathered: np.ndarray,
    expected_pass: bool,
) -> None:
    evidence = _decode_device_memory_words(gathered)
    assert evidence["passes_15_percent_headroom"] is expected_pass


def test_arguments_select_only_the_fixed_ladder(tmp_path: Path) -> None:
    args = parse_args(
        [
            "--maxtext-base-yml",
            str(tmp_path / "base.yml"),
            "--output-directory",
            str(tmp_path / "output"),
            "--mesh",
            "EP32/FSDP2",
            "--seed",
            "7",
        ]
    )
    assert args.mesh == 1
    assert args.seed == 7
    with pytest.raises(argparse.ArgumentTypeError):
        _mesh_argument("EP64/FSDP1")
    with pytest.raises(argparse.ArgumentTypeError):
        _seed_argument("-1")


def test_result_json_is_atomic_and_canonical(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "result.json"
    _atomic_json(path, {"z": 1, "a": [2]})
    assert path.read_bytes() == b'{"a":[2],"z":1}\n'
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": [2], "z": 1}
    assert not list(path.parent.glob(".*.tmp"))


def test_output_validation_rejects_host_values_before_reductions() -> None:
    with pytest.raises(PrefixBenchmarkRunnerError, match="triple"):
        _validate_output(None)
    with pytest.raises(PrefixBenchmarkRunnerError, match="ABI"):
        _validate_output((object(), object(), object()))


def test_runner_initializes_distributed_jax_before_topology_query(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    order: list[str] = []

    def initialized() -> None:
        order.append("initialize")

    def topology() -> None:
        order.append("topology")
        raise RuntimeError("stop after ordering proof")

    monkeypatch.setattr(
        "moe_sae.r1.prefix_benchmark_runner.initialize_v6e_distributed_runtime",
        initialized,
    )
    monkeypatch.setattr(
        "moe_sae.r1.prefix_benchmark_runner.assert_real_v6e64_runtime",
        topology,
    )
    with pytest.raises(RuntimeError, match="ordering proof"):
        run(
            [
                "--maxtext-base-yml",
                str(tmp_path / "base.yml"),
                "--output-directory",
                str(tmp_path / "output"),
            ]
        )
    assert order == ["initialize", "topology"]


def test_runner_samples_memory_before_duration_and_evidence_gathers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    order: list[str] = []
    config = SimpleNamespace()
    mesh = object()
    batch = object()
    compiled = lambda value: value

    monkeypatch.setattr(
        "moe_sae.r1.prefix_benchmark_runner.initialize_v6e_distributed_runtime",
        lambda: None,
    )
    monkeypatch.setattr(
        "moe_sae.r1.prefix_benchmark_runner.assert_real_v6e64_runtime",
        lambda: None,
    )
    monkeypatch.setattr(
        "moe_sae.r1.prefix_benchmark_runner.initialize_maxtext_config",
        lambda _settings: config,
    )
    monkeypatch.setattr(
        "moe_sae.r1.prefix_benchmark_runner.create_prefix_mesh",
        lambda _config: mesh,
    )
    monkeypatch.setattr(
        "moe_sae.r1.prefix_benchmark_runner._physical_mesh_evidence",
        lambda _mesh, _config: {},
    )
    monkeypatch.setattr(
        "moe_sae.r1.prefix_benchmark_runner.create_synthetic_prefix_batch",
        lambda _mesh: batch,
    )
    monkeypatch.setattr(
        "moe_sae.r1.prefix_benchmark_runner.create_compile_synthetic_prefix",
        lambda *_args, **_kwargs: compiled,
    )
    def duration_gather(durations: tuple[int, ...]) -> tuple[int, ...]:
        order.append(
            "measurement-duration-gather"
            if len(durations) == 500
            else "prepare-duration-gather"
        )
        return durations

    monkeypatch.setattr(
        "moe_sae.r1.prefix_benchmark_runner.reduce_max_durations_across_jax_processes",
        duration_gather,
    )
    monkeypatch.setattr(
        "moe_sae.r1.prefix_benchmark_runner._validate_output",
        lambda _output: {},
    )

    def benchmark(*_args: object, **kwargs: object) -> SimpleNamespace:
        order.append("final-synchronized-step")
        reducer = kwargs["reduce_max_durations_ns"]
        assert callable(reducer)
        assert reducer((160_000_000,) * 500) == (160_000_000,) * 500
        order.append("benchmark-return")
        return SimpleNamespace(as_dict=dict)

    sampled_words = object()

    def sample_memory() -> object:
        order.append("device-memory-sample")
        return sampled_words

    def gather_memory(words: object) -> dict[str, object]:
        assert words is sampled_words
        order.append("device-memory-evidence-gather")
        return {"passes_15_percent_headroom": True}

    monkeypatch.setattr(
        "moe_sae.r1.prefix_benchmark_runner.run_synchronized_prefix_benchmark",
        benchmark,
    )
    monkeypatch.setattr(
        "moe_sae.r1.prefix_benchmark_runner._sample_local_device_memory_words",
        sample_memory,
    )
    monkeypatch.setattr(
        "moe_sae.r1.prefix_benchmark_runner._gather_device_memory_evidence",
        gather_memory,
    )

    payload = run(
        [
            "--maxtext-base-yml",
            str(tmp_path / "base.yml"),
            "--output-directory",
            str(tmp_path / "output"),
        ]
    )

    assert order == [
        "prepare-duration-gather",
        "final-synchronized-step",
        "device-memory-sample",
        "measurement-duration-gather",
        "benchmark-return",
        "device-memory-evidence-gather",
    ]
    assert payload is not None
    assert payload["schema"] == "moe-sae.r1.synthetic-prefix-qualification.v2"
    assert payload["device_memory"] == {"passes_15_percent_headroom": True}
