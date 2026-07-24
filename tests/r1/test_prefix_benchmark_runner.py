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
    _mesh_argument,
    _physical_mesh_evidence,
    _seed_argument,
    _validate_output,
    parse_args,
    run,
)


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
