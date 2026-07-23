from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from moe_sae.r1.prefix_benchmark_runner import (
    PrefixBenchmarkRunnerError,
    _atomic_json,
    _mesh_argument,
    _seed_argument,
    _validate_output,
    parse_args,
)


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
