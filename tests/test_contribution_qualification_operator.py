from __future__ import annotations

import argparse
import importlib.util
import json
import os
import py_compile
import sys
from pathlib import Path

import numpy as np
import pytest

from moe_sae.contribution_artifacts import (
    HIDDEN_SIZE,
    build_recomposition_sidecar,
    canonical_sha256,
)
from moe_sae.contribution_config import MAXTEXT_FORK_COMMIT
from moe_sae.contribution_recomposition import SUPPORTED_ACCUMULATION_ORDER

ROOT = Path(__file__).resolve().parents[1]
OPERATOR_PATH = ROOT / "scripts" / "tpu" / "qualify_contribution.py"
DECIDER_PATH = (
    ROOT / "scripts" / "tpu" / "contribution_qualification_decide.py"
)
CONFIG_PATH = ROOT / "configs" / "gemma4_contribution_v1.yaml"


def _load(path: Path, name: str):
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def operator():
    return _load(OPERATOR_PATH, "_test_contribution_qualification_operator")


@pytest.fixture(scope="module")
def decider():
    return _load(DECIDER_PATH, "_test_contribution_qualification_decider")


def _settings(operator, tmp_path: Path):
    return operator.Settings(
        gcp_project="neural-dragnet",
        tpu_name="gemma4-v4-32",
        tpu_zone="us-central2-b",
        project_commit="a" * 40,
        project_root="/home/researcher/src/moe-sae",
        local_buffer_root="/dev/shm/moe-sae-local",
        artifact_root="/home/researcher/src/moe-sae/artifacts/gemma4_v1/production",
        model_checkpoint="gs://bucket/model/0/items",
        checkpoint_sha256="b" * 64,
        remote_uv="/home/researcher/.local/bin/uv",
        ssh_key_file=None,
        run_id="qualification-test",
        local_log_root=tmp_path / "logs",
    )


def _sidecar(fingerprint: str = "e" * 64) -> dict:
    scale = np.full((HIDDEN_SIZE,), 0x3F80, dtype=np.uint16)
    return build_recomposition_sidecar(
        checkpoint_sha256="b" * 64,
        maxtext_fork_commit=MAXTEXT_FORK_COMMIT,
        source_fingerprints={"environment": fingerprint},
        routed_rmsnorm_scale=scale,
        shared_rmsnorm_scale=scale,
        outer_ffn_rmsnorm_scale=scale,
        layer_scalar=np.asarray([0x3F80], dtype=np.uint16),
        accumulation_order=SUPPORTED_ACCUMULATION_ORDER,
    )


def test_operator_scripts_are_executable_and_compile(tmp_path: Path) -> None:
    for index, path in enumerate((OPERATOR_PATH, DECIDER_PATH)):
        assert os.access(path, os.X_OK)
        py_compile.compile(
            str(path),
            cfile=str(tmp_path / f"operator-{index}.pyc"),
            doraise=True,
        )


def test_candidate_order_is_only_b128_b64_b32_on_known_best_mesh(
    operator,
    tmp_path: Path,
) -> None:
    assert operator.MESH == "ep8-fsdp2-tp1"
    assert [
        (candidate.identifier, candidate.per_device_sequence_batch)
        for candidate in operator.CANDIDATES
    ] == [
        ("b128-ep8-fsdp2", 8),
        ("b64-ep8-fsdp2", 4),
        ("b32-ep8-fsdp2", 2),
    ]

    settings = _settings(operator, tmp_path)
    for candidate in operator.CANDIDATES:
        built = [
            operator._qualification_command(settings, candidate, controller_index)
            for controller_index in range(4)
        ]
        assert [
            next(
                argument
                for argument in command
                if argument.startswith("--worker=")
            )
            for command, _, _ in built
        ] == ["--worker=0", "--worker=1", "--worker=2", "--worker=3"]
        joined = "\n".join(" ".join(command) for command, _, _ in built)
        assert "moe_sae.contribution_qualification_worker" in joined
        assert "launch-host" in joined
        assert "--maximum-seconds 600" in joined
        assert "--controller-index" in joined
        assert "--process-index" not in joined
        assert "contribution_harvest_worker" not in joined
        for controller_index, (_, evidence, _) in enumerate(built):
            assert (
                f"/controller-c{controller_index:05d}/host-evidence.json"
                in evidence
            )


def test_operator_source_has_no_workload_launch_path() -> None:
    source = OPERATOR_PATH.read_text(encoding="utf-8")
    forbidden = (
        "contribution_harvest_worker",
        "moe-sae run",
        "activation-farm",
        "launch-workload",
    )
    assert not any(value in source for value in forbidden)
    assert "stop_after_first_pass_no_workload_launch" in source


def test_plan_is_create_only_and_declares_terminal_stop(
    decider,
    tmp_path: Path,
) -> None:
    output = tmp_path / "operator-plan.json"
    args = argparse.Namespace(
        run_id="test",
        project_commit="a" * 40,
        config=str(CONFIG_PATH),
        checkpoint="gs://bucket/model/0/items",
        checkpoint_sha256="b" * 64,
        candidate=[
            "b128-ep8-fsdp2:ep8-fsdp2-tp1:8:128",
            "b64-ep8-fsdp2:ep8-fsdp2-tp1:4:64",
            "b32-ep8-fsdp2:ep8-fsdp2-tp1:2:32",
        ],
        output=str(output),
    )
    assert decider.initialize_plan(args) == 0
    plan = json.loads(output.read_text(encoding="utf-8"))
    claimed = plan.pop("operator_plan_sha256")
    assert claimed == canonical_sha256(plan)
    assert plan["terminal_action"] == "stop_after_first_pass_no_workload_launch"
    assert [item["global_sequence_batch"] for item in plan["candidate_order"]] == [
        128,
        64,
        32,
    ]
    with pytest.raises(decider.QualificationDecisionError, match="already exists"):
        decider.initialize_plan(args)


def test_identical_sidecar_requires_all_four_exact_bytes(
    decider,
    tmp_path: Path,
) -> None:
    expected = _sidecar()
    checked = []
    for index in range(4):
        host_root = tmp_path / f"host-{index:05d}"
        host_root.mkdir()
        (host_root / "recomposition-sidecar.json").write_text(
            json.dumps(expected, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        checked.append(
            (
                host_root / "host-evidence.json",
                {
                    "recomposition_sidecar_sha256": expected["sidecar_sha256"],
                    "recomposition_sidecar_canonical_sha256": canonical_sha256(
                        expected
                    ),
                },
            )
        )
    assert (
        decider._identical_recomposition_sidecar(checked)["sidecar_sha256"]
        == expected["sidecar_sha256"]
    )

    different = _sidecar("f" * 64)
    path, host = checked[-1]
    (path.parent / "recomposition-sidecar.json").write_text(
        json.dumps(different, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    host["recomposition_sidecar_sha256"] = different["sidecar_sha256"]
    host["recomposition_sidecar_canonical_sha256"] = canonical_sha256(different)
    with pytest.raises(
        decider.QualificationDecisionError,
        match="non-identical",
    ):
        decider._identical_recomposition_sidecar(checked)


def test_only_explicit_admission_messages_allow_fallback(decider) -> None:
    assert decider._FALLBACK_ADMISSION_MESSAGES == {
        "qualification observed an OOM": "explicit_oom",
        "writer backlog reached or exceeded its hard bound":
            "persistence_headroom",
        "persisted goodput is below 85% of device output goodput":
            "persistence_headroom",
        "auxiliary capture throughput is below 90% of writer-disabled":
            "persistence_headroom",
    }
    assert any(
        pattern.search("RESOURCE_EXHAUSTED: out of memory")
        for pattern in decider._OOM_PATTERNS
    )
    assert not any(
        pattern.search("checkpoint identity mismatch")
        for pattern in decider._OOM_PATTERNS
    )
