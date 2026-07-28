from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

from moe_sae.object_store import ObjectIdentity


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "tpu" / "prepare_regional_mirror.py"


@pytest.fixture()
def operator():
    name = "_test_prepare_regional_mirror"
    specification = importlib.util.spec_from_file_location(name, SCRIPT)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def identity() -> ObjectIdentity:
    return ObjectIdentity(
        uri="gs://origin/capture/shard.safetensors",
        generation=123,
        size_bytes=456,
        crc32c="AAAAAA==",
        sha256="a" * 64,
    )


def test_copy_command_is_source_pinned_and_destination_create_only(operator):
    command = operator._copy_command(
        project="neural-dragnet",
        source=identity(),
        destination="gs://regional/run/objects/shard.safetensors",
    )
    assert "gs://origin/capture/shard.safetensors#123" in command
    assert "--if-generation-match=0" in command
    assert "--project=neural-dragnet" in command


def test_describe_identity_requires_exact_sha_metadata(operator):
    source = identity()
    destination = "gs://regional/run/objects/shard.safetensors"
    observed = operator._identity_from_describe(
        destination,
        source,
        {
            "generation": "999",
            "size": "456",
            "crc32c": "AAAAAA==",
            "metadata": {"sha256": "a" * 64},
        },
    )
    assert observed.generation == 999
    assert observed.sha256 == source.sha256

    with pytest.raises(operator.MirrorOperatorError, match="differs"):
        operator._identity_from_describe(
            destination,
            source,
            {
                "generation": "999",
                "size": "456",
                "crc32c": "AAAAAA==",
                "metadata": {"sha256": "b" * 64},
            },
        )


def test_copy_and_seal_requires_explicit_apply(operator, tmp_path: Path):
    args = operator._parser().parse_args(
        [
            "copy-and-seal",
            "--project",
            "neural-dragnet",
            "--region",
            "us-east1",
            "--plan",
            str(tmp_path / "plan.json"),
            "--output",
            str(tmp_path / "seal.json"),
        ]
    )
    with pytest.raises(operator.MirrorOperatorError, match="--apply"):
        operator._copy_and_seal(args)


def test_help_is_offline():
    import subprocess

    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert completed.returncode == 0, completed.stderr
    assert "copy-and-seal" in completed.stdout
