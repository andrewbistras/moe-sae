from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = REPO_ROOT / "scripts" / "tpu"
SCRIPTS = (
    SCRIPT_ROOT / "bootstrap_worker.sh",
    SCRIPT_ROOT / "deploy_workers.sh",
    SCRIPT_ROOT / "inspect_workers.sh",
)


@pytest.mark.parametrize("script", SCRIPTS)
def test_tpu_worker_script_is_executable_and_valid_bash(script: Path) -> None:
    assert os.access(script, os.X_OK)
    subprocess.run(["bash", "-n", str(script)], check=True)


@pytest.mark.parametrize("script", SCRIPTS)
def test_tpu_worker_script_help_is_offline(script: Path) -> None:
    completed = subprocess.run(
        [str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert completed.returncode == 0, completed.stderr
    assert "Usage:" in completed.stdout


def test_deploy_refuses_to_infer_required_cloud_inputs() -> None:
    env = {
        "HOME": os.environ.get("HOME", "/tmp"),
        "PATH": os.environ.get("PATH", ""),
    }
    completed = subprocess.run(
        [str(SCRIPT_ROOT / "deploy_workers.sh")],
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert completed.returncode != 0
    assert "required environment variable is unset: GCP_PROJECT" in completed.stderr


def test_bootstrap_contains_no_disk_mutation_or_secret_inputs() -> None:
    bootstrap = (SCRIPT_ROOT / "bootstrap_worker.sh").read_text()
    deploy = (SCRIPT_ROOT / "deploy_workers.sh").read_text()
    forbidden_disk_commands = ("mkfs", "wipefs", "fdisk", "parted")
    assert not any(command in bootstrap for command in forbidden_disk_commands)
    assert "source \"$config_file\"" not in bootstrap
    assert "TOKEN" not in deploy
    assert "PASSWORD" not in deploy


def test_bootstrap_gates_activation_capacity_before_and_after_dependencies() -> None:
    bootstrap = (SCRIPT_ROOT / "bootstrap_worker.sh").read_text()
    assert "readonly MIN_LOCAL_FREE_BYTES=25769803776" in bootstrap
    assert "137438953472" not in bootstrap
    main_body = bootstrap[bootstrap.index("main() {") :]
    before = main_body.index("prepare_local_buffer")
    install = main_body.index("install_python_environment", before)
    after = main_body.index(
        'require_free_bytes "$LOCAL_BUFFER_ROOT" "$MIN_LOCAL_FREE_BYTES"',
        install,
    )
    assert before < install < after
