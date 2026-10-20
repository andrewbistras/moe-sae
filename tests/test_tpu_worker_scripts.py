from __future__ import annotations

import os
import subprocess
from pathlib import Path

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


def test_deploy_requires_explicit_internal_ip_for_regional_slices() -> None:
    deploy = (SCRIPT_ROOT / "deploy_workers.sh").read_text()
    assert "TPU_USE_INTERNAL_IP=${TPU_USE_INTERNAL_IP:-0}" in deploy
    assert '"$TPU_ACCELERATOR_TYPE" == "v6e-16"' in deploy
    assert '"$TPU_ACCELERATOR_TYPE" == "v5litepod-16"' in deploy
    assert (
        "$TPU_ACCELERATOR_TYPE deployment requires explicit "
        "TPU_USE_INTERNAL_IP=1"
    ) in deploy
    assert "gcloud_common+=(--internal-ip)" in deploy
    assert "ssh_command+=(--internal-ip)" in deploy
    inspect = (SCRIPT_ROOT / "inspect_workers.sh").read_text()
    assert "TPU_USE_INTERNAL_IP=${TPU_USE_INTERNAL_IP:-0}" in inspect
    assert "command+=(--internal-ip)" in inspect


def test_worker_operators_use_one_validated_explicit_remote_username() -> None:
    deploy = (SCRIPT_ROOT / "deploy_workers.sh").read_text()
    inspect = (SCRIPT_ROOT / "inspect_workers.sh").read_text()
    for source in (deploy, inspect):
        assert "TPU_SSH_USER=${TPU_SSH_USER:-}" in source
        assert "TPU_SSH_USER must be a safe lowercase Linux username" in source
        assert 'TPU_SSH_TARGET="$TPU_SSH_USER@$TPU_NAME"' in source
        assert 'tpu-vm ssh "$TPU_SSH_TARGET"' in source
    assert '"$TPU_SSH_TARGET:~/.cache/moe-sae-deploy/$deployment_id/"' in deploy


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


def test_bootstrap_enables_and_exactly_verifies_deployment_user_linger() -> None:
    bootstrap = (SCRIPT_ROOT / "bootstrap_worker.sh").read_text()
    function_start = bootstrap.index("ensure_deployment_user_linger() {")
    function_body = bootstrap[
        function_start : bootstrap.index(
            "\n}\n\nverify_worker_identity()", function_start
        )
    ]

    assert "deployment_user=$(id -un)" in function_body
    assert 'sudo -n loginctl enable-linger "$deployment_user"' in function_body
    assert 'loginctl show-user "$deployment_user" -p Linger --value' in function_body
    assert '[[ "$linger" == "yes" ]]' in function_body


def test_bootstrap_enables_linger_before_local_buffer_capability() -> None:
    bootstrap = (SCRIPT_ROOT / "bootstrap_worker.sh").read_text()
    main_body = bootstrap[bootstrap.index("main() {") :]

    packages = main_body.index("install_system_packages")
    linger = main_body.index("ensure_deployment_user_linger")
    local_buffer = main_body.index("prepare_local_buffer")
    assert packages < linger < local_buffer


def test_bootstrap_linger_setup_is_idempotent_when_status_is_exact_yes() -> None:
    completed = _run_bootstrap_function(
        "id() { printf 'deployment-user\\n'; }; "
        "sudo() { return 0; }; "
        "loginctl() { printf 'yes\\n'; }; "
        "ensure_deployment_user_linger; ensure_deployment_user_linger"
    )

    assert completed.returncode == 0, completed.stderr


def test_bootstrap_rejects_nonexact_linger_status() -> None:
    completed = _run_bootstrap_function(
        "id() { printf 'deployment-user\\n'; }; "
        "sudo() { return 0; }; "
        "loginctl() { printf 'yes \\n'; }; "
        "ensure_deployment_user_linger"
    )

    assert completed.returncode != 0
    assert "expected exactly 'yes'" in completed.stderr


def test_bootstrap_disables_all_nfs_attribute_and_lookup_caching() -> None:
    bootstrap = (SCRIPT_ROOT / "bootstrap_worker.sh").read_text()

    assert "rw,hard,timeo=600,retrans=2,noac,lookupcache=none" in bootstrap
    assert 'sudo umount "$artifact_root"' in bootstrap
    assert "remount,noac,lookupcache=none" not in bootstrap
    assert 'has_correctness_first_nfs_options "$mount_options" ||' in bootstrap


def test_bootstrap_requires_both_correctness_first_nfs_cache_options() -> None:
    for safe in (
        "rw,hard,noac,lookupcache=none",
        "rw,hard,noac,lookupcache=none,local_lock=none",
    ):
        accepted = _run_bootstrap_function(
            f"has_correctness_first_nfs_options '{safe}'"
        )
        assert accepted.returncode == 0, accepted.stderr

    for unsafe in (
        "rw,hard,lookupcache=none",
        "rw,hard,noac,lookupcache=positive",
        "rw,hard,noac,lookupcache=pos",
        "rw,hard,noac,lookupcache=all",
        "rw,soft,noac,lookupcache=none",
        "rw,hard,soft,noac,lookupcache=none",
        "rw,hard,softerr,noac,lookupcache=none",
        "rw,hard,nolock,noac,lookupcache=none",
        "rw,hard,noac,lookupcache=none,local_lock=all",
        "rw,hard,noac,lookupcache=none,local_lock=flock",
    ):
        rejected = _run_bootstrap_function(
            f"has_correctness_first_nfs_options '{unsafe}'"
        )
        assert rejected.returncode != 0


def _run_bootstrap_function(command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; eval "$2"',
            "bootstrap-test",
            str(SCRIPT_ROOT / "bootstrap_worker.sh"),
            command,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )


def test_bootstrap_accepts_only_the_dedicated_dev_shm_tmpfs() -> None:
    accepted = _run_bootstrap_function(
        "validate_local_buffer_filesystem "
        "/dev/shm/moe-sae-local /dev/shm tmpfs tmpfs rw,nosuid,nodev,inode64"
    )
    assert accepted.returncode == 0, accepted.stderr

    wrong_mount = _run_bootstrap_function(
        "validate_local_buffer_filesystem "
        "/run/moe-sae-local /run tmpfs tmpfs rw,nosuid,nodev"
    )
    assert wrong_mount.returncode != 0
    assert "strictly below /dev/shm" in wrong_mount.stderr

    unsafe_options = _run_bootstrap_function(
        "validate_local_buffer_filesystem "
        "/dev/shm/moe-sae-local /dev/shm tmpfs tmpfs rw,nosuid"
    )
    assert unsafe_options.returncode != 0
    assert "must be mounted nodev" in unsafe_options.stderr


def test_bootstrap_keeps_local_disk_filesystems_eligible() -> None:
    completed = _run_bootstrap_function(
        "validate_local_buffer_filesystem "
        "/home/user/moe-sae-local / /dev/sda1 ext4 rw,relatime"
    )
    assert completed.returncode == 0, completed.stderr
