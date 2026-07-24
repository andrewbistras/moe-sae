from __future__ import annotations

import hashlib
import json
import re
import runpy
import subprocess
from fractions import Fraction
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAP = ROOT / "scripts/r1/bootstrap_legacy_v6e_worker.sh"
CONTROLLER = ROOT / "scripts/r1/qualify_legacy_v6e.sh"
VALIDATOR = ROOT / "scripts/r1/validate_legacy_v6e_result.py"


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _function(source: str, name: str) -> str:
    match = re.search(
        rf"(?ms)^{re.escape(name)}\(\) \{{\n(?P<body>.*?)^\}}\n",
        source,
    )
    assert match is not None
    return match.group("body")


def test_legacy_scripts_are_executable_bash_with_valid_syntax() -> None:
    for path in (BOOTSTRAP, CONTROLLER):
        assert path.stat().st_mode & 0o111
        assert _source(path).startswith("#!/usr/bin/env bash\n")
        subprocess.run(["bash", "-n", str(path)], check=True)


def test_worker_bootstrap_is_generation_pinned_and_has_no_discovery_read() -> None:
    source = _source(BOOTSTRAP)
    assert (
        'readonly RUNTIME_URI="gs://neural-dragnet-r1-euw4-655343590563/'
        "runtime/6129738baad8d2bb4060c6e19b8a9227341afe9c795304f70fdf3206155f45c9"
        '.tar.gz"'
    ) in source
    assert 'readonly RUNTIME_GENERATION="1784862130885035"' in source
    assert (
        'readonly RUNTIME_SHA256="'
        "6129738baad8d2bb4060c6e19b8a9227341afe9c795304f70fdf3206155f45c9"
        '"'
    ) in source
    assert 'readonly RUNTIME_SIZE_BYTES="1378895492"' in source
    assert 'readonly RUNTIME_MEMBER_COUNT="69566"' in source
    assert (
        'readonly RUNTIME_COMMIT="'
        "0b01bc2b76e375271249eb6f7e0499d5b016a1da"
        '"'
    ) in source
    assert '"${RUNTIME_URI}#${RUNTIME_GENERATION}"' in source
    assert "gcloud storage ls" not in source
    assert "gcloud storage objects list" not in source
    assert "gcloud storage cp" in source


def test_worker_fast_path_precedes_the_only_gcs_get() -> None:
    body = _function(_source(BOOTSTRAP), "main")
    receipt = body.index("if receipt_is_exact; then")
    no_gcs = body.index("no GCS read")
    copy = body.index("gcloud storage cp")
    size = body.index('stat -c %s "$archive_path"')
    digest = body.index('sha256_file "$archive_path"')
    gzip = body.index('gzip --test "$archive_path"')
    extraction = body.index("validate_and_extract_archive")
    publish = body.rindex("publish_receipt")
    assert receipt < no_gcs < copy < size < digest < gzip < extraction < publish
    assert body.count("gcloud storage cp") == 1
    assert "if pending_is_exact; then" in body
    assert body.index("if pending_is_exact; then") < copy


def test_worker_archive_validator_rejects_escape_and_special_members() -> None:
    source = _source(BOOTSTRAP)
    assert "expected_count" in source
    assert "maximum_total" in source
    assert "archive member escapes the allowlist" in source
    assert "member.islnk()" in source
    assert "member.ischr()" in source
    assert "member.isblk()" in source
    assert "member.isfifo()" in source
    assert "archive member descends through symlink" in source
    assert 'resolved.startswith("/opt/moe-sae-r1/")' in source
    assert "member.replace(" in source
    assert "uid=None" in source
    assert "gid=None" in source
    assert "archive.extractall(" in source


def test_worker_installs_opt_atomically_and_publishes_host_receipt_last() -> None:
    source = _source(BOOTSTRAP)
    body = _function(source, "main")
    assert '[[ ! -e "$RUNTIME_ROOT" ]]' in body
    assert 'mv "$staging_root/opt/moe-sae-r1" "$RUNTIME_ROOT"' in body
    install = body.index('mv "$staging_root/opt/moe-sae-r1" "$RUNTIME_ROOT"')
    verify = body.index("verify_runtime_tree", install)
    publish = body.index("publish_receipt", verify)
    assert install < verify < publish
    assert "os.replace(path, destination)" in source
    assert "os.fsync(directory)" in source
    assert '"schema": "moe-sae.r1.legacy-runtime-install.v1"' in source
    assert "/usr/local/lib/moe-sae-r1/image-observation.json" not in source


def test_worker_recovers_both_sides_of_the_pending_rename() -> None:
    source = _source(BOOTSTRAP)
    body = _function(source, "main")
    assert "cleanup_abandoned_staging" in body
    assert body.index("cleanup_abandoned_staging") < body.index("if receipt_is_exact")
    pending = body.index("if pending_is_exact; then")
    runtime = body.index('if [[ -e "$RUNTIME_ROOT" ]]; then', pending)
    recover = body.index("publish_receipt", runtime)
    restart = body.index("discarded an exact pre-rename journal", recover)
    download = body.index("gcloud storage cp", restart)
    assert pending < runtime < recover < restart < download
    cleanup = _function(source, "cleanup_abandoned_staging")
    assert "/opt/.moe-sae-r1-runtime.*.tar.gz" in cleanup
    assert "/opt/.moe-sae-r1-stage.*" in cleanup
    assert "rm -rf --one-file-system" in cleanup


def test_worker_accepts_no_remote_command_or_credential_input() -> None:
    source = _source(BOOTSTRAP)
    assert '[[ "$#" == 0 ]]' in source
    assert "eval " not in source
    assert re.search(r"(?m)^[ \t]*(?:source|\.)[ \t]+", source) is None
    assert "bash -c" not in source
    assert "Authorization:" not in source
    assert "token" not in source.lower()
    assert source.count("curl \\\n") == 2
    assert source.count("$METADATA_ROOT/") == 2


def test_legacy_controller_never_creates_and_deletes_only_bound_slice() -> None:
    source = _source(CONTROLLER)
    assert "queued-resources describe" in source
    assert "tpu-vm describe" in source
    assert "queued-resources create" not in source
    assert "tpu-vm create" not in source
    assert "tpu-vm delete" not in source
    assert source.count('queued-resources delete "$QUEUED_RESOURCE"') == 2
    assert "--force" in source
    assert "SLICE_HARD_DEADLINE_SECONDS=\"7200\"" in source
    assert "persistent formation watchdog is active before resource discovery" in source
    assert "Restart=on-failure" in source
    assert "WantedBy=multi-user.target" in source
    assert 'systemctl enable --now "${REAPER_UNIT}.service"' in source
    assert "write_reaper_deadline_if_earlier" in source
    assert "queued-resource absence independently confirmed" in source
    assert "EXPECTED_ACCELERATOR=\"v6e-64\"" in source
    assert "EXPECTED_RUNTIME_VERSION=\"v6e-ubuntu-2404\"" in source
    assert "EXPECTED_SERVICE_ACCOUNT=" in source
    assert "enableExternalIps == false" in source
    assert "(.networkEndpoints | length) == 16" in source


def test_controller_binds_incarnation_and_isolates_reaper_snapshots() -> None:
    source = _source(CONTROLLER)
    assert 'readonly RESOURCE_CREATE_TIME_PATH=' in source
    assert ".createTime" in source
    assert "bind_or_validate_resource_incarnation" in source
    assert source.count("queued_resource_identity_is_exact") >= 6
    assert 'snapshot_suffix=\'.reaper\'' in source
    assert "resource name now refers to a different identity" in source
    assert "fresh release identity changed; refusing deletion" in source
    assert "attempt identity is already sealed released" in source


def test_controller_derives_every_retry_identity_from_one_sequence() -> None:
    source = _source(CONTROLLER)
    assert 'readonly ATTEMPT_SEQUENCE="${ATTEMPT_SEQUENCE:-3}"' in source
    assert (
        'readonly QUEUED_RESOURCE="r1-v6e64-spot-euw4a-a${ATTEMPT_SEQUENCE}"'
        in source
    )
    assert 'readonly TPU_NAME="${QUEUED_RESOURCE}-node"' in source
    assert (
        'readonly ATTEMPT="euw4a-legacy-a${ATTEMPT_SEQUENCE}-ep8-fsdp8"'
        in source
    )
    assert "fsdp8-euw4a-a${ATTEMPT_SEQUENCE}" in source
    assert "Environment=ATTEMPT_SEQUENCE=${ATTEMPT_SEQUENCE}" in source
    assert "require_positive_integer \"$ATTEMPT_SEQUENCE\"" in source


def test_legacy_controller_fans_out_fixed_systemd_benchmark() -> None:
    source = _source(CONTROLLER)
    assert "for worker in $(seq 0 15)" in source
    assert "--internal-ip" in source
    assert "--worker=\"$worker\"" in source
    assert "--property=RuntimeDirectory=${UNIT}" in source
    assert "--property=RuntimeMaxSec=${BENCHMARK_TIMEOUT_SECONDS}" in source
    assert "--property=KillMode=control-group" in source
    assert "--property=LimitMEMLOCK=infinity" in source
    assert "--property=LimitNOFILE=100000" in source
    assert "prefix_benchmark_runner" in source
    assert "--mesh ${MESH}" in source
    assert 'readonly MESH="EP8/FSDP8"' in source
    assert "legacy bridge is identity-bound to EP8/FSDP8" in source
    assert "all 16 TPU SSH identities are ready" in source
    assert "exact runtime bootstrap" in source
    assert 'WORKER_BOOTSTRAP_TIMEOUT_SECONDS + 180' in source
    assert "queued-resource left ACTIVE during benchmark" in source
    assert "controller exited before the qualification seal" in source
    assert ">=51% qualification completed" in source


def test_controller_binds_exact_bootstrap_and_never_uploads_evidence() -> None:
    source = _source(CONTROLLER)
    bootstrap_source = _source(BOOTSTRAP).encode()

    expected = hashlib.sha256(bootstrap_source).hexdigest()
    assert f'readonly EXPECTED_BOOTSTRAP_SHA256="{expected}"' in source
    assert "sha256sum --check --status" in source
    assert "gcloud storage cp" not in source
    assert "gcloud storage rsync" not in source
    assert "journalctl" in source
    assert "SHA256SUMS" in source


def _fraction(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _qualification(*, duration_ns: int) -> dict[str, object]:
    steps = 500
    tokens = 128 * 1_024
    flops = 4_900_569_495_896_064
    peak = 58_752_000_000_000_000
    durations = [duration_ns] * steps
    total_duration = duration_ns * steps
    aggregate = Fraction(flops * steps * 1_000_000_000, peak * total_duration)
    rate = Fraction(tokens * steps * 1_000_000_000, total_duration)
    devices = []
    for fsdp in range(8):
        for expert in range(8):
            device_id = fsdp * 8 + expert
            devices.append(
                {
                    "logical_index": [0, fsdp, expert, 0],
                    "device_id": device_id,
                    "process_index": device_id // 4,
                    "physical_coordinates": [fsdp, expert, 0],
                }
            )
    return {
        "schema": "moe-sae.r1.synthetic-prefix-qualification.v1",
        "mesh": "EP8/FSDP8",
        "seed": 0,
        "process_count": 16,
        "device_count": 64,
        "physical_mesh": {
            "axis_names": ["data", "fsdp", "expert", "context"],
            "logical_shape": [1, 8, 8, 1],
            "allow_split_physical_axes": False,
            "devices": devices,
        },
        "prepare_seconds": 123.5,
        "output_validation": {
            "shapes": [
                [128, 1_024, 7_168],
                [128, 1_024, 8],
                [128, 1_024, 8],
            ],
            "dtypes": ["bfloat16", "int32", "float32"],
            "activation_finite": True,
            "ids_in_range": True,
            "ids_unique": True,
            "weights_finite": True,
            "weights_positive": True,
            "weights_sum_scaled": True,
        },
        "benchmark": {
            "schema": "moe-sae.r1.prefix-benchmark-result.v1",
            "warmup_steps": 5,
            "measurement_steps": steps,
            "valid_tokens_per_step": tokens,
            "useful_flops_per_step": flops,
            "peak_flops_per_second": peak,
            "durations_ns": durations,
            "total_duration_ns": total_duration,
            "total_valid_tokens": tokens * steps,
            "total_useful_flops": flops * steps,
            "aggregate_mfu": _fraction(aggregate),
            "tokens_per_second": _fraction(rate),
            "median_step_ns": duration_ns,
            "p95_step_ns": duration_ns,
            "passes_51_percent": aggregate >= Fraction(51, 100),
        },
    }


def _validator_namespace() -> dict[str, object]:
    return runpy.run_path(str(VALIDATOR), run_name="legacy_result_validator")


def test_result_validator_accepts_only_derived_ep8_result(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    path.write_text(json.dumps(_qualification(duration_ns=150_000_000)))
    namespace = _validator_namespace()
    summary = namespace["validate_result"](path)  # type: ignore[operator]
    assert summary["mesh"] == "EP8/FSDP8"
    assert summary["passes_51_percent"] is True
    assert summary["aggregate_mfu"] > 0.51


def test_result_validator_rejects_low_mfu_and_derived_tampering(
    tmp_path: Path,
) -> None:
    namespace = _validator_namespace()
    error = namespace["ResultValidationError"]
    low = tmp_path / "low.json"
    low.write_text(json.dumps(_qualification(duration_ns=200_000_000)))
    with pytest.raises(error, match="below 51%"):  # type: ignore[arg-type]
        namespace["validate_result"](low)  # type: ignore[operator]

    tampered_payload = _qualification(duration_ns=150_000_000)
    tampered_payload["benchmark"]["aggregate_mfu"] = {  # type: ignore[index]
        "numerator": 99,
        "denominator": 100,
    }
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(tampered_payload))
    with pytest.raises(error, match="aggregate MFU is not derived"):  # type: ignore[arg-type]
        namespace["validate_result"](tampered)  # type: ignore[operator]


def test_result_validator_rejects_topology_and_output_drift(
    tmp_path: Path,
) -> None:
    namespace = _validator_namespace()
    error = namespace["ResultValidationError"]
    payload = _qualification(duration_ns=150_000_000)
    payload["physical_mesh"]["devices"][63]["process_index"] = 0  # type: ignore[index]
    path = tmp_path / "wrong-process.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(error, match="exactly four devices"):  # type: ignore[arg-type]
        namespace["validate_result"](path)  # type: ignore[operator]

    payload = _qualification(duration_ns=150_000_000)
    payload["output_validation"]["ids_unique"] = False  # type: ignore[index]
    path = tmp_path / "bad-output.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(error, match="ids_unique"):  # type: ignore[arg-type]
        namespace["validate_result"](path)  # type: ignore[operator]


def test_result_validator_rejects_float_identities_and_impossible_mfu(
    tmp_path: Path,
) -> None:
    namespace = _validator_namespace()
    error = namespace["ResultValidationError"]

    payload = _qualification(duration_ns=150_000_000)
    payload["process_count"] = 16.0
    path = tmp_path / "float-process-count.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(error, match="result process count"):  # type: ignore[arg-type]
        namespace["validate_result"](path)  # type: ignore[operator]

    impossible = _qualification(duration_ns=50_000_000)
    path = tmp_path / "impossible-mfu.json"
    path.write_text(json.dumps(impossible))
    with pytest.raises(error, match="exceeds 100%"):  # type: ignore[arg-type]
        namespace["validate_result"](path)  # type: ignore[operator]
