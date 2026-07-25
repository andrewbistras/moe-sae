#!/usr/bin/env python3
"""Bounded four-controller launcher for Experiment-2 contribution harvesting.

Run the public command from zsh on the operator machine.  It creates one
immutable shared attempt plan, starts the same supervised ``launch-host``
command on TPU workers 0--3 concurrently, and publishes on worker 0 only after
all four launch commands report complete immutable evidence.

The operator never lists GCS to infer progress.  Resume and monotonic phase
extension remain functions of the explicit POSIX control roots, range
receipts, and immediate-parent evidence passed to the harvest worker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO

PROCESS_COUNT = 4
GLOBAL_DEVICE_COUNT = 16
TRANSPORT_GRACE_SECONDS = 300
TERMINATION_GRACE_SECONDS = 30

ATTEMPT_PLAN_SCHEMA_VERSION = "gemma4-contribution-harvest-operator-plan-v1"
QUALIFICATION_RECEIPT_SCHEMA_VERSION = (
    "gemma4-contribution-performance-qualification-v2"
)
COORDINATOR_BUNDLE_SCHEMA_VERSION = (
    "gemma4-contribution-qualification-coordinator-bundle-v4"
)
OPERATOR_DECISION_SCHEMA_VERSION = (
    "gemma4-contribution-qualification-operator-decision-v2"
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
_SAFE_PHASE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_SAFE_MESH_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")


class OperatorError(RuntimeError):
    """The harvest operator cannot continue safely."""


@dataclass(frozen=True, slots=True)
class Settings:
    """All explicit, non-secret inputs to one four-host attempt."""

    gcp_project: str
    tpu_name: str
    tpu_zone: str
    project_commit: str
    project_root: str
    remote_uv: str
    local_buffer_root: str
    model_checkpoint: str
    checkpoint_sha256: str
    grain_files: str
    staged_manifest: str
    source_v1_root: str
    source_cache_max_bytes: int
    control_root: str
    object_prefix: str
    recomposition_sidecar: str
    qualification_receipt: str
    qualification_bundle: str
    qualification_decision: str
    maxtext_root: str | None
    ssh_key_file: str | None
    phase: str
    split: str
    attempt_id: str
    mesh: str
    per_device_sequence_batch: int
    maximum_runtime_seconds: int
    prior_control_root: str | None
    prior_phase_evidence: str | None
    local_log_root: Path

    @property
    def config_path(self) -> str:
        return f"{self.project_root}/configs/gemma4_contribution_v1.yaml"

    @property
    def attempt_root(self) -> str:
        return f"{self.control_root}/attempts/{self.attempt_id}"

    @property
    def phase_evidence(self) -> str:
        return f"{self.control_root}/phase-evidence.json"

    @property
    def local_attempt_root(self) -> str:
        return (
            f"{self.local_buffer_root}/experiment2/harvest/"
            f"{self.phase}/{self.split}/{self.attempt_id}"
        )

    def local_controller_buffer(self, controller_index: int) -> str:
        return (
            f"{self.local_attempt_root}/" f"controller-c{controller_index:05d}/buffer"
        )

    def source_cache(self, controller_index: int) -> str:
        # This path is stable across attempts so a retry does not reread every
        # immutable v1 source shard from GCS.
        return (
            f"{self.local_buffer_root}/experiment2/source-cache/"
            f"{self.split}/controller-c{controller_index:05d}"
        )

    def jax_cache(self, controller_index: int) -> str:
        # Graph cache identity is explicit in the path; host suffixes prevent
        # concurrent controllers from mutating one local cache directory.
        return (
            f"{self.local_buffer_root}/experiment2/jax-cache/"
            f"{self.project_commit}/{self.mesh}/"
            f"b{self.per_device_sequence_batch}/"
            f"controller-c{controller_index:05d}"
        )

    def supervised_evidence(self, controller_index: int) -> str:
        return (
            f"{self.attempt_root}/controller-c{controller_index:05d}/"
            "supervised-evidence.json"
        )


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise OperatorError("operator evidence is not finite canonical JSON") from error


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _required(environment: Mapping[str, str], name: str) -> str:
    value = environment.get(name, "")
    if not value:
        raise OperatorError(f"required environment variable is unset: {name}")
    if any(character in value for character in ("\x00", "\n", "\r")):
        raise OperatorError(f"{name} contains an unsafe control character")
    return value


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool):
        raise OperatorError(f"{name} must be a positive integer")
    try:
        selected = int(value)
    except (TypeError, ValueError) as error:
        raise OperatorError(f"{name} must be a positive integer") from error
    if selected <= 0 or str(selected) != str(value):
        raise OperatorError(f"{name} must be a canonical positive integer")
    return selected


def _sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise OperatorError(f"{name} must be a lowercase SHA-256")
    return value


def _absolute_remote_path(name: str, value: str) -> str:
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts or value == "/":
        raise OperatorError(f"{name} must be a non-root absolute POSIX path")
    return str(path)


def _gcs_location(name: str, value: str, *, allow_glob: bool) -> str:
    if not value.startswith("gs://") or value.count("/") < 3:
        raise OperatorError(f"{name} must be a non-root gs:// location")
    if not allow_glob and any(character in value for character in "*?[]"):
        raise OperatorError(f"{name} must not contain a glob")
    if any(character.isspace() for character in value):
        raise OperatorError(f"{name} must not contain whitespace")
    return value.rstrip("/") if not allow_glob else value


def _settings(
    environment: Mapping[str, str],
    args: argparse.Namespace,
) -> Settings:
    project = _required(environment, "GCP_PROJECT")
    tpu_name = _required(environment, "TPU_NAME")
    zone = _required(environment, "TPU_ZONE")
    commit = _required(environment, "PROJECT_COMMIT")
    checkpoint_sha256 = _required(environment, "CHECKPOINT_SHA256")
    if re.fullmatch(r"[a-z][a-z0-9-]{4,61}[a-z0-9]", project) is None:
        raise OperatorError("GCP_PROJECT is invalid")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]*", tpu_name) is None:
        raise OperatorError("TPU_NAME is invalid")
    if re.fullmatch(r"[a-z0-9-]+", zone) is None:
        raise OperatorError("TPU_ZONE is invalid")
    if _GIT_SHA_RE.fullmatch(commit) is None:
        raise OperatorError("PROJECT_COMMIT must be a full lowercase Git SHA")
    _sha256("CHECKPOINT_SHA256", checkpoint_sha256)
    if _SAFE_PHASE_RE.fullmatch(args.phase) is None:
        raise OperatorError("--phase is invalid")
    if _SAFE_ID_RE.fullmatch(args.attempt_id) is None:
        raise OperatorError("--attempt-id is invalid")
    if _SAFE_MESH_RE.fullmatch(args.mesh) is None:
        raise OperatorError("--mesh is invalid")
    per_device = _positive_int(
        "--per-device-sequence-batch",
        args.per_device_sequence_batch,
    )
    maximum_runtime = _positive_int(
        "--maximum-runtime-seconds",
        args.maximum_runtime_seconds,
    )
    prior_control = args.prior_control_root
    prior_evidence = args.prior_phase_evidence
    if (prior_control is None) != (prior_evidence is None):
        raise OperatorError(
            "--prior-control-root and --prior-phase-evidence must be supplied "
            "together"
        )
    if prior_control is not None:
        prior_control = _absolute_remote_path(
            "--prior-control-root",
            prior_control,
        )
        prior_evidence = _absolute_remote_path(
            "--prior-phase-evidence",
            prior_evidence,
        )

    local_default = (
        Path.cwd()
        / "artifacts"
        / "gemma4_v1"
        / "operator-results"
        / "experiment2-harvest"
        / args.attempt_id
    )
    local_log_root = (
        Path(
            environment.get(
                "CONTRIBUTION_LOCAL_LOG_ROOT",
                str(local_default),
            )
        )
        .expanduser()
        .resolve()
    )
    maxtext_root = environment.get("MAXTEXT_ROOT")
    if maxtext_root:
        maxtext_root = _absolute_remote_path("MAXTEXT_ROOT", maxtext_root)
    ssh_key = environment.get("SSH_KEY_FILE") or None

    return Settings(
        gcp_project=project,
        tpu_name=tpu_name,
        tpu_zone=zone,
        project_commit=commit,
        project_root=_absolute_remote_path(
            "TPU_PROJECT_ROOT",
            _required(environment, "TPU_PROJECT_ROOT"),
        ),
        remote_uv=_absolute_remote_path(
            "REMOTE_UV",
            _required(environment, "REMOTE_UV"),
        ),
        local_buffer_root=_absolute_remote_path(
            "LOCAL_BUFFER_ROOT",
            _required(environment, "LOCAL_BUFFER_ROOT"),
        ),
        model_checkpoint=_required(environment, "MODEL_CHECKPOINT"),
        checkpoint_sha256=checkpoint_sha256,
        grain_files=_gcs_location(
            "CONTRIBUTION_GRAIN_FILES",
            _required(environment, "CONTRIBUTION_GRAIN_FILES"),
            allow_glob=True,
        ),
        staged_manifest=_gcs_location(
            "CONTRIBUTION_STAGED_MANIFEST",
            _required(environment, "CONTRIBUTION_STAGED_MANIFEST"),
            allow_glob=False,
        ),
        source_v1_root=_absolute_remote_path(
            "CONTRIBUTION_SOURCE_V1_ROOT",
            _required(environment, "CONTRIBUTION_SOURCE_V1_ROOT"),
        ),
        source_cache_max_bytes=_positive_int(
            "CONTRIBUTION_SOURCE_CACHE_MAX_BYTES",
            _required(environment, "CONTRIBUTION_SOURCE_CACHE_MAX_BYTES"),
        ),
        control_root=_absolute_remote_path(
            "CONTRIBUTION_CONTROL_ROOT",
            _required(environment, "CONTRIBUTION_CONTROL_ROOT"),
        ),
        object_prefix=_gcs_location(
            "CONTRIBUTION_OBJECT_PREFIX",
            _required(environment, "CONTRIBUTION_OBJECT_PREFIX"),
            allow_glob=False,
        ),
        recomposition_sidecar=_absolute_remote_path(
            "CONTRIBUTION_RECOMPOSITION_SIDECAR",
            _required(environment, "CONTRIBUTION_RECOMPOSITION_SIDECAR"),
        ),
        qualification_receipt=_absolute_remote_path(
            "CONTRIBUTION_QUALIFICATION_RECEIPT",
            _required(environment, "CONTRIBUTION_QUALIFICATION_RECEIPT"),
        ),
        qualification_bundle=_absolute_remote_path(
            "CONTRIBUTION_QUALIFICATION_BUNDLE",
            _required(environment, "CONTRIBUTION_QUALIFICATION_BUNDLE"),
        ),
        qualification_decision=_absolute_remote_path(
            "CONTRIBUTION_QUALIFICATION_DECISION",
            _required(environment, "CONTRIBUTION_QUALIFICATION_DECISION"),
        ),
        maxtext_root=maxtext_root,
        ssh_key_file=ssh_key,
        phase=args.phase,
        split=args.split,
        attempt_id=args.attempt_id,
        mesh=args.mesh,
        per_device_sequence_batch=per_device,
        maximum_runtime_seconds=maximum_runtime,
        prior_control_root=prior_control,
        prior_phase_evidence=prior_evidence,
        local_log_root=local_log_root,
    )


def _gcloud(settings: Settings, *, worker: int, remote_command: str) -> list[str]:
    command = [
        "gcloud",
        "compute",
        "tpus",
        "tpu-vm",
        "ssh",
        settings.tpu_name,
        f"--project={settings.gcp_project}",
        f"--zone={settings.tpu_zone}",
        f"--worker={worker}",
        f"--command={remote_command}",
        "--ssh-flag=-oBatchMode=yes",
        "--ssh-flag=-oConnectTimeout=30",
        "--ssh-flag=-oConnectionAttempts=3",
        "--ssh-flag=-oServerAliveInterval=30",
        "--ssh-flag=-oServerAliveCountMax=4",
        "--quiet",
    ]
    if settings.ssh_key_file:
        command.insert(-1, f"--ssh-key-file={settings.ssh_key_file}")
    return command


def _bash_payload(lines: Sequence[str]) -> str:
    return "bash -lc " + shlex.quote("\n".join(lines))


def _clean_checkout_lines(settings: Settings) -> list[str]:
    return [
        f"cd {shlex.quote(settings.project_root)}",
        ('test "$(git rev-parse HEAD)" = ' f"{shlex.quote(settings.project_commit)}"),
        'test -z "$(git status --porcelain=v1 --untracked-files=all)"',
    ]


def _common_harvest_arguments(settings: Settings) -> list[str]:
    arguments = [
        "--config",
        settings.config_path,
        "--phase",
        settings.phase,
        "--split",
        settings.split,
        "--checkpoint",
        settings.model_checkpoint,
        "--checkpoint-sha256",
        settings.checkpoint_sha256,
        "--grain-files",
        settings.grain_files,
        "--staged-manifest",
        settings.staged_manifest,
        "--source-v1-root",
        settings.source_v1_root,
        "--source-cache-max-bytes",
        str(settings.source_cache_max_bytes),
        "--control-root",
        settings.control_root,
        "--object-prefix",
        settings.object_prefix,
        "--recomposition-sidecar",
        settings.recomposition_sidecar,
        "--qualification-receipt",
        settings.qualification_receipt,
        "--repo-root",
        settings.project_root,
        "--mesh",
        settings.mesh,
        "--per-device-sequence-batch",
        str(settings.per_device_sequence_batch),
        "--maximum-runtime-seconds",
        str(settings.maximum_runtime_seconds),
    ]
    if settings.prior_control_root is not None:
        arguments.extend(
            (
                "--prior-control-root",
                settings.prior_control_root,
                "--prior-phase-evidence",
                settings.prior_phase_evidence,
            )
        )
    if settings.maxtext_root is not None:
        arguments.extend(("--maxtext-root", settings.maxtext_root))
    return arguments


def _init_command(settings: Settings) -> list[str]:
    helper = f"{settings.project_root}/scripts/tpu/harvest_contribution.py"
    command = [
        settings.remote_uv,
        "run",
        "--frozen",
        "--extra",
        "tpu",
        "python",
        helper,
        "_init-attempt",
        "--attempt-id",
        settings.attempt_id,
        "--attempt-root",
        settings.attempt_root,
        "--project-commit",
        settings.project_commit,
        "--config",
        settings.config_path,
        "--phase",
        settings.phase,
        "--split",
        settings.split,
        "--checkpoint",
        settings.model_checkpoint,
        "--checkpoint-sha256",
        settings.checkpoint_sha256,
        "--grain-files",
        settings.grain_files,
        "--staged-manifest",
        settings.staged_manifest,
        "--source-v1-root",
        settings.source_v1_root,
        "--source-cache-max-bytes",
        str(settings.source_cache_max_bytes),
        "--control-root",
        settings.control_root,
        "--object-prefix",
        settings.object_prefix,
        "--recomposition-sidecar",
        settings.recomposition_sidecar,
        "--qualification-receipt",
        settings.qualification_receipt,
        "--qualification-bundle",
        settings.qualification_bundle,
        "--qualification-decision",
        settings.qualification_decision,
        "--mesh",
        settings.mesh,
        "--per-device-sequence-batch",
        str(settings.per_device_sequence_batch),
        "--maximum-runtime-seconds",
        str(settings.maximum_runtime_seconds),
        "--phase-evidence",
        settings.phase_evidence,
    ]
    if settings.prior_control_root is not None:
        command.extend(
            (
                "--prior-control-root",
                settings.prior_control_root,
                "--prior-phase-evidence",
                settings.prior_phase_evidence,
            )
        )
    if settings.maxtext_root is not None:
        command.extend(("--maxtext-root", settings.maxtext_root))
    return _gcloud(
        settings,
        worker=0,
        remote_command=_bash_payload(
            [
                "set -eu",
                *_clean_checkout_lines(settings),
                shlex.join(command),
            ]
        ),
    )


def _host_command(
    settings: Settings,
    controller_index: int,
) -> tuple[list[str], str]:
    local_buffer = settings.local_controller_buffer(controller_index)
    source_cache = settings.source_cache(controller_index)
    jax_cache = settings.jax_cache(controller_index)
    evidence = settings.supervised_evidence(controller_index)
    command = [
        settings.remote_uv,
        "run",
        "--frozen",
        "--extra",
        "tpu",
        "python",
        "-m",
        "moe_sae.contribution_harvest_worker",
        "launch-host",
        *_common_harvest_arguments(settings),
        "--source-cache-root",
        source_cache,
        "--local-buffer-root",
        local_buffer,
        "--jax-cache-dir",
        jax_cache,
        "--controller-index",
        str(controller_index),
        "--attempt-root",
        settings.attempt_root,
        "--termination-grace-seconds",
        str(TERMINATION_GRACE_SECONDS),
    ]
    lines = [
        "set -eu",
        *_clean_checkout_lines(settings),
        f"test -s {shlex.quote(settings.attempt_root + '/operator-plan.json')}",
        (
            "install -d -m 0700 "
            f"{shlex.quote(local_buffer)} "
            f"{shlex.quote(source_cache)} "
            f"{shlex.quote(jax_cache)}"
        ),
        "set +e",
        shlex.join(command),
        "harvest_status=$?",
        "set -e",
        (
            f"if test ! -s {shlex.quote(evidence)}; then "
            'echo "missing immutable supervised host evidence" >&2; '
            "exit 91; fi"
        ),
        (
            'printf "terminal phase=%s split=%s controller=%s status=%s\\n" '
            f"{shlex.quote(settings.phase)} "
            f"{shlex.quote(settings.split)} "
            f'{controller_index} "$harvest_status"'
        ),
        'exit "$harvest_status"',
    ]
    return (
        _gcloud(
            settings,
            worker=controller_index,
            remote_command=_bash_payload(lines),
        ),
        evidence,
    )


def _publish_command(
    settings: Settings,
    evidence_paths: Sequence[str],
) -> list[str]:
    if len(evidence_paths) != PROCESS_COUNT:
        raise OperatorError("publication requires exactly four evidence paths")
    command = [
        settings.remote_uv,
        "run",
        "--frozen",
        "--extra",
        "tpu",
        "python",
        "-m",
        "moe_sae.contribution_harvest_worker",
        "publish",
        "--config",
        settings.config_path,
        "--phase",
        settings.phase,
        "--split",
        settings.split,
        "--control-root",
        settings.control_root,
        "--source-v1-root",
        settings.source_v1_root,
        "--source-cache-root",
        settings.source_cache(0),
        "--source-cache-max-bytes",
        str(settings.source_cache_max_bytes),
        "--object-prefix",
        settings.object_prefix,
        "--recomposition-sidecar",
        settings.recomposition_sidecar,
    ]
    for path in evidence_paths:
        command.extend(("--host-evidence", path))
    command.extend(("--phase-evidence", settings.phase_evidence))
    return _gcloud(
        settings,
        worker=0,
        remote_command=_bash_payload(
            [
                "set -eu",
                *_clean_checkout_lines(settings),
                *[f"test -s {shlex.quote(path)}" for path in evidence_paths],
                shlex.join(command),
                f"test -s {shlex.quote(settings.control_root + '/manifest.json')}",
                f"test -s {shlex.quote(settings.phase_evidence)}",
            ]
        ),
    )


def _read_json(path: Path, *, name: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OperatorError(f"cannot read {name}: {path}") from error
    if not isinstance(value, dict):
        raise OperatorError(f"{name} must be a JSON object")
    return value


def _validate_self_hash(
    value: Mapping[str, object],
    *,
    hash_field: str,
    name: str,
) -> tuple[dict[str, object], str]:
    unsigned = dict(value)
    claimed = _sha256(name, unsigned.pop(hash_field, None))
    if _canonical_sha256(unsigned) != claimed:
        raise OperatorError(f"{name} self-hash is invalid")
    return unsigned, claimed


def _qualification_linkage(args: argparse.Namespace) -> dict[str, object]:
    receipt = _read_json(Path(args.qualification_receipt), name="qualification receipt")
    receipt_unsigned, receipt_sha256 = _validate_self_hash(
        receipt,
        hash_field="receipt_sha256",
        name="qualification receipt",
    )
    if (
        receipt_unsigned.get("schema_version") != QUALIFICATION_RECEIPT_SCHEMA_VERSION
        or receipt_unsigned.get("status") != "passed"
    ):
        raise OperatorError("qualification receipt is not a passing v2 receipt")

    sidecar = _read_json(
        Path(args.recomposition_sidecar),
        name="recomposition sidecar",
    )
    sidecar_sha256 = _sha256(
        "recomposition sidecar semantic SHA-256",
        sidecar.get("sidecar_sha256"),
    )
    sidecar_canonical_sha256 = _canonical_sha256(sidecar)

    bundle = _read_json(Path(args.qualification_bundle), name="coordinator bundle")
    bundle_unsigned, bundle_sha256 = _validate_self_hash(
        bundle,
        hash_field="coordinator_bundle_sha256",
        name="qualification coordinator bundle",
    )
    qualified_mapping = bundle_unsigned.get("controller_to_jax_process_index")
    mapping_shape_is_exact = (
        isinstance(qualified_mapping, list)
        and len(qualified_mapping) == PROCESS_COUNT
        and all(
            isinstance(item, Mapping)
            and set(item) == {"controller_index", "process_index"}
            for item in qualified_mapping
        )
    )
    mapping_is_exact = False
    if mapping_shape_is_exact:
        controller_indices = [item["controller_index"] for item in qualified_mapping]
        process_indices = [item["process_index"] for item in qualified_mapping]
        mapping_is_exact = (
            all(type(value) is int for value in controller_indices)
            and all(type(value) is int for value in process_indices)
            and sorted(controller_indices) == list(range(PROCESS_COUNT))
            and sorted(process_indices) == list(range(PROCESS_COUNT))
        )
    if (
        bundle_unsigned.get("schema_version") != COORDINATOR_BUNDLE_SCHEMA_VERSION
        or not mapping_is_exact
        or _canonical_json_bytes(bundle_unsigned.get("qualification_receipt"))
        != _canonical_json_bytes(receipt)
        or bundle_unsigned.get("recomposition_sidecar_sha256") != sidecar_sha256
        or bundle_unsigned.get("recomposition_sidecar_canonical_sha256")
        != sidecar_canonical_sha256
    ):
        raise OperatorError(
            "coordinator bundle does not bind the exact receipt and sidecar"
        )

    decision = _read_json(
        Path(args.qualification_decision),
        name="qualification operator decision",
    )
    decision_unsigned, decision_sha256 = _validate_self_hash(
        decision,
        hash_field="operator_decision_sha256",
        name="qualification operator decision",
    )
    expected_global_batch = (
        _positive_int(
            "per_device_sequence_batch",
            args.per_device_sequence_batch,
        )
        * GLOBAL_DEVICE_COUNT
    )
    if (
        decision_unsigned.get("schema_version") != OPERATOR_DECISION_SCHEMA_VERSION
        or decision_unsigned.get("outcome") != "passed"
        or decision_unsigned.get("next_action") != "stop_no_workload_launch"
        or decision_unsigned.get("mesh") != args.mesh
        or decision_unsigned.get("per_device_sequence_batch")
        != args.per_device_sequence_batch
        or decision_unsigned.get("global_sequence_batch") != expected_global_batch
        or decision_unsigned.get("coordinator_bundle_sha256") != bundle_sha256
        or decision_unsigned.get("recomposition_sidecar_sha256") != sidecar_sha256
    ):
        raise OperatorError(
            "explicit mesh/batch does not match the passing qualification decision"
        )
    return {
        "qualification_receipt_sha256": receipt_sha256,
        "qualification_bundle_sha256": bundle_sha256,
        "qualification_decision_sha256": decision_sha256,
        "recomposition_sidecar_sha256": sidecar_sha256,
        "recomposition_sidecar_canonical_sha256": sidecar_canonical_sha256,
        "qualified_controller_to_jax_process_index": qualified_mapping,
    }


def _local_absolute(value: str, *, name: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise OperatorError(f"{name} must be absolute")
    return path.resolve()


def _atomic_create(path: Path, value: Mapping[str, object]) -> None:
    payload = _canonical_json_bytes(dict(value))
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OperatorError("immutable attempt-plan write made no progress")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def initialize_attempt(args: argparse.Namespace) -> int:
    """Validate qualification lineage and create one immutable attempt plan."""

    attempt_root = _local_absolute(args.attempt_root, name="attempt_root")
    control_root = _local_absolute(args.control_root, name="control_root")
    phase_evidence = _local_absolute(args.phase_evidence, name="phase_evidence")
    source_v1_root = _local_absolute(args.source_v1_root, name="source_v1_root")
    expected_attempt = control_root / "attempts" / args.attempt_id
    if attempt_root != expected_attempt:
        raise OperatorError("attempt_root is not control_root/attempts/attempt_id")
    if phase_evidence != control_root / "phase-evidence.json":
        raise OperatorError("phase_evidence must be control_root/phase-evidence.json")
    if not (source_v1_root / "manifest.json").is_file():
        raise OperatorError("source-v1 root has no sealed manifest.json")
    if not _local_absolute(args.config, name="config").is_file():
        raise OperatorError("Experiment-2 config is missing")
    if control_root.is_symlink():
        raise OperatorError("control_root must not be a symbolic link")
    if (control_root / "manifest.json").exists() or phase_evidence.exists():
        raise OperatorError("requested phase/split is already sealed")

    prior_control = args.prior_control_root
    prior_evidence = args.prior_phase_evidence
    if (prior_control is None) != (prior_evidence is None):
        raise OperatorError(
            "prior control root and prior phase evidence must be supplied together"
        )
    normalized_prior_control: str | None = None
    normalized_prior_evidence: str | None = None
    if prior_control is not None:
        parent_root = _local_absolute(prior_control, name="prior_control_root")
        parent_evidence = _local_absolute(
            prior_evidence,
            name="prior_phase_evidence",
        )
        if not (parent_root / "manifest.json").is_file():
            raise OperatorError("immediate parent has no sealed manifest.json")
        if not parent_evidence.is_file():
            raise OperatorError("immediate parent phase evidence is missing")
        normalized_prior_control = str(parent_root)
        normalized_prior_evidence = str(parent_evidence)

    for name in (
        "recomposition_sidecar",
        "qualification_receipt",
        "qualification_bundle",
        "qualification_decision",
    ):
        path = _local_absolute(getattr(args, name), name=name)
        if not path.is_file():
            raise OperatorError(f"{name} is missing")
        setattr(args, name, str(path))

    linkage = _qualification_linkage(args)
    body: dict[str, object] = {
        "schema_version": ATTEMPT_PLAN_SCHEMA_VERSION,
        "terminal_action": (
            "publish_only_after_four_complete_immutable_supervised_records"
        ),
        "attempt_id": args.attempt_id,
        "attempt_root": str(attempt_root),
        "project_commit": args.project_commit,
        "config": str(_local_absolute(args.config, name="config")),
        "phase": args.phase,
        "split": args.split,
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": args.checkpoint_sha256,
        "grain_files": args.grain_files,
        "staged_manifest": args.staged_manifest,
        "source_v1_root": str(source_v1_root),
        "source_cache_max_bytes": args.source_cache_max_bytes,
        "control_root": str(control_root),
        "object_prefix": args.object_prefix,
        "recomposition_sidecar": args.recomposition_sidecar,
        "qualification_receipt": args.qualification_receipt,
        "qualification_bundle": args.qualification_bundle,
        "qualification_decision": args.qualification_decision,
        **linkage,
        "mesh": args.mesh,
        "per_device_sequence_batch": args.per_device_sequence_batch,
        "global_sequence_batch": (args.per_device_sequence_batch * GLOBAL_DEVICE_COUNT),
        "process_count": PROCESS_COUNT,
        "maximum_runtime_seconds": args.maximum_runtime_seconds,
        "prior_control_root": normalized_prior_control,
        "prior_phase_evidence": normalized_prior_evidence,
        "maxtext_root": args.maxtext_root,
        "phase_evidence": str(phase_evidence),
        "progress_authority": (
            "explicit_range_receipts_and_sealed_manifests_never_bucket_listing"
        ),
    }
    plan = {**body, "operator_plan_sha256": _canonical_sha256(body)}
    control_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    attempts_root = control_root / "attempts"
    attempts_root.mkdir(mode=0o700, exist_ok=True)
    try:
        attempt_root.mkdir(mode=0o700, exist_ok=False)
    except FileExistsError as error:
        raise OperatorError(
            f"immutable attempt ID already exists: {args.attempt_id}"
        ) from error
    for controller_index in range(PROCESS_COUNT):
        (attempt_root / f"controller-c{controller_index:05d}").mkdir(
            mode=0o700,
            exist_ok=False,
        )
    _atomic_create(attempt_root / "operator-plan.json", plan)
    print(plan["operator_plan_sha256"], flush=True)
    return 0


def _run_logged(
    command: Sequence[str],
    log_path: Path,
) -> subprocess.CompletedProcess[bytes]:
    with log_path.open("xb") as log:
        return subprocess.run(
            list(command),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )


def _verify_local_checkout(settings: Settings) -> None:
    repository = Path(__file__).resolve().parents[2]
    try:
        head = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
    except (
        OSError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ) as error:
        raise OperatorError("cannot verify the local Git checkout") from error
    if head != settings.project_commit:
        raise OperatorError(f"local HEAD {head!r} differs from PROJECT_COMMIT")
    if status:
        raise OperatorError(
            "local checkout is dirty; commit and deploy the exact operator "
            "before harvesting"
        )


def _terminate_processes(
    processes: Sequence[tuple[subprocess.Popen[bytes], BinaryIO]],
) -> None:
    pending = [process for process, _ in processes if process.poll() is None]
    for process in pending:
        process.terminate()
    for process in pending:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def _fanout(
    commands: Sequence[Sequence[str]],
    log_paths: Sequence[Path],
    *,
    maximum_runtime_seconds: int,
) -> list[int]:
    if len(commands) != PROCESS_COUNT or len(log_paths) != PROCESS_COUNT:
        raise OperatorError("harvest fanout requires exactly four commands")
    processes: list[tuple[subprocess.Popen[bytes], BinaryIO]] = []
    try:
        for command, log_path in zip(commands, log_paths, strict=True):
            handle = log_path.open("xb")
            try:
                process = subprocess.Popen(
                    list(command),
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
            except BaseException:
                handle.close()
                _terminate_processes(processes)
                raise
            processes.append((process, handle))
        deadline = time.monotonic() + maximum_runtime_seconds + TRANSPORT_GRACE_SECONDS
        pending = {process for process, _ in processes}
        while pending:
            for process in tuple(pending):
                if process.poll() is not None:
                    pending.remove(process)
            if not pending:
                break
            if time.monotonic() >= deadline:
                _terminate_processes(processes)
                raise OperatorError(
                    "gcloud transports exceeded the workload deadline plus "
                    f"{TRANSPORT_GRACE_SECONDS}s grace; publication was not "
                    "attempted"
                )
            time.sleep(0.25)
        return [int(process.returncode) for process, _ in processes]
    finally:
        for _, handle in processes:
            handle.close()


def _terminal_harvest_status(
    log_path: Path,
    *,
    settings: Settings,
    controller_index: int,
) -> int:
    """Read one exact remote harvest terminal marker from an SSH log.

    A completed remote shell can be followed by a nonzero local
    ``gcloud ... ssh`` transport status.  This marker authorizes only the
    immutable four-host publication check; it does not replace that check.
    """

    try:
        lines = log_path.read_bytes().splitlines()
    except OSError as error:
        raise OperatorError(
            f"cannot read controller {controller_index} SSH log: {log_path}"
        ) from error
    marker_lines = [
        line for line in lines if line.startswith(b"terminal phase=")
    ]
    if len(marker_lines) != 1:
        raise OperatorError(
            f"controller {controller_index} SSH log has "
            f"{len(marker_lines)} terminal markers; exactly one is required"
        )
    expected_prefix = (
        f"terminal phase={settings.phase} split={settings.split} "
        f"controller={controller_index} status="
    ).encode("ascii")
    marker = marker_lines[0]
    if not marker.startswith(expected_prefix):
        raise OperatorError(
            f"controller {controller_index} SSH terminal marker does not "
            "match the exact phase/split/controller"
        )
    raw_status = marker[len(expected_prefix) :]
    if re.fullmatch(rb"(?:0|[1-9][0-9]{0,2})", raw_status) is None:
        raise OperatorError(
            f"controller {controller_index} SSH terminal marker has an "
            "invalid harvest status"
        )
    status = int(raw_status)
    if status > 255:
        raise OperatorError(
            f"controller {controller_index} SSH terminal marker harvest "
            "status is outside the shell exit-code range"
        )
    return status


def _validated_transport_completion(
    *,
    settings: Settings,
    returncodes: Sequence[int],
    log_paths: Sequence[Path],
) -> list[int]:
    """Require exact remote completion before attempting publication."""

    if len(returncodes) != PROCESS_COUNT or len(log_paths) != PROCESS_COUNT:
        raise OperatorError(
            "transport completion requires exactly four codes and logs"
        )
    harvest_statuses = [
        _terminal_harvest_status(
            log_path,
            settings=settings,
            controller_index=controller_index,
        )
        for controller_index, log_path in enumerate(log_paths)
    ]
    nonzero = [int(code) for code in returncodes if int(code) != 0]
    if nonzero:
        print(
            "[exp2-harvest] local gcloud transport returned nonzero codes "
            f"{list(returncodes)} after all four exact terminal markers; "
            "continuing only to immutable four-host publication validation",
            flush=True,
        )
    return harvest_statuses


def _dry_run(settings: Settings) -> int:
    built = [
        _host_command(settings, controller_index)
        for controller_index in range(PROCESS_COUNT)
    ]
    payload = {
        "attempt_id": settings.attempt_id,
        "attempt_root": settings.attempt_root,
        "phase": settings.phase,
        "split": settings.split,
        "mesh": settings.mesh,
        "per_device_sequence_batch": settings.per_device_sequence_batch,
        "global_sequence_batch": (
            settings.per_device_sequence_batch * GLOBAL_DEVICE_COUNT
        ),
        "process_count": PROCESS_COUNT,
        "maximum_runtime_seconds": settings.maximum_runtime_seconds,
        "transport_grace_seconds": TRANSPORT_GRACE_SECONDS,
        "init_attempt": _init_command(settings),
        "fanout": [command for command, _ in built],
        "publish": _publish_command(
            settings,
            [evidence for _, evidence in built],
        ),
        "phase_evidence": settings.phase_evidence,
        "local_logs": str(settings.local_log_root),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def run(settings: Settings) -> int:
    if shutil.which("gcloud") is None:
        raise OperatorError("gcloud is unavailable")
    _verify_local_checkout(settings)
    settings.local_log_root.mkdir(parents=True, exist_ok=False, mode=0o700)

    init_log = settings.local_log_root / "init-attempt.ssh.log"
    initialized = _run_logged(_init_command(settings), init_log)
    if initialized.returncode != 0:
        raise OperatorError(
            f"could not create the immutable attempt plan; inspect {init_log}"
        )

    built = [
        _host_command(settings, controller_index)
        for controller_index in range(PROCESS_COUNT)
    ]
    commands = [command for command, _ in built]
    evidence_paths = [evidence for _, evidence in built]
    print(
        "[exp2-harvest] launching four controllers concurrently: "
        f"phase={settings.phase} split={settings.split} "
        f"mesh={settings.mesh} "
        f"per-device={settings.per_device_sequence_batch} "
        f"attempt={settings.attempt_id}",
        flush=True,
    )
    log_paths = [
        settings.local_log_root / f"host-{index:05d}.ssh.log"
        for index in range(PROCESS_COUNT)
    ]
    returncodes = _fanout(
        commands,
        log_paths,
        maximum_runtime_seconds=settings.maximum_runtime_seconds,
    )
    _validated_transport_completion(
        settings=settings,
        returncodes=returncodes,
        log_paths=log_paths,
    )

    publish_log = settings.local_log_root / "publish.ssh.log"
    published = _run_logged(
        _publish_command(settings, evidence_paths),
        publish_log,
    )
    if published.returncode != 0:
        raise OperatorError(
            "worker 0 rejected phase publication; no manifest was accepted; "
            f"inspect {publish_log}"
        )
    print(
        "[exp2-harvest] SEALED\n"
        f"manifest: {settings.control_root}/manifest.json\n"
        f"phase evidence: {settings.phase_evidence}\n"
        f"attempt evidence: {settings.attempt_root}\n"
        f"Mac logs: {settings.local_log_root}",
        flush=True,
    )
    return 0


def _public_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fan out one bounded Experiment-2 contribution harvest and publish "
            "only after four complete supervised records"
        )
    )
    parser.add_argument("--phase", required=True)
    parser.add_argument("--split", choices=("train", "eval"), required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--mesh", required=True)
    parser.add_argument(
        "--per-device-sequence-batch",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--maximum-runtime-seconds",
        type=int,
        required=True,
    )
    parser.add_argument("--prior-control-root")
    parser.add_argument("--prior-phase-evidence")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print exact init, fanout, and publication commands",
    )
    return parser


def _init_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--attempt-root", required=True)
    parser.add_argument("--project-commit", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--split", choices=("train", "eval"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--grain-files", required=True)
    parser.add_argument("--staged-manifest", required=True)
    parser.add_argument("--source-v1-root", required=True)
    parser.add_argument("--source-cache-max-bytes", type=int, required=True)
    parser.add_argument("--control-root", required=True)
    parser.add_argument("--object-prefix", required=True)
    parser.add_argument("--recomposition-sidecar", required=True)
    parser.add_argument("--qualification-receipt", required=True)
    parser.add_argument("--qualification-bundle", required=True)
    parser.add_argument("--qualification-decision", required=True)
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--per-device-sequence-batch", type=int, required=True)
    parser.add_argument("--maximum-runtime-seconds", type=int, required=True)
    parser.add_argument("--prior-control-root")
    parser.add_argument("--prior-phase-evidence")
    parser.add_argument("--maxtext-root")
    parser.add_argument("--phase-evidence", required=True)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    environment: Mapping[str, str] | None = None,
) -> int:
    selected = list(sys.argv[1:] if argv is None else argv)
    try:
        if selected and selected[0] == "_init-attempt":
            return initialize_attempt(_init_parser().parse_args(selected[1:]))
        args = _public_parser().parse_args(selected)
        settings = _settings(
            os.environ if environment is None else environment,
            args,
        )
        if args.dry_run:
            return _dry_run(settings)
        return run(settings)
    except (OperatorError, OSError, ValueError) as error:
        print(f"[exp2-harvest] ERROR: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
