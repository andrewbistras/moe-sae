#!/usr/bin/env python3
"""Fan out one cell or resident panel-seed ladder bundle across a TPU slice."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import uuid
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath

from moe_sae.contribution_artifacts import canonical_sha256
from moe_sae.experiment5_routed_ladder import PROTOCOL_ID as V2_PROTOCOL_ID
from moe_sae.experiment5_routed_ladder_v2_config import (
    CONDITIONS as V2_CONDITIONS,
    PHASE_NAME,
)
from moe_sae.experiment5_routed_ladder_v3_config import (
    ANALYZED_CONDITIONS as V3_ANALYZED_CONDITIONS,
    PROTOCOL_ID as V3_PROTOCOL_ID,
)
from moe_sae.experiment5_routed_ladder_evaluation_worker import (
    BUNDLE_CONDITIONS,
    BUNDLE_RESULT_SCHEMA_VERSION,
    BUNDLE_SPEC_SCHEMA_VERSION,
    CHECKPOINT_TOKENS,
    RESULT_SCHEMA_VERSION,
    V3_RESULT_SCHEMA_VERSION,
    load_merged_scores,
    parse_resident_bundle_spec,
    resident_bundle_conditions,
    resident_bundle_result_schema,
    resident_bundle_spec_schema,
)
from moe_sae.runtime import (
    REGIONAL_ACCELERATOR_TYPES,
    SUPPORTED_TRAINING_ACCELERATOR_TYPES,
    V5LITEPOD_DEVICE_KIND_ENV,
    RuntimeDependencyError,
    accelerator_peak_flops_per_second,
    v5litepod_device_kind_from_environment,
)

PROCESS_COUNT = 4
PANELS = ("sae", "transcoder")
PROTOCOL_ID = V2_PROTOCOL_ID
PROTOCOL_IDS = (V2_PROTOCOL_ID, V3_PROTOCOL_ID)
EVALUATED_CONDITIONS = tuple(
    dict.fromkeys((*V2_CONDITIONS, *V3_ANALYZED_CONDITIONS, "H"))
)
REGIONAL_PROCESS_CACHE_MAX_BYTES = 160 * 1024**3
WORKER_TIMEOUT_SECONDS = 7_500
OPERATOR_BUNDLE_MANIFEST_SCHEMA_VERSION = (
    "gemma4-experiment5-routed-activation-ladder-operator-bundle-manifest-v1"
)
V3_OPERATOR_BUNDLE_MANIFEST_SCHEMA_VERSION = (
    "gemma4-experiment5-routed-activation-policy-operator-bundle-manifest-v1"
)
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_SSH_USER = re.compile(r"^[a-z_][a-z0-9_-]{0,31}$")


class OperatorError(RuntimeError):
    """One activation-ladder cell cannot be safely orchestrated."""


def _operator_bundle_manifest_schema(protocol_id: str) -> str:
    if protocol_id == V2_PROTOCOL_ID:
        return OPERATOR_BUNDLE_MANIFEST_SCHEMA_VERSION
    if protocol_id == V3_PROTOCOL_ID:
        return V3_OPERATOR_BUNDLE_MANIFEST_SCHEMA_VERSION
    raise OperatorError("resident bundle protocol is invalid")


def _required(environment: Mapping[str, str], name: str) -> str:
    value = environment.get(name, "")
    if not value or any(character in value for character in "\x00\n\r"):
        raise OperatorError(f"required environment variable {name} is unset or unsafe")
    return value


def _absolute_remote(name: str, value: str) -> str:
    path = PurePosixPath(value)
    if not path.is_absolute() or value == "/" or ".." in path.parts:
        raise OperatorError(f"{name} must be a non-root absolute POSIX path")
    return str(path)


def _absolute_local(name: str, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or value == "/" or ".." in path.parts:
        raise OperatorError(f"{name} must be a non-root absolute path")
    return path


def _gcs_prefix(name: str, value: str) -> str:
    if (
        not value.startswith("gs://")
        or value.endswith("/")
        or any(character in value for character in "\x00\n\r")
    ):
        raise OperatorError(f"{name} must be a non-root GCS prefix")
    return value


def _positive_int(name: str, value: str) -> int:
    try:
        selected = int(value)
    except ValueError as error:
        raise OperatorError(f"{name} must be an integer") from error
    if selected <= 0 or str(selected) != value:
        raise OperatorError(f"{name} must be a canonical positive integer")
    return selected


@dataclass(frozen=True, slots=True)
class Settings:
    panel: str
    condition: str
    seed: int
    checkpoint_tokens: int
    source_revision: str
    training_source_revision: str
    project: str
    tpu: str
    zone: str
    project_root: str
    uv: str
    config: str
    h_config: str | None
    a256_config: str | None
    reference_run_contract_sha256: str | None
    reference_run_contract_file_sha256: str | None
    parent_training_result_sha256: str
    parent_training_result_file_sha256: str
    parent_run_contract_sha256: str | None
    parent_run_contract_file_sha256: str | None
    parent_checkpoint_pointer_sha256: str
    parent_checkpoint_pointer_file_sha256: str
    mapping_receipt: str
    recomposition_sidecar: str
    train_target_root: str
    eval_target_root: str
    train_target_object_prefix: str
    eval_target_object_prefix: str
    train_source_root: str | None
    eval_source_root: str | None
    train_source_object_prefix: str | None
    eval_source_object_prefix: str | None
    cache_root: str
    cache_max_bytes: int
    checkpoint_payload_root: str
    run_root: str
    result_root: str
    result_root_controller: Path
    log_root: Path
    accelerator_type: str
    peak_flops_per_second: int
    regional_mirror_seal: str | None
    expected_token_identity_sha256: str | None
    internal_ip: bool
    ssh_user: str | None
    ssh_key: str | None
    v5litepod_device_kind: str | None


@dataclass(frozen=True, slots=True)
class BundlePublication:
    condition: str
    checkpoint_tokens: int
    training_source_revision: str
    result_root_remote: str
    result_root_controller: Path


@dataclass(frozen=True, slots=True)
class BundleSettings:
    base: Settings
    protocol_id: str
    manifest_sha256: str
    worker_spec: dict
    worker_spec_sha256: str
    attempt_id: str
    bundle_result_root_remote: str
    bundle_result_root_controller: Path
    publications: tuple[BundlePublication, ...]


def _settings(
    args: argparse.Namespace,
    environment: Mapping[str, str],
) -> Settings:
    if args.panel not in PANELS or args.condition not in EVALUATED_CONDITIONS:
        raise OperatorError("evaluation panel or condition changed")
    if args.seed not in (0, 1, 2) or args.checkpoint_tokens not in CHECKPOINT_TOKENS:
        raise OperatorError("evaluation seed or checkpoint changed")
    commit = _required(environment, "PROJECT_COMMIT")
    if (
        _GIT_SHA.fullmatch(commit) is None
        or args.source_revision != commit
        or _GIT_SHA.fullmatch(args.training_source_revision) is None
    ):
        raise OperatorError("source revisions are invalid or deployment-mismatched")
    historical_a256 = (
        args.panel == "sae"
        and args.condition == "A256ReLU"
        and args.seed == 0
    )
    for name in (
        "parent_training_result_sha256",
        "parent_training_result_file_sha256",
        "parent_checkpoint_pointer_sha256",
        "parent_checkpoint_pointer_file_sha256",
    ):
        value = getattr(args, name)
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise OperatorError(f"--{name.replace('_', '-')} must be a SHA-256")
    parent_run_contract_sha = args.parent_run_contract_sha256
    parent_run_contract_file_sha = args.parent_run_contract_file_sha256
    if args.condition == "H":
        if (
            parent_run_contract_sha is not None
            or parent_run_contract_file_sha is not None
        ):
            raise OperatorError("H cannot claim an A-worker parent run contract")
    elif (
        not isinstance(parent_run_contract_sha, str)
        or _SHA256.fullmatch(parent_run_contract_sha) is None
        or not isinstance(parent_run_contract_file_sha, str)
        or _SHA256.fullmatch(parent_run_contract_file_sha) is None
    ):
        raise OperatorError("A evaluation requires exact parent run-contract pins")
    if args.condition == "H":
        if args.h_config is None:
            raise OperatorError("H evaluation requires --h-config")
        h_config = _absolute_remote("--h-config", args.h_config)
    else:
        if args.h_config is not None:
            raise OperatorError("--h-config is valid only for H")
        h_config = None
        if not historical_a256 and args.training_source_revision != commit:
            raise OperatorError("new A evaluation must match its training revision")
    if historical_a256:
        if args.a256_config is None:
            raise OperatorError(
                "historical SAE A256ReLU seed 0 requires --a256-config"
            )
        a256_config = _absolute_remote("--a256-config", args.a256_config)
        if (
            args.reference_run_contract_sha256 is None
            or _SHA256.fullmatch(args.reference_run_contract_sha256) is None
            or args.reference_run_contract_file_sha256 is None
            or _SHA256.fullmatch(
                args.reference_run_contract_file_sha256
            )
            is None
        ):
            raise OperatorError(
                "historical SAE A256ReLU requires exact run-contract pins"
            )
        reference_run_contract_sha = args.reference_run_contract_sha256
        reference_run_contract_file_sha = (
            args.reference_run_contract_file_sha256
        )
        if (
            reference_run_contract_sha != parent_run_contract_sha
            or reference_run_contract_file_sha != parent_run_contract_file_sha
        ):
            raise OperatorError(
                "historical A256 reference and parent run-contract pins differ"
            )
    else:
        if args.a256_config is not None:
            raise OperatorError(
                "--a256-config is valid only for historical SAE A256ReLU seed 0"
            )
        a256_config = None
        if (
            args.reference_run_contract_sha256 is not None
            or args.reference_run_contract_file_sha256 is not None
        ):
            raise OperatorError(
                "run-contract pins are valid only for historical SAE A256ReLU"
            )
        reference_run_contract_sha = None
        reference_run_contract_file_sha = None
    accelerator = _required(environment, "TPU_ACCELERATOR_TYPE")
    if accelerator not in SUPPORTED_TRAINING_ACCELERATOR_TYPES:
        raise OperatorError("TPU_ACCELERATOR_TYPE is unsupported")
    internal_raw = _required(environment, "TPU_USE_INTERNAL_IP")
    if internal_raw not in {"0", "1"}:
        raise OperatorError("TPU_USE_INTERNAL_IP must be exactly 0 or 1")
    internal = internal_raw == "1"
    if accelerator in REGIONAL_ACCELERATOR_TYPES and not internal:
        raise OperatorError(f"{accelerator} evaluation requires internal IP")
    mirror = environment.get("CONTRIBUTION_REGIONAL_MIRROR_SEAL") or None
    if accelerator in REGIONAL_ACCELERATOR_TYPES and mirror is None:
        raise OperatorError(f"{accelerator} evaluation requires a regional mirror")
    cache_bytes = _positive_int(
        "CONTRIBUTION_SAE_CACHE_MAX_BYTES",
        _required(environment, "CONTRIBUTION_SAE_CACHE_MAX_BYTES"),
    )
    if (
        accelerator in REGIONAL_ACCELERATOR_TYPES
        and cache_bytes != REGIONAL_PROCESS_CACHE_MAX_BYTES
    ):
        raise OperatorError(f"{accelerator} cache bound must be exactly 160GiB")
    ssh_user = environment.get("TPU_SSH_USER") or None
    if ssh_user is not None and _SAFE_SSH_USER.fullmatch(ssh_user) is None:
        raise OperatorError("TPU_SSH_USER is invalid")
    try:
        device_kind = (
            v5litepod_device_kind_from_environment()
            if accelerator == "v5litepod-16"
            else None
        )
    except RuntimeDependencyError as error:
        raise OperatorError(str(error)) from error
    overlays = tuple(
        environment.get(name) or None
        for name in (
            "CONTRIBUTION_SOURCE_OVERLAY_ROOT",
            "CONTRIBUTION_EVAL_SOURCE_OVERLAY_ROOT",
            "CONTRIBUTION_SOURCE_OVERLAY_PREFIX",
            "CONTRIBUTION_EVAL_SOURCE_OVERLAY_PREFIX",
        )
    )
    if args.panel == "sae" and any(value is not None for value in overlays):
        raise OperatorError("SAE evaluation prohibits source overlays")
    if args.panel == "transcoder" and any(value is None for value in overlays):
        raise OperatorError("transcoder evaluation requires all source overlays")
    expected_identity = environment.get(
        "EXPERIMENT5_ROUTED_ACTIVATION_LADDER_EXPECTED_TOKEN_IDENTITY_SHA256"
    ) or None
    if expected_identity is not None and _SHA256.fullmatch(expected_identity) is None:
        raise OperatorError("expected token identity must be a SHA-256")
    log_name = (
        "CONTRIBUTION_EVALUATION_LOG_ROOT"
        if args.panel == "sae"
        else "TRANSCODER_EVALUATION_LOG_ROOT"
    )
    return Settings(
        panel=args.panel,
        condition=args.condition,
        seed=args.seed,
        checkpoint_tokens=args.checkpoint_tokens,
        source_revision=commit,
        training_source_revision=args.training_source_revision,
        project=_required(environment, "GCP_PROJECT"),
        tpu=_required(environment, "TPU_NAME"),
        zone=_required(environment, "TPU_ZONE"),
        project_root=_absolute_remote(
            "TPU_PROJECT_ROOT", _required(environment, "TPU_PROJECT_ROOT")
        ),
        uv=_absolute_remote("REMOTE_UV", _required(environment, "REMOTE_UV")),
        config=_absolute_remote("--config", args.config),
        h_config=h_config,
        a256_config=a256_config,
        reference_run_contract_sha256=reference_run_contract_sha,
        reference_run_contract_file_sha256=reference_run_contract_file_sha,
        parent_training_result_sha256=args.parent_training_result_sha256,
        parent_training_result_file_sha256=(
            args.parent_training_result_file_sha256
        ),
        parent_run_contract_sha256=parent_run_contract_sha,
        parent_run_contract_file_sha256=parent_run_contract_file_sha,
        parent_checkpoint_pointer_sha256=(
            args.parent_checkpoint_pointer_sha256
        ),
        parent_checkpoint_pointer_file_sha256=(
            args.parent_checkpoint_pointer_file_sha256
        ),
        mapping_receipt=_absolute_remote(
            "CONTRIBUTION_FRESH_CONTROLLER_MAPPING",
            _required(environment, "CONTRIBUTION_FRESH_CONTROLLER_MAPPING"),
        ),
        recomposition_sidecar=_absolute_remote(
            "CONTRIBUTION_RECOMPOSITION_SIDECAR",
            _required(environment, "CONTRIBUTION_RECOMPOSITION_SIDECAR"),
        ),
        train_target_root=_absolute_remote(
            "CONTRIBUTION_TRAIN_CAPTURE_ROOT",
            _required(environment, "CONTRIBUTION_TRAIN_CAPTURE_ROOT"),
        ),
        eval_target_root=_absolute_remote(
            "CONTRIBUTION_EVAL_CAPTURE_ROOT",
            _required(environment, "CONTRIBUTION_EVAL_CAPTURE_ROOT"),
        ),
        train_target_object_prefix=_gcs_prefix(
            "CONTRIBUTION_OBJECT_PREFIX",
            _required(environment, "CONTRIBUTION_OBJECT_PREFIX"),
        ),
        eval_target_object_prefix=_gcs_prefix(
            "CONTRIBUTION_EVAL_OBJECT_PREFIX",
            _required(environment, "CONTRIBUTION_EVAL_OBJECT_PREFIX"),
        ),
        train_source_root=(
            _absolute_remote("train source root", str(overlays[0]))
            if args.panel == "transcoder"
            else None
        ),
        eval_source_root=(
            _absolute_remote("eval source root", str(overlays[1]))
            if args.panel == "transcoder"
            else None
        ),
        train_source_object_prefix=(
            _gcs_prefix("train source prefix", str(overlays[2]))
            if args.panel == "transcoder"
            else None
        ),
        eval_source_object_prefix=(
            _gcs_prefix("eval source prefix", str(overlays[3]))
            if args.panel == "transcoder"
            else None
        ),
        cache_root=_absolute_remote(
            "CONTRIBUTION_SAE_CACHE_ROOT",
            _required(environment, "CONTRIBUTION_SAE_CACHE_ROOT"),
        ),
        cache_max_bytes=cache_bytes,
        checkpoint_payload_root=_gcs_prefix(
            "CONTRIBUTION_SAE_CHECKPOINT_ROOT",
            _required(environment, "CONTRIBUTION_SAE_CHECKPOINT_ROOT"),
        ),
        run_root=_absolute_remote("--run-root", args.run_root),
        result_root=_absolute_remote("--result-root", args.result_root),
        result_root_controller=_absolute_local(
            "EXPERIMENT5_ROUTED_ACTIVATION_LADDER_RESULT_ROOT_CONTROLLER",
            _required(
                environment,
                "EXPERIMENT5_ROUTED_ACTIVATION_LADDER_RESULT_ROOT_CONTROLLER",
            ),
        ),
        log_root=_absolute_local(log_name, _required(environment, log_name)),
        accelerator_type=accelerator,
        peak_flops_per_second=accelerator_peak_flops_per_second(accelerator),
        regional_mirror_seal=mirror,
        expected_token_identity_sha256=expected_identity,
        internal_ip=internal,
        ssh_user=ssh_user,
        ssh_key=environment.get("CONTRIBUTION_SSH_KEY_FILE") or None,
        v5litepod_device_kind=device_kind,
    )


def _bundle_settings(
    args: argparse.Namespace,
    environment: Mapping[str, str],
) -> BundleSettings:
    if (
        args.bundle_manifest is None
        or args.attempt_id is None
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", args.attempt_id)
        is None
    ):
        raise OperatorError(
            "resident bundle mode requires a safe manifest and attempt ID"
        )
    manifest_path = _absolute_local(
        "--bundle-manifest",
        args.bundle_manifest,
    )
    try:
        raw = json.loads(manifest_path.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OperatorError("resident bundle manifest cannot be read") from error
    if not isinstance(raw, dict):
        raise OperatorError("resident bundle manifest must be an object")
    expected_keys = {
        "schema_version",
        "protocol_id",
        "panel",
        "seed",
        "source_revision",
        "config_remote",
        "bundle_result_root_remote",
        "bundle_result_root_controller",
        "cells",
        "manifest_sha256",
    }
    if set(raw) != expected_keys:
        raise OperatorError("resident bundle manifest fields changed")
    claimed = raw.get("manifest_sha256")
    unsigned = {
        name: raw[name] for name in raw if name != "manifest_sha256"
    }
    protocol_id = raw["protocol_id"]
    if (
        not isinstance(claimed, str)
        or _SHA256.fullmatch(claimed) is None
        or canonical_sha256(unsigned) != claimed
        or protocol_id not in PROTOCOL_IDS
        or raw["schema_version"]
        != _operator_bundle_manifest_schema(str(protocol_id))
        or raw["panel"] != args.panel
        or raw["seed"] != args.seed
        or raw["source_revision"] != args.source_revision
        or raw["config_remote"] != args.config
    ):
        raise OperatorError("resident bundle manifest identity is invalid")
    if (
        _GIT_SHA.fullmatch(str(raw["source_revision"])) is None
        or raw["source_revision"] != _required(environment, "PROJECT_COMMIT")
    ):
        raise OperatorError("resident bundle deployment revision changed")
    raw_cells = raw["cells"]
    if not isinstance(raw_cells, list):
        raise OperatorError("resident bundle cells must be a list")
    expected_identities = [
        (condition, checkpoint_tokens)
        for condition in resident_bundle_conditions(
            str(protocol_id),
            seed=int(raw["seed"]),
        )
        for checkpoint_tokens in CHECKPOINT_TOKENS
    ]
    observed_identities: list[tuple[object, object]] = []
    worker_cells: list[dict] = []
    publications: list[BundlePublication] = []
    controller_roots: set[Path] = set()
    for index, selected in enumerate(raw_cells):
        if not isinstance(selected, dict) or set(selected) != {
            "condition",
            "checkpoint_tokens",
            "training_source_revision",
            "run_root_remote",
            "checkpoint_payload_root_gcs",
            "result_root_remote",
            "result_root_controller",
            "reference_kind",
            "reference_config_remote",
            "reference_run_contract_sha256",
            "reference_run_contract_file_sha256",
            "parent_training_result_sha256",
            "parent_training_result_file_sha256",
            "parent_run_contract_sha256",
            "parent_run_contract_file_sha256",
            "parent_checkpoint_pointer_sha256",
            "parent_checkpoint_pointer_file_sha256",
        }:
            raise OperatorError(
                f"resident bundle cell {index} fields changed"
            )
        condition = selected["condition"]
        checkpoint_tokens = selected["checkpoint_tokens"]
        observed_identities.append((condition, checkpoint_tokens))
        training_revision = selected["training_source_revision"]
        if (
            not isinstance(training_revision, str)
            or _GIT_SHA.fullmatch(training_revision) is None
        ):
            raise OperatorError(
                f"resident bundle cell {index} training revision is invalid"
            )
        result_controller = _absolute_local(
            f"cell {index} result_root_controller",
            selected["result_root_controller"],
        )
        if result_controller in controller_roots:
            raise OperatorError(
                "resident bundle controller result roots must be unique"
            )
        controller_roots.add(result_controller)
        result_remote = _absolute_remote(
            f"cell {index} result_root_remote",
            selected["result_root_remote"],
        )
        worker_cells.append(
            {
                "condition": condition,
                "checkpoint_tokens": checkpoint_tokens,
                "training_source_revision": training_revision,
                "run_root": _absolute_remote(
                    f"cell {index} run_root_remote",
                    selected["run_root_remote"],
                ),
                "checkpoint_payload_root": _gcs_prefix(
                    f"cell {index} checkpoint_payload_root_gcs",
                    selected["checkpoint_payload_root_gcs"],
                ),
                "result_root": result_remote,
                "reference_kind": selected["reference_kind"],
                "reference_config": (
                    _absolute_remote(
                        f"cell {index} reference_config_remote",
                        selected["reference_config_remote"],
                    )
                    if selected["reference_config_remote"] is not None
                    else None
                ),
                "reference_run_contract_sha256": selected[
                    "reference_run_contract_sha256"
                ],
                "reference_run_contract_file_sha256": selected[
                    "reference_run_contract_file_sha256"
                ],
                "parent_training_result_sha256": selected[
                    "parent_training_result_sha256"
                ],
                "parent_training_result_file_sha256": selected[
                    "parent_training_result_file_sha256"
                ],
                "parent_run_contract_sha256": selected[
                    "parent_run_contract_sha256"
                ],
                "parent_run_contract_file_sha256": selected[
                    "parent_run_contract_file_sha256"
                ],
                "parent_checkpoint_pointer_sha256": selected[
                    "parent_checkpoint_pointer_sha256"
                ],
                "parent_checkpoint_pointer_file_sha256": selected[
                    "parent_checkpoint_pointer_file_sha256"
                ],
            }
        )
        publications.append(
            BundlePublication(
                condition=str(condition),
                checkpoint_tokens=int(checkpoint_tokens),
                training_source_revision=training_revision,
                result_root_remote=result_remote,
                result_root_controller=result_controller,
            )
        )
    if observed_identities != expected_identities:
        raise OperatorError(
            "resident bundle manifest must contain the exact 12-cell order"
        )
    bundle_remote = _absolute_remote(
        "bundle_result_root_remote",
        raw["bundle_result_root_remote"],
    )
    bundle_controller = _absolute_local(
        "bundle_result_root_controller",
        raw["bundle_result_root_controller"],
    )
    if bundle_controller in controller_roots:
        raise OperatorError("bundle completion aliases a cell completion")

    # Reuse the single-cell environment validator with a known extension-owned
    # A512 cell. Per-cell checkpoint and publication roots remain manifest-bound.
    probe_condition = (
        "A512ReLUInclusiveTopK"
        if protocol_id == V3_PROTOCOL_ID
        else "A512ReLU"
    )
    probe_index = next(
        index
        for index, cell in enumerate(worker_cells)
        if cell["condition"] == probe_condition
        and cell["checkpoint_tokens"] == CHECKPOINT_TOKENS[0]
    )
    probe = worker_cells[probe_index]
    probe_publication = publications[probe_index]
    probe_environment = dict(environment)
    probe_environment["CONTRIBUTION_SAE_CHECKPOINT_ROOT"] = str(
        probe["checkpoint_payload_root"]
    )
    probe_environment[
        "EXPERIMENT5_ROUTED_ACTIVATION_LADDER_RESULT_ROOT_CONTROLLER"
    ] = str(probe_publication.result_root_controller)
    probe_args = argparse.Namespace(
        panel=args.panel,
        config=args.config,
        condition=probe_condition,
        seed=args.seed,
        checkpoint_tokens=CHECKPOINT_TOKENS[0],
        source_revision=args.source_revision,
        training_source_revision=args.source_revision,
        run_root=probe["run_root"],
        result_root=probe["result_root"],
        h_config=None,
        a256_config=None,
        reference_run_contract_sha256=None,
        reference_run_contract_file_sha256=None,
        parent_training_result_sha256=probe[
            "parent_training_result_sha256"
        ],
        parent_training_result_file_sha256=probe[
            "parent_training_result_file_sha256"
        ],
        parent_run_contract_sha256=probe["parent_run_contract_sha256"],
        parent_run_contract_file_sha256=probe[
            "parent_run_contract_file_sha256"
        ],
        parent_checkpoint_pointer_sha256=probe[
            "parent_checkpoint_pointer_sha256"
        ],
        parent_checkpoint_pointer_file_sha256=probe[
            "parent_checkpoint_pointer_file_sha256"
        ],
    )
    base = _settings(probe_args, probe_environment)
    worker_unsigned = {
        "schema_version": resident_bundle_spec_schema(str(protocol_id)),
        "protocol_id": protocol_id,
        "panel": args.panel,
        "seed": args.seed,
        "source_revision": args.source_revision,
        "config": args.config,
        "bundle_result_root": bundle_remote,
        "operator_manifest_sha256": claimed,
        "cells": worker_cells,
    }
    worker_spec = {
        **worker_unsigned,
        "bundle_spec_sha256": canonical_sha256(worker_unsigned),
    }
    try:
        parsed = parse_resident_bundle_spec(worker_spec)
    except Exception as error:
        raise OperatorError(
            f"resident worker bundle is invalid: {error}"
        ) from error
    return BundleSettings(
        base=base,
        protocol_id=str(protocol_id),
        manifest_sha256=claimed,
        worker_spec=worker_spec,
        worker_spec_sha256=parsed.bundle_spec_sha256,
        attempt_id=args.attempt_id,
        bundle_result_root_remote=bundle_remote,
        bundle_result_root_controller=bundle_controller,
        publications=tuple(publications),
    )


def _target(settings: Settings) -> str:
    return (
        f"{settings.ssh_user}@{settings.tpu}"
        if settings.ssh_user is not None
        else settings.tpu
    )


def _gcloud_ssh(
    settings: Settings,
    *,
    controller: int,
    remote_command: str,
) -> list[str]:
    command = [
        "gcloud",
        "compute",
        "tpus",
        "tpu-vm",
        "ssh",
        _target(settings),
        f"--project={settings.project}",
        f"--zone={settings.zone}",
        f"--worker={controller}",
        "--quiet",
        "--command",
        remote_command,
    ]
    if settings.ssh_key is not None:
        command.insert(-2, f"--ssh-key-file={settings.ssh_key}")
    if settings.internal_ip:
        command.insert(-2, "--internal-ip")
    return command


def _subprocess_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in ("LD_PRELOAD", "GLIBCXX_FORCE_NEW", "GLIBCPP_FORCE_NEW"):
        environment.pop(name, None)
    return environment


def _remote_bytes(
    settings: Settings,
    *,
    controller: int,
    remote_path: str,
) -> bytes:
    completed = subprocess.run(
        _gcloud_ssh(
            settings,
            controller=controller,
            remote_command="cat " + shlex.quote(remote_path),
        ),
        check=False,
        capture_output=True,
        env=_subprocess_environment(),
    )
    if completed.returncode != 0:
        raise OperatorError(
            f"cannot read remote artifact {remote_path}: "
            + completed.stderr.decode("utf-8", errors="replace")[-2_000:]
        )
    return completed.stdout


def _mapping(settings: Settings) -> dict[int, int]:
    local = Path(settings.mapping_receipt)
    if local.is_file() and not local.is_symlink():
        payload = local.read_bytes()
    else:
        payload = _remote_bytes(
            settings, controller=0, remote_path=settings.mapping_receipt
        )
    try:
        value = json.loads(payload)
        if settings.accelerator_type in REGIONAL_ACCELERATOR_TYPES:
            from moe_sae.regional_execution import (
                validate_controller_mapping_receipt,
            )

            value = validate_controller_mapping_receipt(value)
            if (
                value["accelerator_type"] != settings.accelerator_type
                or value["process_count"] != PROCESS_COUNT
                or value["global_device_count"] != 16
                or value["local_device_count"] != 4
            ):
                raise OperatorError("controller mapping topology changed")
        rows = value["controller_to_jax_process_index"]
        result = {
            int(row["controller_index"]): int(row["process_index"]) for row in rows
        }
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        raise OperatorError("controller mapping receipt is invalid") from error
    if set(result) != set(range(PROCESS_COUNT)) or set(result.values()) != set(
        range(PROCESS_COUNT)
    ):
        raise OperatorError("controller/process mapping is not a permutation")
    return result


def _worker(settings: Settings, process: int) -> list[str]:
    command = [
        settings.uv,
        "run",
        "--frozen",
        "--extra",
        "tpu",
        "python",
        "-m",
        "moe_sae.experiment5_routed_ladder_evaluation_worker",
        "--panel",
        settings.panel,
        "--config",
        settings.config,
        "--condition",
        settings.condition,
        "--seed",
        str(settings.seed),
        "--checkpoint-tokens",
        str(settings.checkpoint_tokens),
        "--source-revision",
        settings.source_revision,
        "--training-source-revision",
        settings.training_source_revision,
        "--train-target-root",
        settings.train_target_root,
        "--eval-target-root",
        settings.eval_target_root,
        "--train-target-object-prefix",
        settings.train_target_object_prefix,
        "--eval-target-object-prefix",
        settings.eval_target_object_prefix,
        "--cache-root",
        f"{settings.cache_root}/process-p{process:05d}",
        "--cache-max-bytes",
        str(settings.cache_max_bytes),
        "--run-root",
        settings.run_root,
        "--checkpoint-payload-root",
        settings.checkpoint_payload_root,
        "--recomposition-sidecar",
        settings.recomposition_sidecar,
        "--result-root",
        settings.result_root,
        "--process-index",
        str(process),
        "--accelerator-type",
        settings.accelerator_type,
        "--peak-flops-per-second",
        str(settings.peak_flops_per_second),
        "--parent-training-result-sha256",
        settings.parent_training_result_sha256,
        "--parent-training-result-file-sha256",
        settings.parent_training_result_file_sha256,
        "--parent-checkpoint-pointer-sha256",
        settings.parent_checkpoint_pointer_sha256,
        "--parent-checkpoint-pointer-file-sha256",
        settings.parent_checkpoint_pointer_file_sha256,
    ]
    if settings.parent_run_contract_sha256 is not None:
        command.extend(
            (
                "--parent-run-contract-sha256",
                settings.parent_run_contract_sha256,
                "--parent-run-contract-file-sha256",
                str(settings.parent_run_contract_file_sha256),
            )
        )
    if settings.h_config is not None:
        command.extend(("--h-config", settings.h_config))
    if settings.a256_config is not None:
        command.extend(("--a256-config", settings.a256_config))
        command.extend(
            (
                "--reference-run-contract-sha256",
                str(settings.reference_run_contract_sha256),
                "--reference-run-contract-file-sha256",
                str(settings.reference_run_contract_file_sha256),
            )
        )
    if settings.panel == "transcoder":
        overlay = (
            settings.train_source_root,
            settings.eval_source_root,
            settings.train_source_object_prefix,
            settings.eval_source_object_prefix,
        )
        if any(value is None for value in overlay):
            raise OperatorError("transcoder worker lost source overlays")
        command.extend(
            (
                "--train-source-root",
                str(overlay[0]),
                "--eval-source-root",
                str(overlay[1]),
                "--train-source-object-prefix",
                str(overlay[2]),
                "--eval-source-object-prefix",
                str(overlay[3]),
            )
        )
    if settings.regional_mirror_seal is not None:
        command.extend(("--regional-mirror-seal", settings.regional_mirror_seal))
    if settings.expected_token_identity_sha256 is not None:
        command.extend(
            (
                "--expected-token-identity-sha256",
                settings.expected_token_identity_sha256,
            )
        )
    if settings.v5litepod_device_kind is not None:
        command[:0] = [
            "env",
            f"{V5LITEPOD_DEVICE_KIND_ENV}={settings.v5litepod_device_kind}",
        ]
    return command


def _bundle_worker(settings: BundleSettings, process: int) -> list[str]:
    base = settings.base
    command = [
        base.uv,
        "run",
        "--frozen",
        "--extra",
        "tpu",
        "python",
        "-m",
        "moe_sae.experiment5_routed_ladder_evaluation_worker",
        "--panel",
        base.panel,
        "--config",
        base.config,
        "--seed",
        str(base.seed),
        "--source-revision",
        base.source_revision,
        "--train-target-root",
        base.train_target_root,
        "--eval-target-root",
        base.eval_target_root,
        "--train-target-object-prefix",
        base.train_target_object_prefix,
        "--eval-target-object-prefix",
        base.eval_target_object_prefix,
        "--cache-root",
        f"{base.cache_root}/process-p{process:05d}",
        "--cache-max-bytes",
        str(base.cache_max_bytes),
        "--recomposition-sidecar",
        base.recomposition_sidecar,
        "--process-index",
        str(process),
        "--accelerator-type",
        base.accelerator_type,
        "--peak-flops-per-second",
        str(base.peak_flops_per_second),
        "--bundle-spec-json",
        json.dumps(
            settings.worker_spec,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "--attempt-id",
        settings.attempt_id,
    ]
    if base.panel == "transcoder":
        overlays = (
            base.train_source_root,
            base.eval_source_root,
            base.train_source_object_prefix,
            base.eval_source_object_prefix,
        )
        if any(value is None for value in overlays):
            raise OperatorError("transcoder bundle lost source overlays")
        command.extend(
            (
                "--train-source-root",
                str(overlays[0]),
                "--eval-source-root",
                str(overlays[1]),
                "--train-source-object-prefix",
                str(overlays[2]),
                "--eval-source-object-prefix",
                str(overlays[3]),
            )
        )
    if base.regional_mirror_seal is not None:
        command.extend(("--regional-mirror-seal", base.regional_mirror_seal))
    if base.expected_token_identity_sha256 is not None:
        command.extend(
            (
                "--expected-token-identity-sha256",
                base.expected_token_identity_sha256,
            )
        )
    if base.v5litepod_device_kind is not None:
        command[:0] = [
            "env",
            f"{V5LITEPOD_DEVICE_KIND_ENV}={base.v5litepod_device_kind}",
        ]
    return command


def _command(
    settings: Settings,
    *,
    controller: int,
    process: int,
) -> list[str]:
    body = "\n".join(
        (
            "set -euo pipefail",
            f"cd {shlex.quote(settings.project_root)}",
            f'test "$(git rev-parse HEAD)" = {shlex.quote(settings.source_revision)}',
            'test -z "$(git status --porcelain=v1 --untracked-files=all)"',
            "set +e",
            shlex.join(_worker(settings, process)),
            "evaluation_status=$?",
            "set -e",
            (
                'printf "terminal action=activation-ladder-evaluate controller=%s '
                'process=%s status=%s\\n" '
                f'{controller} {process} "$evaluation_status"'
            ),
            'exit "$evaluation_status"',
        )
    )
    return _gcloud_ssh(
        settings,
        controller=controller,
        remote_command="bash -lc " + shlex.quote(body),
    )


def _bundle_command(
    settings: BundleSettings,
    *,
    controller: int,
    process: int,
) -> list[str]:
    base = settings.base
    body = "\n".join(
        (
            "set -euo pipefail",
            f"cd {shlex.quote(base.project_root)}",
            f'test "$(git rev-parse HEAD)" = {shlex.quote(base.source_revision)}',
            'test -z "$(git status --porcelain=v1 --untracked-files=all)"',
            "set +e",
            shlex.join(_bundle_worker(settings, process)),
            "evaluation_status=$?",
            "set -e",
            (
                'printf "terminal action=activation-ladder-evaluate-bundle '
                'controller=%s process=%s status=%s\\n" '
                f'{controller} {process} "$evaluation_status"'
            ),
            'exit "$evaluation_status"',
        )
    )
    return _gcloud_ssh(
        base,
        controller=controller,
        remote_command="bash -lc " + shlex.quote(body),
    )


def _run(command: list[str], log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("xb") as handle:
        process = subprocess.Popen(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=_subprocess_environment(),
        )
        try:
            return process.wait(timeout=WORKER_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
            return 124


def _terminal_status(log: Path, *, controller: int, process: int) -> int:
    prefix = (
        f"terminal action=activation-ladder-evaluate controller={controller} "
        f"process={process} status="
    ).encode("ascii")
    rows = [line for line in log.read_bytes().splitlines() if line.startswith(prefix)]
    if (
        len(rows) != 1
        or re.fullmatch(rb"(?:0|[1-9][0-9]{0,2})", rows[0][len(prefix) :])
        is None
    ):
        raise OperatorError(f"controller {controller} lacks one terminal marker")
    status = int(rows[0][len(prefix) :])
    if status > 255:
        raise OperatorError("terminal status exceeds shell range")
    return status


def _bundle_terminal_status(
    log: Path,
    *,
    controller: int,
    process: int,
) -> int:
    prefix = (
        "terminal action=activation-ladder-evaluate-bundle "
        f"controller={controller} process={process} status="
    ).encode("ascii")
    rows = [
        line for line in log.read_bytes().splitlines() if line.startswith(prefix)
    ]
    if (
        len(rows) != 1
        or re.fullmatch(
            rb"(?:0|[1-9][0-9]{0,2})",
            rows[0][len(prefix) :],
        )
        is None
    ):
        raise OperatorError(
            f"controller {controller} lacks one bundle terminal marker"
        )
    status = int(rows[0][len(prefix) :])
    if status > 255:
        raise OperatorError("bundle terminal status exceeds shell range")
    return status


def _validated_result_envelope(
    settings: Settings,
    payload: bytes,
    *,
    protocol_id: str = V2_PROTOCOL_ID,
) -> tuple[dict, str]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OperatorError("evaluation result is not JSON") from error
    if not isinstance(value, dict):
        raise OperatorError("evaluation result must be an object")
    claimed = value.get("result_sha256")
    unsigned = {key: value[key] for key in value if key != "result_sha256"}
    relative = value.get("score_data_object")
    if (
        not isinstance(claimed, str)
        or _SHA256.fullmatch(claimed) is None
        or canonical_sha256(unsigned) != claimed
        or value.get("schema_version")
        != (
            V3_RESULT_SCHEMA_VERSION
            if protocol_id == V3_PROTOCOL_ID
            else RESULT_SCHEMA_VERSION
        )
        or value.get("protocol_id") != protocol_id
        or value.get("status") != "cell_evaluated"
        or value.get("panel") != settings.panel
        or value.get("condition") != settings.condition
        or value.get("seed") != settings.seed
        or value.get("checkpoint_tokens") != settings.checkpoint_tokens
        or value.get("source_revision") != settings.source_revision
        or value.get("training_source_revision")
        != settings.training_source_revision
        or not isinstance(relative, str)
        or PurePosixPath(relative).is_absolute()
        or ".." in PurePosixPath(relative).parts
        or not relative.startswith("objects/")
        or not relative.endswith(".safetensors")
    ):
        raise OperatorError("evaluation result envelope has the wrong identity")
    return value, relative


def _publish(
    settings: Settings,
    *,
    result_payload: bytes,
    relative: str,
    score_payload: bytes,
) -> Path:
    destination = settings.result_root_controller
    result_path = destination / "evaluation-result.json"
    score_path = destination / relative
    if destination.exists():
        if (
            destination.is_dir()
            and not destination.is_symlink()
            and result_path.is_file()
            and not result_path.is_symlink()
            and score_path.is_file()
            and not score_path.is_symlink()
            and result_path.read_bytes() == result_payload
            and score_path.read_bytes() == score_payload
        ):
            return result_path
        raise OperatorError("controller completion destination is partial or changed")
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    staging = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    try:
        (staging / PurePosixPath(relative).parent).mkdir(
            parents=True, mode=0o700
        )
        (staging / "evaluation-result.json").write_bytes(result_payload)
        (staging / relative).write_bytes(score_payload)
        os.rename(staging, destination)
    except FileExistsError:
        if (
            result_path.is_file()
            and score_path.is_file()
            and result_path.read_bytes() == result_payload
            and score_path.read_bytes() == score_payload
        ):
            return result_path
        raise OperatorError("completion publication raced with different evidence")
    finally:
        if staging.exists():
            for child in sorted(staging.rglob("*"), reverse=True):
                if child.is_file():
                    child.unlink()
                elif child.is_dir():
                    child.rmdir()
            staging.rmdir()
    return result_path


def _validated_bundle_result(
    settings: BundleSettings,
    payload: bytes,
    *,
    expected_cells: Sequence[Mapping[str, object]],
) -> dict:
    try:
        result = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OperatorError("resident bundle result is not JSON") from error
    if not isinstance(result, dict):
        raise OperatorError("resident bundle result must be an object")
    claimed = result.get("result_sha256")
    unsigned = {
        name: result[name] for name in result if name != "result_sha256"
    }
    base = settings.base
    if (
        not isinstance(claimed, str)
        or _SHA256.fullmatch(claimed) is None
        or canonical_sha256(unsigned) != claimed
        or result.get("schema_version")
        != resident_bundle_result_schema(settings.protocol_id)
        or result.get("protocol_id") != settings.protocol_id
        or result.get("status") != "bundle_evaluated"
        or result.get("panel") != base.panel
        or result.get("seed") != base.seed
        or result.get("source_revision") != base.source_revision
        or result.get("operator_manifest_sha256")
        != settings.manifest_sha256
        or result.get("bundle_spec_sha256")
        != settings.worker_spec_sha256
        or result.get("cell_results")
        != [dict(row) for row in expected_cells]
    ):
        raise OperatorError("resident bundle result identity is invalid")
    return result


def _publish_bundle_result(
    settings: BundleSettings,
    payload: bytes,
) -> Path:
    destination = settings.bundle_result_root_controller
    result_path = destination / "bundle-result.json"
    if destination.exists():
        if (
            destination.is_dir()
            and not destination.is_symlink()
            and result_path.is_file()
            and not result_path.is_symlink()
            and result_path.read_bytes() == payload
        ):
            return result_path
        raise OperatorError(
            "controller bundle destination is partial or changed"
        )
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    staging = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    try:
        staging.mkdir(mode=0o700)
        (staging / "bundle-result.json").write_bytes(payload)
        os.rename(staging, destination)
    except FileExistsError:
        if result_path.is_file() and result_path.read_bytes() == payload:
            return result_path
        raise OperatorError(
            "bundle completion publication raced with different evidence"
        )
    finally:
        if staging.exists():
            for child in staging.iterdir():
                if child.is_file():
                    child.unlink()
            staging.rmdir()
    return result_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", required=True, choices=PANELS)
    parser.add_argument("--config", required=True)
    parser.add_argument("--condition", choices=EVALUATED_CONDITIONS)
    parser.add_argument("--h-config")
    parser.add_argument("--a256-config")
    parser.add_argument("--reference-run-contract-sha256")
    parser.add_argument("--reference-run-contract-file-sha256")
    parser.add_argument("--parent-training-result-sha256")
    parser.add_argument("--parent-training-result-file-sha256")
    parser.add_argument("--parent-run-contract-sha256")
    parser.add_argument("--parent-run-contract-file-sha256")
    parser.add_argument("--parent-checkpoint-pointer-sha256")
    parser.add_argument("--parent-checkpoint-pointer-file-sha256")
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--checkpoint-tokens", type=int)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--training-source-revision")
    parser.add_argument("--run-root")
    parser.add_argument("--result-root")
    parser.add_argument("--bundle-manifest")
    parser.add_argument("--attempt-id")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _run_bundle_operator(
    settings: BundleSettings,
    *,
    dry_run: bool,
) -> int:
    base = settings.base
    if base.log_root in {
        settings.bundle_result_root_controller,
        *(row.result_root_controller for row in settings.publications),
    }:
        raise OperatorError("bundle completion and operator logs must be separate")
    if dry_run:
        print(
            json.dumps(
                {
                    "bundle_spec_sha256": settings.worker_spec_sha256,
                    "cells": len(settings.publications),
                    "panel": base.panel,
                    "protocol_id": settings.protocol_id,
                    "seed": base.seed,
                    "worker_process_zero": _bundle_worker(settings, 0),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    mapping = _mapping(base)
    base.log_root.mkdir(parents=True, exist_ok=False, mode=0o700)
    logs = {
        controller: base.log_root
        / f"evaluation-bundle-c{controller:05d}-p{process:05d}.log"
        for controller, process in mapping.items()
    }
    with ThreadPoolExecutor(max_workers=PROCESS_COUNT) as pool:
        futures = {
            controller: pool.submit(
                _run,
                _bundle_command(
                    settings,
                    controller=controller,
                    process=process,
                ),
                logs[controller],
            )
            for controller, process in mapping.items()
        }
        transports = {
            controller: future.result()
            for controller, future in futures.items()
        }
    statuses = {
        controller: _bundle_terminal_status(
            logs[controller],
            controller=controller,
            process=process,
        )
        for controller, process in mapping.items()
    }
    if any(status != 0 for status in statuses.values()):
        raise OperatorError(
            f"resident evaluator worker failure: statuses={statuses}; "
            f"transports={transports}; logs={base.log_root}"
        )
    process_zero = next(
        controller for controller, process in mapping.items() if process == 0
    )
    expected_cells: list[dict[str, object]] = []
    for publication, worker_cell in zip(
        settings.publications,
        settings.worker_spec["cells"],
        strict=True,
    ):
        cell_settings = replace(
            base,
            condition=publication.condition,
            checkpoint_tokens=publication.checkpoint_tokens,
            training_source_revision=publication.training_source_revision,
            h_config=(
                worker_cell["reference_config"]
                if worker_cell["reference_kind"] == "h"
                else None
            ),
            a256_config=(
                worker_cell["reference_config"]
                if worker_cell["reference_kind"] == "a256"
                else None
            ),
            reference_run_contract_sha256=worker_cell[
                "reference_run_contract_sha256"
            ],
            reference_run_contract_file_sha256=worker_cell[
                "reference_run_contract_file_sha256"
            ],
            parent_training_result_sha256=worker_cell[
                "parent_training_result_sha256"
            ],
            parent_training_result_file_sha256=worker_cell[
                "parent_training_result_file_sha256"
            ],
            parent_run_contract_sha256=worker_cell[
                "parent_run_contract_sha256"
            ],
            parent_run_contract_file_sha256=worker_cell[
                "parent_run_contract_file_sha256"
            ],
            parent_checkpoint_pointer_sha256=worker_cell[
                "parent_checkpoint_pointer_sha256"
            ],
            parent_checkpoint_pointer_file_sha256=worker_cell[
                "parent_checkpoint_pointer_file_sha256"
            ],
            checkpoint_payload_root=worker_cell["checkpoint_payload_root"],
            run_root=worker_cell["run_root"],
            result_root=publication.result_root_remote,
            result_root_controller=publication.result_root_controller,
        )
        result_payload = _remote_bytes(
            base,
            controller=process_zero,
            remote_path=(
                publication.result_root_remote.rstrip("/")
                + "/evaluation-result.json"
            ),
        )
        result, relative = _validated_result_envelope(
            cell_settings,
            result_payload,
            protocol_id=settings.protocol_id,
        )
        score_payload = _remote_bytes(
            base,
            controller=process_zero,
            remote_path=(
                publication.result_root_remote.rstrip("/")
                + "/"
                + relative
            ),
        )
        if (
            hashlib.sha256(score_payload).hexdigest()
            != result.get("score_data_sha256")
            or len(score_payload) != result.get("score_data_bytes")
        ):
            raise OperatorError(
                f"resident cell score object changed: "
                f"{publication.condition}/{publication.checkpoint_tokens}"
            )
        result_path = _publish(
            cell_settings,
            result_payload=result_payload,
            relative=relative,
            score_payload=score_payload,
        )
        scores = load_merged_scores(result_path)
        if (
            scores.condition != publication.condition
            or scores.panel != base.panel
            or scores.seed != base.seed
            or scores.checkpoint_tokens != publication.checkpoint_tokens
            or scores.source_revision != base.source_revision
            or scores.score_artifact_sha256
            != result["score_artifact_sha256"]
        ):
            raise OperatorError("published resident cell has the wrong role")
        expected_cells.append(
            {
                "condition": publication.condition,
                "checkpoint_tokens": publication.checkpoint_tokens,
                "training_source_revision": (
                    publication.training_source_revision
                ),
                "parent_training_result_sha256": worker_cell[
                    "parent_training_result_sha256"
                ],
                "parent_training_result_file_sha256": worker_cell[
                    "parent_training_result_file_sha256"
                ],
                "parent_run_contract_sha256": worker_cell[
                    "parent_run_contract_sha256"
                ],
                "parent_run_contract_file_sha256": worker_cell[
                    "parent_run_contract_file_sha256"
                ],
                "parent_checkpoint_pointer_sha256": worker_cell[
                    "parent_checkpoint_pointer_sha256"
                ],
                "parent_checkpoint_pointer_file_sha256": worker_cell[
                    "parent_checkpoint_pointer_file_sha256"
                ],
                "result_root": publication.result_root_remote,
                "result_sha256": result["result_sha256"],
                "result_file_sha256": hashlib.sha256(
                    result_payload
                ).hexdigest(),
                "score_artifact_sha256": result[
                    "score_artifact_sha256"
                ],
                "score_data_sha256": result["score_data_sha256"],
                "score_data_bytes": result["score_data_bytes"],
            }
        )
    bundle_payload = _remote_bytes(
        base,
        controller=process_zero,
        remote_path=(
            settings.bundle_result_root_remote.rstrip("/")
            + "/bundle-result.json"
        ),
    )
    result = _validated_bundle_result(
        settings,
        bundle_payload,
        expected_cells=expected_cells,
    )
    path = _publish_bundle_result(settings, bundle_payload)
    if path.read_bytes() != bundle_payload:
        raise OperatorError("published resident bundle bytes changed")
    print(
        json.dumps(
            {
                "cells": len(expected_cells),
                "panel": base.panel,
                "result_sha256": result["result_sha256"],
                "seed": base.seed,
                "status": "bundle_evaluated",
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def main(argv: Sequence[str] = ()) -> int:
    args = _parser().parse_args(argv)
    if args.bundle_manifest is not None:
        if any(
            value is not None
            for value in (
                args.condition,
                args.h_config,
                args.a256_config,
                args.reference_run_contract_sha256,
                args.reference_run_contract_file_sha256,
                args.parent_training_result_sha256,
                args.parent_training_result_file_sha256,
                args.parent_run_contract_sha256,
                args.parent_run_contract_file_sha256,
                args.parent_checkpoint_pointer_sha256,
                args.parent_checkpoint_pointer_file_sha256,
                args.checkpoint_tokens,
                args.training_source_revision,
                args.run_root,
                args.result_root,
            )
        ):
            raise OperatorError(
                "resident bundle mode prohibits single-cell arguments"
            )
        settings = _bundle_settings(args, os.environ)
        return _run_bundle_operator(settings, dry_run=args.dry_run)
    if (
        args.attempt_id is not None
        or any(
            value is None
            for value in (
                args.condition,
                args.checkpoint_tokens,
                args.training_source_revision,
                args.run_root,
                args.result_root,
            )
        )
    ):
        raise OperatorError("single-cell evaluation arguments are incomplete")
    settings = _settings(args, os.environ)
    if settings.result_root_controller == settings.log_root:
        raise OperatorError("completion and operator logs must be separate")
    if "--dry-run" in argv:
        print(
            json.dumps(
                {
                    "condition": settings.condition,
                    "panel": settings.panel,
                    "protocol_id": PROTOCOL_ID,
                    "worker_process_zero": _worker(settings, 0),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    mapping = _mapping(settings)
    settings.log_root.mkdir(parents=True, exist_ok=False, mode=0o700)
    logs = {
        controller: settings.log_root
        / f"evaluation-c{controller:05d}-p{process:05d}.log"
        for controller, process in mapping.items()
    }
    with ThreadPoolExecutor(max_workers=PROCESS_COUNT) as pool:
        futures = {
            controller: pool.submit(
                _run,
                _command(settings, controller=controller, process=process),
                logs[controller],
            )
            for controller, process in mapping.items()
        }
        transports = {
            controller: future.result() for controller, future in futures.items()
        }
    statuses = {
        controller: _terminal_status(
            logs[controller], controller=controller, process=process
        )
        for controller, process in mapping.items()
    }
    if any(status != 0 for status in statuses.values()):
        raise OperatorError(
            f"evaluator worker failure: statuses={statuses}; "
            f"transports={transports}; logs={settings.log_root}"
        )
    process_zero = next(
        controller for controller, process in mapping.items() if process == 0
    )
    result_payload = _remote_bytes(
        settings,
        controller=process_zero,
        remote_path=settings.result_root.rstrip("/") + "/evaluation-result.json",
    )
    result, relative = _validated_result_envelope(settings, result_payload)
    score_payload = _remote_bytes(
        settings,
        controller=process_zero,
        remote_path=settings.result_root.rstrip("/") + "/" + relative,
    )
    if (
        hashlib.sha256(score_payload).hexdigest()
        != result.get("score_data_sha256")
        or len(score_payload) != result.get("score_data_bytes")
    ):
        raise OperatorError("content-addressed score object differs from result")
    result_path = _publish(
        settings,
        result_payload=result_payload,
        relative=relative,
        score_payload=score_payload,
    )
    scores = load_merged_scores(result_path)
    if (
        scores.condition != settings.condition
        or scores.panel != settings.panel
        or scores.seed != settings.seed
        or scores.checkpoint_tokens != settings.checkpoint_tokens
        or scores.source_revision != settings.source_revision
        or scores.score_artifact_sha256 != result["score_artifact_sha256"]
    ):
        raise OperatorError("published score artifact has the wrong role")
    print(
        json.dumps(
            {
                "condition": settings.condition,
                "panel": settings.panel,
                "result_sha256": result["result_sha256"],
                "score_artifact_sha256": scores.score_artifact_sha256,
                "seed": settings.seed,
                "status": "cell_evaluated",
                "target_tokens": settings.checkpoint_tokens,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except Exception as error:
        print(
            f"ROUTED-LADDER EVALUATION OPERATOR FAILED: {error}",
            file=sys.stderr,
        )
        raise SystemExit(1) from error
