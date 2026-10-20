"""Build the exact Experiment-5 plan from the sealed horizon r8 inventory.

The r8 plan is the deployment evidence source, not a scheduling template.  It
contains eight resources, one of which (``gemma4-horizon-v5e-usc-b``) was
preempted after the horizon campaign.  This builder:

* admits only the seven known surviving TPU slices;
* derives the seven-row historical inventory without treating it as a
  scheduling template;
* copies controller mappings, cache inventories, and regional mirror seals
  byte-for-byte at the YAML-object level for four inactive Europe rows;
* requires one strict, final-commit incarnation override for the three active
  ready slices;
* binds every panel/seed group to one region and accelerator generation; and
* emits the strict 24-training/18-evaluation Experiment-5 matrix.

No cloud discovery is performed here.  ``deployed_commit`` is a plan
commitment which must be made true by deployment before the controller starts.
The Experiment-5 controller independently verifies the clean controller
checkout and all evidence hashes before launching work.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import yaml
from moe_sae.experiment5_campaign import (
    CONDITIONS,
    MILESTONES,
    PANELS,
    PHASE_NAME,
    PLAN_SCHEMA_VERSION,
    SEEDS,
    Experiment5CampaignPlan,
    Experiment5CampaignPlanError,
    parse_experiment5_campaign_plan,
)

SOURCE_SCHEMA_VERSION = "gemma4-horizon-direct-campaign-plan-v1"
RESOURCE_EVIDENCE_OVERRIDE_SCHEMA_VERSION = (
    "gemma4-experiment5-resource-evidence-overrides-v2"
)
EXECUTION_PROFILE = "ready-three-v1"
PREEMPTED_TPU_NAME = "gemma4-horizon-v5e-usc-b"
EXPECTED_SURVIVING_TPUS = frozenset(
    {
        "gemma4-horizon-harvest-0728",
        "gemma4-horizon-v4-spot-trans-b",
        "gemma4-horizon-v5e-usc-a",
        "gemma4-horizon-v5e-euw-a",
        "gemma4-horizon-v5e-euw-r",
        "gemma4-horizon-v5e-euw2-b",
        "gemma4-horizon-v5e-euw2-f",
    }
)
EXPECTED_SOURCE_TPUS = EXPECTED_SURVIVING_TPUS | {PREEMPTED_TPU_NAME}
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

# These are the three ready resources which can execute the whole matrix
# without waiting for the four repeatedly cycling Europe reservations.
# Evidence replacement is deliberately not a general patching surface:
# adding or removing a resource requires an intentional code change.
EXPECTED_RESOURCE_EVIDENCE_OVERRIDES = {
    "slice-sae-a": "gemma4-horizon-v5e-usc-a",
    "slice-sae-b": "gemma4-horizon-harvest-0728",
    "slice-transcoder-b": "gemma4-horizon-v4-spot-trans-b",
}
_RESOURCE_EVIDENCE_PROVISIONING = {
    "slice-sae-a": "SPOT",
    "slice-sae-b": "ON_DEMAND",
    "slice-transcoder-b": "SPOT",
}
_ACTIVE_RESOURCE_IDS = frozenset(EXPECTED_RESOURCE_EVIDENCE_OVERRIDES)
_PAIRED_CACHE_RESOURCE_IDS = _ACTIVE_RESOURCE_IDS
_READY_THREE_REQUIRED_LANE_BYTES = 159_000_000_000
_READY_THREE_REQUIRED_FREE_BYTES = 176 * 1024**3

# Seed zero uses the ready USC1 v5e for both panels.  Seeds one and two use the
# two ready central2 v4 slices as a shared pool.  Every H/F pair therefore has
# one accelerator generation while the four Europe rows remain inventory-only.
_GROUP_TPU_NAMES: dict[tuple[str, int], tuple[str, ...]] = {
    ("sae", 0): ("gemma4-horizon-v5e-usc-a",),
    ("transcoder", 0): ("gemma4-horizon-v5e-usc-a",),
    ("sae", 1): (
        "gemma4-horizon-harvest-0728",
        "gemma4-horizon-v4-spot-trans-b",
    ),
    ("sae", 2): (
        "gemma4-horizon-harvest-0728",
        "gemma4-horizon-v4-spot-trans-b",
    ),
    ("transcoder", 1): (
        "gemma4-horizon-harvest-0728",
        "gemma4-horizon-v4-spot-trans-b",
    ),
    ("transcoder", 2): (
        "gemma4-horizon-harvest-0728",
        "gemma4-horizon-v4-spot-trans-b",
    ),
}

_PRESERVED_GLOBAL_ENVIRONMENT = frozenset(
    {
        "ARTIFACT_ROOT",
        "BULK_GCS_ROOT",
        "CONTRIBUTION_TRAIN_CAPTURE_ROOT",
        "CONTRIBUTION_EVAL_CAPTURE_ROOT",
        "CONTRIBUTION_OBJECT_PREFIX",
        "CONTRIBUTION_EVAL_OBJECT_PREFIX",
        "CONTRIBUTION_RECOMPOSITION_SIDECAR",
        "CONTRIBUTION_SAE_CACHE_ROOT",
        "CONTRIBUTION_SAE_CACHE_MAX_BYTES",
        "TRANSCODER_HARVEST_CHIP_SECONDS",
        "TRANSCODER_PEAK_FLOPS_PER_SECOND",
    }
)
_SOURCE_OVERLAY_KEYS = (
    "CONTRIBUTION_SOURCE_OVERLAY_ROOT",
    "CONTRIBUTION_EVAL_SOURCE_OVERLAY_ROOT",
    "CONTRIBUTION_SOURCE_OVERLAY_PREFIX",
    "CONTRIBUTION_EVAL_SOURCE_OVERLAY_PREFIX",
)


class Experiment5PlanBuildError(RuntimeError):
    """The sealed r8 inventory cannot safely produce an Experiment-5 plan."""


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise Experiment5PlanBuildError(f"{name} must be a string-keyed mapping")
    return dict(value)


def _sequence(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise Experiment5PlanBuildError(f"{name} must be a list")
    return list(value)


def _required_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise Experiment5PlanBuildError(f"{name} must be a nonempty string")
    return value


def _exact_fields(
    value: Any,
    expected: set[str],
    name: str,
) -> dict[str, Any]:
    selected = _mapping(value, name)
    if set(selected) != expected:
        raise Experiment5PlanBuildError(
            f"{name} fields changed: expected={sorted(expected)!r}, "
            f"observed={sorted(selected)!r}"
        )
    return selected


def _sha256(value: Any, name: str) -> str:
    selected = _required_string(value, name)
    if _SHA256.fullmatch(selected) is None:
        raise Experiment5PlanBuildError(f"{name} must be a lowercase SHA-256")
    return selected


def _evidence_path(value: Any, name: str) -> dict[str, str]:
    selected = _exact_fields(
        value,
        {"controller_path", "remote_path", "sha256"},
        name,
    )
    controller_value = _required_string(
        selected["controller_path"], f"{name}.controller_path"
    )
    remote_value = _required_string(selected["remote_path"], f"{name}.remote_path")
    controller = Path(controller_value)
    remote = PurePosixPath(remote_value)
    if (
        not controller.is_absolute()
        or ".." in controller.parts
        or not remote.is_absolute()
        or ".." in remote.parts
    ):
        raise Experiment5PlanBuildError(
            f"{name} controller and remote paths must be absolute and canonical"
        )
    return {
        "controller_path": controller_value,
        "remote_path": remote_value,
        "sha256": _sha256(selected["sha256"], f"{name}.sha256"),
    }


def _read_hashed_json(
    evidence: Mapping[str, str],
    *,
    name: str,
) -> dict[str, Any]:
    path = Path(evidence["controller_path"])
    if path.is_symlink() or not path.is_file():
        raise Experiment5PlanBuildError(
            f"{name} is absent, non-regular, or a symlink: {path}"
        )
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise Experiment5PlanBuildError(f"cannot read {name}: {path}") from error
    if hashlib.sha256(payload).hexdigest() != evidence["sha256"]:
        raise Experiment5PlanBuildError(f"{name} byte SHA-256 differs")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Experiment5PlanBuildError(f"{name} is not JSON") from error
    return _mapping(value, name)


def _parse_resource_evidence_overrides(
    value: Mapping[str, Any] | None,
    *,
    project_commit: str,
    execution_profile: str,
) -> dict[str, dict[str, Any]]:
    if value is None:
        raise Experiment5PlanBuildError(
            f"{execution_profile} requires fresh final-commit evidence "
            "for all three active resources"
        )
    selected = _exact_fields(
        value,
        {"schema_version", "execution_profile", "project_commit", "resources"},
        "resource evidence overrides",
    )
    if selected["schema_version"] != RESOURCE_EVIDENCE_OVERRIDE_SCHEMA_VERSION:
        raise Experiment5PlanBuildError("resource evidence override schema changed")
    if selected["project_commit"] != project_commit:
        raise Experiment5PlanBuildError(
            "resource evidence overrides belong to another project commit"
        )
    if selected["execution_profile"] != execution_profile:
        raise Experiment5PlanBuildError(
            "resource evidence overrides belong to another execution profile"
        )
    resources = _mapping(selected["resources"], "resource evidence overrides.resources")
    expected_ids = set(EXPECTED_RESOURCE_EVIDENCE_OVERRIDES)
    if set(resources) != expected_ids:
        raise Experiment5PlanBuildError(
            "resource evidence override IDs changed: "
            f"expected={sorted(expected_ids)!r}, "
            f"observed={sorted(resources)!r}"
        )
    result: dict[str, dict[str, Any]] = {}
    for resource_id in sorted(expected_ids):
        record = _exact_fields(
            resources[resource_id],
            {
                "tpu_name",
                "tpu_description",
                "accelerator_contract",
                "spot_attestation",
                "controller_mapping",
                "cache_inventory",
            },
            f"resource evidence overrides.resources.{resource_id}",
        )
        expected_tpu = EXPECTED_RESOURCE_EVIDENCE_OVERRIDES[resource_id]
        if record["tpu_name"] != expected_tpu:
            raise Experiment5PlanBuildError(f"{resource_id} override TPU name changed")
        result[resource_id] = {
            "tpu_name": expected_tpu,
            "tpu_description": _evidence_path(
                record["tpu_description"],
                f"{resource_id}.tpu_description override",
            ),
            "accelerator_contract": _evidence_path(
                record["accelerator_contract"],
                f"{resource_id}.accelerator_contract override",
            ),
            "spot_attestation": _evidence_path(
                record["spot_attestation"],
                f"{resource_id}.spot_attestation override",
            ),
            "controller_mapping": _evidence_path(
                record["controller_mapping"],
                f"{resource_id}.controller_mapping override",
            ),
            "cache_inventory": _evidence_path(
                record["cache_inventory"],
                f"{resource_id}.cache_inventory override",
            ),
        }
    return result


def _validated_resource_evidence_override(
    override: Mapping[str, Any],
    *,
    resource_id: str,
    old_evidence: Mapping[str, Any],
    project_commit: str,
    project: str,
    tpu_name: str,
    zone: str,
    accelerator_type: str,
    expected_cache_root: str,
    controller_mount_root: str,
    remote_artifact_root: str,
) -> tuple[dict[str, Any], str]:
    description_evidence = _mapping(
        override.get("tpu_description"),
        f"{resource_id}.tpu_description override",
    )
    contract_evidence = _mapping(
        override.get("accelerator_contract"),
        f"{resource_id}.accelerator_contract override",
    )
    attestation_evidence = _mapping(
        override.get("spot_attestation"),
        f"{resource_id}.spot_attestation override",
    )
    mapping_evidence = _mapping(
        override.get("controller_mapping"),
        f"{resource_id}.controller_mapping override",
    )
    inventory_evidence = _mapping(
        override.get("cache_inventory"),
        f"{resource_id}.cache_inventory override",
    )
    if mapping_evidence == old_evidence.get(
        "controller_mapping"
    ) or inventory_evidence == old_evidence.get("cache_inventory"):
        raise Experiment5PlanBuildError(
            f"{resource_id} override does not replace both stale witnesses"
        )
    mapping_controller = Path(mapping_evidence["controller_path"])
    inventory_controller = Path(inventory_evidence["controller_path"])
    description_controller = Path(description_evidence["controller_path"])
    contract_controller = Path(contract_evidence["controller_path"])
    attestation_controller = Path(attestation_evidence["controller_path"])
    mapping_remote = PurePosixPath(mapping_evidence["remote_path"])
    inventory_remote = PurePosixPath(inventory_evidence["remote_path"])
    description_remote = PurePosixPath(description_evidence["remote_path"])
    contract_remote = PurePosixPath(contract_evidence["remote_path"])
    attestation_remote = PurePosixPath(attestation_evidence["remote_path"])
    try:
        mapping_controller_relative = mapping_controller.relative_to(
            Path(controller_mount_root)
        )
        inventory_controller_relative = inventory_controller.relative_to(
            Path(controller_mount_root)
        )
        description_controller_relative = description_controller.relative_to(
            Path(controller_mount_root)
        )
        contract_controller_relative = contract_controller.relative_to(
            Path(controller_mount_root)
        )
        attestation_controller_relative = attestation_controller.relative_to(
            Path(controller_mount_root)
        )
        mapping_remote_relative = mapping_remote.relative_to(
            PurePosixPath(remote_artifact_root)
        )
        inventory_remote_relative = inventory_remote.relative_to(
            PurePosixPath(remote_artifact_root)
        )
        description_remote_relative = description_remote.relative_to(
            PurePosixPath(remote_artifact_root)
        )
        contract_remote_relative = contract_remote.relative_to(
            PurePosixPath(remote_artifact_root)
        )
        attestation_remote_relative = attestation_remote.relative_to(
            PurePosixPath(remote_artifact_root)
        )
    except ValueError as error:
        raise Experiment5PlanBuildError(
            f"{resource_id} override is outside its exact artifact mounts"
        ) from error
    if (
        mapping_controller.name != "controller-mapping.json"
        or inventory_controller.name != "cache-inventory.json"
        or mapping_controller.parent != inventory_controller.parent
        or mapping_remote.name != "controller-mapping.json"
        or inventory_remote.name != "cache-inventory.json"
        or mapping_remote.parent != inventory_remote.parent
        or mapping_controller.parent.parent.name != "probes"
        or mapping_remote.parent.parent.name != "probes"
        or description_controller.name != "tpu-description.json"
        or contract_controller.name != "accelerator-contract.json"
        or attestation_controller.name != "spot-attestation.json"
        or description_remote.name != "tpu-description.json"
        or contract_remote.name != "accelerator-contract.json"
        or attestation_remote.name != "spot-attestation.json"
        or description_controller.parent != mapping_controller.parent.parent.parent
        or contract_controller.parent != mapping_controller.parent.parent.parent
        or attestation_controller.parent != contract_controller.parent
        or description_remote.parent != mapping_remote.parent.parent.parent
        or contract_remote.parent != mapping_remote.parent.parent.parent
        or attestation_remote.parent != contract_remote.parent
        or PurePosixPath(*mapping_controller_relative.parts) != mapping_remote_relative
        or PurePosixPath(*inventory_controller_relative.parts)
        != inventory_remote_relative
        or PurePosixPath(*description_controller_relative.parts)
        != description_remote_relative
        or PurePosixPath(*contract_controller_relative.parts)
        != contract_remote_relative
        or PurePosixPath(*attestation_controller_relative.parts)
        != attestation_remote_relative
    ):
        raise Experiment5PlanBuildError(
            f"{resource_id} override is not one exact mirrored common probe root"
        )

    raw_description = _read_hashed_json(
        description_evidence,
        name=f"{resource_id} fresh TPU description",
    )
    raw_contract = _read_hashed_json(
        contract_evidence,
        name=f"{resource_id} fresh accelerator contract",
    )
    raw_attestation = _read_hashed_json(
        attestation_evidence,
        name=f"{resource_id} fresh Spot attestation",
    )
    raw_mapping = _read_hashed_json(
        mapping_evidence,
        name=f"{resource_id} fresh controller mapping",
    )
    raw_inventory = _read_hashed_json(
        inventory_evidence,
        name=f"{resource_id} fresh cache inventory",
    )
    try:
        from moe_sae.regional_execution import (
            RegionalExecutionError,
            assert_spot_tpu_description,
            validate_accelerator_contract,
            validate_controller_mapping_receipt,
            validate_local_cache_inventory,
        )

        contract = validate_accelerator_contract(raw_contract)
        mapping = validate_controller_mapping_receipt(
            raw_mapping,
            expected_contract_sha256=contract["contract_sha256"],
        )
        inventory = validate_local_cache_inventory(
            raw_inventory,
            expected_contract_sha256=mapping["contract_sha256"],
            expected_controller_mapping_sha256=mapping["mapping_sha256"],
        )
        attestation = assert_spot_tpu_description(raw_description, contract)
        if attestation != raw_attestation:
            raise Experiment5PlanBuildError(
                f"{resource_id} fresh Spot attestation differs from "
                "its exact saved TPU description"
            )
    except (
        Experiment5PlanBuildError,
        RegionalExecutionError,
        TypeError,
        ValueError,
    ) as error:
        raise Experiment5PlanBuildError(
            f"{resource_id} fresh incarnation evidence is invalid"
        ) from error

    device_kinds = mapping["device_kinds"]
    mapping_pairs = {
        (row["controller_index"], row["process_index"])
        for row in mapping["controller_to_jax_process_index"]
    }
    inventory_pairs = {
        (row["controller_index"], row["process_index"]) for row in inventory["workers"]
    }
    expected_run_id = f"gemma4-experiment5-{EXECUTION_PROFILE}-{project_commit[:12]}"
    if (
        contract["run_id"] != expected_run_id
        or contract["scientific_model"] != "hybrid"
        or contract["condition"] != "all"
        or contract["project"] != project
        or contract["tpu_name"] != tpu_name
        or contract["region"] != _region(zone)
        or contract["zone"] != zone
        or contract["accelerator_type"] != accelerator_type
        or contract["provisioning_model"]
        != _RESOURCE_EVIDENCE_PROVISIONING[resource_id]
        or contract["process_count"] != 4
        or contract["global_device_count"] != 16
        or contract["local_device_count"] != 4
        or mapping["run_id"] != expected_run_id
        or mapping["tpu_name"] != tpu_name
        or mapping["zone"] != zone
        or mapping["accelerator_type"] != accelerator_type
        or mapping["resource_create_time"] != contract["resource_create_time"]
        or mapping["process_count"] != 4
        or mapping["global_device_count"] != 16
        or mapping["local_device_count"] != 4
        or mapping["probe_id"] != mapping_controller.parent.name
        or attestation["resource_create_time"] != mapping["resource_create_time"]
        or inventory["resource_create_time"] != mapping["resource_create_time"]
        or inventory["process_count"] != mapping["process_count"]
        or inventory_pairs != mapping_pairs
        or inventory["cache_kind"] != "host-ram-tmpfs"
        or not isinstance(device_kinds, list)
        or len(device_kinds) != 1
        or not isinstance(device_kinds[0], str)
        or not device_kinds[0].strip()
        or any(
            worker["cache_root"] != expected_cache_root
            or worker["required_lane_bytes"] != _READY_THREE_REQUIRED_LANE_BYTES
            or worker["required_free_bytes"] != _READY_THREE_REQUIRED_FREE_BYTES
            for worker in inventory["workers"]
        )
    ):
        raise Experiment5PlanBuildError(
            f"{resource_id} fresh evidence differs from its exact live role"
        )
    evidence = {
        "controller_mapping": copy.deepcopy(mapping_evidence),
        "cache_inventory": copy.deepcopy(inventory_evidence),
        # Mirror generation is deliberately inherited from the source plan.
        "mirror_seal": copy.deepcopy(old_evidence.get("mirror_seal")),
    }
    return evidence, device_kinds[0]


def _region(zone: str) -> str:
    try:
        region, suffix = zone.rsplit("-", 1)
    except ValueError as error:
        raise Experiment5PlanBuildError(f"invalid TPU zone {zone!r}") from error
    if len(suffix) != 1 or not suffix.isalpha():
        raise Experiment5PlanBuildError(f"invalid TPU zone {zone!r}")
    return region


def _prior_cache_profiles(source: Mapping[str, Any]) -> dict[str, str]:
    """Infer the historical cache role before profile-specific promotion.

    A horizon ``sae`` owner proved the target capture resident; a
    ``transcoder`` owner proved both target and routed-input overlay resident.
    Ready-three then promotes all three active slices to ``paired`` only after
    fresh empty-cache evidence proves capacity for a complete paired lane.
    """

    profiles: dict[str, str] = {}
    for index, raw in enumerate(_sequence(source.get("trajectories"), "trajectories")):
        task = _mapping(raw, f"trajectories[{index}]")
        resource_id = _required_string(
            task.get("resource_id"), f"trajectories[{index}].resource_id"
        )
        model = task.get("scientific_model")
        if model not in {"sae", "transcoder"}:
            raise Experiment5PlanBuildError(
                f"trajectories[{index}].scientific_model is invalid"
            )
        profile = "paired" if model == "transcoder" else "target"
        previous = profiles.setdefault(resource_id, profile)
        if previous != profile:
            raise Experiment5PlanBuildError(
                f"{resource_id} has conflicting prior cache assignments"
            )
    return profiles


def _controller_mount_root(
    source: Mapping[str, Any],
    *,
    resource_id: str,
) -> str:
    marker = "/horizon-10m/"
    for raw in _sequence(source.get("trajectories"), "trajectories"):
        task = _mapping(raw, "trajectory")
        if task.get("resource_id") != resource_id:
            continue
        completion = _required_string(
            task.get("completion_path_controller"),
            f"{resource_id}.completion_path_controller",
        )
        if marker not in completion:
            raise Experiment5PlanBuildError(
                f"{resource_id} completion path lacks the horizon mount marker"
            )
        return completion.split(marker, 1)[0]
    raise Experiment5PlanBuildError(
        f"{resource_id} has no prior assignment for mount discovery"
    )


def _regional_bulk_root(
    source_environment: Mapping[str, Any],
    resource: Mapping[str, Any],
) -> str:
    resource_environment = _mapping(
        resource.get("environment"), f"{resource.get('id')}.environment"
    )
    selected = resource_environment.get(
        "BULK_GCS_ROOT", source_environment.get("BULK_GCS_ROOT")
    )
    root = _required_string(selected, f"{resource.get('id')}.BULK_GCS_ROOT")
    return root.removesuffix("/horizon-10m")


def _remote_root(source: Mapping[str, Any]) -> str:
    environment = _mapping(source.get("environment"), "environment")
    artifact_root = _required_string(environment.get("ARTIFACT_ROOT"), "ARTIFACT_ROOT")
    return str(PurePosixPath(artifact_root) / "experiment5")


def _source_overlays(source: Mapping[str, Any]) -> dict[str, str]:
    environment = _mapping(source.get("environment"), "environment")
    result: dict[str, str] = {}
    for key in _SOURCE_OVERLAY_KEYS:
        result[key] = _required_string(environment.get(key), key)
    return result


def _resource_records(
    source: Mapping[str, Any],
    *,
    project_commit: str,
    resource_evidence_overrides: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    profiles = _prior_cache_profiles(source)
    for resource_id in _PAIRED_CACHE_RESOURCE_IDS:
        if resource_id not in profiles:
            raise Experiment5PlanBuildError(
                f"{resource_id} lacks a cache-capacity lineage"
            )
        profiles[resource_id] = "paired"
    source_environment = _mapping(source.get("environment"), "environment")
    expected_cache_root = _required_string(
        source_environment.get("CONTRIBUTION_SAE_CACHE_ROOT"),
        "CONTRIBUTION_SAE_CACHE_ROOT",
    )
    remote_artifact_root = _required_string(
        source_environment.get("ARTIFACT_ROOT"),
        "ARTIFACT_ROOT",
    )
    raw_resources = _sequence(source.get("resources"), "resources")
    observed_names = {
        _required_string(
            _mapping(raw, f"resources[{index}]").get("tpu_name"),
            f"resources[{index}].tpu_name",
        )
        for index, raw in enumerate(raw_resources)
    }
    if observed_names != EXPECTED_SOURCE_TPUS:
        raise Experiment5PlanBuildError(
            "r8 physical inventory changed: " f"observed={sorted(observed_names)!r}"
        )

    output: list[dict[str, Any]] = []
    by_name: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(raw_resources):
        old = _mapping(raw, f"resources[{index}]")
        tpu_name = _required_string(old.get("tpu_name"), f"resources[{index}].tpu_name")
        if tpu_name == PREEMPTED_TPU_NAME:
            continue
        resource_id = _required_string(old.get("id"), f"resources[{index}].id")
        if resource_id not in profiles:
            raise Experiment5PlanBuildError(
                f"{resource_id} has no prior cache assignment"
            )
        evidence = _mapping(old.get("evidence"), f"{resource_id}.evidence")
        if set(evidence) != {
            "controller_mapping",
            "cache_inventory",
            "mirror_seal",
        }:
            raise Experiment5PlanBuildError(f"{resource_id} evidence fields changed")
        # Force inspection of the inventory witness rather than classifying
        # solely from a resource name.
        inventory = _mapping(
            evidence.get("cache_inventory"), f"{resource_id}.cache_inventory"
        )
        if set(inventory) != {"controller_path", "remote_path", "sha256"}:
            raise Experiment5PlanBuildError(
                f"{resource_id} cache inventory witness is incomplete"
            )
        zone = _required_string(old.get("zone"), f"{resource_id}.zone")
        accelerator_type = _required_string(
            old.get("accelerator_type"), f"{resource_id}.accelerator_type"
        )
        project = _required_string(old.get("project"), f"{resource_id}.project")
        environment = copy.deepcopy(old.get("environment"))
        controller_mount_root = _controller_mount_root(source, resource_id=resource_id)
        if resource_id in resource_evidence_overrides:
            evidence, device_kind = _validated_resource_evidence_override(
                resource_evidence_overrides[resource_id],
                resource_id=resource_id,
                old_evidence=evidence,
                project_commit=project_commit,
                project=project,
                tpu_name=tpu_name,
                zone=zone,
                accelerator_type=accelerator_type,
                expected_cache_root=expected_cache_root,
                controller_mount_root=controller_mount_root,
                remote_artifact_root=remote_artifact_root,
            )
            environment = _mapping(environment, f"{resource_id}.environment")
            if accelerator_type == "v5litepod-16":
                environment["MOE_SAE_V5LITEPOD_DEVICE_KIND"] = device_kind
            elif "MOE_SAE_V5LITEPOD_DEVICE_KIND" in environment:
                raise Experiment5PlanBuildError(
                    f"{resource_id} non-v5 environment declares a v5 device kind"
                )
        record = {
            "id": resource_id,
            "project": project,
            "tpu_name": tpu_name,
            "zone": zone,
            "region": _region(zone),
            "accelerator_type": accelerator_type,
            "internal_ip": copy.deepcopy(old.get("internal_ip")),
            "ssh_user": copy.deepcopy(old.get("ssh_user")),
            "cache_profile": profiles[resource_id],
            "deployed_commit": project_commit,
            "evidence": copy.deepcopy(evidence),
            "environment": environment,
        }
        record["_controller_mount_root"] = controller_mount_root
        output.append(record)
        by_name[tpu_name] = record
    if set(by_name) != EXPECTED_SURVIVING_TPUS or len(output) != 7:
        raise Experiment5PlanBuildError("surviving r8 inventory is not exactly seven")
    return output, by_name


def _group_plane(
    by_name: Mapping[str, Mapping[str, Any]],
    source_environment: Mapping[str, Any],
    panel: str,
    seed: int,
) -> tuple[list[str], str, str, str]:
    names = _GROUP_TPU_NAMES[(panel, seed)]
    records = [by_name[name] for name in names]
    regions = {record["region"] for record in records}
    accelerators = {record["accelerator_type"] for record in records}
    mounts = {record["_controller_mount_root"] for record in records}
    if len(regions) != 1 or len(accelerators) != 1 or len(mounts) != 1:
        raise Experiment5PlanBuildError(
            f"{panel}.seed{seed} does not occupy one physical artifact plane"
        )
    if panel == "transcoder" and any(
        record["cache_profile"] != "paired" for record in records
    ):
        raise Experiment5PlanBuildError(
            f"{panel}.seed{seed} includes a target-only cache"
        )
    bulk_roots = {_regional_bulk_root(source_environment, record) for record in records}
    if len(bulk_roots) != 1:
        raise Experiment5PlanBuildError(
            f"{panel}.seed{seed} has multiple regional object planes"
        )
    return (
        [str(record["id"]) for record in records],
        str(next(iter(regions))),
        str(next(iter(mounts))),
        str(next(iter(bulk_roots))),
    )


def build_experiment5_plan(
    source: Mapping[str, Any],
    *,
    project_commit: str,
    execution_profile: str,
    campaign_id: str | None = None,
    already_spent_usd: float = 0.0,
    resource_evidence_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the strict ready-three Experiment-5 campaign plan.

    The override is not an arbitrary patch. It must cover exactly the three
    task-eligible resources, replace only incarnation-bound controller/cache
    evidence, and inherit sealed regional mirrors unchanged.
    """

    source = _mapping(source, "source")
    if execution_profile != EXECUTION_PROFILE:
        raise Experiment5PlanBuildError(
            f"execution_profile must be exactly {EXECUTION_PROFILE!r}"
        )
    scheduled_names = {name for names in _GROUP_TPU_NAMES.values() for name in names}
    if scheduled_names != set(EXPECTED_RESOURCE_EVIDENCE_OVERRIDES.values()):
        raise Experiment5PlanBuildError(
            "execution profile task ownership differs from its fresh-evidence set"
        )
    if source.get("schema_version") != SOURCE_SCHEMA_VERSION:
        raise Experiment5PlanBuildError("source is not a horizon r8 plan")
    if _GIT_SHA.fullmatch(project_commit) is None:
        raise Experiment5PlanBuildError("project_commit must be a full Git SHA")
    overrides = _parse_resource_evidence_overrides(
        resource_evidence_overrides,
        project_commit=project_commit,
        execution_profile=execution_profile,
    )
    source_environment = _mapping(source.get("environment"), "environment")
    resources, by_name = _resource_records(
        source,
        project_commit=project_commit,
        resource_evidence_overrides=overrides,
    )

    source_remote_project = _required_string(
        source.get("remote_project_root"), "remote_project_root"
    )
    short_commit = project_commit[:12]
    experiment_root = str(
        PurePosixPath(_remote_root(source)) / execution_profile / short_commit
    )
    overlays = _source_overlays(source)
    trajectories: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []
    for panel in PANELS:
        config_remote = str(
            PurePosixPath(source_remote_project)
            / "configs"
            / f"gemma4_hybrid_{panel}_v1.yaml"
        )
        for seed in SEEDS:
            eligible, _region_name, controller_mount, regional_bulk = _group_plane(
                by_name, source_environment, panel, seed
            )
            control_root = str(
                PurePosixPath(experiment_root) / "runs" / panel / f"seed{seed}"
            )
            checkpoint_root = (
                f"{regional_bulk}/experiment5/{execution_profile}/{short_commit}/"
                "checkpoints/"
                f"{panel}/seed{seed}"
            )
            controller_group_root = (
                Path(controller_mount)
                / "experiment5"
                / execution_profile
                / short_commit
                / "runs"
                / panel
                / f"seed{seed}"
            )
            train_source_root = (
                overlays["CONTRIBUTION_SOURCE_OVERLAY_ROOT"]
                if panel == "transcoder"
                else None
            )
            train_source_prefix = (
                overlays["CONTRIBUTION_SOURCE_OVERLAY_PREFIX"]
                if panel == "transcoder"
                else None
            )
            for condition in CONDITIONS:
                run_id = f"experiment5-{panel}-{condition.lower()}-seed{seed}"
                trajectories.append(
                    {
                        "id": f"train-{panel}-{condition.lower()}-seed{seed}",
                        "panel": panel,
                        "condition": condition,
                        "seed": seed,
                        "eligible_resource_ids": eligible,
                        "run_id": run_id,
                        "config_remote": config_remote,
                        "phase": PHASE_NAME,
                        "control_root_remote": control_root,
                        "checkpoint_root_gcs": checkpoint_root,
                        "source_overlay_control_root_remote": train_source_root,
                        "source_overlay_object_prefix_gcs": train_source_prefix,
                        "completion_path_controller": str(
                            controller_group_root
                            / "runs"
                            / run_id
                            / "training-result.json"
                        ),
                        "environment": {},
                    }
                )
            run_roots = {
                condition: str(
                    PurePosixPath(control_root)
                    / "runs"
                    / f"experiment5-{panel}-{condition.lower()}-seed{seed}"
                )
                for condition in CONDITIONS
            }
            for milestone in MILESTONES:
                remote_result = str(
                    PurePosixPath(experiment_root)
                    / "evaluation"
                    / panel
                    / f"seed{seed}"
                    / str(milestone)
                )
                controller_result = (
                    Path(controller_mount)
                    / "experiment5"
                    / execution_profile
                    / short_commit
                    / "evaluation"
                    / panel
                    / f"seed{seed}"
                    / str(milestone)
                )
                evaluations.append(
                    {
                        "id": f"eval-{panel}-seed{seed}-{milestone}",
                        "panel": panel,
                        "seed": seed,
                        "milestone_tokens": milestone,
                        "eligible_resource_ids": eligible,
                        "config_remote": config_remote,
                        "phase": PHASE_NAME,
                        "control_root_remote": control_root,
                        "checkpoint_root_gcs": checkpoint_root,
                        "train_source_root_remote": train_source_root,
                        "eval_source_root_remote": (
                            overlays["CONTRIBUTION_EVAL_SOURCE_OVERLAY_ROOT"]
                            if panel == "transcoder"
                            else None
                        ),
                        "train_source_object_prefix_gcs": train_source_prefix,
                        "eval_source_object_prefix_gcs": (
                            overlays["CONTRIBUTION_EVAL_SOURCE_OVERLAY_PREFIX"]
                            if panel == "transcoder"
                            else None
                        ),
                        "result_root_remote": remote_result,
                        "completion_path_controller": str(
                            controller_result / "scientific-result.json"
                        ),
                        "paired_scores_path_controller": str(
                            controller_result / "paired-scores.json"
                        ),
                        "run_roots_remote": run_roots,
                        "environment": {},
                    }
                )

    emitted_resources = []
    for record in resources:
        selected = dict(record)
        selected.pop("_controller_mount_root")
        emitted_resources.append(selected)
    environment = {
        key: copy.deepcopy(value)
        for key, value in source_environment.items()
        if key in _PRESERVED_GLOBAL_ENVIRONMENT
    }
    plan = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "campaign_id": campaign_id
        or f"gemma4-experiment5-{execution_profile}-{short_commit}",
        "state_root": (
            f"/mnt/moe-sae-control/experiment5/controller/"
            f"{execution_profile}/{short_commit}"
        ),
        "working_directory": copy.deepcopy(source.get("working_directory")),
        "python_executable": copy.deepcopy(source.get("python_executable")),
        "remote_project_root": source_remote_project,
        "remote_uv": copy.deepcopy(source.get("remote_uv")),
        "project_commit": project_commit,
        "max_parallel_train": 3,
        "max_parallel_eval": 3,
        "retry_policy": {
            "max_transient_attempts": 8,
            "initial_backoff_seconds": 30,
            "maximum_backoff_seconds": 600,
        },
        "cost_policy": {
            "hard_limit_usd": 50.0,
            "contingency_usd": 5.0,
            "already_spent_usd": already_spent_usd,
            "fixed_cost_reserve_usd": 5.0,
            "training_attempt_reserve_usd": 0.5,
            "evaluation_attempt_reserve_usd": 0.2,
        },
        "environment": environment,
        "resources": emitted_resources,
        "trajectories": trajectories,
        "evaluations": evaluations,
    }
    # The parser is the final authority.  Returning only after this call makes
    # the builder incapable of emitting a syntactically valid but unsafe plan.
    try:
        parse_experiment5_campaign_plan(plan)
    except Experiment5CampaignPlanError as error:
        raise Experiment5PlanBuildError(
            f"generated plan violates the Experiment-5 contract: {error}"
        ) from error
    return plan


def _generated_evidence_path(
    controller_path: Path,
    remote_path: PurePosixPath,
    *,
    name: str,
) -> dict[str, str]:
    if controller_path.is_symlink() or not controller_path.is_file():
        raise Experiment5PlanBuildError(
            f"{name} is absent, non-regular, or a symlink: {controller_path}"
        )
    try:
        payload = controller_path.read_bytes()
    except OSError as error:
        raise Experiment5PlanBuildError(
            f"cannot read {name}: {controller_path}"
        ) from error
    return {
        "controller_path": str(controller_path),
        "remote_path": str(remote_path),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def build_resource_evidence_overrides(
    source: Mapping[str, Any],
    *,
    project_commit: str,
    execution_profile: str,
    controller_probe_roots: Mapping[str, str | Path],
) -> dict[str, Any]:
    """Derive the three strict active-incarnation records from probe roots.

    ``controller_probe_roots`` names controller-side Filestore directories
    ending in ``probes/<probe-id>``.  Remote paths and all byte hashes are
    derived from the sealed r8 artifact-plane mapping; operators never
    transcribe them into YAML.
    """

    selected_source = _mapping(source, "source")
    if selected_source.get("schema_version") != SOURCE_SCHEMA_VERSION:
        raise Experiment5PlanBuildError("source is not a horizon r8 plan")
    if _GIT_SHA.fullmatch(project_commit) is None:
        raise Experiment5PlanBuildError("project_commit must be a full Git SHA")
    if execution_profile != EXECUTION_PROFILE:
        raise Experiment5PlanBuildError(
            f"execution_profile must be exactly {EXECUTION_PROFILE!r}"
        )
    roots = _mapping(controller_probe_roots, "controller probe roots")
    expected_ids = set(EXPECTED_RESOURCE_EVIDENCE_OVERRIDES)
    if set(roots) != expected_ids:
        raise Experiment5PlanBuildError(
            "controller probe root IDs changed: "
            f"expected={sorted(expected_ids)!r}, observed={sorted(roots)!r}"
        )
    resources = {
        _required_string(
            _mapping(raw, f"resources[{index}]").get("id"),
            f"resources[{index}].id",
        ): _mapping(raw, f"resources[{index}]")
        for index, raw in enumerate(
            _sequence(selected_source.get("resources"), "resources")
        )
    }
    remote_artifact_root = PurePosixPath(
        _required_string(
            _mapping(selected_source.get("environment"), "environment").get(
                "ARTIFACT_ROOT"
            ),
            "ARTIFACT_ROOT",
        )
    )
    records: dict[str, dict[str, Any]] = {}
    for resource_id in sorted(expected_ids):
        resource = resources.get(resource_id)
        if resource is None:
            raise Experiment5PlanBuildError(
                f"{resource_id} is absent from the r8 inventory"
            )
        expected_tpu = EXPECTED_RESOURCE_EVIDENCE_OVERRIDES[resource_id]
        if resource.get("tpu_name") != expected_tpu:
            raise Experiment5PlanBuildError(f"{resource_id} r8 TPU identity changed")
        raw_root = roots[resource_id]
        if not isinstance(raw_root, (str, Path)):
            raise Experiment5PlanBuildError(
                f"{resource_id} controller probe root must be a path"
            )
        probe_root = Path(raw_root).expanduser()
        if (
            not probe_root.is_absolute()
            or ".." in probe_root.parts
            or probe_root.is_symlink()
            or not probe_root.is_dir()
            or probe_root.parent.name != "probes"
            or not probe_root.name
        ):
            raise Experiment5PlanBuildError(
                f"{resource_id} controller probe root is not one "
                "probes/<probe-id> directory"
            )
        controller_mount = Path(
            _controller_mount_root(
                selected_source,
                resource_id=resource_id,
            )
        )
        try:
            relative_probe = probe_root.relative_to(controller_mount)
        except ValueError as error:
            raise Experiment5PlanBuildError(
                f"{resource_id} controller probe root is outside its r8 mount"
            ) from error
        remote_probe = remote_artifact_root / PurePosixPath(*relative_probe.parts)
        controller_assignment = probe_root.parent.parent
        remote_assignment = remote_probe.parent.parent
        records[resource_id] = {
            "tpu_name": expected_tpu,
            "tpu_description": _generated_evidence_path(
                controller_assignment / "tpu-description.json",
                remote_assignment / "tpu-description.json",
                name=f"{resource_id} TPU description",
            ),
            "accelerator_contract": _generated_evidence_path(
                controller_assignment / "accelerator-contract.json",
                remote_assignment / "accelerator-contract.json",
                name=f"{resource_id} accelerator contract",
            ),
            "spot_attestation": _generated_evidence_path(
                controller_assignment / "spot-attestation.json",
                remote_assignment / "spot-attestation.json",
                name=f"{resource_id} Spot attestation",
            ),
            "controller_mapping": _generated_evidence_path(
                probe_root / "controller-mapping.json",
                remote_probe / "controller-mapping.json",
                name=f"{resource_id} controller mapping",
            ),
            "cache_inventory": _generated_evidence_path(
                probe_root / "cache-inventory.json",
                remote_probe / "cache-inventory.json",
                name=f"{resource_id} cache inventory",
            ),
        }
    result = {
        "schema_version": RESOURCE_EVIDENCE_OVERRIDE_SCHEMA_VERSION,
        "execution_profile": execution_profile,
        "project_commit": project_commit,
        "resources": records,
    }
    # Exercise the same complete semantic authority chain used during plan
    # construction before exposing a serializable override artifact.
    build_experiment5_plan(
        selected_source,
        project_commit=project_commit,
        execution_profile=execution_profile,
        resource_evidence_overrides=result,
    )
    return result


def write_resource_evidence_overrides(
    path: str | Path,
    value: Mapping[str, Any],
) -> None:
    """Create, fsync, and never overwrite one override YAML artifact."""

    selected = Path(path).expanduser().resolve()
    selected.parent.mkdir(parents=True, exist_ok=True)
    serialized = yaml.safe_dump(
        dict(value),
        sort_keys=False,
        default_flow_style=False,
    )
    try:
        with selected.open("x", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise Experiment5PlanBuildError(
            f"refusing to overwrite resource evidence overrides: {selected}"
        ) from error


def load_r8_plan(path: str | Path) -> dict[str, Any]:
    selected = Path(path).expanduser().resolve()
    try:
        payload = yaml.safe_load(selected.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise Experiment5PlanBuildError(f"cannot read r8 plan: {selected}") from error
    return _mapping(payload, "source")


def load_resource_evidence_overrides(path: str | Path) -> dict[str, Any]:
    """Load the strict fresh-incarnation override without interpreting it."""

    selected = Path(path).expanduser().resolve()
    try:
        payload = yaml.safe_load(selected.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise Experiment5PlanBuildError(
            f"cannot read resource evidence overrides: {selected}"
        ) from error
    return _mapping(payload, "resource evidence overrides")


def _probe_root_argument(value: str) -> tuple[str, Path]:
    try:
        resource_id, raw_path = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "--probe-root must be RESOURCE_ID=/absolute/controller/probe/root"
        ) from error
    if not resource_id or not raw_path:
        raise argparse.ArgumentTypeError(
            "--probe-root must be RESOURCE_ID=/absolute/controller/probe/root"
        )
    return resource_id, Path(raw_path)


def _evidence_override_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create the strict three-resource Experiment-5 evidence override "
            "from controller-side fresh probe roots."
        )
    )
    parser.add_argument("--r8-plan", required=True, type=Path)
    parser.add_argument("--project-commit", required=True)
    parser.add_argument(
        "--execution-profile",
        required=True,
        choices=(EXECUTION_PROFILE,),
    )
    parser.add_argument(
        "--probe-root",
        required=True,
        action="append",
        type=_probe_root_argument,
        metavar="RESOURCE_ID=/ABSOLUTE/CONTROLLER/PROBE/ROOT",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser


def evidence_override_main(argv: Sequence[str] | None = None) -> int:
    args = _evidence_override_parser().parse_args(argv)
    roots: dict[str, Path] = {}
    for resource_id, path in args.probe_root:
        if resource_id in roots:
            raise SystemExit(
                f"Experiment-5 evidence override build failed: duplicate "
                f"--probe-root for {resource_id}"
            )
        roots[resource_id] = path
    try:
        value = build_resource_evidence_overrides(
            load_r8_plan(args.r8_plan),
            project_commit=args.project_commit,
            execution_profile=args.execution_profile,
            controller_probe_roots=roots,
        )
        write_resource_evidence_overrides(args.output, value)
    except Experiment5PlanBuildError as error:
        raise SystemExit(
            f"Experiment-5 evidence override build failed: {error}"
        ) from error
    print(
        f"wrote {args.output} project_commit={args.project_commit} "
        f"execution_profile={args.execution_profile} "
        f"resources={len(value['resources'])}"
    )
    return 0


def write_experiment5_plan(
    path: str | Path,
    plan: Mapping[str, Any],
) -> Experiment5CampaignPlan:
    """Create a new YAML plan and return its parsed immutable representation."""

    parsed = parse_experiment5_campaign_plan(plan)
    selected = Path(path).expanduser().resolve()
    selected.parent.mkdir(parents=True, exist_ok=True)
    if selected.exists():
        raise Experiment5PlanBuildError(f"refusing to overwrite plan: {selected}")
    serialized = yaml.safe_dump(
        dict(plan),
        sort_keys=False,
        default_flow_style=False,
    )
    selected.write_text(serialized, encoding="utf-8")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r8-plan", required=True, type=Path)
    parser.add_argument("--project-commit", required=True)
    parser.add_argument(
        "--execution-profile",
        required=True,
        choices=(EXECUTION_PROFILE,),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--campaign-id")
    parser.add_argument("--already-spent-usd", type=float, default=0.0)
    parser.add_argument(
        "--resource-evidence-overrides",
        type=Path,
        help=(
            "strict fresh-incarnation mapping/cache evidence for exactly the "
            "three active ready resources; regional mirror evidence "
            "cannot be overridden"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        plan = build_experiment5_plan(
            load_r8_plan(args.r8_plan),
            project_commit=args.project_commit,
            execution_profile=args.execution_profile,
            campaign_id=args.campaign_id,
            already_spent_usd=args.already_spent_usd,
            resource_evidence_overrides=(
                load_resource_evidence_overrides(args.resource_evidence_overrides)
                if args.resource_evidence_overrides is not None
                else None
            ),
        )
        parsed = write_experiment5_plan(args.output, plan)
    except (
        Experiment5PlanBuildError,
        Experiment5CampaignPlanError,
    ) as error:
        raise SystemExit(f"Experiment-5 plan build failed: {error}") from error
    print(
        f"wrote {args.output} plan_sha256={parsed.plan_sha256} "
        f"execution_profile={args.execution_profile} "
        f"resources={len(parsed.resources)} trajectories={len(parsed.trajectories)} "
        f"evaluations={len(parsed.evaluations)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
