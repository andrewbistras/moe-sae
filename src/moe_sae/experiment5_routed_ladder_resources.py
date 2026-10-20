"""Immutable N-way resource admission for the Experiment-5 ladder.

The scientific campaign plan embeds the admitted resource rows, but resource
selection happens one layer earlier.  This module validates a self-hashed
resource-pool manifest containing:

* between three and eighteen already-deployed TPU slices;
* the immutable mapping/cache/mirror evidence for each slice;
* the artifact plane owned by each slice; and
* the exact compatible resource set for every ``(panel, seed)`` cohort.

It performs no cloud discovery or mutation.  A resource can enter a campaign
only through a new manifest whose hash, deployment commit, evidence paths, and
cohort assignment are all bound before the controller starts.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from moe_sae.contribution_artifacts import canonical_sha256
from moe_sae.experiment5_campaign import (
    CampaignResource,
    _controller_path,
    _exact_mapping,
    _git_sha,
    _remote_path,
    _resource,
    _safe_id,
    _sha256,
)

RESOURCE_POOL_SCHEMA_VERSION = (
    "gemma4-experiment5-routed-activation-ladder-resource-pool-v1"
)
MIN_RESOURCE_COUNT = 3
MAX_RESOURCE_COUNT = 18
PANELS = ("sae", "transcoder")
SEEDS = (0, 1, 2)
COHORT_KEYS = tuple(
    f"{panel}.seed{seed}" for panel in PANELS for seed in SEEDS
)


class RoutedLadderResourcePoolError(RuntimeError):
    """The immutable resource-pool manifest is incomplete or inconsistent."""


@dataclass(frozen=True, slots=True)
class RoutedLadderArtifactPlane:
    resource_id: str
    controller_artifact_root: Path
    remote_artifact_root: str
    bulk_gcs_root: str

    @property
    def compatibility_key(self) -> tuple[str, str, str]:
        return (
            str(self.controller_artifact_root),
            self.remote_artifact_root,
            self.bulk_gcs_root,
        )


@dataclass(frozen=True, slots=True)
class RoutedLadderResourcePool:
    project_commit: str
    resources: tuple[CampaignResource, ...]
    resource_rows: tuple[Mapping[str, Any], ...]
    planes: tuple[RoutedLadderArtifactPlane, ...]
    cohorts: tuple[tuple[str, tuple[str, ...]], ...]
    pool_sha256: str
    source: Mapping[str, Any]

    def plane(self, resource_id: str) -> RoutedLadderArtifactPlane:
        for plane in self.planes:
            if plane.resource_id == resource_id:
                return plane
        raise RoutedLadderResourcePoolError(
            f"resource pool has no artifact plane for {resource_id!r}"
        )

    def cohort(self, panel: str, seed: int) -> tuple[str, ...]:
        key = f"{panel}.seed{seed}"
        for observed, resource_ids in self.cohorts:
            if observed == key:
                return resource_ids
        raise RoutedLadderResourcePoolError(
            f"resource pool has no cohort {key!r}"
        )


def _pool_mapping(value: Any, fields: set[str], name: str) -> dict[str, Any]:
    try:
        return _exact_mapping(value, fields, name)
    except Exception as error:
        raise RoutedLadderResourcePoolError(str(error)) from error


def _pool_path(value: Any, name: str) -> Path:
    try:
        return _controller_path(value, name)
    except Exception as error:
        raise RoutedLadderResourcePoolError(str(error)) from error


def _pool_remote(value: Any, name: str) -> str:
    try:
        return _remote_path(value, name)
    except Exception as error:
        raise RoutedLadderResourcePoolError(str(error)) from error


def _pool_gcs(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(
            r"gs://[a-z0-9][a-z0-9._-]*(?:/[^/\s].*)?",
            value,
        )
        is None
        or value.endswith("/")
    ):
        raise RoutedLadderResourcePoolError(
            f"{name} must be a canonical GCS bucket or object prefix"
        )
    return value


def _relative_to_controller(
    path: Path,
    root: Path,
    *,
    name: str,
) -> None:
    try:
        path.relative_to(root)
    except ValueError as error:
        raise RoutedLadderResourcePoolError(
            f"{name} is outside its controller artifact root"
        ) from error


def _relative_to_remote(
    path: str,
    root: str,
    *,
    name: str,
) -> None:
    try:
        PurePosixPath(path).relative_to(PurePosixPath(root))
    except ValueError as error:
        raise RoutedLadderResourcePoolError(
            f"{name} is outside its remote artifact root"
        ) from error


def _parse_plane(
    value: Any,
    *,
    resource_id: str,
    resource: CampaignResource,
) -> RoutedLadderArtifactPlane:
    selected = _pool_mapping(
        value,
        {
            "controller_artifact_root",
            "remote_artifact_root",
            "bulk_gcs_root",
        },
        f"planes.{resource_id}",
    )
    plane = RoutedLadderArtifactPlane(
        resource_id=resource_id,
        controller_artifact_root=_pool_path(
            selected["controller_artifact_root"],
            f"planes.{resource_id}.controller_artifact_root",
        ),
        remote_artifact_root=_pool_remote(
            selected["remote_artifact_root"],
            f"planes.{resource_id}.remote_artifact_root",
        ),
        bulk_gcs_root=_pool_gcs(
            selected["bulk_gcs_root"],
            f"planes.{resource_id}.bulk_gcs_root",
        ),
    )
    evidence = (
        ("controller mapping", resource.controller_mapping),
        ("cache inventory", resource.cache_inventory),
    )
    if resource.mirror_seal is not None:
        evidence += (("mirror seal", resource.mirror_seal),)
    for label, reference in evidence:
        _relative_to_controller(
            reference.controller_path,
            plane.controller_artifact_root,
            name=f"{resource_id} {label}",
        )
        _relative_to_remote(
            reference.remote_path,
            plane.remote_artifact_root,
            name=f"{resource_id} {label}",
        )
    declared_bulk = resource.environment_dict().get("BULK_GCS_ROOT")
    if declared_bulk is not None:
        normalized = declared_bulk.removesuffix("/horizon-10m")
        if normalized != plane.bulk_gcs_root:
            raise RoutedLadderResourcePoolError(
                f"{resource_id} bulk object plane differs from its environment"
            )
    return plane


def parse_routed_ladder_resource_pool(
    value: Any,
    *,
    project_commit: str,
) -> RoutedLadderResourcePool:
    """Validate and freeze one exact, self-hashed N-way resource pool."""

    selected = _pool_mapping(
        value,
        {
            "schema_version",
            "project_commit",
            "resources",
            "planes",
            "cohorts",
            "pool_sha256",
        },
        "resource_pool",
    )
    if selected["schema_version"] != RESOURCE_POOL_SCHEMA_VERSION:
        raise RoutedLadderResourcePoolError(
            "resource-pool schema identity changed"
        )
    try:
        selected_commit = _git_sha(
            selected["project_commit"],
            "resource_pool.project_commit",
        )
        expected_commit = _git_sha(project_commit, "project_commit")
        claimed_hash = _sha256(
            selected["pool_sha256"],
            "resource_pool.pool_sha256",
        )
    except Exception as error:
        raise RoutedLadderResourcePoolError(str(error)) from error
    if selected_commit != expected_commit:
        raise RoutedLadderResourcePoolError(
            "resource pool belongs to another deployed commit"
        )
    body = {
        key: selected[key] for key in selected if key != "pool_sha256"
    }
    if canonical_sha256(body) != claimed_hash:
        raise RoutedLadderResourcePoolError(
            "resource-pool self-hash changed"
        )
    raw_resources = selected["resources"]
    if not isinstance(raw_resources, list):
        raise RoutedLadderResourcePoolError(
            "resource_pool.resources must be a list"
        )
    if not MIN_RESOURCE_COUNT <= len(raw_resources) <= MAX_RESOURCE_COUNT:
        raise RoutedLadderResourcePoolError(
            "resource pool must contain between 3 and 18 resources"
        )
    try:
        resources = tuple(
            _resource(row, index, selected_commit)
            for index, row in enumerate(raw_resources)
        )
    except Exception as error:
        raise RoutedLadderResourcePoolError(str(error)) from error
    resource_ids = [resource.resource_id for resource in resources]
    tpu_names = [resource.tpu_name for resource in resources]
    if len(set(resource_ids)) != len(resource_ids):
        raise RoutedLadderResourcePoolError(
            "resource-pool IDs are not unique"
        )
    if len(set(tpu_names)) != len(tpu_names):
        raise RoutedLadderResourcePoolError(
            "resource-pool TPU names are not unique"
        )
    evidence_controller_paths = [
        str(reference.controller_path)
        for resource in resources
        for reference in (
            resource.controller_mapping,
            resource.cache_inventory,
        )
    ]
    evidence_remote_paths = [
        reference.remote_path
        for resource in resources
        for reference in (
            resource.controller_mapping,
            resource.cache_inventory,
        )
    ]
    if (
        len(set(evidence_controller_paths)) != len(evidence_controller_paths)
        or len(set(evidence_remote_paths)) != len(evidence_remote_paths)
    ):
        raise RoutedLadderResourcePoolError(
            "resource-pool mapping/cache evidence paths are not unique"
        )
    if any(resource.cache_profile != "paired" for resource in resources):
        raise RoutedLadderResourcePoolError(
            "every admitted ladder resource requires a paired cache"
        )
    by_id = {resource.resource_id: resource for resource in resources}

    raw_planes = selected["planes"]
    if not isinstance(raw_planes, Mapping) or set(raw_planes) != set(resource_ids):
        raise RoutedLadderResourcePoolError(
            "resource-pool planes must cover exactly the admitted resources"
        )
    planes = tuple(
        _parse_plane(
            raw_planes[resource_id],
            resource_id=resource_id,
            resource=by_id[resource_id],
        )
        for resource_id in resource_ids
    )
    plane_by_id = {plane.resource_id: plane for plane in planes}

    raw_cohorts = selected["cohorts"]
    if not isinstance(raw_cohorts, Mapping) or set(raw_cohorts) != set(COHORT_KEYS):
        raise RoutedLadderResourcePoolError(
            "resource-pool cohorts must cover exactly both panels and all seeds"
        )
    cohorts: list[tuple[str, tuple[str, ...]]] = []
    used_ids: set[str] = set()
    for key in COHORT_KEYS:
        raw_ids = raw_cohorts[key]
        if not isinstance(raw_ids, list) or not raw_ids or len(raw_ids) > 3:
            raise RoutedLadderResourcePoolError(
                f"resource_pool.cohorts.{key} must contain one to three resources"
            )
        try:
            ids = tuple(
                _safe_id(item, f"resource_pool.cohorts.{key}[{index}]")
                for index, item in enumerate(raw_ids)
            )
        except Exception as error:
            raise RoutedLadderResourcePoolError(str(error)) from error
        if len(set(ids)) != len(ids):
            raise RoutedLadderResourcePoolError(
                f"resource_pool.cohorts.{key} contains duplicates"
            )
        if any(resource_id not in by_id for resource_id in ids):
            raise RoutedLadderResourcePoolError(
                f"resource_pool.cohorts.{key} names an unknown resource"
            )
        compatibility = {
            (
                by_id[resource_id].region,
                by_id[resource_id].accelerator_type,
                plane_by_id[resource_id].compatibility_key,
            )
            for resource_id in ids
        }
        if len(compatibility) != 1:
            raise RoutedLadderResourcePoolError(
                f"resource_pool.cohorts.{key} crosses an artifact or accelerator plane"
            )
        cohorts.append((key, ids))
        used_ids.update(ids)
    if used_ids != set(resource_ids):
        raise RoutedLadderResourcePoolError(
            "every admitted resource must own at least one cohort"
        )
    return RoutedLadderResourcePool(
        project_commit=selected_commit,
        resources=resources,
        resource_rows=tuple(copy.deepcopy(raw_resources)),
        planes=planes,
        cohorts=tuple(cohorts),
        pool_sha256=claimed_hash,
        source=copy.deepcopy(selected),
    )


def load_routed_ladder_resource_pool(
    path: str | Path,
    *,
    project_commit: str,
) -> RoutedLadderResourcePool:
    selected = Path(path).expanduser().resolve()
    try:
        value = yaml.safe_load(selected.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise RoutedLadderResourcePoolError(
            f"cannot read routed-ladder resource pool: {selected}"
        ) from error
    return parse_routed_ladder_resource_pool(
        value,
        project_commit=project_commit,
    )


def seal_routed_ladder_resource_pool(
    body: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a self-hashed manifest after strict field validation at parse."""

    selected = dict(body)
    if "pool_sha256" in selected:
        raise RoutedLadderResourcePoolError(
            "resource-pool body already contains pool_sha256"
        )
    return {**selected, "pool_sha256": canonical_sha256(selected)}


__all__ = [
    "COHORT_KEYS",
    "MAX_RESOURCE_COUNT",
    "MIN_RESOURCE_COUNT",
    "RESOURCE_POOL_SCHEMA_VERSION",
    "RoutedLadderArtifactPlane",
    "RoutedLadderResourcePool",
    "RoutedLadderResourcePoolError",
    "load_routed_ladder_resource_pool",
    "parse_routed_ladder_resource_pool",
    "seal_routed_ladder_resource_pool",
]
