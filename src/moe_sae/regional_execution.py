"""Regional, generation-pinned execution contracts for accelerator farms.

This module deliberately does not mutate cloud resources.  It provides the
small contracts needed by the Experiment-4 operator:

* one immutable source-object copy specification per destination region;
* a sealed mapping from original scientific objects to exact mirror
  generations;
* cache prewarming through those mirrors while retaining the original
  manifest identities;
* fresh physical-controller -> JAX-process identity receipts for every TPU
  slice incarnation;
* fail-closed Spot/topology and region-local path assertions; and
* an auditable, dependency-ordered cleanup plan.

The scientific contribution and routed-input manifests are never rewritten.
Their exact ``ObjectIdentity`` values remain the keys used by
:class:`~moe_sae.object_store.ShardCache`; only the byte source used to fill
the content-addressed local cache changes.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from moe_sae.contribution_artifacts import canonical_json_bytes, canonical_sha256
from moe_sae.object_store import (
    ImmutableObjectStore,
    ObjectIdentity,
    ObjectStoreError,
    ShardCache,
    join_gcs_uri,
    object_identity_from_mapping,
    parse_gcs_uri,
)

MIRROR_SPEC_SCHEMA_VERSION = "gemma4-regional-mirror-spec-v1"
SEALED_MIRROR_SCHEMA_VERSION = "gemma4-regional-mirror-seal-v1"
DELIVERY_LEDGER_SCHEMA_VERSION = "gemma4-regional-delivery-ledger-v1"
PREWARM_RECEIPT_SCHEMA_VERSION = "gemma4-regional-prewarm-receipt-v1"
ACCELERATOR_CONTRACT_SCHEMA_VERSION = "gemma4-accelerator-contract-v1"
CONTROLLER_OBSERVATION_SCHEMA_VERSION = "gemma4-controller-observation-v1"
CONTROLLER_MAPPING_SCHEMA_VERSION = "gemma4-controller-mapping-v1"
LOCAL_CACHE_OBSERVATION_SCHEMA_VERSION = "gemma4-local-cache-observation-v1"
LOCAL_CACHE_INVENTORY_SCHEMA_VERSION = "gemma4-local-cache-inventory-v1"
REGIONAL_PLACEMENT_SCHEMA_VERSION = "gemma4-regional-placement-v1"
CLEANUP_PLAN_SCHEMA_VERSION = "gemma4-regional-cleanup-plan-v1"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_REGION = re.compile(r"^[a-z]+-[a-z0-9]+[0-9]$")
_ZONE = re.compile(r"^[a-z]+-[a-z0-9]+[0-9]-[a-z]$")
_CREATE_TIME = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T"
    r"[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?Z$"
)
# A 180-GiB /dev/shm remount leaves a four-GiB operational margin above the
# 160-GiB resident-cache cap plus its exact ten-percent capacity headroom.
MINIMUM_10M_PROCESS_CACHE_FREE_BYTES = 176 * 1024**3
MINIMUM_10M_TMPFS_MEM_AVAILABLE_BYTES = 176 * 1024**3


class RegionalExecutionError(RuntimeError):
    """A regional delivery or runtime identity failed closed."""


def _safe_name(name: str, value: Any) -> str:
    if not isinstance(value, str) or _SAFE_NAME.fullmatch(value) is None:
        raise RegionalExecutionError(f"{name} is not a safe identifier")
    return value


def _sha256(name: str, value: Any) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise RegionalExecutionError(f"{name} must be a lowercase SHA-256")
    return value


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RegionalExecutionError(f"{name} must be a positive integer")
    return value


def _region(value: Any) -> str:
    if not isinstance(value, str) or _REGION.fullmatch(value) is None:
        raise RegionalExecutionError("region is invalid")
    return value


def _zone(value: Any, *, expected_region: str | None = None) -> str:
    if not isinstance(value, str) or _ZONE.fullmatch(value) is None:
        raise RegionalExecutionError("zone is invalid")
    region = value.rsplit("-", 1)[0]
    if expected_region is not None and region != expected_region:
        raise RegionalExecutionError("zone is outside the expected region")
    return value


def _canonical_copy(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(canonical_json_bytes(dict(value)))
    except Exception as error:
        raise RegionalExecutionError(
            "regional evidence must be finite canonical JSON"
        ) from error


def _seal(value: Mapping[str, Any], hash_field: str) -> dict[str, Any]:
    unsigned = _canonical_copy(value)
    if hash_field in unsigned:
        raise RegionalExecutionError(f"unsigned evidence already has {hash_field}")
    return {**unsigned, hash_field: canonical_sha256(unsigned)}


def _validate_seal(
    value: Mapping[str, Any],
    *,
    schema_version: str,
    hash_field: str,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RegionalExecutionError(f"{label} must be an object")
    selected = _canonical_copy(value)
    claimed = _sha256(f"{label}.{hash_field}", selected.pop(hash_field, None))
    if selected.get("schema_version") != schema_version:
        raise RegionalExecutionError(f"{label} has the wrong schema")
    if canonical_sha256(selected) != claimed:
        raise RegionalExecutionError(f"{label} self-hash mismatch")
    return {**selected, hash_field: claimed}


@dataclass(frozen=True, slots=True)
class MirrorDataset:
    """One sealed scientific manifest and its original immutable objects."""

    name: str
    manifest_sha256: str
    source_object_prefix: str
    objects: tuple[ObjectIdentity, ...]

    def __post_init__(self) -> None:
        _safe_name("dataset.name", self.name)
        _sha256("dataset.manifest_sha256", self.manifest_sha256)
        try:
            prefix_bucket, prefix_name = _parse_prefix(self.source_object_prefix)
        except ValueError as error:
            raise RegionalExecutionError(
                "dataset source_object_prefix is invalid"
            ) from error
        if not self.objects:
            raise RegionalExecutionError("mirror dataset cannot be empty")
        seen: set[tuple[str, int]] = set()
        sha_to_object: dict[str, tuple[str, int]] = {}
        for identity in self.objects:
            if not isinstance(identity, ObjectIdentity):
                raise RegionalExecutionError(
                    "dataset objects must be exact ObjectIdentity values"
                )
            bucket, name = parse_gcs_uri(identity.uri)
            if bucket != prefix_bucket or not name.startswith(f"{prefix_name}/"):
                raise RegionalExecutionError(
                    "scientific object is outside its bound source prefix"
                )
            key = (identity.uri, identity.generation)
            if key in seen:
                raise RegionalExecutionError(
                    "dataset repeats one immutable source object"
                )
            prior = sha_to_object.get(identity.sha256)
            if prior is not None and prior != key:
                raise RegionalExecutionError(
                    "distinct source objects unexpectedly share a SHA-256"
                )
            seen.add(key)
            sha_to_object[identity.sha256] = key


def _parse_prefix(value: str) -> tuple[str, str]:
    marker = join_gcs_uri(value.rstrip("/"), "identity-probe")
    bucket, name = parse_gcs_uri(marker)
    return bucket, name.removesuffix("/identity-probe")


def _mirror_uri(mirror_prefix: str, source_uri: str) -> str:
    bucket, name = parse_gcs_uri(source_uri)
    return join_gcs_uri(
        mirror_prefix.rstrip("/"),
        "objects",
        bucket,
        *name.split("/"),
    )


def build_regional_mirror_specification(
    *,
    run_id: str,
    region: str,
    mirror_prefix: str,
    datasets: Sequence[MirrorDataset],
) -> dict[str, Any]:
    """Plan exactly one source transfer for each unique object in a region."""

    selected_run = _safe_name("run_id", run_id)
    selected_region = _region(region)
    try:
        _parse_prefix(mirror_prefix)
    except ValueError as error:
        raise RegionalExecutionError("mirror_prefix is invalid") from error
    if not datasets:
        raise RegionalExecutionError("regional mirror requires at least one dataset")
    names: set[str] = set()
    manifests: set[str] = set()
    by_source: dict[tuple[str, int], dict[str, Any]] = {}
    dataset_rows: list[dict[str, Any]] = []
    sha_to_source: dict[str, tuple[str, int]] = {}
    for dataset in sorted(datasets, key=lambda item: item.name):
        if not isinstance(dataset, MirrorDataset):
            raise RegionalExecutionError("datasets must contain MirrorDataset values")
        if dataset.name in names or dataset.manifest_sha256 in manifests:
            raise RegionalExecutionError(
                "regional mirror dataset names and manifest hashes must be unique"
            )
        names.add(dataset.name)
        manifests.add(dataset.manifest_sha256)
        dataset_bytes = 0
        for identity in dataset.objects:
            key = (identity.uri, identity.generation)
            prior_sha = sha_to_source.get(identity.sha256)
            if prior_sha is not None and prior_sha != key:
                raise RegionalExecutionError(
                    "distinct immutable source identities share one SHA-256"
                )
            sha_to_source[identity.sha256] = key
            dataset_bytes += identity.size_bytes
            prior = by_source.get(key)
            if prior is None:
                by_source[key] = {
                    "source": identity.as_dict(),
                    "mirror_uri": _mirror_uri(mirror_prefix, identity.uri),
                    "datasets": [dataset.name],
                    "copy_preconditions": {
                        "source_generation_match": identity.generation,
                        "destination_generation_match": 0,
                    },
                }
            else:
                if prior["source"] != identity.as_dict():
                    raise RegionalExecutionError(
                        "one source generation has conflicting identities"
                    )
                prior["datasets"].append(dataset.name)
        dataset_rows.append(
            {
                "name": dataset.name,
                "manifest_sha256": dataset.manifest_sha256,
                "source_object_prefix": dataset.source_object_prefix.rstrip("/"),
                "object_count": len(dataset.objects),
                "logical_bytes": dataset_bytes,
            }
        )
    entries = sorted(
        by_source.values(),
        key=lambda item: (
            item["source"]["uri"],
            item["source"]["generation"],
        ),
    )
    for entry in entries:
        entry["datasets"] = sorted(entry["datasets"])
    body = {
        "schema_version": MIRROR_SPEC_SCHEMA_VERSION,
        "run_id": selected_run,
        "region": selected_region,
        "mirror_prefix": mirror_prefix.rstrip("/"),
        "scientific_manifests": dataset_rows,
        "objects": entries,
        "unique_source_objects": len(entries),
        "source_transfer_bytes": sum(
            int(entry["source"]["size_bytes"]) for entry in entries
        ),
        "copy_policy": (
            "one-generation-pinned-create-only-copy-per-source-object-per-region"
        ),
    }
    return _seal(body, "plan_sha256")


def validate_regional_mirror_specification(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    selected = _validate_seal(
        value,
        schema_version=MIRROR_SPEC_SCHEMA_VERSION,
        hash_field="plan_sha256",
        label="regional mirror specification",
    )
    expected_fields = {
        "schema_version",
        "run_id",
        "region",
        "mirror_prefix",
        "scientific_manifests",
        "objects",
        "unique_source_objects",
        "source_transfer_bytes",
        "copy_policy",
        "plan_sha256",
    }
    if set(selected) != expected_fields:
        raise RegionalExecutionError("regional mirror specification fields changed")
    _safe_name("run_id", selected["run_id"])
    _region(selected["region"])
    try:
        _parse_prefix(selected["mirror_prefix"])
    except ValueError as error:
        raise RegionalExecutionError("mirror_prefix is invalid") from error
    manifests = selected["scientific_manifests"]
    objects = selected["objects"]
    if (
        not isinstance(manifests, list)
        or not manifests
        or not isinstance(objects, list)
        or not objects
        or selected["copy_policy"]
        != "one-generation-pinned-create-only-copy-per-source-object-per-region"
    ):
        raise RegionalExecutionError("regional mirror specification is incomplete")
    manifest_names: set[str] = set()
    manifest_hashes: set[str] = set()
    for item in manifests:
        if not isinstance(item, Mapping) or set(item) != {
            "name",
            "manifest_sha256",
            "source_object_prefix",
            "object_count",
            "logical_bytes",
        }:
            raise RegionalExecutionError("scientific manifest binding fields changed")
        name = _safe_name("dataset.name", item["name"])
        digest = _sha256("dataset.manifest_sha256", item["manifest_sha256"])
        if name in manifest_names or digest in manifest_hashes:
            raise RegionalExecutionError("scientific manifest bindings are duplicated")
        manifest_names.add(name)
        manifest_hashes.add(digest)
        _positive_int("dataset.object_count", item["object_count"])
        _positive_int("dataset.logical_bytes", item["logical_bytes"])
        try:
            _parse_prefix(item["source_object_prefix"])
        except ValueError as error:
            raise RegionalExecutionError(
                "scientific source prefix is invalid"
            ) from error
    source_keys: set[tuple[str, int]] = set()
    mirror_uris: set[str] = set()
    sha_to_source: dict[str, tuple[str, int]] = {}
    total = 0
    for item in objects:
        if not isinstance(item, Mapping) or set(item) != {
            "source",
            "mirror_uri",
            "datasets",
            "copy_preconditions",
        }:
            raise RegionalExecutionError("regional mirror object fields changed")
        source = object_identity_from_mapping(item["source"])
        key = (source.uri, source.generation)
        if key in source_keys or item["mirror_uri"] in mirror_uris:
            raise RegionalExecutionError("regional mirror objects are duplicated")
        if item["mirror_uri"] != _mirror_uri(selected["mirror_prefix"], source.uri):
            raise RegionalExecutionError("regional mirror URI is not canonical")
        datasets_for_object = item["datasets"]
        if (
            not isinstance(datasets_for_object, list)
            or not datasets_for_object
            or datasets_for_object != sorted(set(datasets_for_object))
            or not set(datasets_for_object).issubset(manifest_names)
        ):
            raise RegionalExecutionError("mirror object dataset binding is invalid")
        if item["copy_preconditions"] != {
            "source_generation_match": source.generation,
            "destination_generation_match": 0,
        }:
            raise RegionalExecutionError("mirror copy preconditions were weakened")
        prior = sha_to_source.get(source.sha256)
        if prior is not None and prior != key:
            raise RegionalExecutionError(
                "distinct source objects unexpectedly share one SHA-256"
            )
        sha_to_source[source.sha256] = key
        source_keys.add(key)
        mirror_uris.add(item["mirror_uri"])
        total += source.size_bytes
    if (
        selected["unique_source_objects"] != len(objects)
        or selected["source_transfer_bytes"] != total
    ):
        raise RegionalExecutionError("regional mirror accounting changed")
    return selected


def seal_regional_mirror(
    specification: Mapping[str, Any],
    mirror_objects: Mapping[str, ObjectIdentity | Mapping[str, Any]],
) -> dict[str, Any]:
    """Bind every planned destination to the exact generation that exists."""

    plan = validate_regional_mirror_specification(specification)
    expected_uris = {entry["mirror_uri"] for entry in plan["objects"]}
    if set(mirror_objects) != expected_uris:
        raise RegionalExecutionError(
            "mirror observations do not exactly cover the planned destinations"
        )
    entries: list[dict[str, Any]] = []
    for planned in plan["objects"]:
        source = object_identity_from_mapping(planned["source"])
        raw_mirror = mirror_objects[planned["mirror_uri"]]
        mirror = (
            raw_mirror
            if isinstance(raw_mirror, ObjectIdentity)
            else object_identity_from_mapping(raw_mirror)
        )
        if (
            mirror.uri != planned["mirror_uri"]
            or mirror.uri == source.uri
            or mirror.sha256 != source.sha256
            or mirror.size_bytes != source.size_bytes
            or mirror.crc32c != source.crc32c
        ):
            raise RegionalExecutionError(
                "regional mirror differs from its original immutable content"
            )
        entries.append(
            {
                "source": source.as_dict(),
                "mirror": mirror.as_dict(),
                "datasets": list(planned["datasets"]),
            }
        )
    body = {
        "schema_version": SEALED_MIRROR_SCHEMA_VERSION,
        "run_id": plan["run_id"],
        "region": plan["region"],
        "mirror_prefix": plan["mirror_prefix"],
        "mirror_plan_sha256": plan["plan_sha256"],
        "scientific_manifests": plan["scientific_manifests"],
        "objects": entries,
        "unique_source_objects": plan["unique_source_objects"],
        "source_transfer_bytes": plan["source_transfer_bytes"],
        "scientific_identity_policy": (
            "original-manifest-and-object-identities-remain-authoritative"
        ),
    }
    return _seal(body, "mirror_sha256")


def validate_sealed_regional_mirror(value: Mapping[str, Any]) -> dict[str, Any]:
    selected = _validate_seal(
        value,
        schema_version=SEALED_MIRROR_SCHEMA_VERSION,
        hash_field="mirror_sha256",
        label="sealed regional mirror",
    )
    expected_fields = {
        "schema_version",
        "run_id",
        "region",
        "mirror_prefix",
        "mirror_plan_sha256",
        "scientific_manifests",
        "objects",
        "unique_source_objects",
        "source_transfer_bytes",
        "scientific_identity_policy",
        "mirror_sha256",
    }
    if set(selected) != expected_fields:
        raise RegionalExecutionError("sealed regional mirror fields changed")
    _safe_name("run_id", selected["run_id"])
    _region(selected["region"])
    _sha256("mirror_plan_sha256", selected["mirror_plan_sha256"])
    _parse_prefix(selected["mirror_prefix"])
    if (
        selected["scientific_identity_policy"]
        != "original-manifest-and-object-identities-remain-authoritative"
        or not isinstance(selected["scientific_manifests"], list)
        or not isinstance(selected["objects"], list)
        or not selected["objects"]
    ):
        raise RegionalExecutionError("sealed regional mirror is incomplete")
    source_keys: set[tuple[str, int]] = set()
    mirror_keys: set[tuple[str, int]] = set()
    total = 0
    for item in selected["objects"]:
        if not isinstance(item, Mapping) or set(item) != {
            "source",
            "mirror",
            "datasets",
        }:
            raise RegionalExecutionError("sealed mirror object fields changed")
        source = object_identity_from_mapping(item["source"])
        mirror = object_identity_from_mapping(item["mirror"])
        if (
            mirror.uri != _mirror_uri(selected["mirror_prefix"], source.uri)
            or mirror.uri == source.uri
            or (mirror.sha256, mirror.size_bytes, mirror.crc32c)
            != (source.sha256, source.size_bytes, source.crc32c)
        ):
            raise RegionalExecutionError("sealed mirror content identity changed")
        source_key = (source.uri, source.generation)
        mirror_key = (mirror.uri, mirror.generation)
        if source_key in source_keys or mirror_key in mirror_keys:
            raise RegionalExecutionError("sealed mirror repeats an object")
        if (
            not isinstance(item["datasets"], list)
            or not item["datasets"]
            or item["datasets"] != sorted(set(item["datasets"]))
        ):
            raise RegionalExecutionError("sealed mirror dataset bindings changed")
        source_keys.add(source_key)
        mirror_keys.add(mirror_key)
        total += source.size_bytes
    if (
        selected["unique_source_objects"] != len(source_keys)
        or selected["source_transfer_bytes"] != total
    ):
        raise RegionalExecutionError("sealed mirror accounting changed")
    return selected


class RegionalMirrorStore:
    """Read original identities from exact regional mirror generations."""

    def __init__(
        self,
        sealed_mirror: Mapping[str, Any],
        backing_store: ImmutableObjectStore,
    ) -> None:
        self.sealed_mirror = validate_sealed_regional_mirror(sealed_mirror)
        self.backing_store = backing_store
        self._by_source = {
            (item["source"]["uri"], item["source"]["generation"]): (
                object_identity_from_mapping(item["source"]),
                object_identity_from_mapping(item["mirror"]),
            )
            for item in self.sealed_mirror["objects"]
        }

    def _mirror_for(self, value: ObjectIdentity) -> ObjectIdentity:
        pair = self._by_source.get((value.uri, value.generation))
        if pair is None or pair[0] != value:
            raise ObjectStoreError(
                "scientific object is absent from the sealed regional mirror"
            )
        return pair[1]

    def put_if_absent(
        self,
        source: Path,
        uri: str,
        *,
        sha256: str,
        source_fingerprint: Any | None = None,
    ) -> ObjectIdentity:
        del source, uri, sha256, source_fingerprint
        raise ObjectStoreError("regional mirror store is read-only")

    def download(self, identity: ObjectIdentity, destination: Path) -> None:
        self.backing_store.download(self._mirror_for(identity), destination)

    def verify(self, identity: ObjectIdentity) -> None:
        self.backing_store.verify(self._mirror_for(identity))

    def delete(self, identity: ObjectIdentity) -> None:
        del identity
        raise ObjectStoreError("regional mirror store is read-only")


def prewarm_regional_cache(
    *,
    sealed_mirror: Mapping[str, Any],
    backing_store: ImmutableObjectStore,
    cache_root: str | Path,
    cache_max_bytes: int,
    datasets: Iterable[str] | None = None,
    max_workers: int = 4,
) -> dict[str, Any]:
    """Populate a SHA-addressed cache from mirrors under original identities."""

    mirror = validate_sealed_regional_mirror(sealed_mirror)
    available = {
        item["name"]: item["manifest_sha256"]
        for item in mirror["scientific_manifests"]
    }
    selected_names = (
        sorted(available)
        if datasets is None
        else sorted({_safe_name("dataset", name) for name in datasets})
    )
    if not selected_names or not set(selected_names).issubset(available):
        raise RegionalExecutionError(
            "prewarm datasets are empty or absent from the sealed mirror"
        )
    identities = tuple(
        object_identity_from_mapping(item["source"])
        for item in mirror["objects"]
        if set(item["datasets"]).intersection(selected_names)
    )
    if not identities:
        raise RegionalExecutionError("prewarm selected no immutable objects")
    required_bytes = sum(identity.size_bytes for identity in identities)
    if cache_max_bytes < required_bytes:
        raise RegionalExecutionError(
            "cache bound is smaller than the selected resident corpus"
        )
    cache = ShardCache(
        cache_root,
        RegionalMirrorStore(mirror, backing_store),
        max_bytes=cache_max_bytes,
    )
    paths = cache.materialize_many(identities, max_workers=max_workers)
    resident = cache.require_resident_many(identities)
    if paths != resident:
        raise RegionalExecutionError(
            "regional prewarm differs from its resident-only proof"
        )
    body = {
        "schema_version": PREWARM_RECEIPT_SCHEMA_VERSION,
        "run_id": mirror["run_id"],
        "region": mirror["region"],
        "mirror_sha256": mirror["mirror_sha256"],
        "datasets": selected_names,
        "scientific_manifests": [
            {"name": name, "manifest_sha256": available[name]}
            for name in selected_names
        ],
        "original_objects": [identity.as_dict() for identity in identities],
        "object_count": len(identities),
        "object_bytes": required_bytes,
        "cache_root": str(Path(cache_root).expanduser().resolve()),
        "resident_only_complete": True,
        "source_policy": (
            "mirror-download-original-identity-sha-addressed-resident-cache"
        ),
    }
    return _seal(body, "receipt_sha256")


def build_delivery_ledger(
    sealed_mirrors: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Prove that every region receives at most one combined source copy."""

    if not sealed_mirrors:
        raise RegionalExecutionError("delivery ledger cannot be empty")
    mirrors = [validate_sealed_regional_mirror(item) for item in sealed_mirrors]
    run_ids = {item["run_id"] for item in mirrors}
    regions = [item["region"] for item in mirrors]
    if len(run_ids) != 1 or len(regions) != len(set(regions)):
        raise RegionalExecutionError(
            "delivery ledger requires one run and one combined mirror per region"
        )
    rows = [
        {
            "region": item["region"],
            "mirror_sha256": item["mirror_sha256"],
            "unique_source_objects": item["unique_source_objects"],
            "source_transfer_bytes": item["source_transfer_bytes"],
            "scientific_manifests": item["scientific_manifests"],
        }
        for item in sorted(mirrors, key=lambda row: row["region"])
    ]
    body = {
        "schema_version": DELIVERY_LEDGER_SCHEMA_VERSION,
        "run_id": next(iter(run_ids)),
        "regions": rows,
        "total_cross_region_bytes": sum(
            int(row["source_transfer_bytes"]) for row in rows
        ),
        "accounting_policy": (
            "each-unique-source-generation-counted-once-per-destination-region"
        ),
    }
    return _seal(body, "ledger_sha256")


def build_accelerator_contract(
    *,
    run_id: str,
    scientific_model: str,
    condition: str,
    project: str,
    tpu_name: str,
    resource_create_time: str,
    region: str,
    zone: str,
    accelerator_type: str = "v6e-16",
    provisioning_model: str = "SPOT",
    process_count: int = 4,
    global_device_count: int = 16,
    local_device_count: int = 4,
) -> dict[str, Any]:
    """Freeze one TPU slice incarnation and its required logical topology."""

    selected_region = _region(region)
    selected_zone = _zone(zone, expected_region=selected_region)
    if (
        not isinstance(project, str)
        or re.fullmatch(r"[a-z][a-z0-9-]{4,61}[a-z0-9]", project) is None
    ):
        raise RegionalExecutionError("project is invalid")
    if (
        not isinstance(resource_create_time, str)
        or _CREATE_TIME.fullmatch(resource_create_time) is None
    ):
        raise RegionalExecutionError("resource_create_time is not canonical UTC")
    processes = _positive_int("process_count", process_count)
    global_devices = _positive_int("global_device_count", global_device_count)
    local_devices = _positive_int("local_device_count", local_device_count)
    if processes * local_devices != global_devices:
        raise RegionalExecutionError("accelerator topology does not multiply exactly")
    if accelerator_type == "v4-32":
        if provisioning_model not in {"ON_DEMAND", "SPOT"}:
            raise RegionalExecutionError(
                "approved v4-32 accelerator contract requires ON_DEMAND or Spot"
            )
    elif accelerator_type in {"v6e-16", "v5litepod-16"}:
        if provisioning_model != "SPOT":
            raise RegionalExecutionError(
                "approved regional accelerator contract requires Spot"
            )
    else:
        raise RegionalExecutionError("accelerator type is unsupported")
    body = {
        "schema_version": ACCELERATOR_CONTRACT_SCHEMA_VERSION,
        "run_id": _safe_name("run_id", run_id),
        "scientific_model": _safe_name("scientific_model", scientific_model),
        "condition": _safe_name("condition", condition),
        "project": project,
        "tpu_name": _safe_name("tpu_name", tpu_name),
        "resource_create_time": resource_create_time,
        "region": selected_region,
        "zone": selected_zone,
        "accelerator_type": _safe_name("accelerator_type", accelerator_type),
        "provisioning_model": provisioning_model,
        "process_count": processes,
        "global_device_count": global_devices,
        "local_device_count": local_devices,
    }
    return _seal(body, "contract_sha256")


def validate_accelerator_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    selected = _validate_seal(
        value,
        schema_version=ACCELERATOR_CONTRACT_SCHEMA_VERSION,
        hash_field="contract_sha256",
        label="accelerator contract",
    )
    required = {
        "schema_version",
        "run_id",
        "scientific_model",
        "condition",
        "project",
        "tpu_name",
        "resource_create_time",
        "region",
        "zone",
        "accelerator_type",
        "provisioning_model",
        "process_count",
        "global_device_count",
        "local_device_count",
        "contract_sha256",
    }
    if set(selected) != required:
        raise RegionalExecutionError("accelerator contract fields changed")
    rebuilt = build_accelerator_contract(
        **{
            key: selected[key]
            for key in (
                "run_id",
                "scientific_model",
                "condition",
                "project",
                "tpu_name",
                "resource_create_time",
                "region",
                "zone",
                "accelerator_type",
                "provisioning_model",
                "process_count",
                "global_device_count",
                "local_device_count",
            )
        }
    )
    if rebuilt != selected:
        raise RegionalExecutionError("accelerator contract is not canonical")
    return selected


def assert_spot_tpu_description(
    description: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate saved ``gcloud ... describe --format=json`` evidence.

    The historical name is retained for callers.  Regional v5/v6 slices are
    required to be Spot, while the already-allocated central v4-32 horizon
    slice is required to be on-demand.
    """

    expected = validate_accelerator_contract(contract)
    if not isinstance(description, Mapping):
        raise RegionalExecutionError("TPU description must be a JSON object")
    name = str(description.get("name", "")).rstrip("/").split("/")[-1]
    scheduling = description.get("schedulingConfig")
    create_time = description.get("createTime")
    expected_spot = expected["provisioning_model"] == "SPOT"
    scheduling_matches = (
        isinstance(scheduling, Mapping)
        and (
            (
                scheduling.get("spot") is True
                and (
                    "preemptible" not in scheduling
                    or scheduling.get("preemptible") is True
                )
            )
            if expected_spot
            else (
                scheduling.get("spot") is not True
                and scheduling.get("preemptible") is not True
            )
        )
    )
    if (
        name != expected["tpu_name"]
        or description.get("acceleratorType") != expected["accelerator_type"]
        or description.get("state") != "READY"
        or create_time != expected["resource_create_time"]
        or not scheduling_matches
    ):
        raise RegionalExecutionError(
            "TPU description is not the exact ready provisioned slice incarnation"
        )
    full_name = str(description.get("name", ""))
    if (
        f"/projects/{expected['project']}/" not in f"/{full_name}"
        and not full_name.startswith(
            f"projects/{expected['project']}/"
        )
    ):
        raise RegionalExecutionError("TPU description project changed")
    location = ""
    match = re.search(r"/locations/([^/]+)/", full_name)
    if match is not None:
        location = match.group(1)
    else:
        location = str(description.get("zone", "")).split("/")[-1]
    if location != expected["zone"]:
        raise RegionalExecutionError("TPU description zone changed")
    body = {
        "schema_version": "gemma4-accelerator-description-attestation-v1",
        "contract_sha256": expected["contract_sha256"],
        "description_sha256": canonical_sha256(description),
        "tpu_name": expected["tpu_name"],
        "resource_create_time": expected["resource_create_time"],
        "zone": expected["zone"],
        "accelerator_type": expected["accelerator_type"],
        "state": "READY",
        "provisioning_model": expected["provisioning_model"],
        "spot": expected_spot,
    }
    return _seal(body, "attestation_sha256")


def build_controller_observation(
    *,
    contract: Mapping[str, Any],
    probe_id: str,
    controller_index: int,
    process_index: int,
    process_count: int,
    global_device_count: int,
    local_device_count: int,
    platforms: Sequence[str],
    device_kinds: Sequence[str],
    boot_id: str,
    hostname: str,
) -> dict[str, Any]:
    """Record one physical controller's identity after JAX initialization."""

    expected = validate_accelerator_contract(contract)
    controller = _positive_int(
        "controller_index", controller_index + 1
    ) - 1
    process = _positive_int("process_index", process_index + 1) - 1
    if (
        controller >= expected["process_count"]
        or process >= expected["process_count"]
        or process_count != expected["process_count"]
        or global_device_count != expected["global_device_count"]
        or local_device_count != expected["local_device_count"]
    ):
        raise RegionalExecutionError(
            "controller observation differs from the required topology"
        )
    selected_platforms = sorted(set(platforms))
    selected_kinds = sorted(set(device_kinds))
    if selected_platforms != ["tpu"] or not selected_kinds:
        raise RegionalExecutionError("controller observation is not TPU-only")
    if not isinstance(boot_id, str) or re.fullmatch(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
        boot_id,
    ) is None:
        raise RegionalExecutionError("controller boot_id is invalid")
    body = {
        "schema_version": CONTROLLER_OBSERVATION_SCHEMA_VERSION,
        "contract_sha256": expected["contract_sha256"],
        "resource_create_time": expected["resource_create_time"],
        "probe_id": _safe_name("probe_id", probe_id),
        "controller_index": controller,
        "process_index": process,
        "process_count": process_count,
        "global_device_count": global_device_count,
        "local_device_count": local_device_count,
        "platforms": selected_platforms,
        "device_kinds": selected_kinds,
        "boot_id": boot_id,
        "hostname": _safe_name("hostname", hostname),
    }
    return _seal(body, "observation_sha256")


def validate_controller_observation(value: Mapping[str, Any]) -> dict[str, Any]:
    selected = _validate_seal(
        value,
        schema_version=CONTROLLER_OBSERVATION_SCHEMA_VERSION,
        hash_field="observation_sha256",
        label="controller observation",
    )
    required = {
        "schema_version",
        "contract_sha256",
        "resource_create_time",
        "probe_id",
        "controller_index",
        "process_index",
        "process_count",
        "global_device_count",
        "local_device_count",
        "platforms",
        "device_kinds",
        "boot_id",
        "hostname",
        "observation_sha256",
    }
    if set(selected) != required:
        raise RegionalExecutionError("controller observation fields changed")
    _sha256("contract_sha256", selected["contract_sha256"])
    _safe_name("probe_id", selected["probe_id"])
    _safe_name("hostname", selected["hostname"])
    for name in (
        "controller_index",
        "process_index",
        "process_count",
        "global_device_count",
        "local_device_count",
    ):
        value = selected[name]
        if name.endswith("_index"):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise RegionalExecutionError(f"{name} must be nonnegative")
        else:
            _positive_int(name, value)
    if (
        selected["platforms"] != ["tpu"]
        or not isinstance(selected["device_kinds"], list)
        or not selected["device_kinds"]
        or selected["device_kinds"] != sorted(set(selected["device_kinds"]))
    ):
        raise RegionalExecutionError("controller device inventory changed")
    return selected


def seal_controller_mapping(
    *,
    contract: Mapping[str, Any],
    probe_id: str,
    observations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Seal a fresh four-controller permutation for one slice incarnation."""

    expected = validate_accelerator_contract(contract)
    selected_probe = _safe_name("probe_id", probe_id)
    checked = [validate_controller_observation(item) for item in observations]
    if len(checked) != expected["process_count"]:
        raise RegionalExecutionError("controller mapping is incomplete")
    controllers: set[int] = set()
    processes: set[int] = set()
    boot_ids: set[str] = set()
    hostnames: set[str] = set()
    rows: list[dict[str, int]] = []
    observation_hashes: list[str] = []
    device_kinds: set[str] = set()
    for item in checked:
        if (
            item["contract_sha256"] != expected["contract_sha256"]
            or item["resource_create_time"] != expected["resource_create_time"]
            or item["probe_id"] != selected_probe
            or item["process_count"] != expected["process_count"]
            or item["global_device_count"] != expected["global_device_count"]
            or item["local_device_count"] != expected["local_device_count"]
        ):
            raise RegionalExecutionError(
                "controller observation belongs to another slice or topology"
            )
        controller = int(item["controller_index"])
        process = int(item["process_index"])
        if (
            controller in controllers
            or process in processes
            or item["boot_id"] in boot_ids
            or item["hostname"] in hostnames
        ):
            raise RegionalExecutionError(
                "controller observations are not a one-to-one physical mapping"
            )
        controllers.add(controller)
        processes.add(process)
        boot_ids.add(item["boot_id"])
        hostnames.add(item["hostname"])
        rows.append(
            {"controller_index": controller, "process_index": process}
        )
        observation_hashes.append(item["observation_sha256"])
        device_kinds.update(item["device_kinds"])
    exact = set(range(expected["process_count"]))
    if controllers != exact or processes != exact:
        raise RegionalExecutionError(
            "controller/JAX process mapping is not the exact permutation"
        )
    body = {
        "schema_version": CONTROLLER_MAPPING_SCHEMA_VERSION,
        "contract_sha256": expected["contract_sha256"],
        "run_id": expected["run_id"],
        "tpu_name": expected["tpu_name"],
        "resource_create_time": expected["resource_create_time"],
        "zone": expected["zone"],
        "accelerator_type": expected["accelerator_type"],
        "probe_id": selected_probe,
        "process_count": expected["process_count"],
        "global_device_count": expected["global_device_count"],
        "local_device_count": expected["local_device_count"],
        "device_kinds": sorted(device_kinds),
        "controller_to_jax_process_index": sorted(
            rows, key=lambda item: item["controller_index"]
        ),
        "observation_sha256": sorted(observation_hashes),
        "mapping_policy": "fresh-per-slice-incarnation-before-training",
    }
    return _seal(body, "mapping_sha256")


def validate_controller_mapping_receipt(
    value: Mapping[str, Any],
    *,
    expected_contract_sha256: str | None = None,
) -> dict[str, Any]:
    selected = _validate_seal(
        value,
        schema_version=CONTROLLER_MAPPING_SCHEMA_VERSION,
        hash_field="mapping_sha256",
        label="controller mapping receipt",
    )
    required = {
        "schema_version",
        "contract_sha256",
        "run_id",
        "tpu_name",
        "resource_create_time",
        "zone",
        "accelerator_type",
        "probe_id",
        "process_count",
        "global_device_count",
        "local_device_count",
        "device_kinds",
        "controller_to_jax_process_index",
        "observation_sha256",
        "mapping_policy",
        "mapping_sha256",
    }
    if set(selected) != required:
        raise RegionalExecutionError("controller mapping receipt fields changed")
    contract_sha = _sha256("contract_sha256", selected["contract_sha256"])
    if (
        expected_contract_sha256 is not None
        and contract_sha != _sha256(
            "expected_contract_sha256", expected_contract_sha256
        )
    ):
        raise RegionalExecutionError("controller mapping contract changed")
    count = _positive_int("process_count", selected["process_count"])
    if (
        selected["global_device_count"]
        != count * selected["local_device_count"]
        or selected["mapping_policy"]
        != "fresh-per-slice-incarnation-before-training"
        or len(selected["observation_sha256"]) != count
        or len(set(selected["observation_sha256"])) != count
    ):
        raise RegionalExecutionError("controller mapping topology changed")
    mapping = selected["controller_to_jax_process_index"]
    if not isinstance(mapping, list) or len(mapping) != count:
        raise RegionalExecutionError("controller mapping is incomplete")
    controllers: set[int] = set()
    processes: set[int] = set()
    for item in mapping:
        if not isinstance(item, Mapping) or set(item) != {
            "controller_index",
            "process_index",
        }:
            raise RegionalExecutionError("controller mapping row fields changed")
        controller = item["controller_index"]
        process = item["process_index"]
        if (
            isinstance(controller, bool)
            or not isinstance(controller, int)
            or isinstance(process, bool)
            or not isinstance(process, int)
            or not 0 <= controller < count
            or not 0 <= process < count
        ):
            raise RegionalExecutionError("controller mapping index is invalid")
        controllers.add(controller)
        processes.add(process)
    if controllers != set(range(count)) or processes != set(range(count)):
        raise RegionalExecutionError(
            "controller mapping is not the exact permutation"
        )
    return selected


def observe_local_jax_controller(
    *,
    contract: Mapping[str, Any],
    probe_id: str,
    controller_index: int,
) -> dict[str, Any]:
    """Initialize JAX and create one observation; intended for all workers."""

    try:
        import jax
    except ImportError as error:  # pragma: no cover - TPU dependency boundary
        raise RegionalExecutionError("controller probe requires JAX") from error
    devices = tuple(jax.devices())
    local_devices = tuple(jax.local_devices())
    try:
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(
            encoding="ascii"
        ).strip()
    except OSError as error:
        raise RegionalExecutionError("cannot read controller boot identity") from error
    return build_controller_observation(
        contract=contract,
        probe_id=probe_id,
        controller_index=controller_index,
        process_index=int(jax.process_index()),
        process_count=int(jax.process_count()),
        global_device_count=int(jax.device_count()),
        local_device_count=int(jax.local_device_count()),
        platforms=[str(device.platform) for device in devices],
        device_kinds=[
            str(getattr(device, "device_kind", "")) for device in local_devices
        ],
        boot_id=boot_id,
        hostname=socket.gethostname(),
    )


def build_local_cache_observation(
    *,
    contract: Mapping[str, Any],
    controller_mapping: Mapping[str, Any],
    controller_index: int,
    cache_root: str,
    required_lane_bytes: int,
    required_free_bytes: int,
    available_bytes: int,
    mem_available_bytes: int,
    mount_target: str,
    mount_source: str,
    filesystem_type: str,
    mount_options: Sequence[str],
    boot_id: str,
) -> dict[str, Any]:
    """Bind one process lane to actually observed private cache capacity."""

    accelerator = validate_accelerator_contract(contract)
    mapping = validate_controller_mapping_receipt(
        controller_mapping,
        expected_contract_sha256=accelerator["contract_sha256"],
    )
    if (
        isinstance(controller_index, bool)
        or not isinstance(controller_index, int)
        or not 0 <= controller_index < accelerator["process_count"]
    ):
        raise RegionalExecutionError("cache controller_index is invalid")
    process_by_controller = {
        int(item["controller_index"]): int(item["process_index"])
        for item in mapping["controller_to_jax_process_index"]
    }
    root = Path(cache_root)
    target = Path(mount_target)
    if (
        not root.is_absolute()
        or ".." in root.parts
        or not target.is_absolute()
        or ".." in target.parts
        or target == Path("/")
    ):
        raise RegionalExecutionError(
            "cache and mount paths must be non-root absolute paths"
        )
    try:
        root.relative_to(target)
    except ValueError as error:
        raise RegionalExecutionError(
            "cache root is outside its observed mount"
        ) from error
    lane = _positive_int("required_lane_bytes", required_lane_bytes)
    minimum = _positive_int("required_free_bytes", required_free_bytes)
    available = _positive_int("available_bytes", available_bytes)
    memory = _positive_int("mem_available_bytes", mem_available_bytes)
    if (
        minimum < MINIMUM_10M_PROCESS_CACHE_FREE_BYTES
        or minimum < (lane * 11 + 9) // 10
        or available < minimum
    ):
        raise RegionalExecutionError(
            "local cache lacks the 10M process lane plus required headroom"
        )
    options = sorted(set(mount_options))
    if not options or "rw" not in options:
        raise RegionalExecutionError("local cache mount is not read/write")
    if filesystem_type == "tmpfs":
        if (
            target != Path("/dev/shm")
            or mount_source != "tmpfs"
            or "nodev" not in options
            or memory < max(
                minimum,
                MINIMUM_10M_TMPFS_MEM_AVAILABLE_BYTES,
            )
        ):
            raise RegionalExecutionError(
                "tmpfs cache is not a read/write, sufficiently backed "
                "/dev/shm; v5litepod requires a 180GiB /dev/shm remount "
                "before the capacity probe"
            )
        cache_kind = "host-ram-tmpfs"
    elif filesystem_type in {"ext4", "xfs"}:
        if (
            target.parts[:2] not in {("/", "mnt"), ("/", "media")}
            or not mount_source.startswith("/dev/")
        ):
            raise RegionalExecutionError(
                "disk cache is not a dedicated attached local mount"
            )
        cache_kind = "attached-zonal-disk"
    else:
        raise RegionalExecutionError("unsupported local cache filesystem")
    if not isinstance(boot_id, str) or re.fullmatch(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
        boot_id,
    ) is None:
        raise RegionalExecutionError("cache observation boot_id is invalid")
    body = {
        "schema_version": LOCAL_CACHE_OBSERVATION_SCHEMA_VERSION,
        "contract_sha256": accelerator["contract_sha256"],
        "controller_mapping_sha256": mapping["mapping_sha256"],
        "resource_create_time": accelerator["resource_create_time"],
        "controller_index": controller_index,
        "process_index": process_by_controller[controller_index],
        "cache_root": str(root),
        "required_lane_bytes": lane,
        "required_free_bytes": minimum,
        "available_bytes": available,
        "mem_available_bytes": memory,
        "mount_target": str(target),
        "mount_source": mount_source,
        "filesystem_type": filesystem_type,
        "mount_options": options,
        "cache_kind": cache_kind,
        "boot_id": boot_id,
    }
    return _seal(body, "observation_sha256")


def observe_local_cache_capacity(
    *,
    contract: Mapping[str, Any],
    controller_mapping: Mapping[str, Any],
    controller_index: int,
    cache_root: str | Path,
    required_lane_bytes: int,
    required_free_bytes: int,
) -> dict[str, Any]:
    """Measure one fresh worker's existing cache mount without modifying it."""

    root = Path(cache_root).expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise RegionalExecutionError(
            "cache root must already be a regular directory"
        )
    try:
        completed = subprocess.run(
            [
                "findmnt",
                "--json",
                "--target",
                str(root),
                "--output",
                "TARGET,SOURCE,FSTYPE,OPTIONS",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        payload = json.loads(completed.stdout)
        filesystems = payload["filesystems"]
        if not isinstance(filesystems, list) or len(filesystems) != 1:
            raise RegionalExecutionError("findmnt returned an ambiguous cache mount")
        mount = filesystems[0]
        options = str(mount["options"]).split(",")
        meminfo = Path("/proc/meminfo").read_text(encoding="ascii")
        match = re.search(r"^MemAvailable:\s+([0-9]+)\s+kB$", meminfo, re.MULTILINE)
        if match is None:
            raise RegionalExecutionError("MemAvailable is absent from /proc/meminfo")
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(
            encoding="ascii"
        ).strip()
    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        subprocess.SubprocessError,
    ) as error:
        raise RegionalExecutionError(
            "cannot measure local cache mount and memory"
        ) from error
    return build_local_cache_observation(
        contract=contract,
        controller_mapping=controller_mapping,
        controller_index=controller_index,
        cache_root=str(root),
        required_lane_bytes=required_lane_bytes,
        required_free_bytes=required_free_bytes,
        available_bytes=shutil.disk_usage(root).free,
        mem_available_bytes=int(match.group(1)) * 1024,
        mount_target=str(mount["target"]),
        mount_source=str(mount["source"]),
        filesystem_type=str(mount["fstype"]),
        mount_options=options,
        boot_id=boot_id,
    )


def seal_local_cache_inventory(
    *,
    contract: Mapping[str, Any],
    controller_mapping: Mapping[str, Any],
    observations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Prove every mapped process has a sufficiently large private cache."""

    accelerator = validate_accelerator_contract(contract)
    mapping = validate_controller_mapping_receipt(
        controller_mapping,
        expected_contract_sha256=accelerator["contract_sha256"],
    )
    if len(observations) != accelerator["process_count"]:
        raise RegionalExecutionError("local cache inventory is incomplete")
    checked = [
        _validate_seal(
            item,
            schema_version=LOCAL_CACHE_OBSERVATION_SCHEMA_VERSION,
            hash_field="observation_sha256",
            label="local cache observation",
        )
        for item in observations
    ]
    controllers: set[int] = set()
    processes: set[int] = set()
    boot_ids: set[str] = set()
    kinds: set[str] = set()
    rows: list[dict[str, Any]] = []
    for item in checked:
        if (
            item["contract_sha256"] != accelerator["contract_sha256"]
            or item["controller_mapping_sha256"] != mapping["mapping_sha256"]
            or item["resource_create_time"] != accelerator["resource_create_time"]
            or item["available_bytes"] < item["required_free_bytes"]
        ):
            raise RegionalExecutionError(
                "local cache observation belongs to another slice or is undersized"
            )
        controller = int(item["controller_index"])
        process = int(item["process_index"])
        if (
            controller in controllers
            or process in processes
            or item["boot_id"] in boot_ids
        ):
            raise RegionalExecutionError(
                "local caches are not one private cache per mapped process"
            )
        controllers.add(controller)
        processes.add(process)
        boot_ids.add(item["boot_id"])
        kinds.add(item["cache_kind"])
        rows.append(
            {
                "controller_index": controller,
                "process_index": process,
                "cache_root": item["cache_root"],
                "required_lane_bytes": item["required_lane_bytes"],
                "required_free_bytes": item["required_free_bytes"],
                "available_bytes": item["available_bytes"],
                "cache_kind": item["cache_kind"],
                "observation_sha256": item["observation_sha256"],
            }
        )
    exact = set(range(accelerator["process_count"]))
    if controllers != exact or processes != exact or len(kinds) != 1:
        raise RegionalExecutionError(
            "local cache inventory is not one uniform exact topology"
        )
    body = {
        "schema_version": LOCAL_CACHE_INVENTORY_SCHEMA_VERSION,
        "contract_sha256": accelerator["contract_sha256"],
        "controller_mapping_sha256": mapping["mapping_sha256"],
        "resource_create_time": accelerator["resource_create_time"],
        "process_count": accelerator["process_count"],
        "cache_kind": next(iter(kinds)),
        "workers": sorted(rows, key=lambda item: item["controller_index"]),
        "resident_policy": (
            "complete-process-lane-hash-verified-before-training-no-refill"
        ),
    }
    return _seal(body, "inventory_sha256")


def validate_local_cache_inventory(
    value: Mapping[str, Any],
    *,
    expected_contract_sha256: str | None = None,
    expected_controller_mapping_sha256: str | None = None,
) -> dict[str, Any]:
    selected = _validate_seal(
        value,
        schema_version=LOCAL_CACHE_INVENTORY_SCHEMA_VERSION,
        hash_field="inventory_sha256",
        label="local cache inventory",
    )
    required = {
        "schema_version",
        "contract_sha256",
        "controller_mapping_sha256",
        "resource_create_time",
        "process_count",
        "cache_kind",
        "workers",
        "resident_policy",
        "inventory_sha256",
    }
    if set(selected) != required:
        raise RegionalExecutionError("local cache inventory fields changed")
    contract_sha = _sha256("contract_sha256", selected["contract_sha256"])
    mapping_sha = _sha256(
        "controller_mapping_sha256",
        selected["controller_mapping_sha256"],
    )
    if (
        expected_contract_sha256 is not None
        and contract_sha != _sha256(
            "expected_contract_sha256", expected_contract_sha256
        )
    ) or (
        expected_controller_mapping_sha256 is not None
        and mapping_sha != _sha256(
            "expected_controller_mapping_sha256",
            expected_controller_mapping_sha256,
        )
    ):
        raise RegionalExecutionError("local cache inventory authority changed")
    count = _positive_int("process_count", selected["process_count"])
    workers = selected["workers"]
    if (
        selected["cache_kind"] not in {
            "host-ram-tmpfs",
            "attached-zonal-disk",
        }
        or selected["resident_policy"]
        != "complete-process-lane-hash-verified-before-training-no-refill"
        or not isinstance(workers, list)
        or len(workers) != count
    ):
        raise RegionalExecutionError("local cache inventory policy changed")
    controllers: set[int] = set()
    processes: set[int] = set()
    for item in workers:
        if not isinstance(item, Mapping) or set(item) != {
            "controller_index",
            "process_index",
            "cache_root",
            "required_lane_bytes",
            "required_free_bytes",
            "available_bytes",
            "cache_kind",
            "observation_sha256",
        }:
            raise RegionalExecutionError("local cache worker fields changed")
        controller = item["controller_index"]
        process = item["process_index"]
        if (
            isinstance(controller, bool)
            or not isinstance(controller, int)
            or isinstance(process, bool)
            or not isinstance(process, int)
            or not 0 <= controller < count
            or not 0 <= process < count
            or item["cache_kind"] != selected["cache_kind"]
            or item["required_free_bytes"]
            < MINIMUM_10M_PROCESS_CACHE_FREE_BYTES
            or item["available_bytes"] < item["required_free_bytes"]
            or item["required_free_bytes"]
            < (item["required_lane_bytes"] * 11 + 9) // 10
        ):
            raise RegionalExecutionError(
                "local cache worker is invalid or undersized"
            )
        _sha256("observation_sha256", item["observation_sha256"])
        controllers.add(controller)
        processes.add(process)
    if controllers != set(range(count)) or processes != set(range(count)):
        raise RegionalExecutionError(
            "local cache inventory is not the exact mapped topology"
        )
    return selected


def build_regional_placement(
    *,
    contract: Mapping[str, Any],
    mirror: Mapping[str, Any],
    control_root: str | Path,
    control_zone: str,
    checkpoint_prefix: str,
    checkpoint_bucket_region: str,
) -> dict[str, Any]:
    """Bind control and checkpoint state to the accelerator's own region."""

    accelerator = validate_accelerator_contract(contract)
    sealed_mirror = validate_sealed_regional_mirror(mirror)
    root = Path(control_root).expanduser()
    if not root.is_absolute() or ".." in root.parts or str(root) == "/":
        raise RegionalExecutionError("control_root must be a non-root absolute path")
    _zone(control_zone, expected_region=accelerator["region"])
    if (
        sealed_mirror["run_id"] != accelerator["run_id"]
        or sealed_mirror["region"] != accelerator["region"]
        or _region(checkpoint_bucket_region) != accelerator["region"]
    ):
        raise RegionalExecutionError(
            "mirror, control plane, checkpoint bucket, and TPU must be regional"
        )
    try:
        _parse_prefix(checkpoint_prefix)
    except ValueError as error:
        raise RegionalExecutionError("checkpoint_prefix is invalid") from error
    expected_suffix = (
        f"/{accelerator['run_id']}/{accelerator['scientific_model']}/"
        f"{accelerator['condition']}"
    )
    if not checkpoint_prefix.rstrip("/").endswith(expected_suffix):
        raise RegionalExecutionError(
            "checkpoint prefix is not condition-local and canonical"
        )
    body = {
        "schema_version": REGIONAL_PLACEMENT_SCHEMA_VERSION,
        "run_id": accelerator["run_id"],
        "region": accelerator["region"],
        "zone": accelerator["zone"],
        "scientific_model": accelerator["scientific_model"],
        "condition": accelerator["condition"],
        "accelerator_contract_sha256": accelerator["contract_sha256"],
        "mirror_sha256": sealed_mirror["mirror_sha256"],
        "control_root": str(root),
        "control_zone": control_zone,
        "checkpoint_prefix": checkpoint_prefix.rstrip("/"),
        "checkpoint_bucket_region": checkpoint_bucket_region,
        "placement_policy": "region-local-control-checkpoints-and-corpus",
    }
    return _seal(body, "placement_sha256")


def build_cleanup_plan(
    *,
    run_id: str,
    central_result_prefix: str,
    placements: Sequence[Mapping[str, Any]],
    resources: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Describe teardown order; executing it is intentionally out of scope."""

    selected_run = _safe_name("run_id", run_id)
    _parse_prefix(central_result_prefix)
    if not placements or not resources:
        raise RegionalExecutionError("cleanup plan needs placements and resources")
    placement_rows: list[dict[str, Any]] = []
    placement_hashes: set[str] = set()
    regions: set[str] = set()
    for value in placements:
        selected = _validate_seal(
            value,
            schema_version=REGIONAL_PLACEMENT_SCHEMA_VERSION,
            hash_field="placement_sha256",
            label="regional placement",
        )
        if selected["run_id"] != selected_run:
            raise RegionalExecutionError("cleanup placement belongs to another run")
        if selected["placement_sha256"] in placement_hashes:
            raise RegionalExecutionError("cleanup placement is duplicated")
        placement_hashes.add(selected["placement_sha256"])
        regions.add(_region(selected["region"]))
        placement_rows.append(
            {
                "region": selected["region"],
                "scientific_model": selected["scientific_model"],
                "condition": selected["condition"],
                "placement_sha256": selected["placement_sha256"],
            }
        )
    resource_rows: list[dict[str, Any]] = []
    identities: set[tuple[str, str]] = set()
    allowed_types = {
        "tpu",
        "filestore",
        "gcs_mirror",
        "gcs_checkpoint_prefix",
        "boot_disk",
    }
    for item in resources:
        if not isinstance(item, Mapping) or set(item) != {
            "resource_type",
            "resource_name",
            "region",
            "run_label",
        }:
            raise RegionalExecutionError("cleanup resource fields changed")
        resource_type = str(item["resource_type"])
        resource_name = str(item["resource_name"])
        region = _region(item["region"])
        if (
            resource_type not in allowed_types
            or not resource_name
            or item["run_label"] != selected_run
            or region not in regions
            or (resource_type, resource_name) in identities
        ):
            raise RegionalExecutionError(
                "cleanup resource is unowned, duplicated, or out of region"
            )
        identities.add((resource_type, resource_name))
        resource_rows.append(dict(item))
    body = {
        "schema_version": CLEANUP_PLAN_SCHEMA_VERSION,
        "run_id": selected_run,
        "central_result_prefix": central_result_prefix.rstrip("/"),
        "placements": sorted(
            placement_rows,
            key=lambda row: (
                row["region"],
                row["scientific_model"],
                row["condition"],
            ),
        ),
        "resources": sorted(
            resource_rows,
            key=lambda row: (row["resource_type"], row["resource_name"]),
        ),
        "preconditions": [
            "all-final-evaluations-sealed",
            "central-report-and-final-checkpoint-hashes-verified",
            "no-active-writer-or-restore-operation",
        ],
        "ordered_stages": [
            "delete-run-owned-tpu-slices",
            "delete-run-owned-boot-disks",
            "delete-run-owned-regional-filestores",
            "delete-run-owned-regional-checkpoint-prefixes",
            "delete-run-owned-regional-mirror-buckets",
            "verify-run-owned-regional-inventory-empty",
        ],
        "execution_policy": "plan-only-no-implicit-cloud-deletion",
    }
    return _seal(body, "cleanup_sha256")


def atomic_write_evidence(path: str | Path, value: Mapping[str, Any]) -> None:
    """Create immutable evidence locally without overwriting a prior probe."""

    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(dict(value)) + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            if target.read_bytes() != payload:
                raise RegionalExecutionError(
                    "immutable regional evidence differs from replay"
                )
        directory = os.open(
            target.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _load_evidence(path: str | Path) -> dict[str, Any]:
    selected = Path(path).expanduser().resolve()
    if not selected.is_file() or selected.is_symlink():
        raise RegionalExecutionError(
            f"regional evidence is absent, non-regular, or a symlink: {selected}"
        )
    try:
        value = json.loads(selected.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RegionalExecutionError(
            f"cannot read regional evidence: {selected}"
        ) from error
    if not isinstance(value, dict):
        raise RegionalExecutionError("regional evidence must be a JSON object")
    return value


def _observe_command(args: argparse.Namespace) -> int:
    contract = validate_accelerator_contract(_load_evidence(args.contract))
    observation = observe_local_jax_controller(
        contract=contract,
        probe_id=args.probe_id,
        controller_index=args.controller_index,
    )
    atomic_write_evidence(args.output, observation)
    print(canonical_json_bytes(observation).decode("ascii"), flush=True)
    return 0


def _seal_mapping_command(args: argparse.Namespace) -> int:
    contract = validate_accelerator_contract(_load_evidence(args.contract))
    root = Path(args.observation_root).expanduser().resolve()
    observations = [
        _load_evidence(root / f"controller-{index:05d}.json")
        for index in range(contract["process_count"])
    ]
    receipt = seal_controller_mapping(
        contract=contract,
        probe_id=args.probe_id,
        observations=observations,
    )
    atomic_write_evidence(args.output, receipt)
    print(canonical_json_bytes(receipt).decode("ascii"), flush=True)
    return 0


def _observe_cache_command(args: argparse.Namespace) -> int:
    contract = validate_accelerator_contract(_load_evidence(args.contract))
    mapping = validate_controller_mapping_receipt(
        _load_evidence(args.controller_mapping),
        expected_contract_sha256=contract["contract_sha256"],
    )
    observation = observe_local_cache_capacity(
        contract=contract,
        controller_mapping=mapping,
        controller_index=args.controller_index,
        cache_root=args.cache_root,
        required_lane_bytes=args.required_lane_bytes,
        required_free_bytes=args.required_free_bytes,
    )
    atomic_write_evidence(args.output, observation)
    print(canonical_json_bytes(observation).decode("ascii"), flush=True)
    return 0


def _seal_cache_command(args: argparse.Namespace) -> int:
    contract = validate_accelerator_contract(_load_evidence(args.contract))
    mapping = validate_controller_mapping_receipt(
        _load_evidence(args.controller_mapping),
        expected_contract_sha256=contract["contract_sha256"],
    )
    root = Path(args.observation_root).expanduser().resolve()
    observations = [
        _load_evidence(root / f"controller-{index:05d}.json")
        for index in range(contract["process_count"])
    ]
    inventory = seal_local_cache_inventory(
        contract=contract,
        controller_mapping=mapping,
        observations=observations,
    )
    atomic_write_evidence(args.output, inventory)
    print(canonical_json_bytes(inventory).decode("ascii"), flush=True)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m moe_sae.regional_execution"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    observe = commands.add_parser(
        "observe-controller",
        help="initialize JAX and record this physical controller's process index",
    )
    observe.add_argument("--contract", required=True)
    observe.add_argument("--probe-id", required=True)
    observe.add_argument("--controller-index", required=True, type=int)
    observe.add_argument("--output", required=True)
    observe.set_defaults(handler=_observe_command)

    seal = commands.add_parser(
        "seal-controller-map",
        help="seal the exact fresh physical-controller/JAX permutation",
    )
    seal.add_argument("--contract", required=True)
    seal.add_argument("--probe-id", required=True)
    seal.add_argument("--observation-root", required=True)
    seal.add_argument("--output", required=True)
    seal.set_defaults(handler=_seal_mapping_command)

    cache = commands.add_parser(
        "observe-cache",
        help="measure one mapped worker's existing private cache capacity",
    )
    cache.add_argument("--contract", required=True)
    cache.add_argument("--controller-mapping", required=True)
    cache.add_argument("--controller-index", required=True, type=int)
    cache.add_argument("--cache-root", required=True)
    cache.add_argument("--required-lane-bytes", required=True, type=int)
    cache.add_argument("--required-free-bytes", required=True, type=int)
    cache.add_argument("--output", required=True)
    cache.set_defaults(handler=_observe_cache_command)

    seal_cache = commands.add_parser(
        "seal-cache-inventory",
        help="seal all four mapped private-cache capacity observations",
    )
    seal_cache.add_argument("--contract", required=True)
    seal_cache.add_argument("--controller-mapping", required=True)
    seal_cache.add_argument("--observation-root", required=True)
    seal_cache.add_argument("--output", required=True)
    seal_cache.set_defaults(handler=_seal_cache_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (RegionalExecutionError, ObjectStoreError, OSError, ValueError) as error:
        parser.error(str(error))
    return 2


__all__ = [
    "ACCELERATOR_CONTRACT_SCHEMA_VERSION",
    "CLEANUP_PLAN_SCHEMA_VERSION",
    "CONTROLLER_MAPPING_SCHEMA_VERSION",
    "CONTROLLER_OBSERVATION_SCHEMA_VERSION",
    "DELIVERY_LEDGER_SCHEMA_VERSION",
    "LOCAL_CACHE_INVENTORY_SCHEMA_VERSION",
    "LOCAL_CACHE_OBSERVATION_SCHEMA_VERSION",
    "MINIMUM_10M_PROCESS_CACHE_FREE_BYTES",
    "MINIMUM_10M_TMPFS_MEM_AVAILABLE_BYTES",
    "MIRROR_SPEC_SCHEMA_VERSION",
    "PREWARM_RECEIPT_SCHEMA_VERSION",
    "REGIONAL_PLACEMENT_SCHEMA_VERSION",
    "SEALED_MIRROR_SCHEMA_VERSION",
    "MirrorDataset",
    "RegionalExecutionError",
    "RegionalMirrorStore",
    "assert_spot_tpu_description",
    "atomic_write_evidence",
    "build_accelerator_contract",
    "build_cleanup_plan",
    "build_controller_observation",
    "build_delivery_ledger",
    "build_local_cache_observation",
    "build_regional_mirror_specification",
    "build_regional_placement",
    "observe_local_cache_capacity",
    "observe_local_jax_controller",
    "prewarm_regional_cache",
    "seal_controller_mapping",
    "seal_local_cache_inventory",
    "seal_regional_mirror",
    "validate_accelerator_contract",
    "validate_controller_mapping_receipt",
    "validate_controller_observation",
    "validate_local_cache_inventory",
    "validate_regional_mirror_specification",
    "validate_sealed_regional_mirror",
]


if __name__ == "__main__":
    raise SystemExit(main())
