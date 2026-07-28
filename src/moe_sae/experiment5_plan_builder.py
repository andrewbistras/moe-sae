"""Build the exact Experiment-5 plan from the sealed horizon r8 inventory.

The r8 plan is the deployment evidence source, not a scheduling template.  It
contains eight resources, one of which (``gemma4-horizon-v5e-usc-b``) was
preempted after the horizon campaign.  This builder:

* admits only the seven known surviving TPU slices;
* derives target/paired cache capability from each slice's prior assignment;
* copies controller mappings, cache inventories, and regional mirror seals
  byte-for-byte at the YAML-object level;
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

# Each scientific group has one physical artifact plane.  SAE uses the
# target-only resident caches, leaving all paired capacity available to the
# transcoder panel.  The EU paired pool serves two seeds concurrently; the v4
# paired slice serves the third.  This uses all seven surviving resources in
# the first wave without copying captures across regions.
_GROUP_TPU_NAMES: dict[tuple[str, int], tuple[str, ...]] = {
    ("sae", 0): ("gemma4-horizon-v5e-usc-a",),
    ("sae", 1): ("gemma4-horizon-harvest-0728",),
    ("sae", 2): ("gemma4-horizon-v5e-euw2-f",),
    ("transcoder", 0): (
        "gemma4-horizon-v5e-euw-a",
        "gemma4-horizon-v5e-euw-r",
        "gemma4-horizon-v5e-euw2-b",
    ),
    ("transcoder", 1): (
        "gemma4-horizon-v5e-euw-a",
        "gemma4-horizon-v5e-euw-r",
        "gemma4-horizon-v5e-euw2-b",
    ),
    ("transcoder", 2): ("gemma4-horizon-v4-spot-trans-b",),
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
        "TRAINING_PROJECT_COMMIT",
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
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str) for key in value
    ):
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


def _region(zone: str) -> str:
    try:
        region, suffix = zone.rsplit("-", 1)
    except ValueError as error:
        raise Experiment5PlanBuildError(f"invalid TPU zone {zone!r}") from error
    if len(suffix) != 1 or not suffix.isalpha():
        raise Experiment5PlanBuildError(f"invalid TPU zone {zone!r}")
    return region


def _prior_cache_profiles(source: Mapping[str, Any]) -> dict[str, str]:
    """Infer resident data capability from the prior r8 assignments.

    A horizon ``sae`` owner proved the target capture resident; a
    ``transcoder`` owner proved both target and routed-input overlay resident.
    The cache-inventory evidence copied for that resource is the immutable
    witness checked again by the campaign controller.
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
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    profiles = _prior_cache_profiles(source)
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
            "r8 physical inventory changed: "
            f"observed={sorted(observed_names)!r}"
        )

    output: list[dict[str, Any]] = []
    by_name: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(raw_resources):
        old = _mapping(raw, f"resources[{index}]")
        tpu_name = _required_string(
            old.get("tpu_name"), f"resources[{index}].tpu_name"
        )
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
            raise Experiment5PlanBuildError(
                f"{resource_id} evidence fields changed"
            )
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
        record = {
            "id": resource_id,
            "project": copy.deepcopy(old.get("project")),
            "tpu_name": tpu_name,
            "zone": zone,
            "region": _region(zone),
            "accelerator_type": copy.deepcopy(old.get("accelerator_type")),
            "internal_ip": copy.deepcopy(old.get("internal_ip")),
            "ssh_user": copy.deepcopy(old.get("ssh_user")),
            "cache_profile": profiles[resource_id],
            "deployed_commit": project_commit,
            "evidence": copy.deepcopy(evidence),
            "environment": copy.deepcopy(old.get("environment")),
        }
        record["_controller_mount_root"] = _controller_mount_root(
            source, resource_id=resource_id
        )
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
    bulk_roots = {
        _regional_bulk_root(source_environment, record) for record in records
    }
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
    campaign_id: str | None = None,
    already_spent_usd: float = 0.0,
) -> dict[str, Any]:
    """Return a strict Experiment-5 campaign plan from one r8 source plan."""

    source = _mapping(source, "source")
    if source.get("schema_version") != SOURCE_SCHEMA_VERSION:
        raise Experiment5PlanBuildError("source is not a horizon r8 plan")
    if _GIT_SHA.fullmatch(project_commit) is None:
        raise Experiment5PlanBuildError("project_commit must be a full Git SHA")
    source_environment = _mapping(source.get("environment"), "environment")
    resources, by_name = _resource_records(source, project_commit=project_commit)

    source_remote_project = _required_string(
        source.get("remote_project_root"), "remote_project_root"
    )
    short_commit = project_commit[:12]
    experiment_root = str(
        PurePosixPath(_remote_root(source)) / short_commit
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
                f"{regional_bulk}/experiment5/{short_commit}/checkpoints/"
                f"{panel}/seed{seed}"
            )
            controller_group_root = (
                Path(controller_mount)
                / "experiment5"
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
        "campaign_id": campaign_id or f"gemma4-experiment5-{short_commit}",
        "state_root": (
            f"/mnt/moe-sae-control/experiment5/controller/{short_commit}"
        ),
        "working_directory": copy.deepcopy(source.get("working_directory")),
        "python_executable": copy.deepcopy(source.get("python_executable")),
        "remote_project_root": source_remote_project,
        "remote_uv": copy.deepcopy(source.get("remote_uv")),
        "project_commit": project_commit,
        "max_parallel_train": 7,
        "max_parallel_eval": 7,
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


def load_r8_plan(path: str | Path) -> dict[str, Any]:
    selected = Path(path).expanduser().resolve()
    try:
        payload = yaml.safe_load(selected.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise Experiment5PlanBuildError(f"cannot read r8 plan: {selected}") from error
    return _mapping(payload, "source")


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
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--campaign-id")
    parser.add_argument("--already-spent-usd", type=float, default=0.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        plan = build_experiment5_plan(
            load_r8_plan(args.r8_plan),
            project_commit=args.project_commit,
            campaign_id=args.campaign_id,
            already_spent_usd=args.already_spent_usd,
        )
        parsed = write_experiment5_plan(args.output, plan)
    except (
        Experiment5PlanBuildError,
        Experiment5CampaignPlanError,
    ) as error:
        raise SystemExit(f"Experiment-5 plan build failed: {error}") from error
    print(
        f"wrote {args.output} plan_sha256={parsed.plan_sha256} "
        f"resources={len(parsed.resources)} trajectories={len(parsed.trajectories)} "
        f"evaluations={len(parsed.evaluations)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
