"""Bounded, same-slice A/F systems-profile plan."""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from moe_sae.component_closure_config import (
    PANELS,
    PHASE_NAME,
    PROTOCOL_ID,
    SOURCE_CONDITION,
    SOURCE_PHASE_NAME,
    load_component_closure_config,
)
from moe_sae.contribution_artifacts import canonical_sha256

SCHEMA_VERSION = "gemma4-a512-component-closure-af-profile-plan-v1"
_GIT_RE = re.compile(r"^[0-9a-f]{40}$")


class ComponentClosureProfileError(ValueError):
    """The bounded profile would not be an A/F same-hardware comparison."""


def validate_profile_plan(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ComponentClosureProfileError("profile plan must be a mapping")
    selected = copy.deepcopy(dict(value))
    if set(selected) != {
        "schema_version",
        "protocol_id",
        "source_revision",
        "bounds",
        "tasks",
        "plan_sha256",
    }:
        raise ComponentClosureProfileError("profile plan fields changed")
    claimed = selected.pop("plan_sha256")
    if (
        selected["schema_version"] != SCHEMA_VERSION
        or selected["protocol_id"] != PROTOCOL_ID
        or _GIT_RE.fullmatch(str(selected["source_revision"])) is None
        or canonical_sha256(selected) != claimed
    ):
        raise ComponentClosureProfileError("profile plan identity changed")
    if selected["bounds"] != {
        "maximum_seconds_per_task": 600,
        "warmup_steps": 1,
        "measured_steps": 3,
        "training_or_checkpoint_publication_allowed": False,
        "profile_role": "compile_and_steady_step_at_exact_training_shape",
    }:
        raise ComponentClosureProfileError("profile bounds changed")
    tasks = selected["tasks"]
    expected = [(panel, condition) for panel in PANELS for condition in ("A", "F")]
    roles = []
    resources: dict[str, set[str]] = {panel: set() for panel in PANELS}
    if not isinstance(tasks, list) or len(tasks) != 4:
        raise ComponentClosureProfileError("profile requires four A/F tasks")
    for task in tasks:
        roles.append((task.get("panel"), task.get("condition")))
        resources[str(task.get("panel"))].add(str(task.get("resource_id")))
        if set(task) != {
            "task_id",
            "panel",
            "condition",
            "source_condition",
            "source_phase",
            "resource_id",
            "accelerator_type",
            "config",
            "command",
            "required_measurements",
        } or task["required_measurements"] != [
            "compile_seconds",
            "steady_step_seconds",
            "useful_mfu",
            "hbm_peak_bytes",
            "input_wait_fraction",
            "collective_fraction",
            "top_hlo_ops_by_time",
        ]:
            raise ComponentClosureProfileError("profile task contract changed")
    if roles != expected or any(len(rows) != 1 for rows in resources.values()):
        raise ComponentClosureProfileError(
            "A/F must share one exact resource within each panel"
        )
    return {**selected, "plan_sha256": claimed}


def build_profile_plan(
    source: Mapping[str, Any],
    *,
    source_revision: str,
) -> dict[str, Any]:
    if _GIT_RE.fullmatch(source_revision) is None:
        raise ComponentClosureProfileError("source_revision must be a Git SHA")
    working = Path(str(source["working_directory"])).expanduser().resolve()
    remote = PurePosixPath(str(source["remote_project_root"]))
    resources = source["resources"]
    tasks = []
    for panel in PANELS:
        config = load_component_closure_config(
            working / "configs" / f"gemma4_component_closure_{panel}_v1.yaml",
            expected_panel=panel,  # type: ignore[arg-type]
        )
        resource = resources[panel]
        for condition in ("A", "F"):
            historical = condition == "A"
            source_condition = SOURCE_CONDITION if historical else "F"
            source_phase = SOURCE_PHASE_NAME if historical else PHASE_NAME
            config_name = (
                f"gemma4_experiment5_routed_ladder_{panel}_v3.yaml"
                if historical
                else f"gemma4_component_closure_{panel}_v1.yaml"
            )
            launcher = (
                "train_contribution.py" if historical else "train_component_closure.py"
            )
            command = [
                str(working / "scripts/tpu" / launcher),
                "--action",
                "qualify",
                "--scientific-model",
                panel,
                "--config",
                str(remote / "configs" / config_name),
                "--controller-config",
                str(working / "configs" / config_name),
                "--phase",
                source_phase,
                "--condition",
                source_condition,
                "--seed",
                "0",
                "--run-id",
                f"component-closure-profile-{panel}-{condition.lower()}",
                "--attempt-id",
                "<attempt-id>",
                "--maximum-runtime-seconds",
                "600",
            ]
            tasks.append(
                {
                    "task_id": f"profile-{panel}-{condition.lower()}",
                    "panel": panel,
                    "condition": condition,
                    "source_condition": source_condition,
                    "source_phase": source_phase,
                    "resource_id": resource["resource_id"],
                    "accelerator_type": resource["accelerator_type"],
                    "config": (
                        config.existing_a_reference.source_config_sha256
                        if historical
                        else config.config_hash
                    ),
                    "command": command,
                    "required_measurements": [
                        "compile_seconds",
                        "steady_step_seconds",
                        "useful_mfu",
                        "hbm_peak_bytes",
                        "input_wait_fraction",
                        "collective_fraction",
                        "top_hlo_ops_by_time",
                    ],
                }
            )
    body = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "source_revision": source_revision,
        "bounds": {
            "maximum_seconds_per_task": 600,
            "warmup_steps": 1,
            "measured_steps": 3,
            "training_or_checkpoint_publication_allowed": False,
            "profile_role": "compile_and_steady_step_at_exact_training_shape",
        },
        "tasks": tasks,
    }
    return validate_profile_plan({**body, "plan_sha256": canonical_sha256(body)})


__all__ = [
    "SCHEMA_VERSION",
    "ComponentClosureProfileError",
    "build_profile_plan",
    "validate_profile_plan",
]
