"""Immutable historical references for the Experiment-5 v3 quality grid.

``A256Native`` is panel-native context, not one cross-panel treatment.  This
module inventories the completed Experiment-4 seed-zero ``A`` trajectories:

* SAE: ReLU, strict-positive TopK-2;
* transcoder: Softplus, exact-positive TopK-2.

The resulting manifest pins both trajectories at 2M/5M/10M tokens without
copying, restoring, or relabelling their checkpoints.
"""

from __future__ import annotations

import copy
import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from moe_sae.checkpointing import milestone_pointer_path, validate_pointer
from moe_sae.contribution_artifacts import canonical_sha256
from moe_sae.direct_horizon_campaign import (
    PROTOCOL_ID as EXPERIMENT4_PROTOCOL_ID,
    DirectCampaignPlan,
    inspect_training_completion as inspect_horizon_training_completion,
)
from moe_sae.experiment5_campaign import (
    EvidencePath,
    _evidence,
    _exact_mapping,
    _git_sha,
    _read_self_hashed_json,
    _remote_path,
    _safe_id,
    _sha256,
)
from moe_sae.experiment5_routed_ladder_campaign import (
    RoutedLadderCampaignPlanError,
    RoutedLadderCampaignStateError,
)
from moe_sae.experiment5_routed_ladder_v3_config import (
    A256_NATIVE_CONFIG_SHA256,
    A256_NATIVE_LABEL,
    PROTOCOL_ID,
)
from moe_sae.learning_horizon_config import (
    LearningHorizonConfigError,
    load_learning_horizon_config,
)

NATIVE_A256_REFERENCE_MANIFEST_SCHEMA_VERSION = (
    "gemma4-experiment5-routed-activation-ladder-v3-"
    "native-a256-reference-manifest-v1"
)
PANELS = ("sae", "transcoder")
MILESTONES = (2_000_000, 5_000_000, 10_000_000)
TRAIN_TOKENS = 10_000_000


@dataclass(frozen=True, slots=True)
class NativeA256TrainingCost:
    panel: str
    resource_id: str
    accelerator_type: str
    chip_count: int
    marginal_wall_seconds: float
    chip_seconds: float
    chip_seconds_scope: str
    chip_seconds_excludes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class NativeA256CheckpointReference:
    panel: str
    source_label: str
    seed: int
    checkpoint_tokens: int
    training_source_revision: str
    source_config: EvidencePath
    source_config_sha256: str
    run_root_remote: str
    checkpoint_payload_root_gcs: str
    run_contract: EvidencePath
    run_contract_sha256: str
    training_result: EvidencePath
    training_result_sha256: str
    checkpoint_pointer: EvidencePath
    checkpoint_pointer_sha256: str
    resource_id: str
    accelerator_type: str

    @property
    def identity(self) -> tuple[str, str, int, int]:
        return self.panel, "A256Native", self.seed, self.checkpoint_tokens


def _positive_number(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise RoutedLadderCampaignPlanError(
            f"{name} must be finite and positive"
        )
    return float(value)


def _gcs_prefix(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith("gs://")
        or value.endswith("/")
    ):
        raise RoutedLadderCampaignPlanError(
            f"{name} must be a canonical GCS prefix"
        )
    return value


def _training_cost(value: Any, index: int) -> NativeA256TrainingCost:
    name = f"native_a256_reference_manifest.training_costs[{index}]"
    row = _exact_mapping(
        value,
        {
            "panel",
            "resource_id",
            "accelerator_type",
            "chip_count",
            "marginal_wall_seconds",
            "chip_seconds",
            "chip_seconds_scope",
            "chip_seconds_excludes",
        },
        name,
    )
    panel = row["panel"]
    chip_count = row["chip_count"]
    excludes = row["chip_seconds_excludes"]
    if (
        panel not in PANELS
        or isinstance(chip_count, bool)
        or not isinstance(chip_count, int)
        or chip_count <= 0
        or not isinstance(excludes, list)
        or not excludes
        or not all(isinstance(item, str) and item for item in excludes)
        or not isinstance(row["chip_seconds_scope"], str)
        or not row["chip_seconds_scope"]
    ):
        raise RoutedLadderCampaignPlanError(
            f"{name} has invalid measured-cost metadata"
        )
    marginal = _positive_number(
        row["marginal_wall_seconds"],
        f"{name}.marginal_wall_seconds",
    )
    chip_seconds = _positive_number(
        row["chip_seconds"],
        f"{name}.chip_seconds",
    )
    if not math.isclose(
        chip_seconds,
        chip_count * marginal,
        rel_tol=1e-9,
        abs_tol=1e-6,
    ):
        raise RoutedLadderCampaignPlanError(
            f"{name} chip-seconds do not equal chips times wall-seconds"
        )
    return NativeA256TrainingCost(
        panel=str(panel),
        resource_id=_safe_id(row["resource_id"], f"{name}.resource_id"),
        accelerator_type=_safe_id(
            row["accelerator_type"],
            f"{name}.accelerator_type",
        ),
        chip_count=int(chip_count),
        marginal_wall_seconds=marginal,
        chip_seconds=chip_seconds,
        chip_seconds_scope=str(row["chip_seconds_scope"]),
        chip_seconds_excludes=tuple(excludes),
    )


def parse_native_a256_reference_manifest(
    value: Any,
) -> tuple[
    Mapping[str, Any],
    tuple[NativeA256CheckpointReference, ...],
    tuple[NativeA256TrainingCost, ...],
]:
    selected = _exact_mapping(
        value,
        {
            "schema_version",
            "source_protocol_id",
            "target_protocol_id",
            "source_plan_sha256",
            "training_costs",
            "cells",
            "manifest_sha256",
        },
        "native_a256_reference_manifest",
    )
    if (
        selected["schema_version"]
        != NATIVE_A256_REFERENCE_MANIFEST_SCHEMA_VERSION
        or selected["source_protocol_id"] != EXPERIMENT4_PROTOCOL_ID
        or selected["target_protocol_id"] != PROTOCOL_ID
    ):
        raise RoutedLadderCampaignPlanError(
            "native A256 reference manifest identity changed"
        )
    _sha256(
        selected["source_plan_sha256"],
        "native_a256_reference_manifest.source_plan_sha256",
    )
    body = {
        key: selected[key] for key in selected if key != "manifest_sha256"
    }
    claimed = _sha256(
        selected["manifest_sha256"],
        "native_a256_reference_manifest.manifest_sha256",
    )
    if canonical_sha256(body) != claimed:
        raise RoutedLadderCampaignPlanError(
            "native A256 reference manifest self-hash changed"
        )
    raw_costs = selected["training_costs"]
    raw_cells = selected["cells"]
    if not isinstance(raw_costs, list) or not isinstance(raw_cells, list):
        raise RoutedLadderCampaignPlanError(
            "native A256 costs and cells must be lists"
        )
    costs = tuple(
        _training_cost(row, index) for index, row in enumerate(raw_costs)
    )
    if (
        len(costs) != len(PANELS)
        or {row.panel for row in costs} != set(PANELS)
    ):
        raise RoutedLadderCampaignPlanError(
            "native A256 manifest requires one measured cost per panel"
        )
    cells: list[NativeA256CheckpointReference] = []
    for index, raw in enumerate(raw_cells):
        name = f"native_a256_reference_manifest.cells[{index}]"
        row = _exact_mapping(
            raw,
            {
                "panel",
                "target_condition",
                "source_condition",
                "source_label",
                "seed",
                "checkpoint_tokens",
                "training_source_revision",
                "source_config",
                "source_config_sha256",
                "run_root_remote",
                "checkpoint_payload_root_gcs",
                "run_contract",
                "run_contract_sha256",
                "training_result",
                "training_result_sha256",
                "checkpoint_pointer",
                "checkpoint_pointer_sha256",
                "resource_id",
                "accelerator_type",
            },
            name,
        )
        panel = row["panel"]
        milestone = row["checkpoint_tokens"]
        if (
            panel not in PANELS
            or row["target_condition"] != "A256Native"
            or row["source_condition"] != "A"
            or row["source_label"] != A256_NATIVE_LABEL.get(str(panel))
            or row["seed"] != 0
            or milestone not in MILESTONES
        ):
            raise RoutedLadderCampaignPlanError(
                f"{name} is not a panel-native Experiment-4 A milestone"
            )
        cells.append(
            NativeA256CheckpointReference(
                panel=str(panel),
                source_label=str(row["source_label"]),
                seed=0,
                checkpoint_tokens=int(milestone),
                training_source_revision=_git_sha(
                    row["training_source_revision"],
                    f"{name}.training_source_revision",
                ),
                source_config=_evidence(
                    row["source_config"],
                    f"{name}.source_config",
                ),
                source_config_sha256=_sha256(
                    row["source_config_sha256"],
                    f"{name}.source_config_sha256",
                ),
                run_root_remote=_remote_path(
                    row["run_root_remote"],
                    f"{name}.run_root_remote",
                ),
                checkpoint_payload_root_gcs=_gcs_prefix(
                    row["checkpoint_payload_root_gcs"],
                    f"{name}.checkpoint_payload_root_gcs",
                ),
                run_contract=_evidence(
                    row["run_contract"],
                    f"{name}.run_contract",
                ),
                run_contract_sha256=_sha256(
                    row["run_contract_sha256"],
                    f"{name}.run_contract_sha256",
                ),
                training_result=_evidence(
                    row["training_result"],
                    f"{name}.training_result",
                ),
                training_result_sha256=_sha256(
                    row["training_result_sha256"],
                    f"{name}.training_result_sha256",
                ),
                checkpoint_pointer=_evidence(
                    row["checkpoint_pointer"],
                    f"{name}.checkpoint_pointer",
                ),
                checkpoint_pointer_sha256=_sha256(
                    row["checkpoint_pointer_sha256"],
                    f"{name}.checkpoint_pointer_sha256",
                ),
                resource_id=_safe_id(
                    row["resource_id"], f"{name}.resource_id"
                ),
                accelerator_type=_safe_id(
                    row["accelerator_type"],
                    f"{name}.accelerator_type",
                ),
            )
        )
    expected = {
        (panel, "A256Native", 0, milestone)
        for panel in PANELS
        for milestone in MILESTONES
    }
    identities = {cell.identity for cell in cells}
    if len(cells) != len(expected) or identities != expected:
        raise RoutedLadderCampaignPlanError(
            "native A256 manifest requires both panels at 2M/5M/10M"
        )
    cost_by_panel = {cost.panel: cost for cost in costs}
    for panel in PANELS:
        panel_cells = [cell for cell in cells if cell.panel == panel]
        if (
            len({cell.training_source_revision for cell in panel_cells}) != 1
            or len({cell.source_config for cell in panel_cells}) != 1
            or len({cell.source_config_sha256 for cell in panel_cells}) != 1
            or len({cell.run_root_remote for cell in panel_cells}) != 1
            or len(
                {cell.checkpoint_payload_root_gcs for cell in panel_cells}
            )
            != 1
            or len({cell.run_contract for cell in panel_cells}) != 1
            or len({cell.run_contract_sha256 for cell in panel_cells}) != 1
            or len({cell.training_result for cell in panel_cells}) != 1
            or len({cell.training_result_sha256 for cell in panel_cells}) != 1
            or len({cell.checkpoint_pointer for cell in panel_cells})
            != len(MILESTONES)
            or len(
                {cell.checkpoint_pointer_sha256 for cell in panel_cells}
            )
            != len(MILESTONES)
            or {
                (cell.resource_id, cell.accelerator_type)
                for cell in panel_cells
            }
            != {
                (
                    cost_by_panel[panel].resource_id,
                    cost_by_panel[panel].accelerator_type,
                )
            }
        ):
            raise RoutedLadderCampaignPlanError(
                f"native A256 {panel} milestones do not share one parent"
            )
    if len({cell.training_source_revision for cell in cells}) != 1:
        raise RoutedLadderCampaignPlanError(
            "native A256 panels must share one training source revision"
        )
    return (
        copy.deepcopy(selected),
        tuple(
            sorted(
                cells,
                key=lambda row: (row.panel, row.checkpoint_tokens),
            )
        ),
        tuple(sorted(costs, key=lambda row: row.panel)),
    )


def _expected_progress(
    result: Mapping[str, Any],
    checkpoint_tokens: int,
) -> tuple[int, int]:
    identity = result.get("run_identity")
    plan = identity.get("plan") if isinstance(identity, Mapping) else None
    if not isinstance(plan, Mapping):
        raise RoutedLadderCampaignStateError(
            "native A256 result lacks its immutable worker plan"
        )
    microbatch = plan.get("global_microbatch_tokens")
    accumulation = plan.get("microbatches_per_update")
    total = plan.get("total_tokens")
    planned_updates = plan.get("planned_updates")
    if (
        isinstance(microbatch, bool)
        or not isinstance(microbatch, int)
        or microbatch <= 0
        or isinstance(accumulation, bool)
        or not isinstance(accumulation, int)
        or accumulation <= 0
        or total != TRAIN_TOKENS
        or planned_updates
        != math.ceil(total / (microbatch * accumulation))
    ):
        raise RoutedLadderCampaignStateError(
            "native A256 result has an invalid exact-token plan"
        )
    batch = microbatch * accumulation
    step = math.ceil(checkpoint_tokens / batch)
    return step, min(step * batch, total)


def validate_native_a256_checkpoint_lineage(
    cell: NativeA256CheckpointReference,
    cost: NativeA256TrainingCost,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        config_payload = cell.source_config.controller_path.read_bytes()
    except OSError as error:
        raise RoutedLadderCampaignStateError(
            f"cannot read native A256 {cell.panel} config"
        ) from error
    if hashlib.sha256(config_payload).hexdigest() != cell.source_config.sha256:
        raise RoutedLadderCampaignStateError(
            f"native A256 {cell.panel} config file changed"
        )
    try:
        config = load_learning_horizon_config(
            cell.source_config.controller_path,
            expected_scientific_model=cell.panel,
        )
    except LearningHorizonConfigError as error:
        raise RoutedLadderCampaignStateError(
            f"native A256 {cell.panel} config is invalid"
        ) from error
    if (
        config.config_hash != cell.source_config_sha256
        or config.config_hash != A256_NATIVE_CONFIG_SHA256[cell.panel]
    ):
        raise RoutedLadderCampaignStateError(
            f"native A256 {cell.panel} semantic config changed"
        )
    contract, contract_payload = _read_self_hashed_json(
        cell.run_contract.controller_path,
        hash_field="run_contract_sha256",
    )
    result, result_payload = _read_self_hashed_json(
        cell.training_result.controller_path,
        hash_field="result_sha256",
    )
    pointer, pointer_payload = _read_self_hashed_json(
        cell.checkpoint_pointer.controller_path,
        hash_field="pointer_sha256",
    )
    identity = result.get("run_identity")
    run_plan = identity.get("plan") if isinstance(identity, Mapping) else None
    if (
        hashlib.sha256(contract_payload).hexdigest()
        != cell.run_contract.sha256
        or hashlib.sha256(result_payload).hexdigest()
        != cell.training_result.sha256
        or hashlib.sha256(pointer_payload).hexdigest()
        != cell.checkpoint_pointer.sha256
        or contract.get("run_contract_sha256")
        != cell.run_contract_sha256
        or result.get("result_sha256") != cell.training_result_sha256
        or result.get("schema_version")
        != "gemma4-contribution-sae-training-result-v1"
        or result.get("status") not in {"completed", "already_complete"}
        or result.get("tokens") != TRAIN_TOKENS
        or result.get("source_revision")
        != cell.training_source_revision
        or not isinstance(identity, Mapping)
        or not isinstance(run_plan, Mapping)
        or run_plan.get("phase_name") != "learning_horizon_10m"
        or run_plan.get("condition") != "A"
        or run_plan.get("seed") != 0
        or contract.get("project_commit")
        != cell.training_source_revision
        or contract.get("phase") != "learning_horizon_10m"
        or contract.get("scientific_model") != cell.panel
        or contract.get("condition") != "A"
        or contract.get("seed") != 0
        or contract.get("config_file_sha256")
        != cell.source_config.sha256
        or contract.get("checkpoint_payload_root")
        != cell.checkpoint_payload_root_gcs
        or cost.panel != cell.panel
        or cost.resource_id != cell.resource_id
        or cost.accelerator_type != cell.accelerator_type
        or not math.isclose(
            float(result.get("marginal_wall_seconds", -1)),
            cost.marginal_wall_seconds,
            rel_tol=1e-9,
            abs_tol=1e-6,
        )
        or not math.isclose(
            float(result.get("chip_seconds", -1)),
            cost.chip_seconds,
            rel_tol=1e-9,
            abs_tol=1e-6,
        )
        or result.get("chip_seconds_scope") != cost.chip_seconds_scope
        or result.get("chip_seconds_excludes")
        != list(cost.chip_seconds_excludes)
    ):
        raise RoutedLadderCampaignStateError(
            f"native A256 lineage changed: {cell.identity}"
        )
    expected_step, expected_tokens = _expected_progress(
        result,
        cell.checkpoint_tokens,
    )
    try:
        checked = validate_pointer(
            pointer,
            expected_kind="latest",
            payload_root=cell.checkpoint_payload_root_gcs,
            run_identity=identity,
        )
    except (RuntimeError, TypeError, ValueError) as error:
        raise RoutedLadderCampaignStateError(
            f"native A256 checkpoint is invalid: {cell.identity}"
        ) from error
    if (
        checked["pointer_sha256"] != cell.checkpoint_pointer_sha256
        or checked["step"] != expected_step
        or checked["tokens_seen"] != expected_tokens
    ):
        raise RoutedLadderCampaignStateError(
            f"native A256 checkpoint progress changed: {cell.identity}"
        )
    return result, checked


def build_native_a256_reference_manifest(
    horizon_plan: DirectCampaignPlan,
) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    costs: list[dict[str, Any]] = []
    environment = horizon_plan.environment_dict()
    training_revision = _git_sha(
        environment.get("TRAINING_PROJECT_COMMIT"),
        "horizon_plan.environment.TRAINING_PROJECT_COMMIT",
    )
    for panel in PANELS:
        matches = [
            task
            for task in horizon_plan.trajectories
            if task.scientific_model == panel
            and task.condition == "A"
            and task.seed == 0
        ]
        if len(matches) != 1:
            raise RoutedLadderCampaignPlanError(
                f"horizon plan lacks one exact historical {panel} A seed-zero "
                "trajectory"
            )
        task = matches[0]
        if inspect_horizon_training_completion(task) is None:
            raise RoutedLadderCampaignPlanError(
                f"historical {panel} A256 training is incomplete"
            )
        result, result_payload = _read_self_hashed_json(
            task.completion_path_controller,
            hash_field="result_sha256",
        )
        remote_horizon_root = PurePosixPath(
            task.control_root_remote
        ).parents[1]
        try:
            relative_config = PurePosixPath(task.config_remote).relative_to(
                remote_horizon_root
            )
            controller_horizon_root = (
                task.completion_path_controller.parents[4]
            )
        except (ValueError, IndexError) as error:
            raise RoutedLadderCampaignPlanError(
                f"historical {panel} config is outside its artifact plane"
            ) from error
        controller_config = (
            controller_horizon_root / Path(*relative_config.parts)
        )
        try:
            config = load_learning_horizon_config(
                controller_config,
                expected_scientific_model=panel,
            )
            config_payload = controller_config.read_bytes()
        except (LearningHorizonConfigError, OSError) as error:
            raise RoutedLadderCampaignPlanError(
                f"historical {panel} A256 config is invalid"
            ) from error
        if config.config_hash != A256_NATIVE_CONFIG_SHA256[panel]:
            raise RoutedLadderCampaignPlanError(
                f"historical {panel} A256 semantic config differs from the "
                "frozen v3 reference metadata: "
                f"expected={A256_NATIVE_CONFIG_SHA256[panel]} "
                f"observed={config.config_hash}; an explicit protocol "
                "addendum is required before this optional context can enter "
                "the report"
            )
        resource = horizon_plan.resource(task.resource_id)
        marginal = _positive_number(
            result.get("marginal_wall_seconds"),
            f"historical {panel} A256 marginal_wall_seconds",
        )
        chip_seconds = _positive_number(
            result.get("chip_seconds"),
            f"historical {panel} A256 chip_seconds",
        )
        raw_chips = chip_seconds / marginal
        chip_count = round(raw_chips)
        exclusions = result.get("chip_seconds_excludes")
        scope = result.get("chip_seconds_scope")
        if (
            chip_count <= 0
            or not math.isclose(
                raw_chips,
                chip_count,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            or not isinstance(scope, str)
            or not scope
            or not isinstance(exclusions, list)
            or not exclusions
            or not all(isinstance(item, str) and item for item in exclusions)
        ):
            raise RoutedLadderCampaignPlanError(
                f"historical {panel} A256 cost ledger is invalid"
            )
        run_root = str(
            PurePosixPath(task.control_root_remote) / "runs" / task.run_id
        )
        run_contract_path = (
            task.completion_path_controller.parent / "run-contract.json"
        )
        contract, contract_payload = _read_self_hashed_json(
            run_contract_path,
            hash_field="run_contract_sha256",
        )
        config_file_sha = hashlib.sha256(config_payload).hexdigest()
        if (
            contract.get("project_commit") != training_revision
            or contract.get("phase") != "learning_horizon_10m"
            or contract.get("scientific_model") != panel
            or contract.get("condition") != "A"
            or contract.get("seed") != 0
            or contract.get("config_file_sha256") != config_file_sha
            or contract.get("checkpoint_payload_root")
            != task.checkpoint_root_gcs
        ):
            raise RoutedLadderCampaignPlanError(
                f"historical {panel} A256 run contract changed"
            )
        costs.append(
            {
                "panel": panel,
                "resource_id": resource.resource_id,
                "accelerator_type": resource.accelerator_type,
                "chip_count": chip_count,
                "marginal_wall_seconds": marginal,
                "chip_seconds": chip_seconds,
                "chip_seconds_scope": scope,
                "chip_seconds_excludes": exclusions,
            }
        )
        common = {
            "panel": panel,
            "target_condition": "A256Native",
            "source_condition": "A",
            "source_label": A256_NATIVE_LABEL[panel],
            "seed": 0,
            "training_source_revision": training_revision,
            "source_config": {
                "controller_path": str(controller_config),
                "remote_path": task.config_remote,
                "sha256": config_file_sha,
            },
            "source_config_sha256": config.config_hash,
            "run_root_remote": run_root,
            "checkpoint_payload_root_gcs": task.checkpoint_root_gcs,
            "run_contract": {
                "controller_path": str(run_contract_path),
                "remote_path": str(
                    PurePosixPath(run_root) / "run-contract.json"
                ),
                "sha256": hashlib.sha256(
                    contract_payload
                ).hexdigest(),
            },
            "run_contract_sha256": contract["run_contract_sha256"],
            "training_result": {
                "controller_path": str(task.completion_path_controller),
                "remote_path": str(
                    PurePosixPath(run_root) / "training-result.json"
                ),
                "sha256": hashlib.sha256(result_payload).hexdigest(),
            },
            "training_result_sha256": result["result_sha256"],
            "resource_id": resource.resource_id,
            "accelerator_type": resource.accelerator_type,
        }
        controller_checkpoints = (
            task.completion_path_controller.parent / "checkpoints"
        )
        remote_checkpoints = str(
            PurePosixPath(run_root) / "checkpoints"
        )
        for milestone in MILESTONES:
            pointer_path = milestone_pointer_path(
                controller_checkpoints,
                milestone,
            )
            pointer, pointer_payload = _read_self_hashed_json(
                pointer_path,
                hash_field="pointer_sha256",
            )
            cells.append(
                {
                    **common,
                    "checkpoint_tokens": milestone,
                    "checkpoint_pointer": {
                        "controller_path": str(pointer_path),
                        "remote_path": str(
                            milestone_pointer_path(
                                remote_checkpoints,
                                milestone,
                            )
                        ),
                        "sha256": hashlib.sha256(
                            pointer_payload
                        ).hexdigest(),
                    },
                    "checkpoint_pointer_sha256": pointer[
                        "pointer_sha256"
                    ],
                }
            )
    body = {
        "schema_version": (
            NATIVE_A256_REFERENCE_MANIFEST_SCHEMA_VERSION
        ),
        "source_protocol_id": EXPERIMENT4_PROTOCOL_ID,
        "target_protocol_id": PROTOCOL_ID,
        "source_plan_sha256": horizon_plan.plan_sha256,
        "training_costs": costs,
        "cells": cells,
    }
    manifest = {**body, "manifest_sha256": canonical_sha256(body)}
    _parsed, references, parsed_costs = (
        parse_native_a256_reference_manifest(manifest)
    )
    costs_by_panel = {cost.panel: cost for cost in parsed_costs}
    for cell in references:
        validate_native_a256_checkpoint_lineage(
            cell,
            costs_by_panel[cell.panel],
        )
    return manifest


def load_native_a256_reference_manifest(
    path: str | Path,
) -> Mapping[str, Any]:
    selected = Path(path).expanduser().resolve()
    try:
        value = yaml.safe_load(selected.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise RoutedLadderCampaignPlanError(
            f"cannot read native A256 reference manifest: {selected}"
        ) from error
    parsed, _cells, _costs = parse_native_a256_reference_manifest(value)
    return parsed


def write_native_a256_reference_manifest(
    path: str | Path,
    manifest: Mapping[str, Any],
) -> tuple[
    Mapping[str, Any],
    tuple[NativeA256CheckpointReference, ...],
    tuple[NativeA256TrainingCost, ...],
]:
    parsed, cells, costs = parse_native_a256_reference_manifest(manifest)
    selected = Path(path).expanduser().resolve()
    selected.parent.mkdir(parents=True, exist_ok=True)
    if selected.exists():
        raise RoutedLadderCampaignPlanError(
            f"refusing to overwrite native A256 manifest: {selected}"
        )
    selected.write_text(
        yaml.safe_dump(
            dict(manifest),
            sort_keys=False,
            default_flow_style=False,
        ),
        encoding="utf-8",
    )
    return parsed, cells, costs


__all__ = [
    "MILESTONES",
    "NATIVE_A256_REFERENCE_MANIFEST_SCHEMA_VERSION",
    "NativeA256CheckpointReference",
    "NativeA256TrainingCost",
    "build_native_a256_reference_manifest",
    "load_native_a256_reference_manifest",
    "parse_native_a256_reference_manifest",
    "validate_native_a256_checkpoint_lineage",
    "write_native_a256_reference_manifest",
]
