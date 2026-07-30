#!/usr/bin/env python3
"""Build or execute mixed-parent component-closure resident bundles."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
import time
import uuid
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Sequence
from concurrent.futures import ThreadPoolExecutor

import yaml

from moe_sae.component_closure_campaign import (
    load_training_plan,
    parse_training_campaign,
)
from moe_sae.component_closure_evaluation import (
    build_evaluation_plan,
    collect_score_grid,
    collect_training_costs,
    load_evaluation_plan,
    run_resident_bundle,
)
from moe_sae.component_closure_references import load_a_reference_manifest
from moe_sae.contribution_artifacts import canonical_sha256
from moe_sae.experiment5_campaign import (
    ResourceReadinessStatus,
    _default_resource_readiness_probe,
)

PROCESS_COUNT = 4
DEFAULT_OPERATOR_TIMEOUT_SECONDS = 43_200
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class ComponentClosureEvaluationOperatorError(RuntimeError):
    """One controller-side resident-bundle launch is unsafe or incomplete."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--training-plan", required=True, type=Path)
    build.add_argument("--a-references", required=True, type=Path)
    build.add_argument("--runtime", required=True, type=Path)
    build.add_argument("--evaluation-source-revision", required=True)
    build.add_argument("--state-root", required=True, type=Path)
    build.add_argument("--output", required=True, type=Path)
    run = commands.add_parser("run-bundle")
    run.add_argument("--plan", required=True, type=Path)
    run.add_argument("--bundle-index", required=True, type=int)
    run.add_argument("--process-index", required=True, type=int)
    run.add_argument("--attempt-id", required=True)
    run_json = commands.add_parser("run-bundle-json")
    run_json.add_argument("--bundle-json", required=True)
    run_json.add_argument("--process-index", required=True, type=int)
    run_json.add_argument("--attempt-id", required=True)
    operator = commands.add_parser("run-bundle-operator")
    operator.add_argument("--plan", required=True, type=Path)
    operator.add_argument("--training-plan", required=True, type=Path)
    operator.add_argument("--bundle-index", required=True, type=int)
    operator.add_argument("--attempt-id", required=True)
    operator.add_argument("--log-root", required=True, type=Path)
    operator.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_OPERATOR_TIMEOUT_SECONDS,
    )
    operator.add_argument("--dry-run", action="store_true")
    run_all = commands.add_parser("run-all-operators")
    run_all.add_argument("--plan", required=True, type=Path)
    run_all.add_argument("--training-plan", required=True, type=Path)
    run_all.add_argument(
        "--maximum-wall-seconds",
        type=int,
        default=DEFAULT_OPERATOR_TIMEOUT_SECONDS,
    )
    run_all.add_argument("--dry-run", action="store_true")
    validate = commands.add_parser("validate")
    validate.add_argument("--plan", required=True, type=Path)
    collect = commands.add_parser("collect")
    collect.add_argument("--plan", required=True, type=Path)
    collect.add_argument("--costs", required=True, type=Path)
    collect.add_argument("--output", required=True, type=Path)
    collect_costs = commands.add_parser("collect-costs")
    collect_costs.add_argument("--training-plan", required=True, type=Path)
    collect_costs.add_argument("--a-references", required=True, type=Path)
    collect_costs.add_argument("--output", required=True, type=Path)
    return parser


def _load_routed_operator() -> ModuleType:
    path = Path(__file__).with_name("evaluate_experiment5_routed_ladder.py")
    spec = importlib.util.spec_from_file_location(
        "_component_closure_routed_operator",
        path,
    )
    if spec is None or spec.loader is None:
        raise ComponentClosureEvaluationOperatorError(
            "cannot load mature routed evaluator operator"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _operator_binding(
    evaluation: dict,
    training: dict,
    *,
    bundle_index: int,
) -> tuple[dict, object, object]:
    if evaluation["training_plan_sha256"] != training["plan_sha256"]:
        raise ComponentClosureEvaluationOperatorError(
            "evaluation plan belongs to another training plan"
        )
    if bundle_index < 0 or bundle_index >= len(evaluation["bundles"]):
        raise ComponentClosureEvaluationOperatorError(
            "bundle index is outside the six-bundle matrix"
        )
    bundle = evaluation["bundles"][bundle_index]
    parsed_training = parse_training_campaign(training)
    resource = parsed_training.resource(bundle["resource_id"])
    mapping_path = resource.controller_mapping.controller_path
    if mapping_path.is_symlink() or not mapping_path.is_file():
        raise ComponentClosureEvaluationOperatorError(
            f"{resource.resource_id} controller mapping is unavailable"
        )
    mapping_payload = mapping_path.read_bytes()
    if (
        hashlib.sha256(mapping_payload).hexdigest()
        != resource.controller_mapping.sha256
    ):
        raise ComponentClosureEvaluationOperatorError(
            f"{resource.resource_id} controller mapping hash changed"
        )
    try:
        mapping_value = json.loads(mapping_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ComponentClosureEvaluationOperatorError(
            f"{resource.resource_id} controller mapping is invalid"
        ) from error
    if (
        not isinstance(mapping_value, dict)
        or mapping_value.get("tpu_name") != resource.tpu_name
        or mapping_value.get("zone") != resource.zone
        or mapping_value.get("accelerator_type") != resource.accelerator_type
    ):
        raise ComponentClosureEvaluationOperatorError(
            f"{resource.resource_id} controller mapping identity changed"
        )
    settings = SimpleNamespace(
        project=resource.project,
        tpu=resource.tpu_name,
        zone=resource.zone,
        ssh_user=resource.ssh_user,
        ssh_key=None,
        internal_ip=resource.internal_ip,
        mapping_receipt=str(mapping_path),
        accelerator_type=resource.accelerator_type,
    )
    return bundle, parsed_training, settings


def _worker_command(
    bundle: dict,
    training: object,
    resource: object,
    *,
    process_index: int,
    attempt_id: str,
) -> list[str]:
    command = [
        str(training.remote_uv),
        "run",
        "--frozen",
        "--extra",
        "tpu",
        "python",
        "scripts/tpu/evaluate_component_closure.py",
        "run-bundle-json",
        "--bundle-json",
        json.dumps(bundle, sort_keys=True, separators=(",", ":")),
        "--process-index",
        str(process_index),
        "--attempt-id",
        attempt_id,
    ]
    device_kind = resource.environment_dict().get("MOE_SAE_V5LITEPOD_DEVICE_KIND")
    if device_kind is not None:
        command[:0] = [
            "env",
            f"MOE_SAE_V5LITEPOD_DEVICE_KIND={device_kind}",
        ]
    return command


def _remote_command(
    routed_operator: ModuleType,
    settings: object,
    *,
    training: object,
    resource: object,
    bundle: dict,
    controller_index: int,
    process_index: int,
    attempt_id: str,
) -> list[str]:
    worker = _worker_command(
        bundle,
        training,
        resource,
        process_index=process_index,
        attempt_id=attempt_id,
    )
    body = "\n".join(
        (
            "set -euo pipefail",
            f"cd {shlex.quote(str(training.remote_project_root))}",
            (
                'test "$(git rev-parse HEAD)" = '
                f"{shlex.quote(training.project_commit)}"
            ),
            'test -z "$(git status --porcelain=v1 --untracked-files=all)"',
            "set +e",
            shlex.join(worker),
            "evaluation_status=$?",
            "set -e",
            (
                "printf 'terminal action=component-closure-evaluate-bundle "
                "controller=%s process=%s status=%s\\n' "
                f'{controller_index} {process_index} "$evaluation_status"'
            ),
            'exit "$evaluation_status"',
        )
    )
    return routed_operator._gcloud_ssh(
        settings,
        controller=controller_index,
        remote_command="bash -lc " + shlex.quote(body),
    )


def _run_remote(
    command: list[str],
    log: Path,
    *,
    timeout_seconds: int,
    environment: dict[str, str],
) -> int:
    log.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with log.open("xb") as handle:
        process = subprocess.Popen(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=environment,
        )
        try:
            return process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
            return 124


def _terminal_status(log: Path, *, controller: int, process: int) -> int:
    prefix = (
        "terminal action=component-closure-evaluate-bundle "
        f"controller={controller} process={process} status="
    ).encode("ascii")
    rows = [line for line in log.read_bytes().splitlines() if line.startswith(prefix)]
    if (
        len(rows) != 1
        or re.fullmatch(rb"(?:0|[1-9][0-9]{0,2})", rows[0][len(prefix) :]) is None
    ):
        raise ComponentClosureEvaluationOperatorError(
            f"controller {controller} lacks one terminal marker"
        )
    return int(rows[0][len(prefix) :])


def _validated_remote_cell(
    bundle: dict,
    cell: dict,
    payload: bytes,
) -> tuple[dict, str]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ComponentClosureEvaluationOperatorError(
            "remote evaluation result is not JSON"
        ) from error
    if not isinstance(value, dict):
        raise ComponentClosureEvaluationOperatorError(
            "remote evaluation result is not an object"
        )
    claimed = value.get("result_sha256")
    unsigned = {key: item for key, item in value.items() if key != "result_sha256"}
    relative = value.get("score_data_object")
    if (
        not isinstance(claimed, str)
        or canonical_sha256(unsigned) != claimed
        or value.get("protocol_id") != cell["source_protocol_id"]
        or value.get("status") != "cell_evaluated"
        or value.get("panel") != bundle["panel"]
        or value.get("condition") != cell["source_condition"]
        or value.get("seed") != bundle["seed"]
        or value.get("checkpoint_tokens") != cell["checkpoint_tokens"]
        or value.get("source_revision") != bundle["source_revision"]
        or value.get("parent_training_result_sha256")
        != cell["parent_training_result_sha256"]
        or value.get("parent_checkpoint_pointer_sha256")
        != cell["parent_checkpoint_pointer_sha256"]
        or not isinstance(relative, str)
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
        or not relative.startswith("objects/")
    ):
        raise ComponentClosureEvaluationOperatorError(
            "remote evaluation result identity changed"
        )
    return value, relative


def _publish_cell(
    cell: dict,
    *,
    result_payload: bytes,
    relative: str,
    score_payload: bytes,
) -> None:
    result = json.loads(result_payload)
    if hashlib.sha256(score_payload).hexdigest() != result.get(
        "score_data_sha256"
    ) or len(score_payload) != result.get("score_data_bytes"):
        raise ComponentClosureEvaluationOperatorError(
            "remote score object bytes changed"
        )
    destination = Path(cell["result_root_controller"])
    result_path = destination / "evaluation-result.json"
    score_path = destination / relative
    if destination.exists():
        if (
            destination.is_dir()
            and not destination.is_symlink()
            and result_path.is_file()
            and score_path.is_file()
            and result_path.read_bytes() == result_payload
            and score_path.read_bytes() == score_payload
        ):
            return
        raise ComponentClosureEvaluationOperatorError(
            "controller cell destination is partial or changed"
        )
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    staging = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    try:
        (staging / Path(relative).parent).mkdir(parents=True, mode=0o700)
        (staging / "evaluation-result.json").write_bytes(result_payload)
        (staging / relative).write_bytes(score_payload)
        os.rename(staging, destination)
    finally:
        if staging.exists():
            for child in sorted(staging.rglob("*"), reverse=True):
                if child.is_file():
                    child.unlink()
                elif child.is_dir():
                    child.rmdir()
            staging.rmdir()


def _completion_path(evaluation: dict, bundle: dict) -> Path:
    return (
        Path(evaluation["state_root"])
        / "bundles"
        / f"{bundle['panel']}-seed{bundle['seed']}"
        / "completion.json"
    )


def _existing_operator_completion(
    evaluation: dict,
    bundle: dict,
) -> dict | None:
    path = _completion_path(evaluation, bundle)
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ComponentClosureEvaluationOperatorError(
            "bundle operator completion is invalid"
        ) from error
    if not isinstance(value, dict):
        raise ComponentClosureEvaluationOperatorError(
            "bundle operator completion is not an object"
        )
    claimed = value.get("completion_sha256")
    unsigned = {key: item for key, item in value.items() if key != "completion_sha256"}
    if (
        canonical_sha256(unsigned) != claimed
        or value.get("status") != "bundle_published"
        or value.get("bundle_sha256") != bundle["bundle_sha256"]
        or value.get("resource_id") != bundle["resource_id"]
        or len(value.get("cells", [])) != len(bundle["cells"])
    ):
        raise ComponentClosureEvaluationOperatorError(
            "bundle operator completion identity changed"
        )
    for cell, row in zip(bundle["cells"], value["cells"], strict=True):
        result_path = Path(cell["result_root_controller"]) / "evaluation-result.json"
        if (
            not result_path.is_file()
            or result_path.is_symlink()
            or hashlib.sha256(result_path.read_bytes()).hexdigest()
            != row.get("result_file_sha256")
        ):
            raise ComponentClosureEvaluationOperatorError(
                "bundle controller cell completion changed"
            )
        try:
            result = json.loads(result_path.read_text(encoding="ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ComponentClosureEvaluationOperatorError(
                "bundle controller result is invalid"
            ) from error
        relative = result.get("score_data_object")
        score_path = Path(cell["result_root_controller"]) / str(relative)
        if (
            result.get("result_sha256") != row.get("result_sha256")
            or not isinstance(relative, str)
            or not score_path.is_file()
            or score_path.is_symlink()
            or hashlib.sha256(score_path.read_bytes()).hexdigest()
            != row.get("score_data_sha256")
        ):
            raise ComponentClosureEvaluationOperatorError(
                "bundle controller score completion changed"
            )
    return value


def _seal_operator_completion(
    evaluation: dict,
    bundle: dict,
    *,
    attempt_id: str,
    cells: list[dict],
) -> dict:
    body = {
        "schema_version": "gemma4-component-closure-evaluation-operator-v1",
        "status": "bundle_published",
        "plan_sha256": evaluation["plan_sha256"],
        "bundle_sha256": bundle["bundle_sha256"],
        "resource_id": bundle["resource_id"],
        "panel": bundle["panel"],
        "seed": bundle["seed"],
        "attempt_id": attempt_id,
        "cells": cells,
    }
    value = {**body, "completion_sha256": canonical_sha256(body)}
    path = _completion_path(evaluation, bundle)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open("x", encoding="ascii") as handle:
        json.dump(value, handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
    return value


def _run_bundle_operator(
    evaluation: dict,
    training: dict,
    *,
    bundle_index: int,
    attempt_id: str,
    log_root: Path,
    timeout_seconds: int,
    dry_run: bool,
) -> int:
    if (
        _SAFE_ID.fullmatch(attempt_id) is None
        or timeout_seconds <= 0
        or timeout_seconds > DEFAULT_OPERATOR_TIMEOUT_SECONDS
    ):
        raise ComponentClosureEvaluationOperatorError(
            "operator attempt identity or timeout is invalid"
        )
    bundle, parsed_training, settings = _operator_binding(
        evaluation,
        training,
        bundle_index=bundle_index,
    )
    if _existing_operator_completion(evaluation, bundle) is not None:
        return 0
    resource = parsed_training.resource(bundle["resource_id"])
    routed_operator = _load_routed_operator()
    mapping = routed_operator._mapping(settings)
    commands = {
        controller: _remote_command(
            routed_operator,
            settings,
            training=parsed_training,
            resource=resource,
            bundle=bundle,
            controller_index=controller,
            process_index=process,
            attempt_id=attempt_id,
        )
        for controller, process in mapping.items()
    }
    if dry_run:
        print(
            json.dumps(
                {
                    "bundle_sha256": bundle["bundle_sha256"],
                    "resource_id": bundle["resource_id"],
                    "mapping": mapping,
                    "worker_process_zero": commands[
                        next(
                            controller
                            for controller, process in mapping.items()
                            if process == 0
                        )
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    log_root.mkdir(parents=True, exist_ok=False, mode=0o700)
    logs = {
        controller: log_root / f"bundle-c{controller:05d}-p{process:05d}.log"
        for controller, process in mapping.items()
    }
    environment = routed_operator._subprocess_environment()
    with ThreadPoolExecutor(max_workers=PROCESS_COUNT) as pool:
        futures = {
            controller: pool.submit(
                _run_remote,
                commands[controller],
                logs[controller],
                timeout_seconds=timeout_seconds,
                environment=environment,
            )
            for controller in mapping
        }
        transports = {
            controller: future.result() for controller, future in futures.items()
        }
    statuses = {
        controller: _terminal_status(
            logs[controller],
            controller=controller,
            process=process,
        )
        for controller, process in mapping.items()
    }
    if any(status != 0 for status in statuses.values()):
        raise ComponentClosureEvaluationOperatorError(
            f"bundle workers failed: transports={transports}, "
            f"statuses={statuses}, logs={log_root}"
        )
    transport_anomalies = {
        controller: status
        for controller, status in transports.items()
        if status != 0
    }
    if transport_anomalies:
        print(
            "bundle worker transports exited nonzero after exact remote "
            f"success markers: transports={transports}, statuses={statuses}, "
            f"logs={log_root}",
            file=sys.stderr,
            flush=True,
        )
    process_zero = next(
        controller for controller, process in mapping.items() if process == 0
    )
    published: list[dict] = []
    for cell in bundle["cells"]:
        result_payload = routed_operator._remote_bytes(
            settings,
            controller=process_zero,
            remote_path=f"{cell['result_root']}/evaluation-result.json",
        )
        result, relative = _validated_remote_cell(
            bundle,
            cell,
            result_payload,
        )
        score_payload = routed_operator._remote_bytes(
            settings,
            controller=process_zero,
            remote_path=f"{cell['result_root']}/{relative}",
        )
        _publish_cell(
            cell,
            result_payload=result_payload,
            relative=relative,
            score_payload=score_payload,
        )
        published.append(
            {
                "logical_condition": cell["logical_condition"],
                "checkpoint_tokens": cell["checkpoint_tokens"],
                "result_sha256": result["result_sha256"],
                "result_file_sha256": hashlib.sha256(result_payload).hexdigest(),
                "score_data_sha256": result["score_data_sha256"],
            }
        )
    _seal_operator_completion(
        evaluation,
        bundle,
        attempt_id=attempt_id,
        cells=published,
    )
    return 0


def _run_all_operators(
    evaluation: dict,
    training: dict,
    *,
    maximum_wall_seconds: int,
    dry_run: bool,
) -> int:
    if maximum_wall_seconds <= 0 or maximum_wall_seconds > 43_200:
        raise ComponentClosureEvaluationOperatorError(
            "evaluation wall must be in (0, 43200]"
        )
    groups: dict[str, list[int]] = {}
    for index, bundle in enumerate(evaluation["bundles"]):
        groups.setdefault(bundle["resource_id"], []).append(index)
    if len(groups) != 5 or sorted(len(rows) for rows in groups.values()) != [
        1,
        1,
        1,
        1,
        2,
    ]:
        raise ComponentClosureEvaluationOperatorError(
            "six bundles are not queued over the exact five resources"
        )
    deadline = time.monotonic() + maximum_wall_seconds
    parsed_training = parse_training_campaign(training)

    def resource_worker(resource_id: str, indexes: list[int]) -> None:
        resource = parsed_training.resource(resource_id)
        for bundle_index in indexes:
            bundle = evaluation["bundles"][bundle_index]
            for attempt_number in range(2):
                remaining = int(deadline - time.monotonic())
                if remaining <= 0:
                    raise ComponentClosureEvaluationOperatorError(
                        "evaluation campaign reached its 12h wall"
                    )
                if not dry_run:
                    observation = _default_resource_readiness_probe(resource)
                    if observation.status is not ResourceReadinessStatus.READY:
                        if attempt_number == 0:
                            time.sleep(min(30, max(1, remaining)))
                            continue
                        raise ComponentClosureEvaluationOperatorError(
                            f"{resource_id} is not ready: {observation.detail}"
                        )
                attempt_id = (
                    f"closure-eval-{bundle['panel']}-seed{bundle['seed']}-"
                    f"a{attempt_number + 1}-{uuid.uuid4().hex[:12]}"
                )
                log_root = (
                    Path(evaluation["state_root"])
                    / "operator-logs"
                    / f"{bundle['panel']}-seed{bundle['seed']}"
                    / attempt_id
                )
                try:
                    _run_bundle_operator(
                        evaluation,
                        training,
                        bundle_index=bundle_index,
                        attempt_id=attempt_id,
                        log_root=log_root,
                        timeout_seconds=min(
                            remaining,
                            DEFAULT_OPERATOR_TIMEOUT_SECONDS,
                        ),
                        dry_run=dry_run,
                    )
                except ComponentClosureEvaluationOperatorError:
                    if attempt_number != 0:
                        raise
                    time.sleep(min(30, max(1, int(deadline - time.monotonic()))))
                    continue
                break

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = [
            pool.submit(resource_worker, resource_id, indexes)
            for resource_id, indexes in sorted(groups.items())
        ]
        for future in futures:
            future.result()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build":
        runtime = yaml.safe_load(args.runtime.read_text(encoding="utf-8"))
        plan = build_evaluation_plan(
            load_training_plan(args.training_plan),
            load_a_reference_manifest(args.a_references),
            runtime,
            evaluation_source_revision=args.evaluation_source_revision,
            state_root=args.state_root,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="ascii") as handle:
            json.dump(plan, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
        return 0
    if args.command == "collect-costs":
        value = collect_training_costs(
            load_training_plan(args.training_plan),
            load_a_reference_manifest(args.a_references),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="ascii") as handle:
            json.dump(value, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
        return 0
    if args.command == "run-bundle-json":
        try:
            bundle = json.loads(args.bundle_json)
        except json.JSONDecodeError as error:
            raise ComponentClosureEvaluationOperatorError(
                "bundle JSON is invalid"
            ) from error
        if not isinstance(bundle, dict):
            raise ComponentClosureEvaluationOperatorError(
                "bundle JSON is not an object"
            )
        return run_resident_bundle(
            bundle,
            process_index=args.process_index,
            attempt_id=args.attempt_id,
        )
    plan = load_evaluation_plan(args.plan)
    if args.command in {"run-bundle-operator", "run-all-operators"}:
        training = load_training_plan(args.training_plan)
        if args.command == "run-bundle-operator":
            return _run_bundle_operator(
                plan,
                training,
                bundle_index=args.bundle_index,
                attempt_id=args.attempt_id,
                log_root=args.log_root,
                timeout_seconds=args.timeout_seconds,
                dry_run=args.dry_run,
            )
        return _run_all_operators(
            plan,
            training,
            maximum_wall_seconds=args.maximum_wall_seconds,
            dry_run=args.dry_run,
        )
    if args.command == "run-bundle":
        return run_resident_bundle(
            plan["bundles"][args.bundle_index],
            process_index=args.process_index,
            attempt_id=args.attempt_id,
        )
    if args.command == "collect":
        costs = json.loads(args.costs.read_text(encoding="ascii"))
        value = collect_score_grid(plan, costs)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="ascii") as handle:
            json.dump(value, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
