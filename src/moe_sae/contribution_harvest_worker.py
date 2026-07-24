"""Supervised production worker for Experiment-2 contribution harvesting.

The qualification worker and this worker deliberately meet at one boundary:
the exact lowered :class:`PreparedContributionExecutable`.  A long harvest is
admitted only after the just-lowered StableHLO, global v4-32 inventory, and
canonical performance launch manifest reproduce a passing qualification
receipt's identity.  The admitted callable is the callable passed directly to
``harvest_contribution_host``; this module never recompiles a second graph.

Bulk contribution tensors are uploaded through
:class:`ContributionShardWriter`.  Shared POSIX state contains only immutable
receipts, host indexes, supervised evidence, and the final manifest.  A later
phase adopts the preceding sealed phase's receipts and appends the frozen
source-range suffix without listing or copying GCS objects.

Importing this module is CPU-safe.  JAX, MaxText, and GCS clients are reached
only through injectable production dependencies after the child initializes
distributed JAX.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from moe_sae.artifacts import (
    ArtifactDataset,
    StagedDatasetProvenance,
    load_manifest,
    verify_staged_dataset_manifest,
)
from moe_sae.contribution_artifacts import (
    ContributionMetadata,
    canonical_json_bytes,
    canonical_sha256,
    validate_metadata,
    validate_recomposition_sidecar,
)
from moe_sae.contribution_config import (
    MAXTEXT_ROUTED_MOE_PATCHED_SHA256,
    ContributionPhase,
    Experiment2Config,
    load_contribution_config,
)
from moe_sae.contribution_harvest import (
    ContributionHarvestResult,
    harvest_contribution_host,
)
from moe_sae.contribution_maxtext import (
    PreparedContributionExecutable,
    build_contribution_maxtext_argv,
    effective_performance_runtime_manifest,
    exact_performance_identity,
    extract_recomposition_sidecar,
    hardware_manifest,
    observed_performance_launch_manifest,
)
from moe_sae.contribution_performance import (
    AcceleratorRunAuthorization,
    ExactPerformanceIdentity,
    authorize_accelerator_run,
    validate_qualification_receipt,
)
from moe_sae.contribution_qualification_worker import (
    AuthorizedContributionExecutable,
    authorize_extended_workload_after_lowering,
)
from moe_sae.contribution_qualification_worker import (
    production_dependencies as qualification_production_dependencies,
)
from moe_sae.contribution_ranges import (
    ContributionRangePlan,
    build_contribution_range_plan,
)
from moe_sae.contribution_storage import (
    ContributionAdoptionResult,
    ContributionShardWriter,
    adopt_contribution_phase,
    load_contribution_manifest,
    publish_contribution_manifest,
)
from moe_sae.contribution_supervisor import (
    DEFAULT_TERMINATION_GRACE_SECONDS,
    supervise_authorized_process,
    validate_execution_record,
    validate_supervision_result,
)
from moe_sae.harvest import GrainBatch, extract_grain_batch
from moe_sae.model import checkpoint_inventory_sha256
from moe_sae.object_store import (
    GCSObjectStore,
    ImmutableObjectStore,
    ShardCache,
    join_gcs_uri,
    object_identity_from_mapping,
)
from moe_sae.runtime import (
    HARVEST_MESHES,
    MeshCandidate,
    create_maxtext_runtime,
    environment_fingerprint,
    initialize_jax_distributed,
    maxtext_mesh_context,
)

CHILD_EVIDENCE_SCHEMA_VERSION = "gemma4-contribution-harvest-child-evidence-v2"
SUPERVISED_EVIDENCE_SCHEMA_VERSION = (
    "gemma4-contribution-harvest-supervised-evidence-v2"
)
PHASE_EVIDENCE_SCHEMA_VERSION = "gemma4-contribution-harvest-phase-evidence-v2"
EXPECTED_PROCESS_COUNT = 4
EXPECTED_GLOBAL_CHIPS = 16

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_SPLIT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


class ContributionHarvestWorkerError(RuntimeError):
    """Production admission, execution, or evidence failed closed."""


class HarvestCallable(Protocol):
    def __call__(
        self,
        batches: Iterable[Mapping[str, Any]],
        adapter_step: Callable[[Any, Any, Any, Any], Any],
        *,
        writer: ContributionShardWriter,
        source_v1_dataset: ArtifactDataset,
        range_plan: ContributionRangePlan,
        phase_total_tokens: int,
        process_count: int,
        process_index: int,
        recomposition_sidecar: Mapping[str, Any],
        sequence_length: int,
        max_pending_writes: int,
        batch_collective: Callable[[bool, bool], tuple[bool, bool]] | None,
        clock: Callable[[], float],
    ) -> ContributionHarvestResult: ...


@dataclass(frozen=True, slots=True)
class HarvestInvocation:
    """All explicit non-secret inputs for one already-supervised TPU child."""

    config_path: Path
    phase: str
    split: str
    checkpoint_path: str
    checkpoint_sha256: str
    grain_files: str
    staged_manifest: str
    source_v1_root: Path
    source_cache_root: Path
    source_cache_max_bytes: int
    control_root: Path
    prior_control_root: Path | None
    prior_phase_evidence_path: Path | None
    local_buffer_root: Path
    object_prefix: str
    recomposition_sidecar_path: Path
    qualification_receipt_path: Path
    repo_root: Path
    jax_cache_dir: Path
    maxtext_root: Path | None
    mesh: MeshCandidate
    per_device_sequence_batch: int
    controller_index: int
    maximum_runtime_seconds: int
    child_evidence_path: Path
    execution_record_path: Path


@dataclass(frozen=True, slots=True)
class HarvestWorkerDependencies:
    """Injectable accelerator, storage, and persistence boundaries."""

    initialize_distributed: Callable[[], None]
    process_context: Callable[[], tuple[Sequence[Any], int, int, int]]
    verify_maxtext: Callable[[Path | None], Mapping[str, str]]
    verify_checkpoint: Callable[[str], str]
    create_object_store: Callable[[], ImmutableObjectStore]
    verify_staged: Callable[
        [HarvestInvocation, Experiment2Config, ImmutableObjectStore],
        StagedDatasetProvenance,
    ]
    create_source_dataset: Callable[
        [HarvestInvocation, ImmutableObjectStore],
        ArtifactDataset,
    ]
    create_runtime: Callable[[Sequence[str], Sequence[str]], Any]
    close_runtime: Callable[[Any], None]
    mesh_context: Callable[[Any], AbstractContextManager[Any]]
    create_adapter: Callable[[Any, Path | None], Any]
    prepare: Callable[
        [Any, Any, GrainBatch, Callable[[], float]],
        PreparedContributionExecutable,
    ]
    effective_runtime: Callable[
        [Any, Experiment2Config, MeshCandidate, int],
        Mapping[str, Any],
    ]
    performance_launch: Callable[
        [
            Experiment2Config,
            str,
            MeshCandidate,
            int,
            Path,
            Mapping[str, Any],
            Mapping[str, str],
            Any,
        ],
        Mapping[str, Any],
    ]
    verify_sidecar: Callable[
        [Any, str, Mapping[str, Any]],
        Mapping[str, Any],
    ]
    environment: Callable[[Path], Mapping[str, Any]]
    create_writer: Callable[
        [
            HarvestInvocation,
            str,
            Mapping[str, Any],
            ImmutableObjectStore,
            int,
        ],
        ContributionShardWriter,
    ]
    adopt_phase: Callable[
        [ContributionShardWriter, Path, Mapping[str, Any]],
        ContributionAdoptionResult,
    ]
    harvest: HarvestCallable
    batch_collective: Callable[[bool, bool], tuple[bool, bool]]
    clock: Callable[[], float]


def _sha256(name: str, value: Any) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ContributionHarvestWorkerError(f"{name} must be a lowercase SHA-256")
    return value


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContributionHarvestWorkerError(f"{name} must be a positive integer")
    return value


def _bounded_index(name: str, value: Any) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value < EXPECTED_PROCESS_COUNT
    ):
        raise ContributionHarvestWorkerError(
            f"{name} must lie in [0,{EXPECTED_PROCESS_COUNT - 1}]"
        )
    return value


def _process_index(value: Any) -> int:
    return _bounded_index("process_index", value)


def _controller_index(value: Any) -> int:
    return _bounded_index("controller_index", value)


def _read_json(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContributionHarvestWorkerError(f"cannot read {name}: {path}") from error
    if not isinstance(value, dict):
        raise ContributionHarvestWorkerError(f"{name} must be a JSON object")
    return value


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise ContributionHarvestWorkerError(
                "immutable evidence write made no progress"
            )
        offset += written


def _atomic_create_or_match(path: Path, value: Mapping[str, Any]) -> None:
    """Create one immutable JSON object, accepting only an identical replay."""

    payload = canonical_json_bytes(dict(value))
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError:
        try:
            existing = path.read_bytes()
        except OSError as error:
            raise ContributionHarvestWorkerError(
                f"cannot read existing immutable evidence: {path}"
            ) from error
        if existing != payload:
            raise ContributionHarvestWorkerError(
                f"immutable evidence already exists with different bytes: {path}"
            )
        return
    try:
        _write_all(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _self_hash(
    value: Mapping[str, Any],
    *,
    hash_field: str,
) -> dict[str, Any]:
    unsigned = dict(value)
    unsigned.pop(hash_field, None)
    return {**unsigned, hash_field: canonical_sha256(unsigned)}


def _identity_from_receipt(
    receipt: Mapping[str, Any],
) -> ExactPerformanceIdentity:
    identity = receipt.get("identity")
    if not isinstance(identity, Mapping):
        raise ContributionHarvestWorkerError(
            "qualification receipt has no performance identity"
        )
    try:
        observed = ExactPerformanceIdentity(
            graph_sha256=str(identity["graph_sha256"]),
            hardware_sha256=str(identity["hardware_sha256"]),
            config_sha256=str(identity["config_sha256"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ContributionHarvestWorkerError(
            "qualification receipt identity is malformed"
        ) from error
    if receipt.get("identity_sha256") != observed.identity_sha256:
        raise ContributionHarvestWorkerError(
            "qualification receipt identity hash is inconsistent"
        )
    return observed


def _phase_tokens(phase: ContributionPhase, split: str) -> int:
    if split == "train":
        return phase.train_tokens
    if split == "eval":
        return phase.eval_tokens
    raise ContributionHarvestWorkerError("split must be 'train' or 'eval'")


def _phase_totals(config: Experiment2Config, split: str) -> tuple[int, ...]:
    return tuple(_phase_tokens(phase, split) for phase in config.phases)


def _source_split(config: Experiment2Config, split: str) -> str:
    if split == "train":
        return config.provenance.train_split
    if split == "eval":
        return config.provenance.eval_split
    raise ContributionHarvestWorkerError("split must be 'train' or 'eval'")


def _validate_source_identity(
    source: ArtifactDataset,
    *,
    invocation: HarvestInvocation,
    config: Experiment2Config,
    provenance: StagedDatasetProvenance,
) -> None:
    manifest = load_manifest(source.root)
    if canonical_json_bytes(manifest) != canonical_json_bytes(source.manifest):
        raise ContributionHarvestWorkerError(
            "source-v1 dataset differs from its sealed manifest"
        )
    metadata = manifest.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ContributionHarvestWorkerError("source-v1 metadata is missing")
    expected = {
        "checkpoint_sha256": invocation.checkpoint_sha256,
        "dataset_fingerprint": provenance.fingerprint_sha256,
        "split": invocation.split,
        "model_id": config.provenance.model_id,
        "target_layer_one_indexed": config.model.target_layer_one_indexed,
        "sequence_length": config.model.sequence_length,
        "maxtext_commit": config.provenance.maxtext_base_commit,
    }
    failures = {
        key: {"expected": wanted, "observed": metadata.get(key)}
        for key, wanted in expected.items()
        if metadata.get(key) != wanted
    }
    final_total = _phase_totals(config, invocation.split)[-1]
    if manifest.get("total_tokens") != final_total:
        failures["total_tokens"] = {
            "expected": final_total,
            "observed": manifest.get("total_tokens"),
        }
    if failures:
        raise ContributionHarvestWorkerError(
            "sealed source-v1 identity differs from Experiment 2: "
            + json.dumps(failures, sort_keys=True, separators=(",", ":"))
        )


def _metadata(
    *,
    invocation: HarvestInvocation,
    config: Experiment2Config,
    source: ArtifactDataset,
    staged: StagedDatasetProvenance,
    sidecar: Mapping[str, Any],
    environment_sha256: str,
) -> dict[str, Any]:
    return ContributionMetadata(
        checkpoint_sha256=invocation.checkpoint_sha256,
        maxtext_fork_commit=config.provenance.maxtext_fork_commit,
        dataset_fingerprint=staged.fingerprint_sha256,
        split=invocation.split,
        config_sha256=config.config_hash,
        environment_sha256=environment_sha256,
        source_v1_manifest_sha256=source.manifest_sha256,
        recomposition_sidecar_sha256=str(sidecar["sidecar_sha256"]),
    ).to_dict()


def _validate_sidecar_run_linkage(
    sidecar: Mapping[str, Any],
    *,
    config: Experiment2Config,
    environment_sha256: str,
    maxtext_provenance: Mapping[str, str],
) -> None:
    fingerprints = sidecar.get("source_fingerprints")
    if not isinstance(fingerprints, Mapping):
        raise ContributionHarvestWorkerError(
            "recomposition sidecar lacks source fingerprints"
        )
    expected = {
        "experiment_config": config.config_hash,
        "environment": environment_sha256,
        "maxtext_routed_moe": maxtext_provenance["routed_moe_sha256"],
    }
    failures = {
        name: {"expected": wanted, "observed": fingerprints.get(name)}
        for name, wanted in expected.items()
        if fingerprints.get(name) != wanted
    }
    if failures:
        raise ContributionHarvestWorkerError(
            "recomposition sidecar is not bound to this qualified run: "
            + json.dumps(failures, sort_keys=True, separators=(",", ":"))
        )


def _adoption_manifest(
    value: ContributionAdoptionResult | None,
) -> dict[str, Any] | None:
    return None if value is None else asdict(value)


def _load_launch_receipt(
    invocation: HarvestInvocation,
    config: Experiment2Config,
) -> tuple[dict[str, Any], ExactPerformanceIdentity, dict[str, Any]]:
    """Validate the receipt and the supervisor's pre-launch authorization."""

    try:
        receipt = validate_qualification_receipt(
            _read_json(
                invocation.qualification_receipt_path,
                name="qualification receipt",
            ),
            minimum_useful_mfu=config.runtime.accelerator_min_useful_mfu,
            qualification_max_seconds=(
                config.runtime.accelerator_qualification_max_seconds
            ),
        )
    except ValueError as error:
        raise ContributionHarvestWorkerError(
            "qualification receipt failed pre-launch validation"
        ) from error
    identity = _identity_from_receipt(receipt)
    try:
        execution = validate_execution_record(
            _read_json(
                invocation.execution_record_path,
                name="supervisor execution record",
            )
        )
    except (ValueError, RuntimeError) as error:
        raise ContributionHarvestWorkerError(
            "harvest child lacks a valid supervisor execution record"
        ) from error
    authorization = execution["authorization"]
    if (
        authorization["purpose"] != "workload"
        or authorization["basis"] != "qualified_exact_identity"
        or authorization["maximum_runtime_seconds"]
        != invocation.maximum_runtime_seconds
        or authorization["identity_sha256"] != identity.identity_sha256
        or authorization["qualification_receipt_sha256"] != receipt["receipt_sha256"]
    ):
        raise ContributionHarvestWorkerError(
            "supervisor authorization differs from the qualification receipt "
            "or requested workload deadline"
        )
    return receipt, identity, execution


def _validate_prior_phase(
    invocation: HarvestInvocation,
    *,
    config: Experiment2Config,
    phase: ContributionPhase,
    range_plan: ContributionRangePlan,
    metadata: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Bind an extension to the immediate sealed predecessor, if any."""

    if phase.extends_phase is None:
        if (
            invocation.prior_control_root is not None
            or invocation.prior_phase_evidence_path is not None
        ):
            raise ContributionHarvestWorkerError(
                "Phase 1 must not receive a prior phase root or evidence"
            )
        return None, None
    if (
        invocation.prior_control_root is None
        or invocation.prior_phase_evidence_path is None
    ):
        raise ContributionHarvestWorkerError(
            f"{phase.name} requires its immediate predecessor root and evidence"
        )
    prior_phase = config.phase(phase.extends_phase)
    prior_manifest = load_contribution_manifest(invocation.prior_control_root)
    prior_evidence = validate_phase_evidence(
        _read_json(
            invocation.prior_phase_evidence_path,
            name="prior phase evidence",
        )
    )
    expected_tokens = _phase_tokens(prior_phase, invocation.split)
    failures = (
        prior_evidence["status"] != "sealed",
        prior_evidence["phase"] != prior_phase.name,
        prior_evidence["phase_ordinal"] != prior_phase.ordinal,
        prior_evidence["split"] != invocation.split,
        prior_evidence["config_sha256"] != config.config_hash,
        prior_evidence["source_v1_manifest_sha256"]
        != range_plan.source_manifest_sha256,
        prior_evidence["range_plan_sha256"] != range_plan.plan_sha256,
        prior_evidence["phase_total_tokens"] != expected_tokens,
        prior_evidence["contribution_manifest_sha256"]
        != prior_manifest["manifest_sha256"],
        prior_manifest["total_tokens"] != expected_tokens,
        canonical_json_bytes(prior_manifest["metadata"])
        != canonical_json_bytes(metadata),
    )
    if any(failures):
        raise ContributionHarvestWorkerError(
            "prior phase is not the exact sealed immediate predecessor"
        )
    return prior_manifest, prior_evidence


def _authorize_prepared(
    prepared: PreparedContributionExecutable,
    *,
    hardware: Mapping[str, Any],
    performance_launch: Mapping[str, Any],
    receipt: Mapping[str, Any],
    invocation: HarvestInvocation,
    config: Experiment2Config,
) -> AuthorizedContributionExecutable:
    """Revalidate exact identity before returning the only harvest callable."""

    if (
        invocation.maximum_runtime_seconds
        > config.runtime.accelerator_ungated_max_seconds
    ):
        return authorize_extended_workload_after_lowering(
            prepared,
            hardware=hardware,
            launch_config=performance_launch,
            qualification_receipt=receipt,
            maximum_runtime_seconds=invocation.maximum_runtime_seconds,
            config=config,
        )
    observed = exact_performance_identity(
        graph=prepared.graph,
        hardware=hardware,
        launch_config=performance_launch,
    )
    authorization = authorize_accelerator_run(
        identity=observed,
        maximum_runtime_seconds=invocation.maximum_runtime_seconds,
        purpose="workload",
        qualification_receipt=receipt,
        minimum_useful_mfu=config.runtime.accelerator_min_useful_mfu,
        ungated_max_seconds=config.runtime.accelerator_ungated_max_seconds,
        qualification_max_seconds=(
            config.runtime.accelerator_qualification_max_seconds
        ),
    )
    return AuthorizedContributionExecutable(
        executable=prepared.executable,
        observed_executable_identity=observed,
        authorization=authorization,
    )


def _child_evidence(
    *,
    invocation: HarvestInvocation,
    process_index: int,
    config: Experiment2Config,
    phase: ContributionPhase,
    execution: Mapping[str, Any],
    receipt: Mapping[str, Any],
    prepared: PreparedContributionExecutable,
    authorized: AuthorizedContributionExecutable,
    hardware: Mapping[str, Any],
    performance_launch: Mapping[str, Any],
    maxtext_provenance: Mapping[str, str],
    effective_runtime: Mapping[str, Any],
    metadata: Mapping[str, Any],
    sidecar: Mapping[str, Any],
    range_plan: ContributionRangePlan,
    adoption: ContributionAdoptionResult | None,
    parent_evidence: Mapping[str, Any] | None,
    harvest: ContributionHarvestResult,
    host_index: Mapping[str, Any],
) -> dict[str, Any]:
    unsigned = {
        "schema_version": CHILD_EVIDENCE_SCHEMA_VERSION,
        "status": "complete",
        "protocol_id": config.protocol_id,
        "config_sha256": config.config_hash,
        "phase": phase.name,
        "phase_ordinal": phase.ordinal,
        "split": invocation.split,
        "phase_total_tokens": _phase_tokens(phase, invocation.split),
        "process_count": EXPECTED_PROCESS_COUNT,
        "controller_index": invocation.controller_index,
        "process_index": process_index,
        "host_id": harvest.host_id,
        "checkpoint_sha256": invocation.checkpoint_sha256,
        "source_v1_manifest_sha256": range_plan.source_manifest_sha256,
        "range_plan_sha256": range_plan.plan_sha256,
        "recomposition_sidecar_sha256": sidecar["sidecar_sha256"],
        "metadata": dict(metadata),
        "metadata_sha256": canonical_sha256(metadata),
        "prepared_graph": dict(prepared.graph),
        "observed_executable_identity": (
            authorized.observed_executable_identity.to_dict()
        ),
        "observed_identity_sha256": (
            authorized.observed_executable_identity.identity_sha256
        ),
        "performance_launch": dict(performance_launch),
        "hardware_sha256": canonical_sha256(hardware),
        "qualification_receipt_sha256": receipt["receipt_sha256"],
        "qualification_useful_prefix_mfu": receipt["performance"]["useful_prefix_mfu"],
        "authorization": asdict(authorized.authorization),
        "supervisor_execution_sha256": execution["execution_sha256"],
        "maximum_runtime_seconds": invocation.maximum_runtime_seconds,
        "maxtext_provenance": dict(maxtext_provenance),
        "effective_runtime": dict(effective_runtime),
        "adopted_parent": _adoption_manifest(adoption),
        "parent_phase_evidence_sha256": (
            None
            if parent_evidence is None
            else parent_evidence["phase_evidence_sha256"]
        ),
        "harvest": harvest.to_dict(),
        "host_index_sha256": host_index["index_sha256"],
    }
    return _self_hash(unsigned, hash_field="child_evidence_sha256")


def validate_child_evidence(value: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "schema_version",
        "status",
        "protocol_id",
        "config_sha256",
        "phase",
        "phase_ordinal",
        "split",
        "phase_total_tokens",
        "process_count",
        "controller_index",
        "process_index",
        "host_id",
        "checkpoint_sha256",
        "source_v1_manifest_sha256",
        "range_plan_sha256",
        "recomposition_sidecar_sha256",
        "metadata",
        "metadata_sha256",
        "prepared_graph",
        "observed_executable_identity",
        "observed_identity_sha256",
        "performance_launch",
        "hardware_sha256",
        "qualification_receipt_sha256",
        "qualification_useful_prefix_mfu",
        "authorization",
        "supervisor_execution_sha256",
        "maximum_runtime_seconds",
        "maxtext_provenance",
        "effective_runtime",
        "adopted_parent",
        "parent_phase_evidence_sha256",
        "harvest",
        "host_index_sha256",
        "child_evidence_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ContributionHarvestWorkerError(
            "harvest child evidence fields are not exact"
        )
    selected = dict(value)
    claimed = _sha256(
        "child_evidence_sha256",
        selected.pop("child_evidence_sha256"),
    )
    if (
        selected["schema_version"] != CHILD_EVIDENCE_SCHEMA_VERSION
        or selected["status"] != "complete"
        or canonical_sha256(selected) != claimed
    ):
        raise ContributionHarvestWorkerError(
            "harvest child evidence is incompatible or has an invalid self-hash"
        )
    for name in (
        "config_sha256",
        "checkpoint_sha256",
        "source_v1_manifest_sha256",
        "range_plan_sha256",
        "recomposition_sidecar_sha256",
        "metadata_sha256",
        "observed_identity_sha256",
        "hardware_sha256",
        "qualification_receipt_sha256",
        "supervisor_execution_sha256",
        "host_index_sha256",
    ):
        _sha256(name, selected[name])
    _positive_int("phase_ordinal", selected["phase_ordinal"])
    _positive_int("phase_total_tokens", selected["phase_total_tokens"])
    if selected["process_count"] != EXPECTED_PROCESS_COUNT:
        raise ContributionHarvestWorkerError(
            "harvest evidence is not from a four-controller v4-32"
        )
    _controller_index(selected["controller_index"])
    _process_index(selected["process_index"])
    if (
        not isinstance(selected["split"], str)
        or _SAFE_SPLIT_RE.fullmatch(selected["split"]) is None
        or not isinstance(selected["phase"], str)
        or not selected["phase"]
        or not isinstance(selected["host_id"], str)
        or not selected["host_id"]
    ):
        raise ContributionHarvestWorkerError(
            "harvest child phase/split/host identity is invalid"
        )
    metadata = validate_metadata(selected["metadata"])
    if (
        canonical_sha256(metadata) != selected["metadata_sha256"]
        or metadata["checkpoint_sha256"] != selected["checkpoint_sha256"]
        or metadata["config_sha256"] != selected["config_sha256"]
        or metadata["split"] != selected["split"]
        or metadata["source_v1_manifest_sha256"]
        != selected["source_v1_manifest_sha256"]
        or metadata["recomposition_sidecar_sha256"]
        != selected["recomposition_sidecar_sha256"]
    ):
        raise ContributionHarvestWorkerError(
            "harvest child metadata identity is inconsistent"
        )
    identity = selected["observed_executable_identity"]
    if not isinstance(identity, Mapping):
        raise ContributionHarvestWorkerError(
            "harvest child observed identity is missing"
        )
    parsed_identity = ExactPerformanceIdentity(
        graph_sha256=str(identity.get("graph_sha256")),
        hardware_sha256=str(identity.get("hardware_sha256")),
        config_sha256=str(identity.get("config_sha256")),
    )
    if (
        parsed_identity.identity_sha256 != selected["observed_identity_sha256"]
        or parsed_identity.graph_sha256 != canonical_sha256(selected["prepared_graph"])
        or parsed_identity.hardware_sha256 != selected["hardware_sha256"]
        or parsed_identity.config_sha256
        != canonical_sha256(selected["performance_launch"])
    ):
        raise ContributionHarvestWorkerError(
            "harvest child observed identity components are inconsistent"
        )
    authorization = selected["authorization"]
    if (
        not isinstance(authorization, Mapping)
        or authorization.get("purpose") != "workload"
        or authorization.get("basis") != "qualified_exact_identity"
        or authorization.get("identity_sha256") != parsed_identity.identity_sha256
        or authorization.get("qualification_receipt_sha256")
        != selected["qualification_receipt_sha256"]
        or authorization.get("maximum_runtime_seconds")
        != selected["maximum_runtime_seconds"]
    ):
        raise ContributionHarvestWorkerError(
            "harvest child authorization is inconsistent"
        )
    harvest = selected["harvest"]
    if (
        not isinstance(harvest, Mapping)
        or harvest.get("range_plan_sha256") != selected["range_plan_sha256"]
        or harvest.get("source_v1_manifest_sha256")
        != selected["source_v1_manifest_sha256"]
        or harvest.get("phase_total_tokens") != selected["phase_total_tokens"]
        or harvest.get("process_count") != EXPECTED_PROCESS_COUNT
        or harvest.get("process_index") != selected["process_index"]
        or harvest.get("host_id") != selected["host_id"]
        or harvest.get("split") != selected["split"]
        or harvest.get("host_tokens")
        != selected["phase_total_tokens"] // EXPECTED_PROCESS_COUNT
        or harvest.get("new_tokens", -1) + harvest.get("resumed_tokens", -1)
        != harvest.get("host_tokens")
        or harvest.get("max_writer_backlog_batches", 2) >= 2
    ):
        raise ContributionHarvestWorkerError(
            "harvest result is not bound to the child evidence"
        )
    adoption = selected["adopted_parent"]
    parent_evidence = selected["parent_phase_evidence_sha256"]
    if adoption is None:
        if parent_evidence is not None:
            raise ContributionHarvestWorkerError(
                "harvest child has parent evidence without adopted receipts"
            )
    else:
        if (
            not isinstance(adoption, Mapping)
            or set(adoption)
            != {
                "prior_manifest_sha256",
                "host_id",
                "adopted_shards",
                "adopted_tokens",
            }
            or adoption["host_id"] != selected["host_id"]
            or not isinstance(adoption["adopted_shards"], int)
            or adoption["adopted_shards"] <= 0
            or not isinstance(adoption["adopted_tokens"], int)
            or adoption["adopted_tokens"] <= 0
            or parent_evidence is None
        ):
            raise ContributionHarvestWorkerError(
                "harvest child adopted-parent evidence is malformed"
            )
        _sha256("adopted prior manifest", adoption["prior_manifest_sha256"])
        _sha256("parent phase evidence", parent_evidence)
    selected["child_evidence_sha256"] = claimed
    return selected


def run_harvest_child(
    invocation: HarvestInvocation,
    *,
    dependencies: HarvestWorkerDependencies | None = None,
) -> dict[str, Any]:
    """Run one host, revalidating its receipt before the first harvest step."""

    deps = dependencies or production_dependencies()
    config = load_contribution_config(invocation.config_path)
    phase = config.phase(invocation.phase)
    phase_total = _phase_tokens(phase, invocation.split)
    _controller_index(invocation.controller_index)
    _positive_int(
        "maximum_runtime_seconds",
        invocation.maximum_runtime_seconds,
    )
    receipt, receipt_identity, execution = _load_launch_receipt(
        invocation,
        config,
    )

    # This is the first dependency allowed to query an accelerator backend.
    deps.initialize_distributed()
    devices, process_count, observed_index, global_chips = deps.process_context()
    if process_count != EXPECTED_PROCESS_COUNT or global_chips != EXPECTED_GLOBAL_CHIPS:
        raise ContributionHarvestWorkerError(
            "distributed JAX identity differs from the required v4-32 host: "
            f"count={process_count}, index={observed_index}, chips={global_chips}"
        )
    process_index = _process_index(observed_index)
    maxtext_provenance = dict(deps.verify_maxtext(invocation.maxtext_root))
    if set(maxtext_provenance) != {
        "mode",
        "commit",
        "routed_moe_sha256",
    }:
        raise ContributionHarvestWorkerError(
            "MaxText verifier returned incomplete provenance"
        )
    if (
        maxtext_provenance["commit"] != config.provenance.maxtext_fork_commit
        or maxtext_provenance["routed_moe_sha256"] != MAXTEXT_ROUTED_MOE_PATCHED_SHA256
    ):
        raise ContributionHarvestWorkerError(
            "installed MaxText differs from the published Experiment-2 fork"
        )
    config.validate_authoritative_tpu_capture(
        phase.name,
        observed_maxtext_fork_commit=maxtext_provenance["commit"],
    )
    checkpoint = deps.verify_checkpoint(invocation.checkpoint_path)
    if checkpoint != invocation.checkpoint_sha256:
        raise ContributionHarvestWorkerError(
            "checkpoint inventory differs from the requested immutable identity"
        )

    object_store = deps.create_object_store()
    staged = deps.verify_staged(invocation, config, object_store)
    source = deps.create_source_dataset(invocation, object_store)
    _validate_source_identity(
        source,
        invocation=invocation,
        config=config,
        provenance=staged,
    )
    range_plan = build_contribution_range_plan(
        source.manifest,
        _phase_totals(config, invocation.split),
        process_count=process_count,
        max_shard_tokens=config.runtime.max_shard_tokens,
    )
    try:
        sidecar = validate_recomposition_sidecar(
            _read_json(
                invocation.recomposition_sidecar_path,
                name="recomposition sidecar",
            )
        )
    except ValueError as error:
        raise ContributionHarvestWorkerError(
            "recomposition sidecar is invalid"
        ) from error
    environment = dict(deps.environment(invocation.repo_root))
    environment_sha256 = _sha256(
        "environment.sha256",
        environment.get("sha256"),
    )
    _validate_sidecar_run_linkage(
        sidecar,
        config=config,
        environment_sha256=environment_sha256,
        maxtext_provenance=maxtext_provenance,
    )
    metadata = _metadata(
        invocation=invocation,
        config=config,
        source=source,
        staged=staged,
        sidecar=sidecar,
        environment_sha256=environment_sha256,
    )
    _prior_manifest, parent_evidence = _validate_prior_phase(
        invocation,
        config=config,
        phase=phase,
        range_plan=range_plan,
        metadata=metadata,
    )

    argv = build_contribution_maxtext_argv(
        checkpoint_path=invocation.checkpoint_path,
        # MaxText creates both iterators.  Supplying the selected immutable
        # inventory to both avoids an unrelated second input dependency.
        grain_train_files=invocation.grain_files,
        grain_eval_files=invocation.grain_files,
        mesh=invocation.mesh,
        per_device_sequence_batch=invocation.per_device_sequence_batch,
        jax_cache_dir=invocation.jax_cache_dir,
        synthetic=False,
    )
    runtime = deps.create_runtime(argv, staged.files)
    body_error: BaseException | None = None
    try:
        adapter = deps.create_adapter(runtime.model, invocation.maxtext_root)
        verified_sidecar = dict(
            deps.verify_sidecar(
                adapter,
                checkpoint,
                sidecar,
            )
        )
        if canonical_json_bytes(verified_sidecar) != canonical_json_bytes(sidecar):
            raise ContributionHarvestWorkerError(
                "runtime model produces a different recomposition sidecar"
            )
        iterator_value = (
            runtime.train_iterator
            if invocation.split == "train"
            else runtime.eval_iterator
        )
        if iterator_value is None:
            raise ContributionHarvestWorkerError(
                f"MaxText did not construct the {invocation.split} iterator"
            )
        iterator = iter(iterator_value)
        try:
            first_raw = next(iterator)
        except StopIteration as error:
            raise ContributionHarvestWorkerError(
                "staged MaxText iterator is empty"
            ) from error
        first_batch = extract_grain_batch(
            first_raw,
            expected_sequence_length=config.model.sequence_length,
        )
        effective_runtime = dict(
            deps.effective_runtime(
                runtime,
                config,
                invocation.mesh,
                invocation.per_device_sequence_batch,
            )
        )
        hardware = hardware_manifest(
            devices=devices,
            process_count=process_count,
            mesh=invocation.mesh,
        )
        performance_launch = dict(
            deps.performance_launch(
                config,
                checkpoint,
                invocation.mesh,
                invocation.per_device_sequence_batch,
                invocation.jax_cache_dir,
                environment,
                maxtext_provenance,
                runtime,
            )
        )

        with deps.mesh_context(runtime):
            prepared = deps.prepare(
                adapter,
                runtime.mesh,
                first_batch,
                deps.clock,
            )
            authorized = _authorize_prepared(
                prepared,
                hardware=hardware,
                performance_launch=performance_launch,
                receipt=receipt,
                invocation=invocation,
                config=config,
            )
            if (
                authorized.observed_executable_identity != receipt_identity
                or asdict(authorized.authorization) != execution["authorization"]
            ):
                raise ContributionHarvestWorkerError(
                    "post-lowering authorization differs from the supervisor "
                    "or qualification receipt"
                )

            # No ContributionShardWriter or harvest side effect exists before
            # the exact observed identity has passed the line above.
            assignments = range_plan.for_process(
                phase_total,
                process_index,
            )
            if not assignments:
                raise ContributionHarvestWorkerError(
                    "frozen range plan assigned no work to this host"
                )
            writer = deps.create_writer(
                invocation,
                assignments[0].host_id,
                metadata,
                object_store,
                config.runtime.max_shard_tokens,
            )
            adoption: ContributionAdoptionResult | None = None
            if invocation.prior_control_root is not None:
                adoption = deps.adopt_phase(
                    writer,
                    invocation.prior_control_root,
                    sidecar,
                )
            result = deps.harvest(
                itertools.chain((first_raw,), iterator),
                authorized,
                writer=writer,
                source_v1_dataset=source,
                range_plan=range_plan,
                phase_total_tokens=phase_total,
                process_count=process_count,
                process_index=process_index,
                recomposition_sidecar=sidecar,
                sequence_length=config.model.sequence_length,
                max_pending_writes=1,
                batch_collective=deps.batch_collective,
                clock=deps.clock,
            )
            host_index = writer.finalize_host()
        evidence = _child_evidence(
            invocation=invocation,
            process_index=process_index,
            config=config,
            phase=phase,
            execution=execution,
            receipt=receipt,
            prepared=prepared,
            authorized=authorized,
            hardware=hardware,
            performance_launch=performance_launch,
            maxtext_provenance=maxtext_provenance,
            effective_runtime=effective_runtime,
            metadata=metadata,
            sidecar=sidecar,
            range_plan=range_plan,
            adoption=adoption,
            parent_evidence=parent_evidence,
            harvest=result,
            host_index=host_index,
        )
        validate_child_evidence(evidence)
        _atomic_create_or_match(invocation.child_evidence_path, evidence)
        return evidence
    except BaseException as error:
        body_error = error
        raise
    finally:
        try:
            deps.close_runtime(runtime)
        except BaseException as error:
            if body_error is not None:
                try:
                    body_error.add_note(
                        "suppressed MaxText runtime cleanup failure: "
                        f"{type(error).__name__}: {error}"
                    )
                except AttributeError:  # pragma: no cover
                    pass
            else:
                raise


def _supervised_evidence(
    *,
    invocation: HarvestInvocation,
    authorization: AcceleratorRunAuthorization,
    receipt: Mapping[str, Any],
    execution: Mapping[str, Any],
    result: Mapping[str, Any],
    child: Mapping[str, Any] | None,
) -> dict[str, Any]:
    completed = result["status"] == "completed" and child is not None
    checked_child = None if child is None else validate_child_evidence(child)
    if checked_child is not None and (
        checked_child["controller_index"] != invocation.controller_index
        or checked_child["supervisor_execution_sha256"] != execution["execution_sha256"]
        or checked_child["qualification_receipt_sha256"] != receipt["receipt_sha256"]
        or checked_child["authorization"] != asdict(authorization)
    ):
        raise ContributionHarvestWorkerError(
            "child evidence does not belong to this supervised launch"
        )
    unsigned = {
        "schema_version": SUPERVISED_EVIDENCE_SCHEMA_VERSION,
        "status": "complete" if completed else "failed",
        "controller_index": invocation.controller_index,
        "process_index": (
            None if checked_child is None else checked_child["process_index"]
        ),
        "qualification_receipt_sha256": receipt["receipt_sha256"],
        "authorization": asdict(authorization),
        "supervisor_execution_sha256": execution["execution_sha256"],
        "supervisor_result_sha256": result["result_sha256"],
        "supervisor_status": result["status"],
        "supervisor_elapsed_seconds": result["elapsed_seconds"],
        "child_evidence_sha256": (
            None if checked_child is None else checked_child["child_evidence_sha256"]
        ),
        "child_evidence": checked_child,
    }
    return _self_hash(unsigned, hash_field="supervised_evidence_sha256")


def validate_supervised_evidence(value: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "schema_version",
        "status",
        "controller_index",
        "process_index",
        "qualification_receipt_sha256",
        "authorization",
        "supervisor_execution_sha256",
        "supervisor_result_sha256",
        "supervisor_status",
        "supervisor_elapsed_seconds",
        "child_evidence_sha256",
        "child_evidence",
        "supervised_evidence_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ContributionHarvestWorkerError(
            "supervised harvest evidence fields are not exact"
        )
    selected = dict(value)
    claimed = _sha256(
        "supervised_evidence_sha256",
        selected.pop("supervised_evidence_sha256"),
    )
    if (
        selected["schema_version"] != SUPERVISED_EVIDENCE_SCHEMA_VERSION
        or selected["status"] not in {"complete", "failed"}
        or canonical_sha256(selected) != claimed
    ):
        raise ContributionHarvestWorkerError(
            "supervised harvest evidence is incompatible or has a bad self-hash"
        )
    _controller_index(selected["controller_index"])
    for name in (
        "qualification_receipt_sha256",
        "supervisor_execution_sha256",
        "supervisor_result_sha256",
    ):
        _sha256(name, selected[name])
    child = selected["child_evidence"]
    child_hash = selected["child_evidence_sha256"]
    completed = (
        selected["status"] == "complete"
        and selected["supervisor_status"] == "completed"
        and selected["process_index"] is not None
        and child is not None
        and child_hash is not None
    )
    failed = (
        selected["status"] == "failed"
        and selected["supervisor_status"] != "completed"
        and selected["process_index"] is None
        and child is None
        and child_hash is None
    )
    if not (completed or failed):
        raise ContributionHarvestWorkerError(
            "supervised harvest terminal fields are inconsistent"
        )
    if completed:
        _process_index(selected["process_index"])
        checked = validate_child_evidence(child)
        if (
            checked["child_evidence_sha256"] != child_hash
            or checked["controller_index"] != selected["controller_index"]
            or checked["process_index"] != selected["process_index"]
            or checked["qualification_receipt_sha256"]
            != selected["qualification_receipt_sha256"]
            or checked["supervisor_execution_sha256"]
            != selected["supervisor_execution_sha256"]
            or checked["authorization"] != selected["authorization"]
        ):
            raise ContributionHarvestWorkerError(
                "supervised wrapper and child evidence differ"
            )
        selected["child_evidence"] = checked
    selected["supervised_evidence_sha256"] = claimed
    return selected


def _child_command(invocation: HarvestInvocation) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "moe_sae.contribution_harvest_worker",
        "_run-host",
        "--config",
        str(invocation.config_path),
        "--phase",
        invocation.phase,
        "--split",
        invocation.split,
        "--checkpoint",
        invocation.checkpoint_path,
        "--checkpoint-sha256",
        invocation.checkpoint_sha256,
        "--grain-files",
        invocation.grain_files,
        "--staged-manifest",
        invocation.staged_manifest,
        "--source-v1-root",
        str(invocation.source_v1_root),
        "--source-cache-root",
        str(invocation.source_cache_root),
        "--source-cache-max-bytes",
        str(invocation.source_cache_max_bytes),
        "--control-root",
        str(invocation.control_root),
        "--local-buffer-root",
        str(invocation.local_buffer_root),
        "--object-prefix",
        invocation.object_prefix,
        "--recomposition-sidecar",
        str(invocation.recomposition_sidecar_path),
        "--qualification-receipt",
        str(invocation.qualification_receipt_path),
        "--repo-root",
        str(invocation.repo_root),
        "--jax-cache-dir",
        str(invocation.jax_cache_dir),
        "--mesh",
        invocation.mesh.name,
        "--per-device-sequence-batch",
        str(invocation.per_device_sequence_batch),
        "--controller-index",
        str(invocation.controller_index),
        "--maximum-runtime-seconds",
        str(invocation.maximum_runtime_seconds),
        "--child-evidence",
        str(invocation.child_evidence_path),
        "--execution-record",
        str(invocation.execution_record_path),
    ]
    if invocation.prior_control_root is not None:
        command.extend(("--prior-control-root", str(invocation.prior_control_root)))
    if invocation.prior_phase_evidence_path is not None:
        command.extend(
            (
                "--prior-phase-evidence",
                str(invocation.prior_phase_evidence_path),
            )
        )
    if invocation.maxtext_root is not None:
        command.extend(("--maxtext-root", str(invocation.maxtext_root)))
    return command


def launch_harvest_host(
    invocation: HarvestInvocation,
    *,
    log_path: Path,
    result_path: Path,
    supervised_evidence_path: Path,
    termination_grace_seconds: float = DEFAULT_TERMINATION_GRACE_SECONDS,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Prevalidate a receipt, supervise one child, and seal terminal evidence."""

    config = load_contribution_config(invocation.config_path)
    receipt = validate_qualification_receipt(
        _read_json(
            invocation.qualification_receipt_path,
            name="qualification receipt",
        ),
        minimum_useful_mfu=config.runtime.accelerator_min_useful_mfu,
        qualification_max_seconds=(
            config.runtime.accelerator_qualification_max_seconds
        ),
    )
    identity = _identity_from_receipt(receipt)
    authorization = authorize_accelerator_run(
        identity=identity,
        maximum_runtime_seconds=invocation.maximum_runtime_seconds,
        purpose="workload",
        qualification_receipt=receipt,
        minimum_useful_mfu=config.runtime.accelerator_min_useful_mfu,
        ungated_max_seconds=config.runtime.accelerator_ungated_max_seconds,
        qualification_max_seconds=(
            config.runtime.accelerator_qualification_max_seconds
        ),
    )
    result = supervise_authorized_process(
        authorization=authorization,
        command=_child_command(invocation),
        environment=dict(os.environ if environment is None else environment),
        working_directory=invocation.repo_root,
        log_path=log_path,
        execution_record_path=invocation.execution_record_path,
        result_path=result_path,
        termination_grace_seconds=termination_grace_seconds,
        qualification_receipt=receipt,
    )
    execution = validate_execution_record(
        _read_json(
            invocation.execution_record_path,
            name="supervisor execution record",
        )
    )
    terminal = validate_supervision_result(result)
    child: dict[str, Any] | None = None
    if terminal["status"] == "completed":
        child = _read_json(
            invocation.child_evidence_path,
            name="harvest child evidence",
        )
    evidence = _supervised_evidence(
        invocation=invocation,
        authorization=authorization,
        receipt=receipt,
        execution=execution,
        result=terminal,
        child=child,
    )
    validate_supervised_evidence(evidence)
    _atomic_create_or_match(supervised_evidence_path, evidence)
    return evidence


def _phase_evidence(
    *,
    config: Experiment2Config,
    phase: ContributionPhase,
    split: str,
    manifest: Mapping[str, Any],
    hosts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    children = [host["child_evidence"] for host in hosts]
    first = children[0]
    parent_manifests = {
        (
            None
            if child["adopted_parent"] is None
            else child["adopted_parent"]["prior_manifest_sha256"]
        )
        for child in children
    }
    parent_evidence = {child["parent_phase_evidence_sha256"] for child in children}
    if len(parent_manifests) != 1 or len(parent_evidence) != 1:
        raise ContributionHarvestWorkerError(
            "hosts disagree on their adopted predecessor"
        )
    unsigned = {
        "schema_version": PHASE_EVIDENCE_SCHEMA_VERSION,
        "status": "sealed",
        "protocol_id": config.protocol_id,
        "config_sha256": config.config_hash,
        "phase": phase.name,
        "phase_ordinal": phase.ordinal,
        "split": split,
        "phase_total_tokens": _phase_tokens(phase, split),
        "process_count": EXPECTED_PROCESS_COUNT,
        "source_v1_manifest_sha256": first["source_v1_manifest_sha256"],
        "range_plan_sha256": first["range_plan_sha256"],
        "metadata_sha256": manifest["metadata_sha256"],
        "recomposition_sidecar_sha256": first["recomposition_sidecar_sha256"],
        "adopted_parent_manifest_sha256": next(iter(parent_manifests)),
        "parent_phase_evidence_sha256": next(iter(parent_evidence)),
        "contribution_manifest_sha256": manifest["manifest_sha256"],
        "qualification_receipt_sha256": first["qualification_receipt_sha256"],
        "observed_identity_sha256": first["observed_identity_sha256"],
        "controller_index_by_process": [host["controller_index"] for host in hosts],
        "host_evidence_sha256_by_process": [
            host["supervised_evidence_sha256"] for host in hosts
        ],
    }
    return _self_hash(unsigned, hash_field="phase_evidence_sha256")


def validate_phase_evidence(value: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "schema_version",
        "status",
        "protocol_id",
        "config_sha256",
        "phase",
        "phase_ordinal",
        "split",
        "phase_total_tokens",
        "process_count",
        "source_v1_manifest_sha256",
        "range_plan_sha256",
        "metadata_sha256",
        "recomposition_sidecar_sha256",
        "adopted_parent_manifest_sha256",
        "parent_phase_evidence_sha256",
        "contribution_manifest_sha256",
        "qualification_receipt_sha256",
        "observed_identity_sha256",
        "controller_index_by_process",
        "host_evidence_sha256_by_process",
        "phase_evidence_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ContributionHarvestWorkerError(
            "contribution phase evidence fields are not exact"
        )
    selected = dict(value)
    claimed = _sha256(
        "phase_evidence_sha256",
        selected.pop("phase_evidence_sha256"),
    )
    if (
        selected["schema_version"] != PHASE_EVIDENCE_SCHEMA_VERSION
        or selected["status"] != "sealed"
        or canonical_sha256(selected) != claimed
    ):
        raise ContributionHarvestWorkerError(
            "contribution phase evidence is incompatible or has a bad self-hash"
        )
    for name in (
        "config_sha256",
        "source_v1_manifest_sha256",
        "range_plan_sha256",
        "metadata_sha256",
        "recomposition_sidecar_sha256",
        "contribution_manifest_sha256",
        "qualification_receipt_sha256",
        "observed_identity_sha256",
    ):
        _sha256(name, selected[name])
    for name in (
        "adopted_parent_manifest_sha256",
        "parent_phase_evidence_sha256",
    ):
        if selected[name] is not None:
            _sha256(name, selected[name])
    if selected["process_count"] != EXPECTED_PROCESS_COUNT:
        raise ContributionHarvestWorkerError("phase evidence process count is not four")
    controllers = selected["controller_index_by_process"]
    if (
        not isinstance(controllers, list)
        or len(controllers) != EXPECTED_PROCESS_COUNT
        or any(type(value) is not int for value in controllers)
        or sorted(controllers) != list(range(EXPECTED_PROCESS_COUNT))
    ):
        raise ContributionHarvestWorkerError(
            "phase evidence controller mapping is not an exact permutation"
        )
    hosts = selected["host_evidence_sha256_by_process"]
    if (
        not isinstance(hosts, list)
        or len(hosts) != EXPECTED_PROCESS_COUNT
        or len(set(hosts)) != EXPECTED_PROCESS_COUNT
    ):
        raise ContributionHarvestWorkerError(
            "phase evidence must bind four unique host records"
        )
    for digest in hosts:
        _sha256("host evidence SHA-256", digest)
    selected["phase_evidence_sha256"] = claimed
    return selected


def publish_harvest_phase(
    *,
    config_path: Path,
    phase_name: str,
    split: str,
    control_root: Path,
    source_v1_dataset: ArtifactDataset,
    object_prefix: str,
    object_store: ImmutableObjectStore,
    recomposition_sidecar: Mapping[str, Any],
    supervised_host_evidence: Sequence[Mapping[str, Any]],
    phase_evidence_path: Path,
) -> dict[str, Any]:
    """Coordinator-verify four hosts, seal the v2 manifest, then seal evidence."""

    config = load_contribution_config(config_path)
    phase = config.phase(phase_name)
    validated_hosts = [
        validate_supervised_evidence(value) for value in supervised_host_evidence
    ]
    if (
        len(validated_hosts) != EXPECTED_PROCESS_COUNT
        or any(host["status"] != "complete" for host in validated_hosts)
        or sorted(host["controller_index"] for host in validated_hosts)
        != list(range(EXPECTED_PROCESS_COUNT))
        or sorted(host["process_index"] for host in validated_hosts)
        != list(range(EXPECTED_PROCESS_COUNT))
    ):
        raise ContributionHarvestWorkerError(
            "publication requires successful records with independent exact "
            "controller-index and JAX-process-index permutations"
        )
    hosts = sorted(validated_hosts, key=lambda item: item["process_index"])
    children = [host["child_evidence"] for host in hosts]
    first = children[0]
    common_fields = (
        "protocol_id",
        "config_sha256",
        "phase",
        "phase_ordinal",
        "split",
        "phase_total_tokens",
        "source_v1_manifest_sha256",
        "range_plan_sha256",
        "recomposition_sidecar_sha256",
        "metadata",
        "metadata_sha256",
        "observed_identity_sha256",
        "qualification_receipt_sha256",
        "parent_phase_evidence_sha256",
    )
    if any(
        any(child[name] != first[name] for name in common_fields)
        for child in children[1:]
    ):
        raise ContributionHarvestWorkerError(
            "harvest hosts disagree on their scientific or performance identity"
        )
    if (
        first["protocol_id"] != config.protocol_id
        or first["config_sha256"] != config.config_hash
        or first["phase"] != phase.name
        or first["phase_ordinal"] != phase.ordinal
        or first["split"] != split
        or first["phase_total_tokens"] != _phase_tokens(phase, split)
        or first["source_v1_manifest_sha256"] != source_v1_dataset.manifest_sha256
        or first["recomposition_sidecar_sha256"]
        != recomposition_sidecar.get("sidecar_sha256")
    ):
        raise ContributionHarvestWorkerError(
            "host evidence differs from the requested publication"
        )
    adopted = [child["adopted_parent"] for child in children]
    if phase.extends_phase is None:
        if any(value is not None for value in adopted):
            raise ContributionHarvestWorkerError(
                "Phase 1 host unexpectedly adopted a parent"
            )
    elif (
        any(value is None for value in adopted)
        or len(
            {value["prior_manifest_sha256"] for value in adopted if value is not None}
        )
        != 1
    ):
        raise ContributionHarvestWorkerError(
            "extension hosts did not adopt one common predecessor"
        )
    publish_contribution_manifest(
        control_root,
        first["metadata"],
        source_v1_dataset=source_v1_dataset,
        object_prefix=object_prefix,
        object_store=object_store,
        expected_hosts=EXPECTED_PROCESS_COUNT,
        expected_tokens=_phase_tokens(phase, split),
        recomposition_sidecar=recomposition_sidecar,
    )
    manifest = load_contribution_manifest(
        control_root,
        recomposition_sidecar=recomposition_sidecar,
    )
    evidence = _phase_evidence(
        config=config,
        phase=phase,
        split=split,
        manifest=manifest,
        hosts=hosts,
    )
    validate_phase_evidence(evidence)
    _atomic_create_or_match(phase_evidence_path, evidence)
    return evidence


def _default_process_context() -> tuple[Sequence[Any], int, int, int]:
    import jax

    return (
        tuple(jax.devices()),
        int(jax.process_count()),
        int(jax.process_index()),
        int(jax.device_count()),
    )


def _default_verify_staged(
    invocation: HarvestInvocation,
    config: Experiment2Config,
    object_store: ImmutableObjectStore,
) -> StagedDatasetProvenance:
    """Validate the frozen inventory and cheaply pin every GCS generation."""

    provenance = verify_staged_dataset_manifest(
        invocation.staged_manifest,
        invocation.grain_files,
        expected_split=_source_split(config, invocation.split),
        expected_tokens=_phase_totals(config, invocation.split)[-1],
        tokenizer_id=config.provenance.tokenizer_path,
        tokenizer_revision=config.provenance.model_revision,
        sequence_length=config.model.sequence_length,
        # Full byte hashing happened before the sealed v1 capture.  Repeating
        # multi-terabyte reads on every host would be pure GCS cost.  Instead,
        # the exact generation-bound object identities are revalidated below,
        # and token keys are checked against v1 before every model step.
        verify_file_hashes=False,
        verify_arrayrecord_count=False,
    )
    for entry in provenance.manifest["files"]:
        identity = entry.get("object")
        if identity is not None:
            object_store.verify(object_identity_from_mapping(identity))
    return provenance


def _source_dataset(
    root: Path,
    *,
    cache_root: Path,
    cache_max_bytes: int,
    object_store: ImmutableObjectStore,
) -> ArtifactDataset:
    manifest = load_manifest(root)
    remote = any("object" in entry for entry in manifest["shards"])
    cache = (
        ShardCache(
            cache_root,
            object_store,
            max_bytes=_positive_int("source_cache_max_bytes", cache_max_bytes),
        )
        if remote
        else None
    )
    return ArtifactDataset(
        root,
        verify_hashes=True,
        verify_statistics=True,
        decode_bf16=False,
        cache=cache,
    )


def _default_create_source_dataset(
    invocation: HarvestInvocation,
    object_store: ImmutableObjectStore,
) -> ArtifactDataset:
    return _source_dataset(
        invocation.source_v1_root,
        cache_root=invocation.source_cache_root,
        cache_max_bytes=invocation.source_cache_max_bytes,
        object_store=object_store,
    )


def _default_create_runtime(argv: Sequence[str], files: Sequence[str]) -> Any:
    return create_maxtext_runtime(
        argv,
        packed_train_files=tuple(files),
        packed_eval_files=tuple(files),
    )


def _default_effective_runtime(
    runtime: Any,
    config: Experiment2Config,
    mesh: MeshCandidate,
    per_device_sequence_batch: int,
) -> Mapping[str, Any]:
    dataset_type = getattr(runtime.config, "dataset_type", None)
    if dataset_type != "grain":
        raise ContributionHarvestWorkerError(
            "production harvest requires the real Grain data path"
        )
    return effective_performance_runtime_manifest(
        runtime.config,
        runtime.mesh,
        config=config,
        mesh=mesh,
        per_device_sequence_batch=per_device_sequence_batch,
    )


def _default_performance_launch(
    config: Experiment2Config,
    checkpoint_sha256: str,
    mesh: MeshCandidate,
    per_device_sequence_batch: int,
    jax_cache_dir: Path,
    environment: Mapping[str, Any],
    maxtext_provenance: Mapping[str, str],
    runtime: Any,
) -> Mapping[str, Any]:
    return observed_performance_launch_manifest(
        config,
        checkpoint_sha256=checkpoint_sha256,
        mesh=mesh,
        per_device_sequence_batch=per_device_sequence_batch,
        writer_backlog_limit_batches=(config.runtime.writer_backlog_max_batches),
        jax_cache_dir=jax_cache_dir,
        environment=environment,
        maxtext_provenance=maxtext_provenance,
        runtime_config=runtime.config,
        runtime_mesh=runtime.mesh,
    )


def _default_verify_sidecar(
    adapter: Any,
    checkpoint_sha256: str,
    sidecar: Mapping[str, Any],
) -> Mapping[str, Any]:
    source_fingerprints = sidecar.get("source_fingerprints")
    if not isinstance(source_fingerprints, Mapping):
        raise ContributionHarvestWorkerError(
            "recomposition sidecar lacks source fingerprints"
        )
    return extract_recomposition_sidecar(
        adapter,
        checkpoint_sha256=checkpoint_sha256,
        source_fingerprints={
            str(key): str(value) for key, value in source_fingerprints.items()
        },
    )


def _default_create_writer(
    invocation: HarvestInvocation,
    host_id: str,
    metadata: Mapping[str, Any],
    object_store: ImmutableObjectStore,
    max_tokens: int,
) -> ContributionShardWriter:
    return ContributionShardWriter(
        invocation.control_root,
        host_id,
        metadata,
        local_buffer_root=invocation.local_buffer_root,
        object_prefix=invocation.object_prefix,
        object_store=object_store,
        max_tokens=max_tokens,
    )


def _default_adopt_phase(
    writer: ContributionShardWriter,
    prior_root: Path,
    sidecar: Mapping[str, Any],
) -> ContributionAdoptionResult:
    return adopt_contribution_phase(
        writer,
        prior_root,
        recomposition_sidecar=sidecar,
    )


def _jax_batch_collective(
    local_complete_after_batch: bool,
    local_needs_execution: bool,
) -> tuple[bool, bool]:
    try:
        import numpy as np
        from jax.experimental import multihost_utils
    except ImportError as error:  # pragma: no cover - TPU boundary
        raise ContributionHarvestWorkerError(
            "multi-host harvest coordination requires JAX"
        ) from error
    local = np.asarray(
        [local_complete_after_batch, local_needs_execution],
        dtype=np.uint8,
    )
    gathered = np.asarray(
        multihost_utils.process_allgather(local, tiled=False),
        dtype=np.uint8,
    ).reshape((-1, 2))
    return bool(np.all(gathered[:, 0])), bool(np.any(gathered[:, 1]))


def production_dependencies() -> HarvestWorkerDependencies:
    """Return lazy production seams; installed MaxText is the default."""

    qualifier = qualification_production_dependencies()
    return HarvestWorkerDependencies(
        initialize_distributed=initialize_jax_distributed,
        process_context=_default_process_context,
        verify_maxtext=qualifier.verify_maxtext,
        verify_checkpoint=checkpoint_inventory_sha256,
        create_object_store=GCSObjectStore,
        verify_staged=_default_verify_staged,
        create_source_dataset=_default_create_source_dataset,
        create_runtime=_default_create_runtime,
        close_runtime=qualifier.close_runtime,
        mesh_context=maxtext_mesh_context,
        create_adapter=qualifier.create_adapter,
        prepare=qualifier.prepare,
        effective_runtime=_default_effective_runtime,
        performance_launch=_default_performance_launch,
        verify_sidecar=_default_verify_sidecar,
        environment=environment_fingerprint,
        create_writer=_default_create_writer,
        adopt_phase=_default_adopt_phase,
        harvest=harvest_contribution_host,
        batch_collective=_jax_batch_collective,
        clock=time.perf_counter,
    )


def _absolute_path(value: str | Path, *, name: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_absolute():
        raise ContributionHarvestWorkerError(f"{name} must be absolute")
    return path


def _mesh_by_name(name: str) -> MeshCandidate:
    for mesh in HARVEST_MESHES:
        if mesh.name == name:
            return mesh
    raise ContributionHarvestWorkerError(f"unknown harvest mesh {name!r}")


def _add_common_host_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--split", choices=("train", "eval"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--grain-files", required=True)
    parser.add_argument("--staged-manifest", required=True)
    parser.add_argument("--source-v1-root", required=True)
    parser.add_argument("--source-cache-root", required=True)
    parser.add_argument("--source-cache-max-bytes", type=int, required=True)
    parser.add_argument("--control-root", required=True)
    parser.add_argument("--prior-control-root")
    parser.add_argument("--prior-phase-evidence")
    parser.add_argument("--local-buffer-root", required=True)
    parser.add_argument("--object-prefix", required=True)
    parser.add_argument("--recomposition-sidecar", required=True)
    parser.add_argument("--qualification-receipt", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--jax-cache-dir", required=True)
    parser.add_argument("--maxtext-root")
    parser.add_argument(
        "--mesh",
        choices=tuple(mesh.name for mesh in HARVEST_MESHES),
        required=True,
    )
    parser.add_argument(
        "--per-device-sequence-batch",
        type=int,
        required=True,
    )
    parser.add_argument("--controller-index", type=int, required=True)
    parser.add_argument("--maximum-runtime-seconds", type=int, required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m moe_sae.contribution_harvest_worker",
        description=("Supervised, receipt-gated Experiment-2 contribution harvesting"),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    launch = commands.add_parser(
        "launch-host",
        help="supervise one TPU controller's qualified harvest lane",
    )
    _add_common_host_arguments(launch)
    launch.add_argument("--attempt-root", required=True)
    launch.add_argument(
        "--termination-grace-seconds",
        type=float,
        default=DEFAULT_TERMINATION_GRACE_SECONDS,
    )

    child = commands.add_parser("_run-host", help=argparse.SUPPRESS)
    _add_common_host_arguments(child)
    child.add_argument("--child-evidence", required=True)
    child.add_argument("--execution-record", required=True)

    publish = commands.add_parser(
        "publish",
        help="verify four host records and seal one immutable phase manifest",
    )
    publish.add_argument("--config", required=True)
    publish.add_argument("--phase", required=True)
    publish.add_argument("--split", choices=("train", "eval"), required=True)
    publish.add_argument("--control-root", required=True)
    publish.add_argument("--source-v1-root", required=True)
    publish.add_argument("--source-cache-root", required=True)
    publish.add_argument("--source-cache-max-bytes", type=int, required=True)
    publish.add_argument("--object-prefix", required=True)
    publish.add_argument("--recomposition-sidecar", required=True)
    publish.add_argument(
        "--host-evidence",
        action="append",
        default=[],
        help="repeat exactly four times, in any controller order",
    )
    publish.add_argument("--phase-evidence", required=True)
    return parser


def _invocation_from_args(
    args: argparse.Namespace,
    *,
    child_evidence_path: Path,
    execution_record_path: Path,
) -> HarvestInvocation:
    checkpoint_sha256 = _sha256(
        "checkpoint_sha256",
        args.checkpoint_sha256,
    )
    join_gcs_uri(args.object_prefix, "identity-probe")
    return HarvestInvocation(
        config_path=_absolute_path(args.config, name="config"),
        phase=args.phase,
        split=args.split,
        checkpoint_path=args.checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        grain_files=args.grain_files,
        staged_manifest=args.staged_manifest,
        source_v1_root=_absolute_path(
            args.source_v1_root,
            name="source_v1_root",
        ),
        source_cache_root=_absolute_path(
            args.source_cache_root,
            name="source_cache_root",
        ),
        source_cache_max_bytes=_positive_int(
            "source_cache_max_bytes",
            args.source_cache_max_bytes,
        ),
        control_root=_absolute_path(args.control_root, name="control_root"),
        prior_control_root=(
            None
            if args.prior_control_root is None
            else _absolute_path(
                args.prior_control_root,
                name="prior_control_root",
            )
        ),
        prior_phase_evidence_path=(
            None
            if args.prior_phase_evidence is None
            else _absolute_path(
                args.prior_phase_evidence,
                name="prior_phase_evidence",
            )
        ),
        local_buffer_root=_absolute_path(
            args.local_buffer_root,
            name="local_buffer_root",
        ),
        object_prefix=args.object_prefix.rstrip("/"),
        recomposition_sidecar_path=_absolute_path(
            args.recomposition_sidecar,
            name="recomposition_sidecar",
        ),
        qualification_receipt_path=_absolute_path(
            args.qualification_receipt,
            name="qualification_receipt",
        ),
        repo_root=_absolute_path(args.repo_root, name="repo_root"),
        jax_cache_dir=_absolute_path(
            args.jax_cache_dir,
            name="jax_cache_dir",
        ),
        maxtext_root=(
            None
            if args.maxtext_root is None
            else _absolute_path(args.maxtext_root, name="maxtext_root")
        ),
        mesh=_mesh_by_name(args.mesh),
        per_device_sequence_batch=_positive_int(
            "per_device_sequence_batch",
            args.per_device_sequence_batch,
        ),
        controller_index=_controller_index(args.controller_index),
        maximum_runtime_seconds=_positive_int(
            "maximum_runtime_seconds",
            args.maximum_runtime_seconds,
        ),
        child_evidence_path=child_evidence_path,
        execution_record_path=execution_record_path,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "_run-host":
        invocation = _invocation_from_args(
            args,
            child_evidence_path=_absolute_path(
                args.child_evidence,
                name="child_evidence",
            ),
            execution_record_path=_absolute_path(
                args.execution_record,
                name="execution_record",
            ),
        )
        evidence = run_harvest_child(invocation)
        print(evidence["child_evidence_sha256"], flush=True)
        return 0
    if args.command == "launch-host":
        attempt = _absolute_path(args.attempt_root, name="attempt_root")
        host = attempt / (f"controller-c{_controller_index(args.controller_index):05d}")
        invocation = _invocation_from_args(
            args,
            child_evidence_path=host / "child-evidence.json",
            execution_record_path=host / "execution.json",
        )
        evidence = launch_harvest_host(
            invocation,
            log_path=host / "stdout.log",
            result_path=host / "result.json",
            supervised_evidence_path=host / "supervised-evidence.json",
            termination_grace_seconds=args.termination_grace_seconds,
        )
        print(evidence["supervised_evidence_sha256"], flush=True)
        if evidence["status"] != "complete":
            raise ContributionHarvestWorkerError(
                "supervised harvest host did not complete; inspect its immutable "
                f"record below {host}"
            )
        return 0
    if args.command == "publish":
        if len(args.host_evidence) != EXPECTED_PROCESS_COUNT:
            raise ContributionHarvestWorkerError(
                "--host-evidence must be repeated exactly four times"
            )
        object_prefix = args.object_prefix.rstrip("/")
        join_gcs_uri(object_prefix, "identity-probe")
        store = GCSObjectStore()
        source = _source_dataset(
            _absolute_path(args.source_v1_root, name="source_v1_root"),
            cache_root=_absolute_path(
                args.source_cache_root,
                name="source_cache_root",
            ),
            cache_max_bytes=_positive_int(
                "source_cache_max_bytes",
                args.source_cache_max_bytes,
            ),
            object_store=store,
        )
        sidecar = validate_recomposition_sidecar(
            _read_json(
                _absolute_path(
                    args.recomposition_sidecar,
                    name="recomposition_sidecar",
                ),
                name="recomposition sidecar",
            )
        )
        hosts = [
            _read_json(
                _absolute_path(path, name="host_evidence"),
                name="supervised host evidence",
            )
            for path in args.host_evidence
        ]
        evidence = publish_harvest_phase(
            config_path=_absolute_path(args.config, name="config"),
            phase_name=args.phase,
            split=args.split,
            control_root=_absolute_path(
                args.control_root,
                name="control_root",
            ),
            source_v1_dataset=source,
            object_prefix=object_prefix,
            object_store=store,
            recomposition_sidecar=sidecar,
            supervised_host_evidence=hosts,
            phase_evidence_path=_absolute_path(
                args.phase_evidence,
                name="phase_evidence",
            ),
        )
        print(evidence["phase_evidence_sha256"], flush=True)
        return 0
    raise AssertionError(f"unhandled command {args.command!r}")


if __name__ == "__main__":  # pragma: no cover - exercised on TPU
    raise SystemExit(main())


__all__ = [
    "CHILD_EVIDENCE_SCHEMA_VERSION",
    "EXPECTED_GLOBAL_CHIPS",
    "EXPECTED_PROCESS_COUNT",
    "PHASE_EVIDENCE_SCHEMA_VERSION",
    "SUPERVISED_EVIDENCE_SCHEMA_VERSION",
    "ContributionHarvestWorkerError",
    "HarvestInvocation",
    "HarvestWorkerDependencies",
    "launch_harvest_host",
    "main",
    "production_dependencies",
    "publish_harvest_phase",
    "run_harvest_child",
    "validate_child_evidence",
    "validate_phase_evidence",
    "validate_supervised_evidence",
]
