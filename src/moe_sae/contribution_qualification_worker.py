"""Supervised, executable Experiment-2 MaxText qualification seam.

The qualification has two deliberately distinct identities:

``authorization_contract_identity``
    A pre-launch hash of the pinned source graph contract, required v4-32
    hardware contract, and complete requested launch configuration.  It is
    sufficient only to authorize one process-group-bounded qualification.

``observed_executable_identity``
    The StableHLO, physical device inventory, and effective runtime identity
    observed inside that already-deadlined process.  Only this identity is
    eligible for a performance receipt and later work beyond 600 seconds.

The distinction avoids pretending that a StableHLO digest is knowable before
JAX has initialized and lowered the graph.  The outer process never imports
JAX or MaxText.  It publishes the contract and delegates the child to
``contribution_supervisor``, which starts a new process group and enforces the
hard wall deadline.  The child initializes distributed JAX before any backend
query, verifies the exact clean MaxText fork, restores Gemma, reuses one static
synthetic batch, lowers/compiles the contribution graph once, and writes one
immutable host observation.  A final immutable wrapper binds that observation
to the supervisor's execution and terminal records for coordinator
aggregation.

This module does not provision or delete accelerator resources.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, fields, replace
from importlib import util as importlib_util
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from moe_sae.contribution_artifacts import (
    ContributionSchemaError,
    canonical_json_bytes,
    canonical_sha256,
    validate_recomposition_sidecar,
)
from moe_sae.contribution_config import (
    MAXTEXT_FORK_COMMIT,
    MAXTEXT_ROUTED_MOE_PATCHED_SHA256,
    Experiment2Config,
    load_contribution_config,
)
from moe_sae.contribution_harvest import (
    contribution_capture_to_wire,
)
from moe_sae.contribution_maxtext import (
    CONTRIBUTION_GRAPH_SCHEMA_VERSION,
    InstrumentationObservation,
    PreparedContributionExecutable,
    QualificationDiagnostics,
    QualificationHostObservation,
    QualificationPlan,
    aggregate_qualification_hosts,
    build_contribution_maxtext_argv,
    contribution_adapter_from_runtime,
    exact_performance_identity,
    extract_recomposition_sidecar,
    hardware_manifest,
    launch_config_manifest,
    observed_performance_launch_manifest,
    prepare_contribution_executable,
    run_qualification_host,
)
from moe_sae.contribution_performance import (
    AcceleratorRunAuthorization,
    ExactPerformanceIdentity,
    authorize_accelerator_run,
)
from moe_sae.contribution_supervisor import (
    DEFAULT_TERMINATION_GRACE_SECONDS,
    supervise_authorized_process,
    validate_execution_record,
    validate_supervision_result,
)
from moe_sae.harvest import (
    GrainBatch,
    _block_until_ready,
    _close_maxtext_runtime,
    _process_local_array,
    extract_grain_batch,
)
from moe_sae.model import checkpoint_inventory_sha256
from moe_sae.runtime import (
    HARVEST_MESHES,
    V4_32_BF16_PEAK_FLOPS,
    MeshCandidate,
    create_maxtext_runtime,
    environment_fingerprint,
    initialize_jax_distributed,
    installed_maxtext_commit,
    maxtext_mesh_context,
)

_MODULE_STARTED = time.perf_counter()
CONTRIBUTION_PER_DEVICE_SEQUENCE_BATCHES = (1, 2, 4, 8)
DEFAULT_QUALIFICATION_PLAN = QualificationPlan()

AUTHORIZATION_CONTRACT_SCHEMA_VERSION = (
    "gemma4-contribution-qualification-authorization-contract-v1"
)
CHILD_EVIDENCE_SCHEMA_VERSION = "gemma4-contribution-qualification-child-evidence-v3"
SUPERVISED_HOST_EVIDENCE_SCHEMA_VERSION = (
    "gemma4-contribution-supervised-host-evidence-v3"
)
COORDINATOR_BUNDLE_SCHEMA_VERSION = (
    "gemma4-contribution-qualification-coordinator-bundle-v3"
)
QUALIFICATION_WRITER_SCHEMA_VERSION = (
    "gemma4-contribution-qualification-local-writer-v1"
)
SOURCE_GRAPH_CONTRACT_SCHEMA_VERSION = "gemma4-contribution-source-graph-contract-v1"
EXPECTED_HARDWARE_CONTRACT_SCHEMA_VERSION = (
    "gemma4-contribution-expected-hardware-contract-v1"
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_GRAPH_FILES = (
    "src/moe_sae/model.py",
    "src/moe_sae/harvest.py",
    "src/moe_sae/contribution_model.py",
    "src/moe_sae/contribution_maxtext.py",
    "src/moe_sae/contribution_qualification_worker.py",
    "patches/maxtext/0001-gemma4-return-selected-outputs.patch",
    "patches/maxtext/selected_outputs_manifest.json",
)
_RUNTIME_CONFIG_FIELDS = (
    "model_name",
    "load_parameters_path",
    "use_multimodal",
    "scan_layers",
    "pure_nnx",
    "enable_nnx",
    "pure_nnx_decoder",
    "enable_dropout",
    "load_balance_loss_weight",
    "use_ragged_sort",
    "ragged_buffer_factor",
    "sparse_matmul",
    "use_ring_of_experts",
    "float32_weight_sum",
    "matmul_precision",
    "attention",
    "dtype",
    "weight_dtype",
    "max_target_length",
    "per_device_batch_size",
    "ici_expert_parallelism",
    "ici_fsdp_parallelism",
    "ici_tensor_parallelism",
    "dataset_type",
    "jax_cache_dir",
)


class ContributionQualificationWorkerError(RuntimeError):
    """A supervised qualification invariant was not satisfied."""


class QualificationPersist(Protocol):
    """Persistence interface consumed by ``run_qualification_host``."""

    backlog_limit: int
    max_observed_backlog: int
    queue_drained: bool

    def __call__(self, output: Any) -> int: ...

    def flush(self) -> None: ...

    def close(self) -> None: ...

    def summary(self) -> Mapping[str, Any]: ...


@dataclass(frozen=True, slots=True)
class ChildInvocation:
    """All non-secret inputs needed by one already-supervised TPU child."""

    config_path: Path
    checkpoint_path: str
    checkpoint_sha256: str
    maxtext_root: Path | None
    repo_root: Path
    jax_cache_dir: Path
    scratch_root: Path
    contract_path: Path
    execution_record_path: Path
    child_evidence_path: Path
    recomposition_sidecar_path: Path
    mesh: MeshCandidate
    per_device_sequence_batch: int
    process_index: int
    plan: QualificationPlan


@dataclass(frozen=True, slots=True)
class QualificationWorkerDependencies:
    """Injectable TPU boundaries used by deterministic CPU tests."""

    initialize_distributed: Callable[[], None]
    process_context: Callable[[], tuple[Sequence[Any], int, int, int]]
    verify_maxtext: Callable[[Path | None], Mapping[str, str]]
    verify_checkpoint: Callable[[str], str]
    create_runtime: Callable[[Sequence[str]], Any]
    close_runtime: Callable[[Any], None]
    mesh_context: Callable[[Any], Any]
    create_adapter: Callable[[Any, Path | None], Any]
    extract_sidecar: Callable[
        [Any, str, Mapping[str, str]],
        Mapping[str, Any],
    ]
    prepare: Callable[
        [Any, Any, GrainBatch, Callable[[], float]],
        PreparedContributionExecutable,
    ]
    instrumentation_probe: Callable[
        [Any, GrainBatch, int, int],
        Callable[[], InstrumentationObservation],
    ]
    create_persist: Callable[
        [Path, int, int, int],
        QualificationPersist,
    ]
    environment: Callable[[Path], Mapping[str, Any]]
    clock: Callable[[], float]


@dataclass(frozen=True, slots=True)
class AuthorizedContributionExecutable:
    """A lowered executable admitted for work beyond the ungated window.

    Construct this wrapper after lowering and before the first workload step.
    Its authorization is derived from the observed StableHLO identity, never
    from the pre-launch source contract used by a bounded qualifier.
    """

    executable: Callable[[Any, Any, Any, Any], Any]
    observed_executable_identity: ExactPerformanceIdentity
    authorization: AcceleratorRunAuthorization

    def __call__(
        self,
        token_ids: Any,
        positions: Any,
        segment_ids: Any,
        sequence_ids: Any,
    ) -> Any:
        return self.executable(
            token_ids,
            positions,
            segment_ids,
            sequence_ids,
        )


def _sha256(name: str, value: Any) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ContributionQualificationWorkerError(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContributionQualificationWorkerError(f"{name} must be a positive integer")
    return value


def _process_index(value: Any, *, process_count: int = 4) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value < process_count
    ):
        raise ContributionQualificationWorkerError(
            f"process_index must lie in [0, {process_count - 1}]"
        )
    return value


def _canonical_hash(value: Any, *, name: str) -> str:
    try:
        return canonical_sha256(value)
    except ContributionSchemaError as error:
        raise ContributionQualificationWorkerError(
            f"{name} is not finite canonical JSON"
        ) from error


def _absolute_directory(value: str | Path, *, name: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_absolute():
        raise ContributionQualificationWorkerError(f"{name} must be absolute")
    return path


def _absolute_file(value: str | Path, *, name: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.name:
        raise ContributionQualificationWorkerError(f"{name} must name a file")
    return path


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise ContributionQualificationWorkerError(
                "immutable record write made no progress"
            )
        offset += written


def _atomic_create_json(path: Path, value: Mapping[str, Any]) -> None:
    """Publish canonical JSON atomically without replacing prior evidence."""

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        payload = canonical_json_bytes(dict(value))
    except ContributionSchemaError as error:
        raise ContributionQualificationWorkerError(
            "immutable evidence is not canonical JSON"
        ) from error
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        _write_all(descriptor, payload)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as error:
            raise ContributionQualificationWorkerError(
                f"immutable evidence already exists: {path}"
            ) from error
        directory = os.open(
            path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _read_json(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContributionQualificationWorkerError(
            f"{name} is missing or invalid: {path}"
        ) from error
    if not isinstance(value, dict):
        raise ContributionQualificationWorkerError(f"{name} must be an object")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 << 20), b""):
                digest.update(block)
    except OSError as error:
        raise ContributionQualificationWorkerError(
            f"cannot hash graph source {path}"
        ) from error
    return digest.hexdigest()


def _mesh_by_name(name: str) -> MeshCandidate:
    for mesh in HARVEST_MESHES:
        if mesh.name == name:
            return mesh
    raise ContributionQualificationWorkerError(
        f"unknown mesh {name!r}; expected one of "
        + ", ".join(item.name for item in HARVEST_MESHES)
    )


def _source_graph_contract(
    config: Experiment2Config,
    *,
    repo_root: Path,
) -> dict[str, Any]:
    inventory: list[dict[str, str]] = []
    for relative in _SOURCE_GRAPH_FILES:
        path = repo_root / relative
        if not path.is_file():
            raise ContributionQualificationWorkerError(
                f"source graph contract file is absent: {path}"
            )
        inventory.append(
            {
                "path": relative,
                "sha256": _file_sha256(path),
            }
        )
    return {
        "schema_version": SOURCE_GRAPH_CONTRACT_SCHEMA_VERSION,
        "compiled_graph_schema_version": CONTRIBUTION_GRAPH_SCHEMA_VERSION,
        "adapter": ("moe_sae.contribution_model.Gemma4ContributionPrefixAdapter"),
        "maxtext_fork_commit": config.provenance.maxtext_fork_commit,
        "selected_outputs_patch_sha256": (
            config.provenance.maxtext_selected_outputs_patch_sha256
        ),
        "source_files": inventory,
    }


def _expected_hardware_contract(mesh: MeshCandidate) -> dict[str, Any]:
    return {
        "schema_version": EXPECTED_HARDWARE_CONTRACT_SCHEMA_VERSION,
        "accelerator": "v4-32",
        "platform": "tpu",
        "device_kind_contains": "v4",
        "global_chip_count": 16,
        "process_count": 4,
        "peak_bf16_flops_per_second": int(V4_32_BF16_PEAK_FLOPS),
        "mesh": asdict(mesh),
    }


def _plan_manifest(plan: QualificationPlan) -> dict[str, int]:
    return {
        "warmup_steps": plan.warmup_steps,
        "writer_disabled_steps": plan.writer_disabled_steps,
        "measured_steps": plan.measured_steps,
        "maximum_seconds": plan.maximum_seconds,
    }


def build_authorization_contract(
    config: Experiment2Config,
    *,
    checkpoint_sha256: str,
    mesh: MeshCandidate,
    per_device_sequence_batch: int,
    jax_cache_dir: str | Path,
    repo_root: str | Path,
    plan: QualificationPlan = DEFAULT_QUALIFICATION_PLAN,
) -> dict[str, Any]:
    """Build the exact pre-launch contract for a bounded qualification."""

    config.validate()
    checkpoint = _sha256("checkpoint_sha256", checkpoint_sha256)
    root = _absolute_directory(repo_root, name="repo_root")
    launch = launch_config_manifest(
        config,
        checkpoint_sha256=checkpoint,
        mesh=mesh,
        per_device_sequence_batch=per_device_sequence_batch,
        sequence_length=config.model.sequence_length,
        writer_backlog_limit_batches=(config.runtime.writer_backlog_max_batches),
        jax_cache_dir=jax_cache_dir,
    )
    launch = {
        **launch,
        "qualification_plan": _plan_manifest(plan),
        "writer_implementation": QUALIFICATION_WRITER_SCHEMA_VERSION,
        "synthetic_static_batch_reused": True,
        "source_graph_contract_sha256": _canonical_hash(
            _source_graph_contract(config, repo_root=root),
            name="source graph contract",
        ),
    }
    graph = _source_graph_contract(config, repo_root=root)
    hardware = _expected_hardware_contract(mesh)
    identity = ExactPerformanceIdentity.from_components(
        graph=graph,
        hardware=hardware,
        config=launch,
    )
    unsigned = {
        "schema_version": AUTHORIZATION_CONTRACT_SCHEMA_VERSION,
        "identity_semantics": (
            "prelaunch-source-hardware-config-contract-not-observed-stablehlo"
        ),
        "graph_contract": graph,
        "hardware_contract": hardware,
        "launch_contract": launch,
        "authorization_contract_identity": {
            **identity.to_dict(),
            "identity_sha256": identity.identity_sha256,
        },
    }
    return {
        **unsigned,
        "authorization_contract_sha256": _canonical_hash(
            unsigned,
            name="authorization contract",
        ),
    }


def validate_authorization_contract(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    expected = {
        "schema_version",
        "identity_semantics",
        "graph_contract",
        "hardware_contract",
        "launch_contract",
        "authorization_contract_identity",
        "authorization_contract_sha256",
    }
    if set(value) != expected:
        raise ContributionQualificationWorkerError(
            "authorization contract fields are not exact"
        )
    selected = dict(value)
    claimed = _sha256(
        "authorization_contract_sha256",
        selected.pop("authorization_contract_sha256"),
    )
    if selected["schema_version"] != AUTHORIZATION_CONTRACT_SCHEMA_VERSION:
        raise ContributionQualificationWorkerError(
            "authorization contract schema is incompatible"
        )
    if (
        selected["identity_semantics"]
        != "prelaunch-source-hardware-config-contract-not-observed-stablehlo"
    ):
        raise ContributionQualificationWorkerError(
            "authorization contract identity semantics changed"
        )
    if _canonical_hash(selected, name="authorization contract") != claimed:
        raise ContributionQualificationWorkerError(
            "authorization contract self-hash is invalid"
        )
    raw_identity = selected["authorization_contract_identity"]
    if not isinstance(raw_identity, Mapping):
        raise ContributionQualificationWorkerError(
            "authorization contract identity is invalid"
        )
    identity = ExactPerformanceIdentity(
        graph_sha256=_sha256(
            "authorization graph_sha256",
            raw_identity.get("graph_sha256"),
        ),
        hardware_sha256=_sha256(
            "authorization hardware_sha256",
            raw_identity.get("hardware_sha256"),
        ),
        config_sha256=_sha256(
            "authorization config_sha256",
            raw_identity.get("config_sha256"),
        ),
    )
    if (
        raw_identity.get("schema_version") != identity.to_dict()["schema_version"]
        or raw_identity.get("identity_sha256") != identity.identity_sha256
        or identity.graph_sha256
        != _canonical_hash(
            selected["graph_contract"],
            name="authorization graph contract",
        )
        or identity.hardware_sha256
        != _canonical_hash(
            selected["hardware_contract"],
            name="authorization hardware contract",
        )
        or identity.config_sha256
        != _canonical_hash(
            selected["launch_contract"],
            name="authorization launch contract",
        )
    ):
        raise ContributionQualificationWorkerError(
            "authorization contract identity is not derived from its components"
        )
    return {
        **selected,
        "authorization_contract_sha256": claimed,
    }


def _contract_identity(contract: Mapping[str, Any]) -> ExactPerformanceIdentity:
    checked = validate_authorization_contract(contract)
    raw = checked["authorization_contract_identity"]
    return ExactPerformanceIdentity(
        graph_sha256=raw["graph_sha256"],
        hardware_sha256=raw["hardware_sha256"],
        config_sha256=raw["config_sha256"],
    )


def _verify_maxtext_source(
    maxtext_root: Path | None,
) -> dict[str, str]:
    """Verify the locked install, or an explicitly requested clean checkout."""

    spec = importlib_util.find_spec("maxtext")
    origin = None if spec is None else spec.origin
    if origin is None:
        raise ContributionQualificationWorkerError(
            "the exact MaxText fork is not importable"
        )
    imported_package = Path(origin).resolve().parent
    if maxtext_root is None:
        head = installed_maxtext_commit()
        mode = "locked_pep610_install"
        routed_moe = imported_package / "layers" / "moe.py"
    else:
        root = maxtext_root.expanduser().resolve()
        try:
            head = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            ).stdout.strip()
            dirty = subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            ).stdout.strip()
        except (
            OSError,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
        ) as error:
            raise ContributionQualificationWorkerError(
                "cannot verify the Experiment-2 MaxText checkout"
            ) from error
        if dirty:
            raise ContributionQualificationWorkerError(
                "MaxText checkout has uncommitted changes"
            )
        try:
            imported_package.relative_to(root)
        except ValueError as error:
            raise ContributionQualificationWorkerError(
                "importable MaxText does not come from --maxtext-root"
            ) from error
        mode = "explicit_clean_checkout"
        routed_moe = root / "src" / "maxtext" / "layers" / "moe.py"
    if head != MAXTEXT_FORK_COMMIT:
        raise ContributionQualificationWorkerError(
            f"MaxText is not the published Experiment-2 fork {MAXTEXT_FORK_COMMIT}"
        )
    digest = _file_sha256(routed_moe)
    if digest != MAXTEXT_ROUTED_MOE_PATCHED_SHA256:
        raise ContributionQualificationWorkerError(
            "MaxText routed-MoE source differs from the manifest-verified "
            "fork post-image"
        )
    return {
        "mode": mode,
        "commit": head,
        "routed_moe_sha256": digest,
    }


def _default_process_context() -> tuple[Sequence[Any], int, int, int]:
    import jax

    devices = tuple(jax.devices())
    return (
        devices,
        int(jax.process_count()),
        int(jax.process_index()),
        int(jax.device_count()),
    )


def _default_create_adapter(model: Any, maxtext_root: Path | None) -> Any:
    return contribution_adapter_from_runtime(
        model,
        maxtext_root=maxtext_root,
    )


def _default_extract_sidecar(
    adapter: Any,
    checkpoint_sha256: str,
    source_fingerprints: Mapping[str, str],
) -> Mapping[str, Any]:
    return extract_recomposition_sidecar(
        adapter,
        checkpoint_sha256=checkpoint_sha256,
        source_fingerprints=source_fingerprints,
    )


def _default_prepare(
    adapter: Any,
    mesh: Any,
    batch: GrainBatch,
    clock: Callable[[], float],
) -> PreparedContributionExecutable:
    return prepare_contribution_executable(
        adapter,
        mesh,
        batch,
        clock=clock,
    )


def _array_local_digest(value: Any) -> str:
    """Hash this controller's exact addressable shards without an all-gather."""

    digest = hashlib.sha256()
    shards = getattr(value, "addressable_shards", None)
    if shards is None:
        array = np.asarray(value)
        header = {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "index": None,
        }
        digest.update(canonical_json_bytes(header))
        digest.update(np.ascontiguousarray(array).tobytes())
        return digest.hexdigest()
    records: list[tuple[str, Any]] = []
    for shard in shards:
        index = tuple(
            (
                0 if item.start is None else int(item.start),
                int(value.shape[axis]) if item.stop is None else int(item.stop),
                1 if item.step is None else int(item.step),
            )
            for axis, item in enumerate(tuple(shard.index))
        )
        records.append((repr(index), shard))
    if not records:
        raise ContributionQualificationWorkerError(
            "instrumentation output has no addressable shards"
        )
    for _, shard in sorted(records, key=lambda item: item[0]):
        array = np.asarray(shard.data)
        header = {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "index": repr(tuple(shard.index)),
        }
        digest.update(canonical_json_bytes(header))
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _default_instrumentation_probe(
    adapter: Any,
    batch: GrainBatch,
    process_count: int,
    process_index: int,
) -> Callable[[], InstrumentationObservation]:
    """Compile and run the exact full-block parity oracle once."""

    del process_count, process_index
    try:
        from flax import nnx
    except ImportError as error:  # pragma: no cover - TPU dependency boundary
        raise ContributionQualificationWorkerError(
            "instrumentation parity requires the pinned Flax NNX runtime"
        ) from error
    adapter_type = type(adapter)
    model = adapter.model
    config = adapter.config

    def explicit_model_probe(
        dynamic_model: Any,
        token_ids: Any,
        positions: Any,
        segment_ids: Any,
        sequence_ids: Any,
    ) -> Any:
        rebound = adapter_type(dynamic_model, config)
        return rebound.instrumentation_evidence(
            token_ids,
            positions,
            segment_ids,
            sequence_ids,
            model_mode="train",
        )

    compiled_probe = nnx.jit(explicit_model_probe)

    def probe() -> InstrumentationObservation:
        evidence = compiled_probe(
            model,
            batch.token_ids,
            batch.positions,
            batch.segment_ids,
            batch.segment_ids,
        )
        _block_until_ready(evidence)
        try:
            equal_array = np.asarray(evidence.equal)
        except (TypeError, ValueError, RuntimeError) as error:
            raise ContributionQualificationWorkerError(
                "instrumentation equality scalar is not host-addressable"
            ) from error
        if equal_array.size != 1:
            raise ContributionQualificationWorkerError(
                "instrumentation equality must be one scalar"
            )
        disabled = _array_local_digest(evidence.unmodified_full_block)
        enabled = _array_local_digest(evidence.instrumented_full_block)
        equal = bool(equal_array.reshape(()).item())
        return InstrumentationObservation(
            disabled_output_sha256=disabled,
            enabled_output_sha256=enabled,
            equal=equal,
        )

    return probe


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(child)
            for key, child in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_json_value(child) for child in value]
    return str(value)


def _effective_runtime_config(runtime: Any) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for name in _RUNTIME_CONFIG_FIELDS:
        try:
            value = getattr(runtime.config, name)
        except (AttributeError, KeyError, ValueError):
            value = None
        values[name] = _json_value(value)
    mesh = runtime.mesh
    values["mesh_axis_names"] = [str(item) for item in mesh.axis_names]
    values["mesh_shape"] = {
        str(name): int(size)
        for name, size in zip(
            tuple(mesh.axis_names),
            tuple(mesh.devices.shape),
            strict=True,
        )
    }
    return values


def _validate_effective_runtime(
    runtime: Any,
    *,
    config: Experiment2Config,
    mesh: MeshCandidate,
    per_device_sequence_batch: int,
) -> dict[str, Any]:
    observed = _effective_runtime_config(runtime)
    required = {
        "use_multimodal": False,
        "scan_layers": False,
        "pure_nnx_decoder": True,
        "enable_dropout": False,
        "use_ragged_sort": True,
        "sparse_matmul": True,
        "use_ring_of_experts": False,
        "float32_weight_sum": True,
        "max_target_length": config.model.sequence_length,
        "per_device_batch_size": per_device_sequence_batch,
        "ici_expert_parallelism": mesh.expert_parallelism,
        "ici_fsdp_parallelism": mesh.fsdp_parallelism,
        "ici_tensor_parallelism": mesh.tensor_parallelism,
        "dataset_type": "synthetic",
    }
    failures = {
        name: {"expected": wanted, "observed": observed.get(name)}
        for name, wanted in required.items()
        if observed.get(name) != wanted
    }
    if failures:
        raise ContributionQualificationWorkerError(
            "effective MaxText runtime differs from the qualification launch: "
            + json.dumps(failures, sort_keys=True, separators=(",", ":"))
        )
    return observed


def _write_safetensors_create_only(
    destination: Path,
    tensors: Mapping[str, np.ndarray],
    *,
    metadata: Mapping[str, str],
) -> tuple[int, str]:
    try:
        from safetensors.numpy import save_file
    except ImportError as error:  # pragma: no cover - installed base dependency
        raise ContributionQualificationWorkerError(
            "qualification persistence requires safetensors"
        ) from error
    if destination.exists():
        raise ContributionQualificationWorkerError(
            f"qualification payload already exists: {destination}"
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".safetensors.tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.unlink()
        save_file(dict(tensors), str(temporary), metadata=dict(metadata))
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        digest = hashlib.sha256()
        size = 0
        with temporary.open("rb") as handle:
            for block in iter(lambda: handle.read(8 << 20), b""):
                size += len(block)
                digest.update(block)
        try:
            os.link(temporary, destination, follow_symlinks=False)
        except FileExistsError as error:
            raise ContributionQualificationWorkerError(
                f"qualification payload already exists: {destination}"
            ) from error
        directory = os.open(
            destination.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return size, digest.hexdigest()
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


class LocalQualificationPersist:
    """Bounded single-writer sink that exercises the v2 safetensors wire."""

    backlog_limit = 2

    def __init__(
        self,
        root: Path,
        *,
        process_count: int,
        process_index: int,
        sequence_length: int,
    ) -> None:
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.process_count = process_count
        self.process_index = process_index
        self.sequence_length = sequence_length
        self.max_observed_backlog = 0
        self.queue_drained = False
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="contribution-qualification-writer",
        )
        self._pending: list[Future[dict[str, Any]]] = []
        self._records: list[dict[str, Any]] = []
        self._step = 0
        self._closed = False

    def _reap(self, *, wait: bool) -> None:
        remaining: list[Future[dict[str, Any]]] = []
        for future in self._pending:
            if wait or future.done():
                self._records.append(future.result())
            else:
                remaining.append(future)
        self._pending = remaining

    def _local(self, value: Any, name: str) -> np.ndarray:
        return _process_local_array(
            value,
            f"qualification {name}",
            process_count=self.process_count,
            process_index=self.process_index,
            replicated_partition_unit=self.sequence_length,
        )

    def _write(self, step: int, wire: Any) -> dict[str, Any]:
        destination = self.root / f"step-{step:05d}.safetensors"
        tensors = {
            name: np.ascontiguousarray(getattr(wire, name))
            for name in wire.__dataclass_fields__
        }
        size, digest = _write_safetensors_create_only(
            destination,
            tensors,
            metadata={
                "schema_version": QUALIFICATION_WRITER_SCHEMA_VERSION,
                "process_index": str(self.process_index),
                "step": str(step),
            },
        )
        return {
            "step": step,
            "file": destination.name,
            "tokens": int(wire.tokens),
            "bytes": size,
            "sha256": digest,
        }

    def __call__(self, output: Any) -> int:
        if self._closed:
            raise ContributionQualificationWorkerError("qualification writer is closed")
        self.queue_drained = False
        self._reap(wait=False)
        if self._pending:
            self._reap(wait=True)
        wire = contribution_capture_to_wire(
            expert_output=self._local(output.expert_output, "expert_output"),
            expert_ids=self._local(output.expert_ids, "expert_ids"),
            router_weights=self._local(
                output.router_weights,
                "router_weights",
            ),
            shared_output=self._local(output.shared_output, "shared_output"),
            sequence_id=self._local(output.sequence_id, "sequence_id"),
            position=self._local(output.position, "position"),
        )
        step = self._step
        self._step += 1
        self._pending.append(self._executor.submit(self._write, step, wire))
        self.max_observed_backlog = max(
            self.max_observed_backlog,
            len(self._pending),
        )
        return int(wire.tokens)

    def flush(self) -> None:
        self._reap(wait=True)
        self.queue_drained = True

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.flush()
        finally:
            self._executor.shutdown(wait=True, cancel_futures=False)
            self._closed = True

    def summary(self) -> Mapping[str, Any]:
        if not self.queue_drained or self._pending:
            raise ContributionQualificationWorkerError(
                "qualification writer summary requested before drain"
            )
        ordered = sorted(self._records, key=lambda item: item["step"])
        unsigned = {
            "schema_version": QUALIFICATION_WRITER_SCHEMA_VERSION,
            "process_count": self.process_count,
            "process_index": self.process_index,
            "backlog_limit": self.backlog_limit,
            "max_observed_backlog": self.max_observed_backlog,
            "queue_drained": True,
            "files": ordered,
            "tokens": sum(int(item["tokens"]) for item in ordered),
            "bytes": sum(int(item["bytes"]) for item in ordered),
        }
        return {
            **unsigned,
            "writer_evidence_sha256": _canonical_hash(
                unsigned,
                name="qualification writer evidence",
            ),
        }


def _default_create_persist(
    root: Path,
    process_count: int,
    process_index: int,
    sequence_length: int,
) -> QualificationPersist:
    return LocalQualificationPersist(
        root,
        process_count=process_count,
        process_index=process_index,
        sequence_length=sequence_length,
    )


def production_dependencies() -> QualificationWorkerDependencies:
    """Return lazy production boundaries without touching JAX until invoked."""

    return QualificationWorkerDependencies(
        initialize_distributed=initialize_jax_distributed,
        process_context=_default_process_context,
        verify_maxtext=_verify_maxtext_source,
        verify_checkpoint=checkpoint_inventory_sha256,
        create_runtime=create_maxtext_runtime,
        close_runtime=_close_maxtext_runtime,
        mesh_context=maxtext_mesh_context,
        create_adapter=_default_create_adapter,
        extract_sidecar=_default_extract_sidecar,
        prepare=_default_prepare,
        instrumentation_probe=_default_instrumentation_probe,
        create_persist=_default_create_persist,
        environment=environment_fingerprint,
        clock=time.perf_counter,
    )


def authorize_extended_workload_after_lowering(
    prepared: PreparedContributionExecutable,
    *,
    hardware: Mapping[str, Any],
    launch_config: Mapping[str, Any],
    qualification_receipt: Mapping[str, Any],
    maximum_runtime_seconds: int,
    config: Experiment2Config,
) -> AuthorizedContributionExecutable:
    """Bind a >600-second workload to the just-lowered executable.

    This function is intentionally downstream of lowering: it derives the
    observed StableHLO/hardware/launch identity, validates the qualification
    receipt against that exact identity, and only then returns a callable.
    A receipt for the pre-launch authorization contract, another topology, or
    a drifted graph cannot produce the wrapper.
    """

    if type(prepared) is not PreparedContributionExecutable:
        raise ContributionQualificationWorkerError(
            "extended workload requires the exact prepared executable"
        )
    maximum = _positive_int(
        "maximum_runtime_seconds",
        maximum_runtime_seconds,
    )
    if maximum <= config.runtime.accelerator_ungated_max_seconds:
        raise ContributionQualificationWorkerError(
            "extended workload seam is only for work beyond 600 seconds"
        )
    observed = exact_performance_identity(
        graph=prepared.graph,
        hardware=hardware,
        launch_config=launch_config,
    )
    authorization = authorize_accelerator_run(
        identity=observed,
        maximum_runtime_seconds=maximum,
        purpose="workload",
        qualification_receipt=qualification_receipt,
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


def _identity_from_mapping(
    value: Mapping[str, Any],
    *,
    name: str,
) -> ExactPerformanceIdentity:
    expected = {
        "schema_version",
        "graph_sha256",
        "hardware_sha256",
        "config_sha256",
        "identity_sha256",
    }
    if set(value) != expected:
        raise ContributionQualificationWorkerError(f"{name} fields are not exact")
    identity = ExactPerformanceIdentity(
        graph_sha256=_sha256(
            f"{name}.graph_sha256",
            value["graph_sha256"],
        ),
        hardware_sha256=_sha256(
            f"{name}.hardware_sha256",
            value["hardware_sha256"],
        ),
        config_sha256=_sha256(
            f"{name}.config_sha256",
            value["config_sha256"],
        ),
    )
    if (
        value["schema_version"] != identity.to_dict()["schema_version"]
        or value["identity_sha256"] != identity.identity_sha256
    ):
        raise ContributionQualificationWorkerError(f"{name} self-identity is invalid")
    return identity


def _observation_from_dict(
    value: Mapping[str, Any],
) -> QualificationHostObservation:
    expected = {
        "schema_version",
        "identity",
        "identity_sha256",
        "process_count",
        "process_index",
        "qualification_wall_seconds",
        "compile_seconds",
        "measured_device_seconds",
        "persisted_window_seconds",
        "writer_disabled_device_seconds",
        "global_output_tokens",
        "local_persisted_tokens",
        "writer_disabled_global_output_tokens",
        "warmup_steps",
        "synchronized_measurement_steps",
        "measured_capture_compile_count",
        "parity_oracle_compile_count",
        "total_compile_count",
        "max_writer_backlog_batches",
        "writer_backlog_limit_batches",
        "writer_queue_drained",
        "instrumentation",
        "diagnostics",
        "observation_sha256",
    }
    if set(value) != expected:
        raise ContributionQualificationWorkerError(
            "host observation fields are not exact"
        )
    unsigned = dict(value)
    claimed = _sha256(
        "observation_sha256",
        unsigned.pop("observation_sha256"),
    )
    if _canonical_hash(unsigned, name="host observation") != claimed:
        raise ContributionQualificationWorkerError(
            "host observation self-hash is invalid"
        )
    raw_identity = unsigned["identity"]
    if not isinstance(raw_identity, Mapping):
        raise ContributionQualificationWorkerError(
            "host observation identity is invalid"
        )
    identity = ExactPerformanceIdentity(
        graph_sha256=raw_identity["graph_sha256"],
        hardware_sha256=raw_identity["hardware_sha256"],
        config_sha256=raw_identity["config_sha256"],
    )
    if (
        raw_identity != identity.to_dict()
        or unsigned["identity_sha256"] != identity.identity_sha256
    ):
        raise ContributionQualificationWorkerError(
            "host observation identity is inconsistent"
        )
    instrumentation = unsigned["instrumentation"]
    diagnostics = unsigned["diagnostics"]
    if not isinstance(instrumentation, Mapping) or not isinstance(
        diagnostics,
        Mapping,
    ):
        raise ContributionQualificationWorkerError(
            "host observation nested diagnostics are invalid"
        )
    scalar_names = {
        item.name
        for item in fields(QualificationHostObservation)
        if item.name not in {"identity", "instrumentation", "diagnostics"}
    }
    kwargs = {name: unsigned[name] for name in scalar_names}
    return QualificationHostObservation(
        identity=identity,
        instrumentation=InstrumentationObservation(**dict(instrumentation)),
        diagnostics=QualificationDiagnostics(**dict(diagnostics)),
        **kwargs,
    )


def _child_record(
    *,
    contract: Mapping[str, Any],
    execution_sha256: str,
    observation: QualificationHostObservation,
    persistence: Mapping[str, Any],
    checkpoint_sha256: str,
    maxtext_fork_commit: str,
    effective_runtime: Mapping[str, Any],
    recomposition_sidecar: Mapping[str, Any],
) -> dict[str, Any]:
    checked_sidecar = validate_recomposition_sidecar(recomposition_sidecar)
    unsigned = {
        "schema_version": CHILD_EVIDENCE_SCHEMA_VERSION,
        "identity_relationship": (
            "authorization-contract-bounds-discovery-"
            "observed-executable-identity-is-receipt-authority"
        ),
        "authorization_contract_sha256": contract["authorization_contract_sha256"],
        "authorization_contract_identity": contract["authorization_contract_identity"],
        "observed_executable_identity": {
            **observation.identity.to_dict(),
            "identity_sha256": observation.identity.identity_sha256,
        },
        "execution_sha256": execution_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "maxtext_fork_commit": maxtext_fork_commit,
        "recomposition_sidecar_sha256": checked_sidecar["sidecar_sha256"],
        "recomposition_sidecar_canonical_sha256": _canonical_hash(
            checked_sidecar,
            name="recomposition sidecar",
        ),
        "effective_runtime": dict(effective_runtime),
        "observation": observation.to_dict(),
        "persistence": dict(persistence),
    }
    return {
        **unsigned,
        "child_evidence_sha256": _canonical_hash(
            unsigned,
            name="child qualification evidence",
        ),
    }


def validate_child_evidence(value: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "schema_version",
        "identity_relationship",
        "authorization_contract_sha256",
        "authorization_contract_identity",
        "observed_executable_identity",
        "execution_sha256",
        "checkpoint_sha256",
        "maxtext_fork_commit",
        "recomposition_sidecar_sha256",
        "recomposition_sidecar_canonical_sha256",
        "effective_runtime",
        "observation",
        "persistence",
        "child_evidence_sha256",
    }
    if set(value) != expected:
        raise ContributionQualificationWorkerError(
            "child evidence fields are not exact"
        )
    selected = dict(value)
    claimed = _sha256(
        "child_evidence_sha256",
        selected.pop("child_evidence_sha256"),
    )
    if selected["schema_version"] != CHILD_EVIDENCE_SCHEMA_VERSION:
        raise ContributionQualificationWorkerError(
            "child evidence schema is incompatible"
        )
    if (
        selected["identity_relationship"] != "authorization-contract-bounds-discovery-"
        "observed-executable-identity-is-receipt-authority"
    ):
        raise ContributionQualificationWorkerError("child identity semantics changed")
    if _canonical_hash(selected, name="child evidence") != claimed:
        raise ContributionQualificationWorkerError(
            "child evidence self-hash is invalid"
        )
    _sha256(
        "authorization_contract_sha256",
        selected["authorization_contract_sha256"],
    )
    _sha256("execution_sha256", selected["execution_sha256"])
    _sha256("checkpoint_sha256", selected["checkpoint_sha256"])
    _sha256(
        "recomposition_sidecar_sha256",
        selected["recomposition_sidecar_sha256"],
    )
    _sha256(
        "recomposition_sidecar_canonical_sha256",
        selected["recomposition_sidecar_canonical_sha256"],
    )
    if selected["maxtext_fork_commit"] != MAXTEXT_FORK_COMMIT:
        raise ContributionQualificationWorkerError(
            "child evidence uses the wrong MaxText fork"
        )
    authorization = selected["authorization_contract_identity"]
    observed = selected["observed_executable_identity"]
    if not isinstance(authorization, Mapping) or not isinstance(
        observed,
        Mapping,
    ):
        raise ContributionQualificationWorkerError(
            "child evidence identities are invalid"
        )
    _identity_from_mapping(
        authorization,
        name="authorization_contract_identity",
    )
    observed_identity = _identity_from_mapping(
        observed,
        name="observed_executable_identity",
    )
    observation = _observation_from_dict(selected["observation"])
    if observation.identity != observed_identity:
        raise ContributionQualificationWorkerError(
            "child observation and observed executable identity differ"
        )
    persistence = selected["persistence"]
    if not isinstance(persistence, Mapping):
        raise ContributionQualificationWorkerError(
            "child persistence evidence is invalid"
        )
    writer_claim = _sha256(
        "writer_evidence_sha256",
        persistence.get("writer_evidence_sha256"),
    )
    writer_unsigned = dict(persistence)
    writer_unsigned.pop("writer_evidence_sha256", None)
    if (
        writer_unsigned.get("schema_version") != QUALIFICATION_WRITER_SCHEMA_VERSION
        or _canonical_hash(
            writer_unsigned,
            name="qualification writer evidence",
        )
        != writer_claim
    ):
        raise ContributionQualificationWorkerError(
            "qualification writer evidence self-hash is invalid"
        )
    return {
        **selected,
        "child_evidence_sha256": claimed,
    }


def _load_and_match_contract(
    invocation: ChildInvocation,
    config: Experiment2Config,
) -> dict[str, Any]:
    persisted = validate_authorization_contract(
        _read_json(
            invocation.contract_path,
            name="authorization contract",
        )
    )
    rebuilt = build_authorization_contract(
        config,
        checkpoint_sha256=invocation.checkpoint_sha256,
        mesh=invocation.mesh,
        per_device_sequence_batch=invocation.per_device_sequence_batch,
        jax_cache_dir=invocation.jax_cache_dir,
        repo_root=invocation.repo_root,
        plan=invocation.plan,
    )
    if rebuilt != persisted:
        raise ContributionQualificationWorkerError(
            "child launch inputs no longer match the prelaunch authorization contract"
        )
    execution = validate_execution_record(
        _read_json(
            invocation.execution_record_path,
            name="supervisor execution record",
        )
    )
    identity = _contract_identity(persisted)
    authorization = execution["authorization"]
    if (
        authorization["purpose"] != "qualification"
        or authorization["basis"] != "bounded_qualification"
        or authorization["identity_sha256"] != identity.identity_sha256
        or authorization["maximum_runtime_seconds"] != invocation.plan.maximum_seconds
    ):
        raise ContributionQualificationWorkerError(
            "supervisor authorization does not match the child contract"
        )
    return persisted


def run_qualification_child(
    invocation: ChildInvocation,
    *,
    dependencies: QualificationWorkerDependencies | None = None,
    qualification_started: float | None = None,
) -> dict[str, Any]:
    """Execute one host's exact qualification inside a supervisor deadline."""

    deps = dependencies or production_dependencies()
    started = deps.clock() if qualification_started is None else qualification_started
    config = load_contribution_config(invocation.config_path)
    contract = _load_and_match_contract(invocation, config)

    # This must be the first dependency boundary capable of querying a backend.
    deps.initialize_distributed()
    devices, process_count, observed_index, global_chips = deps.process_context()
    if (
        process_count != 4
        or global_chips != 16
        or observed_index != invocation.process_index
    ):
        raise ContributionQualificationWorkerError(
            "distributed JAX process identity does not match the required v4-32 "
            f"host {invocation.process_index}: count={process_count}, "
            f"index={observed_index}, chips={global_chips}"
        )
    maxtext_provenance = dict(deps.verify_maxtext(invocation.maxtext_root))
    if set(maxtext_provenance) != {
        "mode",
        "commit",
        "routed_moe_sha256",
    }:
        raise ContributionQualificationWorkerError(
            "MaxText verifier returned incomplete provenance"
        )
    head = maxtext_provenance["commit"]
    if (
        head != config.provenance.maxtext_fork_commit
        or maxtext_provenance["routed_moe_sha256"] != MAXTEXT_ROUTED_MOE_PATCHED_SHA256
    ):
        raise ContributionQualificationWorkerError(
            "observed MaxText fork differs from the experiment configuration"
        )
    checkpoint = deps.verify_checkpoint(invocation.checkpoint_path)
    if checkpoint != invocation.checkpoint_sha256:
        raise ContributionQualificationWorkerError(
            "checkpoint inventory differs from --checkpoint-sha256"
        )

    argv = build_contribution_maxtext_argv(
        checkpoint_path=invocation.checkpoint_path,
        grain_train_files="",
        grain_eval_files="",
        mesh=invocation.mesh,
        per_device_sequence_batch=invocation.per_device_sequence_batch,
        jax_cache_dir=invocation.jax_cache_dir,
        synthetic=True,
    )
    runtime = deps.create_runtime(argv)
    persist: QualificationPersist | None = None
    body_error: BaseException | None = None
    try:
        qualification_runtime = _validate_effective_runtime(
            runtime,
            config=config,
            mesh=invocation.mesh,
            per_device_sequence_batch=invocation.per_device_sequence_batch,
        )
        adapter = deps.create_adapter(runtime.model, invocation.maxtext_root)
        source = iter(runtime.train_iterator)
        try:
            raw_batch = next(source)
        except StopIteration as error:
            raise ContributionQualificationWorkerError(
                "MaxText synthetic iterator is empty"
            ) from error
        batch = extract_grain_batch(
            raw_batch,
            expected_sequence_length=config.model.sequence_length,
        )
        _block_until_ready(
            (
                batch.token_ids,
                batch.positions,
                batch.segment_ids,
            )
        )
        environment = dict(deps.environment(invocation.repo_root))
        environment_sha256 = _sha256(
            "environment fingerprint",
            environment.get("sha256"),
        )
        source_fingerprints = {
            "authorization_contract": contract[
                "authorization_contract_sha256"
            ],
            "environment": environment_sha256,
            "experiment_config": config.config_hash,
            "maxtext_routed_moe": maxtext_provenance[
                "routed_moe_sha256"
            ],
            "source_graph_contract": _canonical_hash(
                contract["graph_contract"],
                name="source graph contract",
            ),
        }
        sidecar = validate_recomposition_sidecar(
            deps.extract_sidecar(
                adapter,
                checkpoint,
                source_fingerprints,
            )
        )
        if (
            sidecar["checkpoint_sha256"] != checkpoint
            or sidecar["maxtext_fork_commit"] != head
            or sidecar["source_fingerprints"] != source_fingerprints
        ):
            raise ContributionQualificationWorkerError(
                "extracted recomposition sidecar differs from the qualified "
                "checkpoint/config/fork/environment sources"
            )
        _atomic_create_json(invocation.recomposition_sidecar_path, sidecar)
        hardware = hardware_manifest(
            devices=devices,
            process_count=process_count,
            mesh=invocation.mesh,
        )
        observed_launch = observed_performance_launch_manifest(
            config,
            checkpoint_sha256=checkpoint,
            mesh=invocation.mesh,
            per_device_sequence_batch=invocation.per_device_sequence_batch,
            writer_backlog_limit_batches=(config.runtime.writer_backlog_max_batches),
            jax_cache_dir=invocation.jax_cache_dir,
            environment=environment,
            maxtext_provenance=maxtext_provenance,
            runtime_config=runtime.config,
            runtime_mesh=runtime.mesh,
        )
        effective = {
            "graph_runtime": observed_launch["effective_graph_runtime"],
            "qualification_runtime": qualification_runtime,
            "qualification_plan": _plan_manifest(invocation.plan),
            "writer_implementation": QUALIFICATION_WRITER_SCHEMA_VERSION,
            "synthetic_static_batch_reused": True,
            "maxtext_argv": list(argv),
        }
        persist = deps.create_persist(
            invocation.scratch_root,
            process_count,
            observed_index,
            config.model.sequence_length,
        )
        with deps.mesh_context(runtime):
            prepared = deps.prepare(
                adapter,
                runtime.mesh,
                batch,
                deps.clock,
            )
            observed_identity = exact_performance_identity(
                graph=prepared.graph,
                hardware=hardware,
                launch_config=observed_launch,
            )
            probe = deps.instrumentation_probe(
                adapter,
                batch,
                process_count,
                observed_index,
            )
            observation = run_qualification_host(
                prepared,
                batch,
                identity=observed_identity,
                process_count=process_count,
                process_index=observed_index,
                persist=persist,
                instrumentation_probe=probe,
                qualification_started=started,
                plan=invocation.plan,
                diagnostics=QualificationDiagnostics,
                clock=deps.clock,
            )
        persistence = persist.summary()
        execution = validate_execution_record(
            _read_json(
                invocation.execution_record_path,
                name="supervisor execution record",
            )
        )
        record = _child_record(
            contract=contract,
            execution_sha256=execution["execution_sha256"],
            observation=observation,
            persistence=persistence,
            checkpoint_sha256=checkpoint,
            maxtext_fork_commit=head,
            effective_runtime=effective,
            recomposition_sidecar=sidecar,
        )
        validate_child_evidence(record)
        _atomic_create_json(invocation.child_evidence_path, record)
        return record
    except BaseException as error:
        body_error = error
        raise
    finally:
        cleanup_errors: list[BaseException] = []
        if persist is not None:
            try:
                persist.close()
            except BaseException as error:  # noqa: BLE001 - preserve primary failure
                cleanup_errors.append(error)
        try:
            deps.close_runtime(runtime)
        except BaseException as error:  # noqa: BLE001 - cleanup must never mask state
            cleanup_errors.append(error)
        if cleanup_errors:
            if body_error is not None:
                for error in cleanup_errors:
                    try:
                        body_error.add_note(
                            "suppressed qualification cleanup failure: "
                            f"{type(error).__name__}: {error}"
                        )
                    except AttributeError:  # pragma: no cover - Python >=3.11
                        pass
            else:
                raise cleanup_errors[0]


def _child_command(
    invocation: ChildInvocation,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "moe_sae.contribution_qualification_worker",
        "_run-host",
        "--config",
        str(invocation.config_path),
        "--checkpoint",
        invocation.checkpoint_path,
        "--checkpoint-sha256",
        invocation.checkpoint_sha256,
        "--repo-root",
        str(invocation.repo_root),
        "--jax-cache-dir",
        str(invocation.jax_cache_dir),
        "--scratch-root",
        str(invocation.scratch_root),
        "--contract",
        str(invocation.contract_path),
        "--execution-record",
        str(invocation.execution_record_path),
        "--child-evidence",
        str(invocation.child_evidence_path),
        "--recomposition-sidecar",
        str(invocation.recomposition_sidecar_path),
        "--mesh",
        invocation.mesh.name,
        "--per-device-sequence-batch",
        str(invocation.per_device_sequence_batch),
        "--process-index",
        str(invocation.process_index),
        "--warmup-steps",
        str(invocation.plan.warmup_steps),
        "--writer-disabled-steps",
        str(invocation.plan.writer_disabled_steps),
        "--measured-steps",
        str(invocation.plan.measured_steps),
        "--maximum-seconds",
        str(invocation.plan.maximum_seconds),
    ]
    if invocation.maxtext_root is not None:
        command.extend(("--maxtext-root", str(invocation.maxtext_root)))
    return command


def _supervised_host_record(
    *,
    contract: Mapping[str, Any],
    child: Mapping[str, Any] | None,
    execution: Mapping[str, Any],
    result: Mapping[str, Any],
    expected_process_index: int,
) -> dict[str, Any]:
    completed = result["status"] == "completed" and child is not None
    observation: dict[str, Any] | None = None
    observed_identity: Mapping[str, Any] | None = None
    child_hash: str | None = None
    sidecar_sha256: str | None = None
    sidecar_canonical_sha256: str | None = None
    if child is not None:
        checked_child = validate_child_evidence(child)
        parsed = _observation_from_dict(checked_child["observation"])
        # The parent supervisor starts its clock before Popen.  Use the larger
        # wall so Python startup and child import time cannot disappear from a
        # receipt's qualification budget.
        authoritative = replace(
            parsed,
            qualification_wall_seconds=max(
                parsed.qualification_wall_seconds,
                float(result["elapsed_seconds"]),
            ),
        )
        observation = authoritative.to_dict()
        observed_identity = checked_child["observed_executable_identity"]
        child_hash = checked_child["child_evidence_sha256"]
        sidecar_sha256 = checked_child["recomposition_sidecar_sha256"]
        sidecar_canonical_sha256 = checked_child[
            "recomposition_sidecar_canonical_sha256"
        ]
        if (
            checked_child["authorization_contract_sha256"]
            != contract["authorization_contract_sha256"]
            or checked_child["execution_sha256"] != execution["execution_sha256"]
            or parsed.process_index != expected_process_index
        ):
            raise ContributionQualificationWorkerError(
                "child evidence does not belong to this supervised launch"
            )
    unsigned = {
        "schema_version": SUPERVISED_HOST_EVIDENCE_SCHEMA_VERSION,
        "status": "coordinator_ready" if completed else "failed",
        "process_index": expected_process_index,
        "identity_relationship": (
            "authorization_contract_identity_is_prelaunch_only;"
            "observed_executable_identity_is_receipt_authority;"
            "equality_is_not_asserted"
        ),
        "authorization_contract_sha256": contract["authorization_contract_sha256"],
        "authorization_contract_identity": contract["authorization_contract_identity"],
        "observed_executable_identity": observed_identity,
        "child_evidence_sha256": child_hash,
        "recomposition_sidecar_sha256": sidecar_sha256,
        "recomposition_sidecar_canonical_sha256": sidecar_canonical_sha256,
        "supervisor_execution_sha256": execution["execution_sha256"],
        "supervisor_result_sha256": result["result_sha256"],
        "supervisor_status": result["status"],
        "supervisor_elapsed_seconds": result["elapsed_seconds"],
        "observation": observation,
    }
    return {
        **unsigned,
        "host_evidence_sha256": _canonical_hash(
            unsigned,
            name="supervised host evidence",
        ),
    }


def validate_supervised_host_evidence(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    expected = {
        "schema_version",
        "status",
        "process_index",
        "identity_relationship",
        "authorization_contract_sha256",
        "authorization_contract_identity",
        "observed_executable_identity",
        "child_evidence_sha256",
        "recomposition_sidecar_sha256",
        "recomposition_sidecar_canonical_sha256",
        "supervisor_execution_sha256",
        "supervisor_result_sha256",
        "supervisor_status",
        "supervisor_elapsed_seconds",
        "observation",
        "host_evidence_sha256",
    }
    if set(value) != expected:
        raise ContributionQualificationWorkerError(
            "supervised host evidence fields are not exact"
        )
    selected = dict(value)
    claimed = _sha256(
        "host_evidence_sha256",
        selected.pop("host_evidence_sha256"),
    )
    if (
        selected["schema_version"] != SUPERVISED_HOST_EVIDENCE_SCHEMA_VERSION
        or _canonical_hash(selected, name="supervised host evidence") != claimed
    ):
        raise ContributionQualificationWorkerError(
            "supervised host evidence self-hash is invalid"
        )
    _process_index(selected["process_index"])
    _sha256(
        "authorization_contract_sha256",
        selected["authorization_contract_sha256"],
    )
    _sha256(
        "supervisor_execution_sha256",
        selected["supervisor_execution_sha256"],
    )
    _sha256(
        "supervisor_result_sha256",
        selected["supervisor_result_sha256"],
    )
    authorization_identity = selected["authorization_contract_identity"]
    if not isinstance(authorization_identity, Mapping):
        raise ContributionQualificationWorkerError(
            "authorization contract identity is invalid"
        )
    _identity_from_mapping(
        authorization_identity,
        name="authorization_contract_identity",
    )
    if (
        selected["identity_relationship"]
        != "authorization_contract_identity_is_prelaunch_only;"
        "observed_executable_identity_is_receipt_authority;"
        "equality_is_not_asserted"
    ):
        raise ContributionQualificationWorkerError(
            "supervised host identity semantics changed"
        )
    status = selected["status"]
    if status == "coordinator_ready":
        if (
            selected["supervisor_status"] != "completed"
            or selected["child_evidence_sha256"] is None
            or selected["recomposition_sidecar_sha256"] is None
            or selected["recomposition_sidecar_canonical_sha256"] is None
            or selected["observed_executable_identity"] is None
            or selected["observation"] is None
        ):
            raise ContributionQualificationWorkerError(
                "coordinator-ready host evidence is incomplete"
            )
        _sha256(
            "child_evidence_sha256",
            selected["child_evidence_sha256"],
        )
        _sha256(
            "recomposition_sidecar_sha256",
            selected["recomposition_sidecar_sha256"],
        )
        _sha256(
            "recomposition_sidecar_canonical_sha256",
            selected["recomposition_sidecar_canonical_sha256"],
        )
        observed = _identity_from_mapping(
            selected["observed_executable_identity"],
            name="observed_executable_identity",
        )
        observation = _observation_from_dict(selected["observation"])
        if observation.identity != observed:
            raise ContributionQualificationWorkerError(
                "supervised observation identity is inconsistent"
            )
    elif status == "failed":
        if selected["supervisor_status"] == "completed":
            raise ContributionQualificationWorkerError(
                "completed supervisor result cannot be labeled failed"
            )
    else:
        raise ContributionQualificationWorkerError(
            "supervised host evidence status is invalid"
        )
    return {
        **selected,
        "host_evidence_sha256": claimed,
    }


def launch_qualification_host(
    *,
    config_path: str | Path,
    checkpoint_path: str,
    checkpoint_sha256: str,
    maxtext_root: str | Path | None = None,
    repo_root: str | Path,
    jax_cache_dir: str | Path,
    control_root: str | Path,
    scratch_root: str | Path,
    mesh: MeshCandidate,
    per_device_sequence_batch: int,
    process_index: int,
    plan: QualificationPlan = DEFAULT_QUALIFICATION_PLAN,
    environment: Mapping[str, str] | None = None,
    termination_grace_seconds: float = DEFAULT_TERMINATION_GRACE_SECONDS,
    supervisor: Callable[..., Mapping[str, Any]] = (supervise_authorized_process),
) -> dict[str, Any]:
    """Publish intent, supervise one TPU child, and seal coordinator evidence."""

    index = _process_index(process_index)
    config_file = _absolute_file(config_path, name="config_path")
    repo = _absolute_directory(repo_root, name="repo_root")
    maxtext = (
        None
        if maxtext_root is None
        else _absolute_directory(maxtext_root, name="maxtext_root")
    )
    cache = _absolute_directory(jax_cache_dir, name="jax_cache_dir")
    control = _absolute_directory(control_root, name="control_root")
    scratch = _absolute_directory(scratch_root, name="scratch_root")
    host_control = control / f"host-{index:05d}"
    host_scratch = scratch / f"host-{index:05d}"
    host_control.mkdir(parents=True, exist_ok=True)
    host_scratch.mkdir(parents=True, exist_ok=True)
    config = load_contribution_config(config_file)
    contract = build_authorization_contract(
        config,
        checkpoint_sha256=checkpoint_sha256,
        mesh=mesh,
        per_device_sequence_batch=per_device_sequence_batch,
        jax_cache_dir=cache,
        repo_root=repo,
        plan=plan,
    )
    contract_path = host_control / "authorization-contract.json"
    execution_path = host_control / "execution.json"
    result_path = host_control / "supervision-result.json"
    log_path = host_control / "qualification.log"
    child_path = host_control / "child-evidence.json"
    sidecar_path = host_control / "recomposition-sidecar.json"
    final_path = host_control / "host-evidence.json"
    _atomic_create_json(contract_path, contract)
    identity = _contract_identity(contract)
    authorization = authorize_accelerator_run(
        identity=identity,
        maximum_runtime_seconds=plan.maximum_seconds,
        purpose="qualification",
        minimum_useful_mfu=config.runtime.accelerator_min_useful_mfu,
        ungated_max_seconds=(config.runtime.accelerator_ungated_max_seconds),
        qualification_max_seconds=(
            config.runtime.accelerator_qualification_max_seconds
        ),
    )
    invocation = ChildInvocation(
        config_path=config_file,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=_sha256(
            "checkpoint_sha256",
            checkpoint_sha256,
        ),
        maxtext_root=maxtext,
        repo_root=repo,
        jax_cache_dir=cache,
        scratch_root=host_scratch,
        contract_path=contract_path,
        execution_record_path=execution_path,
        child_evidence_path=child_path,
        recomposition_sidecar_path=sidecar_path,
        mesh=mesh,
        per_device_sequence_batch=per_device_sequence_batch,
        process_index=index,
        plan=plan,
    )
    command = _child_command(invocation)
    selected_environment = dict(os.environ if environment is None else environment)
    selected_environment["PYTHONUNBUFFERED"] = "1"
    result = dict(
        supervisor(
            authorization=authorization,
            command=command,
            environment=selected_environment,
            working_directory=repo,
            log_path=log_path,
            execution_record_path=execution_path,
            result_path=result_path,
            termination_grace_seconds=termination_grace_seconds,
        )
    )
    execution = validate_execution_record(
        _read_json(execution_path, name="supervisor execution record")
    )
    result = validate_supervision_result(
        result,
        expected_execution_sha256=execution["execution_sha256"],
    )
    child: Mapping[str, Any] | None = None
    if child_path.exists():
        child = validate_child_evidence(
            _read_json(child_path, name="child qualification evidence")
        )
    host = _supervised_host_record(
        contract=contract,
        child=child,
        execution=execution,
        result=result,
        expected_process_index=index,
    )
    validate_supervised_host_evidence(host)
    _atomic_create_json(final_path, host)
    if host["status"] != "coordinator_ready":
        raise ContributionQualificationWorkerError(
            "qualification child did not complete; immutable supervisor and host "
            f"evidence is under {host_control}"
        )
    return host


def aggregate_supervised_host_evidence(
    host_records: Sequence[Mapping[str, Any]],
    *,
    batch_sequences: int,
    sequence_length: int,
) -> dict[str, Any]:
    """Create one receipt bundle from four supervised host records."""

    checked = [validate_supervised_host_evidence(item) for item in host_records]
    if len(checked) != 4:
        raise ContributionQualificationWorkerError(
            "coordinator requires exactly four host records"
        )
    ordered = sorted(checked, key=lambda item: item["process_index"])
    if [item["process_index"] for item in ordered] != [0, 1, 2, 3]:
        raise ContributionQualificationWorkerError(
            "coordinator host inventory is incomplete or duplicated"
        )
    contract_hashes = {item["authorization_contract_sha256"] for item in ordered}
    if len(contract_hashes) != 1:
        raise ContributionQualificationWorkerError(
            "hosts used different prelaunch authorization contracts"
        )
    sidecar_hashes = {item["recomposition_sidecar_sha256"] for item in ordered}
    sidecar_canonical_hashes = {
        item["recomposition_sidecar_canonical_sha256"] for item in ordered
    }
    if len(sidecar_hashes) != 1 or len(sidecar_canonical_hashes) != 1:
        raise ContributionQualificationWorkerError(
            "controllers extracted different recomposition sidecars"
        )
    observations = [_observation_from_dict(item["observation"]) for item in ordered]
    _, bundle = aggregate_qualification_hosts(
        observations,
        batch_sequences=batch_sequences,
        sequence_length=sequence_length,
    )
    unsigned = {
        "schema_version": COORDINATOR_BUNDLE_SCHEMA_VERSION,
        "identity_relationship": (
            "receipt-binds-observed-executable-identity-not-"
            "authorization-contract-identity"
        ),
        "authorization_contract_sha256": next(iter(contract_hashes)),
        "host_evidence_sha256": [item["host_evidence_sha256"] for item in ordered],
        "observed_executable_identity": ordered[0]["observed_executable_identity"],
        "recomposition_sidecar_sha256": next(iter(sidecar_hashes)),
        "recomposition_sidecar_canonical_sha256": next(
            iter(sidecar_canonical_hashes)
        ),
        "source_evidence": bundle["source_evidence"],
        "useful_flops_ledger": bundle["useful_flops_ledger"],
        "qualification_receipt": bundle["qualification_receipt"],
    }
    return {
        **unsigned,
        "coordinator_bundle_sha256": _canonical_hash(
            unsigned,
            name="qualification coordinator bundle",
        ),
    }


def _add_common_child_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument(
        "--maxtext-root",
        help=(
            "optional clean checkout for development; production verifies the "
            "locked PEP-610 MaxText install"
        ),
    )
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--jax-cache-dir", required=True)
    parser.add_argument(
        "--mesh",
        required=True,
        choices=[item.name for item in HARVEST_MESHES],
    )
    parser.add_argument(
        "--per-device-sequence-batch",
        type=int,
        required=True,
        choices=CONTRIBUTION_PER_DEVICE_SEQUENCE_BATCHES,
    )
    parser.add_argument("--process-index", type=int, required=True)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--writer-disabled-steps", type=int, default=10)
    parser.add_argument("--measured-steps", type=int, default=10)
    parser.add_argument("--maximum-seconds", type=int, default=600)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Supervised Experiment-2 MaxText qualification worker"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    launch = commands.add_parser(
        "launch-host",
        help="launch one host under the hard process-group supervisor",
    )
    _add_common_child_arguments(launch)
    launch.add_argument("--control-root", required=True)
    launch.add_argument("--scratch-root", required=True)
    launch.add_argument(
        "--termination-grace-seconds",
        type=float,
        default=DEFAULT_TERMINATION_GRACE_SECONDS,
    )

    child = commands.add_parser("_run-host", help=argparse.SUPPRESS)
    _add_common_child_arguments(child)
    child.add_argument("--scratch-root", required=True)
    child.add_argument("--contract", required=True)
    child.add_argument("--execution-record", required=True)
    child.add_argument("--child-evidence", required=True)
    child.add_argument("--recomposition-sidecar", required=True)

    aggregate = commands.add_parser(
        "aggregate",
        help="seal four immutable host observations into one receipt bundle",
    )
    aggregate.add_argument(
        "--host-evidence",
        action="append",
        required=True,
        help="repeat exactly four times",
    )
    aggregate.add_argument("--batch-sequences", type=int, required=True)
    aggregate.add_argument("--sequence-length", type=int, default=1_024)
    aggregate.add_argument("--output", required=True)
    return parser


def _plan_from_args(args: argparse.Namespace) -> QualificationPlan:
    return QualificationPlan(
        warmup_steps=args.warmup_steps,
        writer_disabled_steps=args.writer_disabled_steps,
        measured_steps=args.measured_steps,
        maximum_seconds=args.maximum_seconds,
    )


def _invocation_from_args(args: argparse.Namespace) -> ChildInvocation:
    return ChildInvocation(
        config_path=_absolute_file(args.config, name="config"),
        checkpoint_path=args.checkpoint,
        checkpoint_sha256=_sha256(
            "checkpoint_sha256",
            args.checkpoint_sha256,
        ),
        maxtext_root=(
            None
            if args.maxtext_root is None
            else _absolute_directory(
                args.maxtext_root,
                name="maxtext_root",
            )
        ),
        repo_root=_absolute_directory(args.repo_root, name="repo_root"),
        jax_cache_dir=_absolute_directory(
            args.jax_cache_dir,
            name="jax_cache_dir",
        ),
        scratch_root=_absolute_directory(
            args.scratch_root,
            name="scratch_root",
        ),
        contract_path=_absolute_file(args.contract, name="contract"),
        execution_record_path=_absolute_file(
            args.execution_record,
            name="execution_record",
        ),
        child_evidence_path=_absolute_file(
            args.child_evidence,
            name="child_evidence",
        ),
        recomposition_sidecar_path=_absolute_file(
            args.recomposition_sidecar,
            name="recomposition_sidecar",
        ),
        mesh=_mesh_by_name(args.mesh),
        per_device_sequence_batch=args.per_device_sequence_batch,
        process_index=_process_index(args.process_index),
        plan=_plan_from_args(args),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "_run-host":
        record = run_qualification_child(
            _invocation_from_args(args),
            qualification_started=_MODULE_STARTED,
        )
        print(record["child_evidence_sha256"], flush=True)
        return 0
    if args.command == "launch-host":
        record = launch_qualification_host(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            checkpoint_sha256=args.checkpoint_sha256,
            maxtext_root=args.maxtext_root,
            repo_root=args.repo_root,
            jax_cache_dir=args.jax_cache_dir,
            control_root=args.control_root,
            scratch_root=args.scratch_root,
            mesh=_mesh_by_name(args.mesh),
            per_device_sequence_batch=args.per_device_sequence_batch,
            process_index=args.process_index,
            plan=_plan_from_args(args),
            termination_grace_seconds=args.termination_grace_seconds,
        )
        print(record["host_evidence_sha256"], flush=True)
        return 0
    if args.command == "aggregate":
        if len(args.host_evidence) != 4:
            raise ContributionQualificationWorkerError(
                "--host-evidence must be repeated exactly four times"
            )
        records = [
            _read_json(
                _absolute_file(path, name="host_evidence"),
                name="supervised host evidence",
            )
            for path in args.host_evidence
        ]
        bundle = aggregate_supervised_host_evidence(
            records,
            batch_sequences=_positive_int(
                "batch_sequences",
                args.batch_sequences,
            ),
            sequence_length=_positive_int(
                "sequence_length",
                args.sequence_length,
            ),
        )
        output = _absolute_file(args.output, name="output")
        _atomic_create_json(output, bundle)
        print(bundle["coordinator_bundle_sha256"], flush=True)
        return 0
    raise AssertionError(f"unhandled command {args.command!r}")


if __name__ == "__main__":  # pragma: no cover - exercised as a module on TPU
    raise SystemExit(main())


__all__ = [
    "AUTHORIZATION_CONTRACT_SCHEMA_VERSION",
    "CHILD_EVIDENCE_SCHEMA_VERSION",
    "COORDINATOR_BUNDLE_SCHEMA_VERSION",
    "QUALIFICATION_WRITER_SCHEMA_VERSION",
    "SUPERVISED_HOST_EVIDENCE_SCHEMA_VERSION",
    "AuthorizedContributionExecutable",
    "ChildInvocation",
    "ContributionQualificationWorkerError",
    "LocalQualificationPersist",
    "QualificationWorkerDependencies",
    "aggregate_supervised_host_evidence",
    "authorize_extended_workload_after_lowering",
    "build_authorization_contract",
    "launch_qualification_host",
    "main",
    "production_dependencies",
    "run_qualification_child",
    "validate_authorization_contract",
    "validate_child_evidence",
    "validate_supervised_host_evidence",
]
