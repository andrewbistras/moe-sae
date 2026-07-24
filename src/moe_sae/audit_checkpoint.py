"""Identity-bound direct restore of immutable SAE Orbax payloads.

Production training normally restores through a validated POSIX control
pointer.  Retrospective audits instead name an exact historical ``gs://``
payload.  This module is the only direct-payload escape hatch: it requires a
frozen expectation, validates fresh Orbax finalization and custom metadata,
and repeats the inspection after restore.

The descriptor returned by :func:`inspect_audit_checkpoint` is a self-hashed
record of the complete Orbax header.  It deliberately contains no observation
timestamp, so identical immutable payload metadata produces identical
descriptors on every host and across repeated audit attempts.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

AUDIT_EXPECTATION_SCHEMA = "gemma4-audit-checkpoint-expectation-v1"
AUDIT_DESCRIPTOR_SCHEMA = "gemma4-audit-checkpoint-descriptor-v1"
_CONDITIONS = frozenset({"A", "B", "F"})
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_INSTANCE_RE = re.compile(r"[0-9a-f]{32}")
_BUCKET_RE = re.compile(r"[a-z0-9][a-z0-9._-]{1,61}[a-z0-9]")
_PAYLOAD_SUFFIX_RE = re.compile(
    r"(?:^|/)runs/(?P<identity>[0-9a-f]{64})/"
    r"step-(?P<step>[0-9]{8})-tokens-(?P<tokens>[0-9]{12})-"
    r"attempt-(?P<attempt>[0-9a-f]{32})$"
)
_HEADER_KEYS = {
    "checkpoint_version",
    "training_config",
    "training_config_sha256",
    "mesh_shape",
    "metadata",
}
_EXPECTATION_KEYS = {
    "schema_version",
    "checkpoint_uri",
    "condition",
    "seed",
    "tokens_seen",
    "steps",
    "training_config_sha256",
    "mesh_shape",
    "checkpoint_identity",
}
_DESCRIPTOR_KEYS = {
    "schema_version",
    "expectation_sha256",
    "checkpoint_uri",
    "finalized",
    "checkpoint_version",
    "condition",
    "seed",
    "tokens_seen",
    "steps",
    "training_config_sha256",
    "mesh_shape",
    "checkpoint_identity",
    "checkpoint_identity_sha256",
    "custom_metadata_sha256",
    "checkpoint_header",
    "checkpoint_header_sha256",
    "descriptor_sha256",
}
_CHECKPOINT_OBSERVATION_KEYS = {
    "checkpoint_payload_uri",
    "tokens_seen",
    "steps",
    "optimizer_wall_seconds",
    "compile_seconds",
    "checkpoint_seconds",
}


class AuditCheckpointError(RuntimeError):
    """An explicit audit checkpoint failed closed validation."""


def _canonical_bytes(value: Any, *, label: str) -> bytes:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be canonical JSON") from error
    try:
        decoded = json.loads(encoded)
    except json.JSONDecodeError as error:  # pragma: no cover - json encoded it
        raise ValueError(f"{label} must be canonical JSON") from error
    if decoded != value:
        raise ValueError(f"{label} must round-trip through JSON exactly")
    return encoded


def _json_mapping(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} keys must be strings")
    encoded = _canonical_bytes(dict(value), label=label)
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):  # pragma: no cover - guarded above
        raise ValueError(f"{label} must be a JSON object")
    return decoded


def _sha256(value: Any, *, label: str) -> str:
    return hashlib.sha256(_canonical_bytes(value, label=label)).hexdigest()


def _same_json(left: Any, right: Any) -> bool:
    """Compare JSON values without Python's ``True == 1`` coercion."""

    try:
        return _canonical_bytes(left, label="JSON value") == _canonical_bytes(
            right, label="JSON value"
        )
    except ValueError:
        return False


def _require_int(value: Any, label: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if value < (1 if positive else 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{label} must be {qualifier}")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be 64 lowercase hexadecimal digits")
    return value


def _canonical_gcs_uri(value: Any) -> str:
    if not isinstance(value, str) or not value.startswith("gs://"):
        raise ValueError("audit checkpoint URI must be an explicit gs:// URI")
    if any(character in value for character in ("?", "#", "\\", "\n", "\r", "\t")):
        raise ValueError("audit checkpoint URI contains an ambiguous character")
    remainder = value[5:]
    bucket, separator, object_name = remainder.partition("/")
    if (
        not separator
        or _BUCKET_RE.fullmatch(bucket) is None
        or not object_name
        or object_name.endswith("/")
    ):
        raise ValueError("audit checkpoint URI must name a canonical GCS prefix")
    components = object_name.split("/")
    if any(not component or component in {".", ".."} for component in components):
        raise ValueError("audit checkpoint URI contains a non-canonical path segment")
    return value


def _mesh_mapping(value: Mapping[str, Any]) -> dict[str, int]:
    mesh = _json_mapping(value, label="audit checkpoint mesh")
    if not mesh:
        raise ValueError("audit checkpoint mesh may not be empty")
    for axis, size in mesh.items():
        if not axis or isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError("audit checkpoint mesh axes must have positive sizes")
    return {str(axis): int(size) for axis, size in mesh.items()}


def _validate_identity_shapes(identity: Mapping[str, Any]) -> None:
    run_instance_id = identity.get("run_instance_id")
    if run_instance_id is not None and (
        not isinstance(run_instance_id, str)
        or _INSTANCE_RE.fullmatch(run_instance_id) is None
    ):
        raise ValueError("checkpoint run_instance_id must be 32 lowercase hex digits")
    for name, value in identity.items():
        if (
            name.endswith("_sha256")
            and value is not None
            and (not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None)
        ):
            raise ValueError(f"checkpoint identity {name} is not a SHA-256")


@dataclass(frozen=True, slots=True)
class AuditCheckpointExpectation:
    """Frozen operator input for one exact historical checkpoint payload.

    ``checkpoint_identity`` is a required subset of the custom run identity.
    The resulting descriptor still records and hashes every identity field
    found in the Orbax header.
    """

    checkpoint_uri: str
    condition: str
    seed: int
    tokens_seen: int
    steps: int
    training_config_sha256: str
    mesh_shape: Mapping[str, int]
    checkpoint_identity: Mapping[str, Any]

    def __post_init__(self) -> None:
        uri = _canonical_gcs_uri(self.checkpoint_uri)
        if self.condition not in _CONDITIONS:
            raise ValueError("audit checkpoint condition must be A, B, or F")
        seed = _require_int(self.seed, "audit checkpoint seed")
        tokens_seen = _require_int(
            self.tokens_seen, "audit checkpoint tokens_seen", positive=True
        )
        steps = _require_int(self.steps, "audit checkpoint steps", positive=True)
        config_sha256 = _require_sha256(
            self.training_config_sha256, "training_config_sha256"
        )
        mesh = _mesh_mapping(self.mesh_shape)
        identity = _json_mapping(
            self.checkpoint_identity, label="audit checkpoint identity"
        )
        _validate_identity_shapes(identity)
        for name, expected in (("condition", self.condition), ("seed", seed)):
            if name in identity and not _same_json(identity[name], expected):
                raise ValueError(
                    f"audit checkpoint identity {name} conflicts with expectation"
                )
        object.__setattr__(self, "checkpoint_uri", uri)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "tokens_seen", tokens_seen)
        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "training_config_sha256", config_sha256)
        object.__setattr__(self, "mesh_shape", mesh)
        object.__setattr__(self, "checkpoint_identity", identity)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> AuditCheckpointExpectation:
        """Parse one fail-closed YAML/JSON expectation entry."""

        record = _json_mapping(value, label="audit checkpoint expectation")
        if set(record) != _EXPECTATION_KEYS:
            raise ValueError("audit checkpoint expectation fields do not match schema")
        if record.get("schema_version") != AUDIT_EXPECTATION_SCHEMA:
            raise ValueError("unsupported audit checkpoint expectation schema")
        return cls(
            checkpoint_uri=record["checkpoint_uri"],
            condition=record["condition"],
            seed=record["seed"],
            tokens_seen=record["tokens_seen"],
            steps=record["steps"],
            training_config_sha256=record["training_config_sha256"],
            mesh_shape=record["mesh_shape"],
            checkpoint_identity=record["checkpoint_identity"],
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": AUDIT_EXPECTATION_SCHEMA,
            "checkpoint_uri": self.checkpoint_uri,
            "condition": self.condition,
            "seed": self.seed,
            "tokens_seen": self.tokens_seen,
            "steps": self.steps,
            "training_config_sha256": self.training_config_sha256,
            "mesh_shape": dict(self.mesh_shape),
            "checkpoint_identity": _json_mapping(
                self.checkpoint_identity, label="audit checkpoint identity"
            ),
        }

    @property
    def sha256(self) -> str:
        return _sha256(self.to_mapping(), label="audit checkpoint expectation")


@dataclass(frozen=True, slots=True)
class AuditCheckpointInspection:
    """Fresh read-only observation of one Orbax payload."""

    checkpoint_uri: str
    finalized: bool
    checkpoint_header: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class RestoredAuditCheckpoint:
    """State released only after pre/post direct-payload validation."""

    state: Any
    metadata: Mapping[str, Any]
    descriptor: Mapping[str, Any]


def _training_config_sha256(training_config: Mapping[str, Any]) -> str:
    # This is intentionally byte-identical to distributed._checkpoint_header.
    encoded = json.dumps(
        dict(training_config), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _checkpoint_identity(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metadata.items()
        if key not in _CHECKPOINT_OBSERVATION_KEYS
    }


def _validate_inspection(
    inspection: AuditCheckpointInspection,
    expectation: AuditCheckpointExpectation,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not isinstance(inspection, AuditCheckpointInspection):
        raise AuditCheckpointError(
            "audit checkpoint inspector returned the wrong result type"
        )
    if inspection.checkpoint_uri != expectation.checkpoint_uri:
        raise AuditCheckpointError(
            "audit inspector returned a different checkpoint URI"
        )
    if inspection.finalized is not True:
        raise AuditCheckpointError("audit checkpoint payload is not finalized")
    try:
        header = _json_mapping(
            inspection.checkpoint_header, label="Orbax checkpoint header"
        )
    except ValueError as error:
        raise AuditCheckpointError(
            "Orbax checkpoint header is not canonical"
        ) from error
    if set(header) != _HEADER_KEYS:
        raise AuditCheckpointError("Orbax checkpoint header fields do not match schema")
    if (
        isinstance(header.get("checkpoint_version"), bool)
        or header.get("checkpoint_version") != 1
    ):
        raise AuditCheckpointError("unsupported Orbax checkpoint version")

    try:
        training_config = _json_mapping(
            header["training_config"], label="Orbax TrainingConfig"
        )
    except ValueError as error:
        raise AuditCheckpointError("Orbax TrainingConfig is invalid") from error
    computed_config_sha256 = _training_config_sha256(training_config)
    if header.get("training_config_sha256") != computed_config_sha256:
        raise AuditCheckpointError("Orbax TrainingConfig digest is invalid")
    if computed_config_sha256 != expectation.training_config_sha256:
        raise AuditCheckpointError("Orbax TrainingConfig does not match expectation")
    sae_config = training_config.get("sae")
    if not isinstance(sae_config, Mapping):
        raise AuditCheckpointError("Orbax TrainingConfig has no SAE configuration")
    if not _same_json(sae_config.get("condition"), expectation.condition):
        raise AuditCheckpointError("Orbax TrainingConfig condition mismatch")
    if not _same_json(training_config.get("seed"), expectation.seed):
        raise AuditCheckpointError("Orbax TrainingConfig seed mismatch")

    try:
        mesh = _mesh_mapping(header["mesh_shape"])
    except ValueError as error:
        raise AuditCheckpointError("Orbax checkpoint mesh is invalid") from error
    if mesh != dict(expectation.mesh_shape):
        raise AuditCheckpointError("Orbax checkpoint mesh mismatch")

    try:
        metadata = _json_mapping(header["metadata"], label="Orbax custom metadata")
    except ValueError as error:
        raise AuditCheckpointError("Orbax custom metadata is invalid") from error
    for name, expected in (
        ("checkpoint_payload_uri", expectation.checkpoint_uri),
        ("condition", expectation.condition),
        ("seed", expectation.seed),
        ("tokens_seen", expectation.tokens_seen),
        ("steps", expectation.steps),
    ):
        if name not in metadata or not _same_json(metadata[name], expected):
            raise AuditCheckpointError(f"Orbax custom metadata {name} mismatch")
    for name, expected in expectation.checkpoint_identity.items():
        if name not in metadata or not _same_json(metadata[name], expected):
            raise AuditCheckpointError(
                f"Orbax custom identity {name} does not match expectation"
            )
    for name in (
        "optimizer_wall_seconds",
        "compile_seconds",
        "checkpoint_seconds",
    ):
        if name in metadata:
            value = metadata[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise AuditCheckpointError(f"Orbax custom metadata {name} is invalid")

    identity = _checkpoint_identity(metadata)
    try:
        _validate_identity_shapes(identity)
        if not _same_json(identity.get("condition"), expectation.condition):
            raise ValueError("checkpoint identity condition is invalid")
        if not _same_json(identity.get("seed"), expectation.seed):
            raise ValueError("checkpoint identity seed is invalid")
    except ValueError as error:
        raise AuditCheckpointError("Orbax custom run identity is invalid") from error
    payload_match = _PAYLOAD_SUFFIX_RE.search(expectation.checkpoint_uri[5:])
    if payload_match is None:
        raise AuditCheckpointError(
            "audit checkpoint URI is not an immutable SAE payload path"
        )
    identity_sha256 = _sha256(identity, label="checkpoint identity")
    if payload_match.group("identity") != identity_sha256:
        raise AuditCheckpointError(
            "checkpoint URI namespace does not match custom run identity"
        )
    if (
        int(payload_match.group("step")) != expectation.steps
        or int(payload_match.group("tokens")) != expectation.tokens_seen
    ):
        raise AuditCheckpointError(
            "checkpoint URI progress does not match frozen expectation"
        )
    return header, metadata, identity


def _descriptor_from_inspection(
    inspection: AuditCheckpointInspection,
    expectation: AuditCheckpointExpectation,
) -> dict[str, Any]:
    header, metadata, identity = _validate_inspection(inspection, expectation)
    descriptor: dict[str, Any] = {
        "schema_version": AUDIT_DESCRIPTOR_SCHEMA,
        "expectation_sha256": expectation.sha256,
        "checkpoint_uri": expectation.checkpoint_uri,
        "finalized": True,
        "checkpoint_version": 1,
        "condition": expectation.condition,
        "seed": expectation.seed,
        "tokens_seen": expectation.tokens_seen,
        "steps": expectation.steps,
        "training_config_sha256": expectation.training_config_sha256,
        "mesh_shape": dict(expectation.mesh_shape),
        "checkpoint_identity": identity,
        "checkpoint_identity_sha256": _sha256(identity, label="checkpoint identity"),
        "custom_metadata_sha256": _sha256(metadata, label="checkpoint custom metadata"),
        "checkpoint_header": header,
        "checkpoint_header_sha256": _sha256(header, label="checkpoint header"),
    }
    descriptor["descriptor_sha256"] = _sha256(
        descriptor, label="audit checkpoint descriptor"
    )
    return descriptor


def validate_audit_checkpoint_descriptor(
    descriptor: Mapping[str, Any],
    expectation: AuditCheckpointExpectation,
) -> dict[str, Any]:
    """Validate a persisted descriptor against its frozen expectation."""

    try:
        record = _json_mapping(descriptor, label="audit checkpoint descriptor")
    except ValueError as error:
        raise AuditCheckpointError("audit checkpoint descriptor is invalid") from error
    if set(record) != _DESCRIPTOR_KEYS:
        raise AuditCheckpointError(
            "audit checkpoint descriptor fields do not match schema"
        )
    if record.get("schema_version") != AUDIT_DESCRIPTOR_SCHEMA:
        raise AuditCheckpointError("unsupported audit checkpoint descriptor schema")
    supplied_sha256 = record.get("descriptor_sha256")
    if (
        not isinstance(supplied_sha256, str)
        or _SHA256_RE.fullmatch(supplied_sha256) is None
    ):
        raise AuditCheckpointError("audit checkpoint descriptor hash is invalid")
    unsigned = dict(record)
    del unsigned["descriptor_sha256"]
    if _sha256(unsigned, label="audit checkpoint descriptor") != supplied_sha256:
        raise AuditCheckpointError("audit checkpoint descriptor hash mismatch")
    if record.get("expectation_sha256") != expectation.sha256:
        raise AuditCheckpointError(
            "audit checkpoint descriptor belongs to another expectation"
        )
    rebuilt = _descriptor_from_inspection(
        AuditCheckpointInspection(
            checkpoint_uri=record.get("checkpoint_uri"),
            finalized=record.get("finalized"),
            checkpoint_header=record.get("checkpoint_header"),
        ),
        expectation,
    )
    if rebuilt != record:
        raise AuditCheckpointError("audit checkpoint descriptor fields disagree")
    return record


def _default_inspector(uri: str) -> AuditCheckpointInspection:
    try:
        import orbax.checkpoint as ocp
    except ImportError as error:  # pragma: no cover - TPU-only dependency
        raise RuntimeError(
            "audit checkpoint inspection requires `moe-sae[tpu]`"
        ) from error
    finalized = getattr(ocp.utils, "is_path_finalized", None)
    if finalized is None:
        finalized = getattr(ocp.utils, "is_checkpoint_finalized", None)
    if finalized is None:
        raise RuntimeError("installed Orbax has no public finalization check")
    is_finalized = bool(finalized(uri))
    checkpointer = ocp.StandardCheckpointer()
    try:
        metadata = checkpointer.metadata(uri).custom_metadata or {}
        header = dict(metadata)
    finally:
        checkpointer.close()
    return AuditCheckpointInspection(
        checkpoint_uri=uri,
        finalized=is_finalized,
        checkpoint_header=header,
    )


def _collective_decision(operation: str, error: Exception | None) -> None:
    # Keep audit failures synchronized with ordinary distributed checkpoint
    # save/restore failures.  The import is lazy so descriptor parsing remains
    # usable in the base CPU environment.
    from moe_sae.distributed import _checkpoint_phase_decision

    _checkpoint_phase_decision(operation, error)


def _collective_descriptor_consensus(descriptor_sha256: str) -> None:
    """Require every JAX process to have observed the identical descriptor."""

    import jax
    import numpy as np

    if jax.process_count() == 1:
        return
    from jax.experimental import multihost_utils

    local = np.frombuffer(bytes.fromhex(descriptor_sha256), dtype=np.uint8).copy()
    gathered = np.asarray(
        multihost_utils.process_allgather(local, tiled=False), dtype=np.uint8
    )
    local_error: Exception | None = None
    if gathered.size != jax.process_count() * local.size:
        local_error = AuditCheckpointError(
            "audit descriptor consensus returned the wrong shape"
        )
    else:
        rows = gathered.reshape(jax.process_count(), local.size)
        if not np.all(rows == rows[0]):
            local_error = AuditCheckpointError(
                "hosts observed different audit checkpoint descriptors"
            )
    _collective_decision("audit descriptor consensus", local_error)


def inspect_audit_checkpoint(
    uri: str,
    expectation: AuditCheckpointExpectation,
    inspector: Callable[[str], AuditCheckpointInspection] | None = None,
) -> dict[str, Any]:
    """Collectively inspect and describe one exact finalized GCS payload."""

    local_error: Exception | None = None
    descriptor: dict[str, Any] | None = None
    try:
        if not isinstance(expectation, AuditCheckpointExpectation):
            raise TypeError("expectation must be an AuditCheckpointExpectation")
        if _canonical_gcs_uri(uri) != expectation.checkpoint_uri:
            raise AuditCheckpointError(
                "explicit checkpoint URI does not match frozen expectation"
            )
        observation = (inspector or _default_inspector)(uri)
        descriptor = _descriptor_from_inspection(observation, expectation)
    except Exception as error:
        local_error = error
    _collective_decision("audit payload inspection", local_error)
    assert descriptor is not None
    _collective_descriptor_consensus(str(descriptor["descriptor_sha256"]))
    return descriptor


def _training_config_mapping(training_config: object) -> dict[str, Any]:
    if not dataclasses.is_dataclass(training_config) or isinstance(
        training_config, type
    ):
        raise TypeError("training_config must be a dataclass instance")
    value = dataclasses.asdict(training_config)
    return _json_mapping(value, label="restore TrainingConfig")


def _mesh_shape(mesh: object) -> dict[str, int]:
    shape = getattr(mesh, "shape", None)
    if not isinstance(shape, Mapping):
        raise TypeError("mesh must expose a shape mapping")
    return _mesh_mapping(shape)


def _state_scalar(state: object, name: str) -> int:
    value = getattr(state, name, None)
    try:
        import jax
        import numpy as np

        array = np.asarray(jax.device_get(value))
    except Exception as error:
        raise AuditCheckpointError(f"cannot read restored state {name}") from error
    if array.shape != () or not (
        np.issubdtype(array.dtype, np.integer)
        or np.issubdtype(array.dtype, np.unsignedinteger)
    ):
        raise AuditCheckpointError(f"restored state {name} is not an integer scalar")
    return int(array)


def _default_restore_payload(
    uri: str, template: object, mesh: object, training_config: object
) -> tuple[object, Mapping[str, Any]]:
    from moe_sae.distributed import _restore_distributed_payload

    return _restore_distributed_payload(uri, template, mesh, training_config)


def restore_audit_checkpoint_payload(
    uri: str,
    template: object,
    mesh: object,
    training_config: object,
    expectation: AuditCheckpointExpectation,
    *,
    inspector: Callable[[str], AuditCheckpointInspection] | None = None,
    restore_payload: Callable[
        [str, object, object, object], tuple[object, Mapping[str, Any]]
    ]
    | None = None,
) -> RestoredAuditCheckpoint:
    """Restore an explicit payload only after exact collective audit checks.

    The returned state is suitable for the ordinary distributed evaluator.
    A second live inspection after restore ensures the finalized metadata did
    not change across the operation.
    """

    descriptor = inspect_audit_checkpoint(uri, expectation, inspector)

    local_error: Exception | None = None
    try:
        config_mapping = _training_config_mapping(training_config)
        if (
            _training_config_sha256(config_mapping)
            != expectation.training_config_sha256
        ):
            raise AuditCheckpointError(
                "restore TrainingConfig does not match frozen expectation"
            )
        if config_mapping != descriptor["checkpoint_header"]["training_config"]:
            raise AuditCheckpointError(
                "restore TrainingConfig does not match Orbax header"
            )
        if _mesh_shape(mesh) != dict(expectation.mesh_shape):
            raise AuditCheckpointError("restore mesh does not match frozen expectation")
    except Exception as error:
        local_error = error
    _collective_decision("audit restore preparation", local_error)

    restored: object | None = None
    metadata: Mapping[str, Any] | None = None
    local_error = None
    try:
        restored, metadata = (restore_payload or _default_restore_payload)(
            uri, template, mesh, training_config
        )
        observed_metadata = _json_mapping(
            metadata, label="restored checkpoint metadata"
        )
        if not _same_json(
            observed_metadata, descriptor["checkpoint_header"]["metadata"]
        ):
            raise AuditCheckpointError(
                "restored metadata differs from inspected Orbax metadata"
            )
        if _state_scalar(restored, "tokens_seen") != expectation.tokens_seen:
            raise AuditCheckpointError("restored state token count mismatch")
        if _state_scalar(restored, "step") != expectation.steps:
            raise AuditCheckpointError("restored state step mismatch")
        metadata = observed_metadata
    except Exception as error:
        local_error = error
    _collective_decision("audit payload restore", local_error)
    assert restored is not None and metadata is not None

    after = inspect_audit_checkpoint(uri, expectation, inspector)
    local_error = None
    if after != descriptor:
        local_error = AuditCheckpointError(
            "audit checkpoint descriptor changed across restore"
        )
    _collective_decision("audit post-restore inspection", local_error)
    return RestoredAuditCheckpoint(
        state=restored,
        metadata=metadata,
        descriptor=descriptor,
    )
