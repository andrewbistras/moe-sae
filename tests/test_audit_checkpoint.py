from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import pytest

from moe_sae.audit_checkpoint import (
    AUDIT_DESCRIPTOR_SCHEMA,
    AUDIT_EXPECTATION_SCHEMA,
    AuditCheckpointError,
    AuditCheckpointExpectation,
    AuditCheckpointInspection,
    inspect_audit_checkpoint,
    restore_audit_checkpoint_payload,
    validate_audit_checkpoint_descriptor,
)
from moe_sae.pipeline import TrainingConfig
from moe_sae.sae import SAEConfig

IDENTITY = {
    "operation_sha256": "c" * 64,
    "config_sha256": "d" * 64,
    "capture_manifest_sha256": "e" * 64,
    "capture_checkpoint_sha256": "f" * 64,
    "condition": "A",
    "seed": 1,
    "run_instance_id": "1" * 32,
}
IDENTITY_SHA256 = hashlib.sha256(
    json.dumps(
        IDENTITY, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
).hexdigest()
URI = (
    f"gs://moe-sae-bucket/checkpoints/runs/{IDENTITY_SHA256}"
    "/step-00003052-tokens-000050000000-attempt-" + "b" * 32
)


def _training_config() -> TrainingConfig:
    return TrainingConfig(
        sae=SAEConfig(condition="A"),
        seed=1,
        a_capacity=1128,
    )


def _config_mapping(config: TrainingConfig) -> dict[str, Any]:
    return dataclasses.asdict(config)


def _config_sha256(config: TrainingConfig) -> str:
    encoded = json.dumps(
        _config_mapping(config), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _identity() -> dict[str, Any]:
    return dict(IDENTITY)


def _expectation(config: TrainingConfig | None = None) -> AuditCheckpointExpectation:
    config = config or _training_config()
    return AuditCheckpointExpectation(
        checkpoint_uri=URI,
        condition="A",
        seed=1,
        tokens_seen=50_000_000,
        steps=3_052,
        training_config_sha256=_config_sha256(config),
        mesh_shape={"data": 4, "feature": 4},
        checkpoint_identity={
            "capture_manifest_sha256": "e" * 64,
            "capture_checkpoint_sha256": "f" * 64,
            "run_instance_id": "1" * 32,
        },
    )


def _header(config: TrainingConfig | None = None) -> dict[str, Any]:
    config = config or _training_config()
    metadata = {
        **_identity(),
        "checkpoint_payload_uri": URI,
        "tokens_seen": 50_000_000,
        "steps": 3_052,
        "optimizer_wall_seconds": 6_688.0,
        "compile_seconds": 100.0,
        "checkpoint_seconds": 25.0,
    }
    return {
        "checkpoint_version": 1,
        "training_config": _config_mapping(config),
        "training_config_sha256": _config_sha256(config),
        "mesh_shape": {"data": 4, "feature": 4},
        "metadata": metadata,
    }


def _inspection(
    *,
    header: dict[str, Any] | None = None,
    finalized: bool = True,
    uri: str = URI,
) -> AuditCheckpointInspection:
    return AuditCheckpointInspection(
        checkpoint_uri=uri,
        finalized=finalized,
        checkpoint_header=header or _header(),
    )


def test_expectation_mapping_roundtrip_is_strict_and_content_addressed() -> None:
    expectation = _expectation()
    mapping = expectation.to_mapping()

    assert mapping["schema_version"] == AUDIT_EXPECTATION_SCHEMA
    assert AuditCheckpointExpectation.from_mapping(mapping) == expectation
    assert len(expectation.sha256) == 64

    unknown = dict(mapping)
    unknown["surprise"] = True
    with pytest.raises(ValueError, match="fields"):
        AuditCheckpointExpectation.from_mapping(unknown)

    wrong_schema = dict(mapping)
    wrong_schema["schema_version"] = "future"
    with pytest.raises(ValueError, match="unsupported"):
        AuditCheckpointExpectation.from_mapping(wrong_schema)


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"checkpoint_uri": "https://example.test/checkpoint"}, "gs://"),
        ({"checkpoint_uri": "gs://bucket/a//b"}, "non-canonical"),
        ({"condition": "C"}, "condition"),
        ({"seed": True}, "integer"),
        ({"training_config_sha256": "A" * 64}, "lowercase"),
        ({"mesh_shape": {"data": 4, "feature": 0}}, "positive"),
        ({"checkpoint_identity": {"run_instance_id": "x"}}, "run_instance"),
    ],
)
def test_expectation_rejects_ambiguous_or_malformed_identity(
    change: dict[str, Any], match: str
) -> None:
    values = {
        "checkpoint_uri": URI,
        "condition": "A",
        "seed": 1,
        "tokens_seen": 50_000_000,
        "steps": 3_052,
        "training_config_sha256": "a" * 64,
        "mesh_shape": {"data": 4, "feature": 4},
        "checkpoint_identity": {},
    }
    values.update(change)
    with pytest.raises(ValueError, match=match):
        AuditCheckpointExpectation(**values)


def test_inspection_builds_reproducible_complete_self_hashed_descriptor() -> None:
    expectation = _expectation()
    calls: list[str] = []

    def inspect(uri: str) -> AuditCheckpointInspection:
        calls.append(uri)
        return _inspection()

    first = inspect_audit_checkpoint(URI, expectation, inspect)
    second = inspect_audit_checkpoint(URI, expectation, inspect)

    assert first == second
    assert first["schema_version"] == AUDIT_DESCRIPTOR_SCHEMA
    assert first["checkpoint_uri"] == URI
    assert first["condition"] == "A"
    assert first["seed"] == 1
    assert first["tokens_seen"] == 50_000_000
    assert first["steps"] == 3_052
    assert first["checkpoint_identity"] == _identity()
    assert first["checkpoint_header"] == _header()
    assert len(first["descriptor_sha256"]) == 64
    assert validate_audit_checkpoint_descriptor(first, expectation) == first
    assert calls == [URI, URI]


def test_descriptor_tampering_fails_even_when_semantic_fields_still_look_valid() -> (
    None
):
    expectation = _expectation()
    descriptor = inspect_audit_checkpoint(URI, expectation, lambda _uri: _inspection())
    tampered = dict(descriptor)
    tampered["tokens_seen"] = 40_000_000

    with pytest.raises(AuditCheckpointError, match="hash mismatch"):
        validate_audit_checkpoint_descriptor(tampered, expectation)


def test_validly_rehashed_descriptor_cannot_disagree_with_bound_header() -> None:
    expectation = _expectation()
    descriptor = inspect_audit_checkpoint(URI, expectation, lambda _uri: _inspection())
    tampered = dict(descriptor)
    tampered["tokens_seen"] = 40_000_000
    unsigned = dict(tampered)
    unsigned.pop("descriptor_sha256")
    tampered["descriptor_sha256"] = hashlib.sha256(
        json.dumps(
            unsigned,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()

    with pytest.raises(AuditCheckpointError, match="fields disagree"):
        validate_audit_checkpoint_descriptor(tampered, expectation)


def test_explicit_uri_must_match_before_inspector_is_called() -> None:
    called = False

    def inspect(_uri: str) -> AuditCheckpointInspection:
        nonlocal called
        called = True
        return _inspection()

    with pytest.raises(RuntimeError, match="audit payload inspection") as caught:
        inspect_audit_checkpoint(URI + "-other", _expectation(), inspect)

    assert called is False
    assert "does not match frozen expectation" in str(caught.value.__cause__)


@pytest.mark.parametrize(
    ("mutate", "cause"),
    [
        (lambda value: dataclasses.replace(value, finalized=False), "not finalized"),
        (
            lambda value: dataclasses.replace(value, checkpoint_uri=URI + "-different"),
            "different checkpoint URI",
        ),
        (
            lambda value: dataclasses.replace(
                value,
                checkpoint_header={
                    **value.checkpoint_header,
                    "training_config_sha256": "0" * 64,
                },
            ),
            "digest",
        ),
        (
            lambda value: dataclasses.replace(
                value,
                checkpoint_header={
                    **value.checkpoint_header,
                    "mesh_shape": {"data": 8, "feature": 2},
                },
            ),
            "mesh mismatch",
        ),
        (
            lambda value: dataclasses.replace(
                value,
                checkpoint_header={
                    **value.checkpoint_header,
                    "metadata": {
                        **value.checkpoint_header["metadata"],
                        "run_instance_id": "2" * 32,
                    },
                },
            ),
            "run_instance_id",
        ),
    ],
)
def test_inspection_fails_closed_on_finalization_header_and_identity(
    mutate: Any, cause: str
) -> None:
    expectation = _expectation()

    with pytest.raises(RuntimeError, match="audit payload inspection") as caught:
        inspect_audit_checkpoint(URI, expectation, lambda _uri: mutate(_inspection()))

    assert caught.value.__cause__ is not None
    assert cause in str(caught.value.__cause__)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seed", True),
        ("steps", 3_052.0),
        ("tokens_seen", 50_000_000.0),
    ],
)
def test_custom_progress_metadata_requires_exact_json_types(
    field: str, value: Any
) -> None:
    header = _header()
    header["metadata"][field] = value

    with pytest.raises(RuntimeError, match="audit payload inspection") as caught:
        inspect_audit_checkpoint(
            URI, _expectation(), lambda _uri: _inspection(header=header)
        )

    assert f"custom metadata {field} mismatch" in str(caught.value.__cause__)


def test_uri_namespace_must_match_complete_custom_identity() -> None:
    wrong_uri = URI.replace(IDENTITY_SHA256, "0" * 64)
    expectation_mapping = _expectation().to_mapping()
    expectation_mapping["checkpoint_uri"] = wrong_uri
    expectation = AuditCheckpointExpectation.from_mapping(expectation_mapping)
    header = _header()
    header["metadata"]["checkpoint_payload_uri"] = wrong_uri

    with pytest.raises(RuntimeError, match="audit payload inspection") as caught:
        inspect_audit_checkpoint(
            wrong_uri,
            expectation,
            lambda _uri: _inspection(header=header, uri=wrong_uri),
        )

    assert "namespace does not match" in str(caught.value.__cause__)


@dataclass(frozen=True)
class _Mesh:
    shape: dict[str, int]


@dataclass(frozen=True)
class _State:
    step: Any
    tokens_seen: Any


def test_direct_restore_reinspects_and_releases_state_for_evaluation() -> None:
    config = _training_config()
    expectation = _expectation(config)
    inspections: list[str] = []
    restores: list[tuple[str, object, object, object]] = []
    template = object()
    mesh = _Mesh({"data": 4, "feature": 4})
    state = _State(
        step=jnp.asarray(3_052, dtype=jnp.int32),
        tokens_seen=jnp.asarray(50_000_000, dtype=jnp.int32),
    )

    def inspect(uri: str) -> AuditCheckpointInspection:
        inspections.append(uri)
        return _inspection()

    def restore(
        uri: str, target: object, observed_mesh: object, observed_config: object
    ) -> tuple[object, dict[str, Any]]:
        restores.append((uri, target, observed_mesh, observed_config))
        return state, _header()["metadata"]

    result = restore_audit_checkpoint_payload(
        URI,
        template,
        mesh,
        config,
        expectation,
        inspector=inspect,
        restore_payload=restore,
    )

    assert result.state is state
    assert result.metadata == _header()["metadata"]
    assert result.descriptor["descriptor_sha256"]
    assert inspections == [URI, URI]
    assert restores == [(URI, template, mesh, config)]


def test_direct_restore_rejects_wrong_state_and_metadata_collectively() -> None:
    config = _training_config()
    expectation = _expectation(config)
    mesh = _Mesh({"data": 4, "feature": 4})

    def wrong_state(
        *_args: object,
    ) -> tuple[object, dict[str, Any]]:
        return (
            _State(
                step=jnp.asarray(3_051),
                tokens_seen=jnp.asarray(50_000_000),
            ),
            _header()["metadata"],
        )

    with pytest.raises(RuntimeError, match="audit payload restore") as caught:
        restore_audit_checkpoint_payload(
            URI,
            object(),
            mesh,
            config,
            expectation,
            inspector=lambda _uri: _inspection(),
            restore_payload=wrong_state,
        )
    assert isinstance(caught.value.__cause__, AuditCheckpointError)
    assert "step mismatch" in str(caught.value.__cause__)

    def wrong_metadata(
        *_args: object,
    ) -> tuple[object, dict[str, Any]]:
        return (
            _State(
                step=jnp.asarray(3_052),
                tokens_seen=jnp.asarray(50_000_000),
            ),
            {**_header()["metadata"], "condition": "B"},
        )

    with pytest.raises(RuntimeError, match="audit payload restore") as caught:
        restore_audit_checkpoint_payload(
            URI,
            object(),
            mesh,
            config,
            expectation,
            inspector=lambda _uri: _inspection(),
            restore_payload=wrong_metadata,
        )
    assert isinstance(caught.value.__cause__, AuditCheckpointError)
    assert "metadata differs" in str(caught.value.__cause__)


def test_direct_restore_detects_header_change_after_restore() -> None:
    config = _training_config()
    expectation = _expectation(config)
    observations = [_inspection(), _inspection(header=_header())]
    observations[1].checkpoint_header["metadata"]["checkpoint_seconds"] = 26.0

    def inspect(_uri: str) -> AuditCheckpointInspection:
        return observations.pop(0)

    def restore(
        *_args: object,
    ) -> tuple[object, dict[str, Any]]:
        return (
            _State(
                step=jnp.asarray(3_052),
                tokens_seen=jnp.asarray(50_000_000),
            ),
            _header()["metadata"],
        )

    with pytest.raises(RuntimeError, match="post-restore inspection") as caught:
        restore_audit_checkpoint_payload(
            URI,
            object(),
            _Mesh({"data": 4, "feature": 4}),
            config,
            expectation,
            inspector=inspect,
            restore_payload=restore,
        )
    assert isinstance(caught.value.__cause__, AuditCheckpointError)
    assert "changed across restore" in str(caught.value.__cause__)
