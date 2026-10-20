from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from moe_sae.gate1_protocol import (
    GATE1_PROTOCOL_SCHEMA,
    GATE1_ROUTE_MODES,
    ROUTE_PERMUTATION_DOMAIN,
    Gate1ProtocolError,
    TokenRange,
    canonical_gate1_protocol_sha256,
    load_gate1_protocol,
    validate_gate1_protocol,
)

PROTOCOL_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "gemma4_gate1_audit.yaml"
)
EXPECTED_PROTOCOL_SHA256 = (
    "142102c565fe024e476d537f92e845dd149294c168b42324c16b619733bf094d"
)


def _valid_mapping() -> dict:
    value = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_authoritative_protocol_loads_with_canonical_identity() -> None:
    protocol = load_gate1_protocol(PROTOCOL_PATH)

    assert protocol.schema_version == GATE1_PROTOCOL_SCHEMA
    assert protocol.sha256 == EXPECTED_PROTOCOL_SHA256
    assert canonical_gate1_protocol_sha256(protocol) == protocol.sha256
    assert protocol.route_permutation.domain == ROUTE_PERMUTATION_DOMAIN
    assert protocol.matrix.router_ablation.route_modes == GATE1_ROUTE_MODES
    assert protocol.capture.train_tokens == 50_000_000
    assert protocol.capture.eval_tokens == 2_000_000
    assert protocol.splits["heldout"].ranges == (TokenRange(0, 2_000_000),)
    assert protocol.splits["train_stratified"].tokens == 2_000_000
    assert set(protocol.checkpoints) == {
        "A_seed1_5m",
        "B_seed1_5m",
        "F_seed1_5m",
        "A_seed1_40m",
        "A_seed1_50m",
        "A_seed2_50m",
        "A_seed3_50m",
        "B_seed1_50m",
        "B_seed2_50m",
        "B_seed3_50m",
    }
    assert protocol.checkpoints["A_seed1_5m"].steps == 306
    assert protocol.checkpoints["A_seed1_5m"].tokens_seen == 5_013_504
    assert protocol.checkpoints["A_seed1_50m"].steps == 3_052
    assert protocol.checkpoints["A_seed1_50m"].tokens_seen == 50_000_000
    assert (
        protocol.checkpoints["A_seed1_50m"].expectation.to_mapping()
        == protocol.to_mapping()["checkpoints"]["A_seed1_50m"]
    )
    assert len(protocol.checkpoints["B_seed3_50m"].run_identity_sha256) == 64
    with pytest.raises(TypeError):
        protocol.splits["new"] = protocol.splits["heldout"]  # type: ignore[index]


def test_canonical_hash_ignores_mapping_order_but_binds_semantics() -> None:
    original = _valid_mapping()
    reversed_top_level = dict(reversed(tuple(original.items())))

    assert (
        canonical_gate1_protocol_sha256(reversed_top_level) == EXPECTED_PROTOCOL_SHA256
    )

    changed = copy.deepcopy(original)
    changed["protocol_id"] = f"{changed['protocol_id']}-amended"
    assert canonical_gate1_protocol_sha256(changed) != EXPECTED_PROTOCOL_SHA256
    assert validate_gate1_protocol(original).to_mapping() == original


def test_json_round_trip_preserves_protocol_hash(tmp_path: Path) -> None:
    path = tmp_path / "gate1.json"
    path.write_text(json.dumps(_valid_mapping(), indent=2), encoding="utf-8")

    assert load_gate1_protocol(path).sha256 == EXPECTED_PROTOCOL_SHA256


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value.update(unexpected=True),
            "unknown keys: unexpected",
        ),
        (
            lambda value: value["capture"].pop("checkpoint_sha256"),
            "missing keys: checkpoint_sha256",
        ),
        (
            lambda value: value["capture"].update(config_sha256="A" * 64),
            "64 lowercase hexadecimal",
        ),
        (
            lambda value: value["capture"].update(config_sha256="0" * 64),
            "production Gemma config",
        ),
        (
            lambda value: value["capture"].update(train_tokens=49_000_000),
            "train_tokens must be 50000000",
        ),
        (
            lambda value: value["capture"].update(
                eval_manifest_sha256=value["capture"]["train_manifest_sha256"]
            ),
            "manifests must be distinct",
        ),
        (
            lambda value: value["route_permutation"].update(mode="random"),
            "mode must be",
        ),
        (
            lambda value: value["route_permutation"].update(domain="other"),
            "domain must be",
        ),
        (
            lambda value: value["route_permutation"].update(key=True),
            "key must be an integer",
        ),
        (
            lambda value: value["statistics"].update(cluster="token"),
            "cluster must be 'sequence_id'",
        ),
        (
            lambda value: value["statistics"].update(extra=1),
            "unknown keys: extra",
        ),
        (
            lambda value: value["statistics"].update(material_effect_ev=0.003),
            "must equal 10%",
        ),
    ],
)
def test_exact_schema_capture_route_and_statistics_fail_closed(
    mutation, message: str
) -> None:
    value = copy.deepcopy(_valid_mapping())
    mutation(value)
    with pytest.raises(Gate1ProtocolError, match=message):
        validate_gate1_protocol(value)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["splits"]["heldout"].update(ranges=[[0, 2_000_004]]),
            "exceeds its declared capture token count",
        ),
        (
            lambda value: value["splits"]["heldout"].update(ranges=[[4, 2_000_000]]),
            "complete declared eval capture",
        ),
        (
            lambda value: value["splits"]["train_stratified"]["ranges"].__setitem__(
                0, [0, 500_002]
            ),
            "divisible by four",
        ),
        (
            lambda value: value["splits"]["train_stratified"]["ranges"].__setitem__(
                1, [499_996, 1_000_000]
            ),
            "ranges must be disjoint",
        ),
        (
            lambda value: value["splits"]["train_stratified"]["ranges"].reverse(),
            "must be ordered",
        ),
        (
            lambda value: value["splits"]["train_stratified"].update(source="eval"),
            "source must be 'train'",
        ),
    ],
)
def test_split_ranges_are_partitionable_disjoint_and_in_bounds(
    mutation, message: str
) -> None:
    value = copy.deepcopy(_valid_mapping())
    mutation(value)
    with pytest.raises(Gate1ProtocolError, match=message):
        validate_gate1_protocol(value)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["checkpoints"]["A_seed1_5m"].update(condition="B"),
            "condition conflicts|disagrees with condition/seed",
        ),
        (
            lambda value: value["checkpoints"]["A_seed1_5m"].update(seed=True),
            "seed must be an integer",
        ),
        (
            lambda value: value["checkpoints"]["A_seed1_5m"].pop(
                "training_config_sha256"
            ),
            "expectation fields do not match schema",
        ),
        (
            lambda value: value["checkpoints"]["A_seed1_5m"].update(extra=True),
            "expectation fields do not match schema",
        ),
        (
            lambda value: value["checkpoints"]["A_seed1_5m"][
                "checkpoint_identity"
            ].update(extra=True),
            "frozen Gate-1 identity schema",
        ),
        (
            lambda value: value["checkpoints"]["A_seed1_5m"][
                "checkpoint_identity"
            ].update(capture_manifest_sha256="0" * 64),
            "does not match the capture contract",
        ),
        (
            lambda value: value["checkpoints"]["A_seed1_5m"].update(
                mesh_shape={"data": 4, "feature": 0}
            ),
            "mesh axes must have positive sizes",
        ),
        (
            lambda value: value["checkpoints"]["A_seed1_5m"].update(steps=307),
            "disagree with its milestone",
        ),
        (
            lambda value: value["checkpoints"]["A_seed1_5m"].update(
                checkpoint_uri=value["checkpoints"]["A_seed1_5m"][
                    "checkpoint_uri"
                ].replace("step-00000306", "step-00000307")
            ),
            "payload name disagrees",
        ),
        (
            lambda value: value["checkpoints"]["A_seed1_5m"].update(
                checkpoint_uri=value["checkpoints"]["A_seed1_5m"][
                    "checkpoint_uri"
                ].replace("/runs/a", "/runs/")
            ),
            "run identity",
        ),
        (
            lambda value: value["checkpoints"]["A_seed1_5m"].update(
                checkpoint_uri=value["checkpoints"]["A_seed1_5m"][
                    "checkpoint_uri"
                ].rsplit("/", 1)[0]
                + "/final",
            ),
            "immutable checkpoint payload",
        ),
        (
            lambda value: value["checkpoints"]["B_seed1_5m"].update(
                checkpoint_uri=value["checkpoints"]["A_seed1_5m"]["checkpoint_uri"]
            ),
            "payload name disagrees|URIs must be unique",
        ),
    ],
)
def test_checkpoint_labels_milestones_and_immutable_uris_agree(
    mutation, message: str
) -> None:
    value = copy.deepcopy(_valid_mapping())
    mutation(value)
    with pytest.raises(Gate1ProtocolError, match=message):
        validate_gate1_protocol(value)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["matrix"]["learning_curve"].update(route_mode=True),
            "route_mode must be a nonempty string",
        ),
        (
            lambda value: value["matrix"]["learning_curve"]["checkpoints"].__setitem__(
                0, "missing"
            ),
            "unknown references: missing",
        ),
        (
            lambda value: value["matrix"]["final_baselines"].update(
                splits=["train_stratified", "heldout"]
            ),
            "must be heldout and train_stratified",
        ),
        (
            lambda value: value["matrix"]["router_ablation"].update(
                route_modes=["true", "all_banks", "shuffled"]
            ),
            "route_modes must be exactly",
        ),
        (
            lambda value: value["matrix"]["router_ablation"]["checkpoints"].__setitem__(
                0, "B_seed1_50m"
            ),
            "frozen audit",
        ),
        (
            lambda value: value["matrix"]["router_ablation"].update(extra=[]),
            "unknown keys: extra",
        ),
    ],
)
def test_matrix_route_modes_and_references_are_exact(mutation, message: str) -> None:
    value = copy.deepcopy(_valid_mapping())
    mutation(value)
    with pytest.raises(Gate1ProtocolError, match=message):
        validate_gate1_protocol(value)


def test_duplicate_yaml_and_json_keys_are_rejected(tmp_path: Path) -> None:
    yaml_path = tmp_path / "duplicate.yaml"
    yaml_path.write_text(
        "schema_version: first\nschema_version: second\n", encoding="utf-8"
    )
    with pytest.raises(Gate1ProtocolError, match="duplicate YAML key"):
        load_gate1_protocol(yaml_path)

    json_path = tmp_path / "duplicate.json"
    json_path.write_text(
        '{"schema_version":"first","schema_version":"second"}',
        encoding="utf-8",
    )
    with pytest.raises(Gate1ProtocolError, match="duplicate JSON key"):
        load_gate1_protocol(json_path)
