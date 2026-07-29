from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

import moe_sae.experiment5_routed_ladder_v3_provenance as provenance_module
from moe_sae.contribution_artifacts import canonical_sha256
from moe_sae.direct_horizon_campaign import PROTOCOL_ID as EXPERIMENT4_PROTOCOL_ID
from moe_sae.experiment5_campaign import EvidencePath
from moe_sae.experiment5_routed_ladder_campaign import (
    RoutedLadderCampaignPlanError,
)
from moe_sae.experiment5_routed_ladder_v3_config import (
    A256_NATIVE_CONFIG_SHA256,
    PROTOCOL_ID,
)
from moe_sae.experiment5_routed_ladder_v3_provenance import (
    A256_PROVENANCE_ADDENDUM_ID,
    A256_PROVENANCE_ADDENDUM_SCHEMA_VERSION,
    MILESTONES,
    A256AddendumCheckpoint,
    A256AddendumCost,
    A256AddendumPanel,
    build_a256_provenance_addendum,
    load_a256_provenance_addendum,
    parse_a256_provenance_addendum,
    validate_a256_addendum_panel,
    write_a256_provenance_addendum,
)


def _evidence(path: Path, remote: str, digest: str) -> dict[str, str]:
    return {
        "controller_path": str(path.resolve()),
        "remote_path": remote,
        "sha256": digest,
    }


def _scope() -> dict:
    return {
        "condition": "A256Native",
        "evidentiary_role": "panel_native_descriptive_context_only",
        "included_in_primary_a512_inference": False,
        "included_in_primary_a512_estimand": False,
        "cross_panel_pooling_allowed": False,
        "across_seed_inference_allowed": False,
        "allowed_use": "within_panel_historical_scale_context",
        "prohibited_uses": [
            "matched_a512_activation_policy_contrast",
            "cross_panel_activation_policy_contrast",
            "three_seed_inference",
            "replacement_for_any_a512_cell",
        ],
    }


def _manifest(tmp_path: Path) -> dict:
    panels = []
    for panel_index, panel in enumerate(("sae", "transcoder")):
        is_sae = panel == "sae"
        declared = A256_NATIVE_CONFIG_SHA256[panel]
        observed = declared if is_sae else "8" * 64
        raw_sha = f"{panel_index + 1:x}" * 64
        root = tmp_path / panel
        checkpoints = []
        for milestone_index, milestone in enumerate(MILESTONES):
            pointer_file_sha = (
                f"{panel_index * len(MILESTONES) + milestone_index + 3:x}"
                * 64
            )
            pointer_sha = (
                f"{panel_index * len(MILESTONES) + milestone_index + 9:x}"
                * 64
            )
            checkpoints.append(
                {
                    "checkpoint_tokens": milestone,
                    "checkpoint_pointer": _evidence(
                        root / "checkpoints" / f"{milestone}.json",
                        f"/remote/{panel}/checkpoints/{milestone}.json",
                        pointer_file_sha,
                    ),
                    "checkpoint_pointer_sha256": pointer_sha,
                }
            )
        marginal = 100.0 + panel_index
        panels.append(
            {
                "panel": panel,
                "target_condition": "A256Native",
                "source_condition": "A",
                "source_label": (
                    "A256ReLU" if is_sae else "A256Softplus"
                ),
                "seed": 0,
                "encoder_activation": "relu" if is_sae else "softplus",
                "selection_policy": (
                    "strict_positive_topk"
                    if is_sae
                    else "exact_positive_topk"
                ),
                "declared_source_config_semantic_sha256": declared,
                "observed_source_config_semantic_sha256": observed,
                "observed_source_config_file_sha256": raw_sha,
                "declaration_status": (
                    "exact_match" if is_sae else "mismatch"
                ),
                "disposition": (
                    "contextual_reference_verified"
                    if is_sae
                    else "contextual_reference_corrected_by_observed_lineage"
                ),
                "training_source_revision": "a" * 40,
                "source_config": _evidence(
                    root / "config.yaml",
                    f"/remote/{panel}/config.yaml",
                    raw_sha,
                ),
                "run_root_remote": f"/remote/{panel}/runs/a",
                "checkpoint_payload_root_gcs": f"gs://bucket/{panel}/a",
                "run_contract": _evidence(
                    root / "run-contract.json",
                    f"/remote/{panel}/runs/a/run-contract.json",
                    "b" * 64,
                ),
                "run_contract_sha256": "c" * 64,
                "training_result": _evidence(
                    root / "training-result.json",
                    f"/remote/{panel}/runs/a/training-result.json",
                    "d" * 64,
                ),
                "training_result_sha256": "e" * 64,
                "cost": {
                    "resource_id": f"slice-{panel}",
                    "accelerator_type": "v4-32",
                    "chip_count": 16,
                    "marginal_wall_seconds": marginal,
                    "chip_seconds": 16 * marginal,
                    "chip_seconds_scope": "optimizer_training",
                    "chip_seconds_excludes": [
                        "compile",
                        "checkpoint_serialization",
                    ],
                },
                "checkpoints": checkpoints,
            }
        )
    body = {
        "schema_version": A256_PROVENANCE_ADDENDUM_SCHEMA_VERSION,
        "addendum_id": A256_PROVENANCE_ADDENDUM_ID,
        "source_protocol_id": EXPERIMENT4_PROTOCOL_ID,
        "target_protocol_id": PROTOCOL_ID,
        "source_plan_sha256": "f" * 64,
        "scientific_scope": _scope(),
        "panels": panels,
    }
    return {**body, "manifest_sha256": canonical_sha256(body)}


def _rehash(manifest: dict) -> None:
    body = {
        key: value
        for key, value in manifest.items()
        if key != "manifest_sha256"
    }
    manifest["manifest_sha256"] = canonical_sha256(body)


def test_addendum_pins_honest_mismatch_and_context_only_scope(
    tmp_path: Path,
) -> None:
    parsed, panels = parse_a256_provenance_addendum(_manifest(tmp_path))
    by_panel = {panel.panel: panel for panel in panels}

    assert parsed["scientific_scope"] == _scope()
    assert by_panel["sae"].declaration_status == "exact_match"
    assert by_panel["transcoder"].declaration_status == "mismatch"
    assert (
        by_panel["transcoder"].declared_source_config_semantic_sha256
        == A256_NATIVE_CONFIG_SHA256["transcoder"]
    )
    assert (
        by_panel["transcoder"].observed_source_config_semantic_sha256
        == "8" * 64
    )
    assert [item.checkpoint_tokens for item in by_panel["sae"].checkpoints] == [
        *MILESTONES
    ]
    assert by_panel["sae"].cost.chip_seconds == 1600.0


def test_addendum_rejects_rewriting_frozen_v3_declaration(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    transcoder = manifest["panels"][1]
    transcoder["declared_source_config_semantic_sha256"] = (
        transcoder["observed_source_config_semantic_sha256"]
    )
    transcoder["declaration_status"] = "exact_match"
    transcoder["disposition"] = "contextual_reference_verified"
    _rehash(manifest)

    with pytest.raises(
        RoutedLadderCampaignPlanError,
        match="frozen v3 declaration",
    ):
        parse_a256_provenance_addendum(manifest)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("included_in_primary_a512_inference", True),
        ("included_in_primary_a512_estimand", True),
        ("cross_panel_pooling_allowed", True),
        ("across_seed_inference_allowed", True),
    ),
)
def test_addendum_cannot_expand_context_into_inference(
    tmp_path: Path,
    field: str,
    value: bool,
) -> None:
    manifest = _manifest(tmp_path)
    manifest["scientific_scope"][field] = value
    _rehash(manifest)

    with pytest.raises(
        RoutedLadderCampaignPlanError,
        match="scientific scope changed",
    ):
        parse_a256_provenance_addendum(manifest)


def test_addendum_requires_transcoder_mismatch(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    transcoder = manifest["panels"][1]
    transcoder["observed_source_config_semantic_sha256"] = (
        transcoder["declared_source_config_semantic_sha256"]
    )
    transcoder["declaration_status"] = "exact_match"
    transcoder["disposition"] = "contextual_reference_verified"
    _rehash(manifest)

    with pytest.raises(
        RoutedLadderCampaignPlanError,
        match="transcoder mismatch",
    ):
        parse_a256_provenance_addendum(manifest)


def test_addendum_rejects_raw_hash_not_bound_to_config_evidence(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    manifest["panels"][1]["observed_source_config_file_sha256"] = "0" * 64
    _rehash(manifest)

    with pytest.raises(
        RoutedLadderCampaignPlanError,
        match="raw config hash differs",
    ):
        parse_a256_provenance_addendum(manifest)


def test_addendum_is_create_only_and_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "contracts" / "a256-provenance-addendum.yaml"
    manifest = _manifest(tmp_path)

    parsed, panels = write_a256_provenance_addendum(path, manifest)

    assert parsed["manifest_sha256"] == manifest["manifest_sha256"]
    assert len(panels) == 2
    assert (
        load_a256_provenance_addendum(path)["manifest_sha256"]
        == manifest["manifest_sha256"]
    )
    with pytest.raises(
        RoutedLadderCampaignPlanError,
        match="refusing to overwrite",
    ):
        write_a256_provenance_addendum(path, manifest)


def test_addendum_self_hash_is_immutable(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    manifest["panels"][0]["cost"]["chip_seconds"] += 1

    with pytest.raises(
        RoutedLadderCampaignPlanError,
        match="self-hash changed",
    ):
        parse_a256_provenance_addendum(manifest)


def test_builder_derives_full_observed_hashes_from_historical_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    revision = "a" * 40
    trajectories = []
    config_payloads = {}
    for panel in ("sae", "transcoder"):
        run_id = f"horizon10m-{panel}-a-seed0"
        completion = (
            tmp_path
            / "plane"
            / "controller"
            / "training"
            / panel
            / run_id
            / "training-result.json"
        )
        config_remote = f"/remote/plane/controller/contracts/{panel}.yaml"
        controller_config = tmp_path / "plane" / "contracts" / f"{panel}.yaml"
        controller_config.parent.mkdir(parents=True, exist_ok=True)
        config_payload = f"historical-finalized-{panel}-config\n".encode()
        controller_config.write_bytes(config_payload)
        config_payloads[panel] = config_payload
        trajectories.append(
            SimpleNamespace(
                scientific_model=panel,
                condition="A",
                seed=0,
                completion_path_controller=completion,
                control_root_remote=(
                    f"/remote/plane/controller/training/{panel}"
                ),
                config_remote=config_remote,
                run_id=run_id,
                checkpoint_root_gcs=f"gs://bucket/{panel}/a",
                resource_id=f"slice-{panel}",
            )
        )

    class _Plan:
        plan_sha256 = "f" * 64

        def __init__(self) -> None:
            self.trajectories = tuple(trajectories)

        def environment_dict(self) -> dict[str, str]:
            return {"TRAINING_PROJECT_COMMIT": revision}

        @staticmethod
        def resource(resource_id: str) -> SimpleNamespace:
            return SimpleNamespace(
                resource_id=resource_id,
                accelerator_type="v4-32",
            )

    def _config(
        path: Path,
        *,
        expected_scientific_model: str,
    ) -> SimpleNamespace:
        assert path.read_bytes() == config_payloads[expected_scientific_model]
        return SimpleNamespace(
            config_hash=(
                A256_NATIVE_CONFIG_SHA256["sae"]
                if expected_scientific_model == "sae"
                else "8" * 64
            )
        )

    def _artifact(path: Path, *, hash_field: str) -> tuple[dict, bytes]:
        panel = "transcoder" if "transcoder" in str(path) else "sae"
        if path.name == "training-result.json":
            value = {
                "result_sha256": "1" * 64,
                "marginal_wall_seconds": 100.0,
                "chip_seconds": 1600.0,
                "chip_seconds_scope": "optimizer_training",
                "chip_seconds_excludes": ["compile", "checkpoint"],
                "source_revision": revision,
                "run_identity": {
                    "plan": {
                        "phase_name": "learning_horizon_10m",
                        "condition": "A",
                        "seed": 0,
                        "global_microbatch_tokens": 4096,
                        "microbatches_per_update": 4,
                        "total_tokens": 10_000_000,
                        "planned_updates": 611,
                    }
                },
            }
        elif path.name == "run-contract.json":
            value = {
                "run_contract_sha256": "2" * 64,
                "project_commit": revision,
                "phase": "learning_horizon_10m",
                "scientific_model": panel,
                "condition": "A",
                "seed": 0,
                "config_file_sha256": hashlib.sha256(
                    config_payloads[panel]
                ).hexdigest(),
                "checkpoint_payload_root": f"gs://bucket/{panel}/a",
            }
        else:
            milestone = int(path.stem.split("-")[-1])
            value = {
                "pointer_sha256": hashlib.sha256(
                    str(milestone).encode()
                ).hexdigest()
            }
        assert hash_field in value
        return value, f"{panel}:{path.name}".encode()

    validated = []
    monkeypatch.setattr(
        provenance_module,
        "inspect_horizon_training_completion",
        lambda _task: object(),
    )
    monkeypatch.setattr(
        provenance_module,
        "load_learning_horizon_config",
        _config,
    )
    monkeypatch.setattr(
        provenance_module,
        "_read_self_hashed_json",
        _artifact,
    )
    monkeypatch.setattr(
        provenance_module,
        "validate_a256_addendum_panel",
        validated.append,
    )

    manifest = build_a256_provenance_addendum(_Plan())
    _parsed, panels = parse_a256_provenance_addendum(manifest)
    by_panel = {panel.panel: panel for panel in panels}

    assert (
        by_panel["transcoder"].observed_source_config_semantic_sha256
        == "8" * 64
    )
    assert (
        by_panel["transcoder"].observed_source_config_file_sha256
        == hashlib.sha256(config_payloads["transcoder"]).hexdigest()
    )
    assert (
        by_panel["transcoder"].observed_source_config_file_sha256
        != by_panel["transcoder"].observed_source_config_semantic_sha256
    )
    assert [panel.panel for panel in validated] == ["sae", "transcoder"]


def test_live_v1_training_result_binds_revision_through_run_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The historical result predates an embedded source_revision field."""

    revision = "a" * 40
    config_path = tmp_path / "sae.yaml"
    config_payload = b"historical-sae-config\n"
    config_path.write_bytes(config_payload)
    contract_path = tmp_path / "run-contract.json"
    result_path = tmp_path / "training-result.json"
    contract_payload = b"contract-payload"
    result_payload = b"result-payload"
    contract = {
        "run_contract_sha256": "1" * 64,
        "project_commit": revision,
        "phase": "learning_horizon_10m",
        "scientific_model": "sae",
        "condition": "A",
        "seed": 0,
        "config_file_sha256": hashlib.sha256(config_payload).hexdigest(),
        "checkpoint_payload_root": "gs://bucket/sae/a",
    }
    result = {
        "result_sha256": "2" * 64,
        "schema_version": "gemma4-contribution-sae-training-result-v1",
        "status": "completed",
        "tokens": 10_000_000,
        # Deliberately no source_revision: this is the live v1 schema.
        "marginal_wall_seconds": 100.0,
        "chip_seconds": 1600.0,
        "chip_seconds_scope": "optimizer_training",
        "chip_seconds_excludes": ["compile", "checkpoint"],
        "run_identity": {
            "plan": {
                "phase_name": "learning_horizon_10m",
                "condition": "A",
                "seed": 0,
                "global_microbatch_tokens": 4096,
                "microbatches_per_update": 4,
                "total_tokens": 10_000_000,
                "planned_updates": 611,
            }
        },
    }
    pointer_rows = {}
    checkpoints = []
    for index, milestone in enumerate(MILESTONES):
        pointer_path = tmp_path / f"pointer-{milestone}.json"
        batch = 4096 * 4
        step = (milestone + batch - 1) // batch
        tokens = min(step * batch, 10_000_000)
        pointer = {
            "pointer_sha256": f"{index + 3:x}" * 64,
            "step": step,
            "tokens_seen": tokens,
        }
        payload = f"pointer-{milestone}".encode()
        pointer_rows[pointer_path] = (pointer, payload)
        checkpoints.append(
            A256AddendumCheckpoint(
                checkpoint_tokens=milestone,
                checkpoint_pointer=EvidencePath(
                    controller_path=pointer_path,
                    remote_path=f"/remote/pointer-{milestone}.json",
                    sha256=hashlib.sha256(payload).hexdigest(),
                ),
                checkpoint_pointer_sha256=pointer["pointer_sha256"],
            )
        )
    panel = A256AddendumPanel(
        panel="sae",
        source_label="A256ReLU",
        seed=0,
        encoder_activation="relu",
        selection_policy="strict_positive_topk",
        declared_source_config_semantic_sha256=(
            A256_NATIVE_CONFIG_SHA256["sae"]
        ),
        observed_source_config_semantic_sha256=(
            A256_NATIVE_CONFIG_SHA256["sae"]
        ),
        observed_source_config_file_sha256=hashlib.sha256(
            config_payload
        ).hexdigest(),
        declaration_status="exact_match",
        disposition="contextual_reference_verified",
        training_source_revision=revision,
        source_config=EvidencePath(
            controller_path=config_path,
            remote_path="/remote/sae.yaml",
            sha256=hashlib.sha256(config_payload).hexdigest(),
        ),
        run_root_remote="/remote/runs/sae-a",
        checkpoint_payload_root_gcs="gs://bucket/sae/a",
        run_contract=EvidencePath(
            controller_path=contract_path,
            remote_path="/remote/run-contract.json",
            sha256=hashlib.sha256(contract_payload).hexdigest(),
        ),
        run_contract_sha256=contract["run_contract_sha256"],
        training_result=EvidencePath(
            controller_path=result_path,
            remote_path="/remote/training-result.json",
            sha256=hashlib.sha256(result_payload).hexdigest(),
        ),
        training_result_sha256=result["result_sha256"],
        cost=A256AddendumCost(
            resource_id="slice-sae",
            accelerator_type="v4-32",
            chip_count=16,
            marginal_wall_seconds=100.0,
            chip_seconds=1600.0,
            chip_seconds_scope="optimizer_training",
            chip_seconds_excludes=("compile", "checkpoint"),
        ),
        checkpoints=tuple(checkpoints),
    )

    def _artifact(path: Path, *, hash_field: str) -> tuple[dict, bytes]:
        if path == contract_path:
            value, payload = contract, contract_payload
        elif path == result_path:
            value, payload = result, result_payload
        else:
            value, payload = pointer_rows[path]
        assert hash_field in value
        return value, payload

    monkeypatch.setattr(
        provenance_module,
        "load_learning_horizon_config",
        lambda *_args, **_kwargs: SimpleNamespace(
            config_hash=A256_NATIVE_CONFIG_SHA256["sae"],
            encoder_activation="relu",
            layouts=(
                SimpleNamespace(
                    condition="A",
                    bank_identity="true_expert_id",
                    bank_count=128,
                    bank_width=256,
                    components_per_token=8,
                    topk_per_component=2,
                    target="raw_selected_expert_output",
                ),
            ),
        ),
    )
    monkeypatch.setattr(
        provenance_module,
        "_read_self_hashed_json",
        _artifact,
    )
    monkeypatch.setattr(
        provenance_module,
        "validate_pointer",
        lambda pointer, **_kwargs: pointer,
    )

    validate_a256_addendum_panel(panel)
