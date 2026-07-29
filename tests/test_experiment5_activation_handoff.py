from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/tpu/handoff_experiment5_activation_ladder.py"
HANDOFF_UNIT = (
    ROOT
    / "deploy/systemd/"
    "moe-sae-experiment5-activation-handoff-4472ff228b43-a1.service"
)


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "experiment5_activation_handoff",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


handoff = _load_script()


def _spec(tmp_path: Path):
    staged = tmp_path / "staged"
    canonical = tmp_path / "canonical"
    staged.mkdir()
    canonical.mkdir()
    (staged / ".venv/bin").mkdir(parents=True)
    (canonical / ".venv/bin").mkdir(parents=True)
    for path in (
        staged / ".venv/bin/python",
        canonical / ".venv/bin/python-direct",
    ):
        path.write_text("#!/bin/sh\n", encoding="ascii")
        path.chmod(0o700)
    for path in (
        tmp_path / "core.yaml",
        tmp_path / "r8.yaml",
        tmp_path / "bundle",
    ):
        path.write_text("sealed\n", encoding="ascii")
    extension = tmp_path / "extension"
    return handoff.ProductionSpec(
        core_plan=tmp_path / "core.yaml",
        staged_checkout=staged,
        canonical_checkout=canonical,
        deploy_bundle=tmp_path / "bundle",
        r8_plan=tmp_path / "r8.yaml",
        extension_root=extension,
        handoff_root=extension / "handoff",
        contract_root=extension / "contracts",
        h_reference_manifest=extension / "contracts/h.yaml",
    )


def test_production_contract_is_the_frozen_core_to_activation_transition() -> None:
    spec = handoff.ProductionSpec()
    assert spec.core_commit == "674cecbb19ba9130a81a4e27249c9862de08ad17"
    assert (
        spec.extension_commit
        == "4472ff228b43d272caeac6a31b547dbe11e7e2be"
    )
    assert str(spec.core_plan).endswith(
        "campaign-ready-three-v1-674cecbb19ba-a1.yaml"
    )
    assert str(spec.staged_checkout).endswith(
        "moe-sae-activation-4472ff228b43"
    )
    assert str(spec.staged_python).endswith(".venv/bin/python")
    assert [row.tpu_name for row in spec.resources] == [
        "gemma4-horizon-v5e-usc-a",
        "gemma4-horizon-harvest-0728",
        "gemma4-horizon-v4-spot-trans-b",
    ]


def test_handoff_unit_is_persistent_and_uses_staged_frozen_source() -> None:
    text = HANDOFF_UNIT.read_text(encoding="ascii")
    assert "WantedBy=default.target" in text
    assert "Restart=on-failure" in text
    assert "/moe-sae-activation-4472ff228b43/.venv/bin/python" in text
    assert "handoff_experiment5_activation_ladder.py run" in text
    assert "python-direct" not in text


def test_campaign_unit_is_persistent_distinct_and_runs_complete_plan(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    plan = tmp_path / "activation-plan.yaml"
    text = handoff.campaign_unit_text(plan=plan, attempt="a1", spec=spec)
    assert "WantedBy=default.target" in text
    assert "Restart=on-failure" in text
    assert "run_experiment5_routed_ladder_campaign.py run" in text
    assert "--stop-after-training" not in text
    assert handoff.campaign_unit_name("a1") == (
        "moe-sae-experiment5-activation-4472ff228b43-a1.service"
    )


def test_missing_core_final_cannot_reach_any_mutating_phase(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec(tmp_path)
    operator = handoff.ActivationHandoff(spec)
    events: list[str] = []
    monkeypatch.setattr(
        operator,
        "validate_payload",
        lambda: {
            "core_plan_sha256": "1" * 64,
            "r8_plan_sha256": "2" * 64,
            "bundle_sha256": "3" * 64,
            "extension_commit": spec.extension_commit,
        },
    )

    def refuse(*, wait: bool):
        del wait
        events.append("gate")
        raise handoff.HandoffError("core is not sealed")

    monkeypatch.setattr(operator, "seal_core_gate", refuse)
    monkeypatch.setattr(
        operator,
        "deploy",
        lambda _gate: events.append("deploy"),
    )
    with pytest.raises(handoff.HandoffError, match="core is not sealed"):
        operator.run(wait=False)
    assert events == ["gate"]
    assert not (spec.handoff_root / "phases/02-extension-deployed.json").exists()


def test_core_gate_rejects_inactive_unsealed_campaign(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = handoff.ActivationHandoff(_spec(tmp_path))
    monkeypatch.setattr(
        operator,
        "_core_status",
        lambda: {
            "campaign_id": "core",
            "plan_sha256": "a" * 64,
            "final": None,
        },
    )
    monkeypatch.setattr(operator, "_active_core_units", lambda: [])
    with pytest.raises(
        handoff.HandoffError,
        match="no core controller unit is active",
    ):
        operator.wait_for_core(wait=True)


def test_partial_probe_attempt_is_preserved_and_next_identity_is_used(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = handoff.ActivationHandoff(_spec(tmp_path))

    def fake_paths(attempt: str):
        root = tmp_path / attempt
        return {
            row.resource_id: (
                f"/remote/{attempt}/{row.resource_id}",
                f"/remote/{attempt}/{row.resource_id}/probe",
                root / row.resource_id / "probes/probe",
            )
            for row in operator.spec.resources
        }

    monkeypatch.setattr(operator, "_attempt_paths", fake_paths)
    a1 = fake_paths("a1")
    first_assignment = next(iter(a1.values()))[2].parent.parent
    first_assignment.mkdir(parents=True)
    (first_assignment / "partial").write_text("do not delete\n", encoding="ascii")

    attempt, _paths, state = operator.select_evidence_attempt()
    assert attempt == "a2"
    assert state == "absent"
    assert (first_assignment / "partial").read_text(encoding="ascii") == (
        "do not delete\n"
    )


def test_phase_record_is_create_only_and_self_hashed(tmp_path: Path) -> None:
    path = tmp_path / "phase.json"
    first = handoff._seal_phase(path, "test", {"value": 7})
    second = handoff._seal_phase(path, "test", {"value": 7})
    assert first == second
    assert json.loads(path.read_text(encoding="ascii")) == first
    with pytest.raises(handoff.HandoffError, match="phase changed"):
        handoff._seal_phase(path, "test", {"value": 8})


def test_restart_reconciles_unit_start_without_requiring_quiescence_again(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec(tmp_path)
    operator = handoff.ActivationHandoff(spec)
    monkeypatch.setattr(
        operator,
        "validate_payload",
        lambda: {
            "core_plan_sha256": "1" * 64,
            "r8_plan_sha256": "2" * 64,
            "bundle_sha256": "3" * 64,
            "extension_commit": spec.extension_commit,
        },
    )
    evidence = handoff._seal_phase(
        handoff._phase_path(spec, "03-extension-evidence"),
        "extension-evidence",
        {
            "evidence_attempt": "a1",
            "record_parent": "a" * 64,
        },
    )
    contracts = handoff._seal_phase(
        handoff._phase_path(spec, "04-extension-contracts"),
        "extension-contracts",
        {
            "activation_plan": str(tmp_path / "plan.yaml"),
            "activation_plan_sha256": "b" * 64,
        },
    )
    # Keep the real home untouched while preserving the exact branch shape.
    fake_home = tmp_path / "home"
    monkeypatch.setattr(handoff.Path, "home", classmethod(lambda cls: fake_home))
    unit = (
        fake_home
        / ".config/systemd/user"
        / handoff.campaign_unit_name("a1")
    )
    unit.parent.mkdir(parents=True)
    unit.write_text("accepted by systemd\n", encoding="ascii")
    events: list[str] = []
    monkeypatch.setattr(
        operator,
        "seal_core_gate",
        lambda **_kwargs: pytest.fail("must not repeat core quiescence"),
    )
    monkeypatch.setattr(
        operator,
        "launch",
        lambda got_contracts, got_evidence: (
            events.append("reconciled")
            or {
                "contracts": got_contracts["record_sha256"],
                "evidence": got_evidence["record_sha256"],
            }
        ),
    )
    result = operator.run(wait=True)
    assert events == ["reconciled"]
    assert result == {
        "contracts": contracts["record_sha256"],
        "evidence": evidence["record_sha256"],
    }


def test_launch_matrix_counts_remain_full_experiment() -> None:
    assert handoff.EXPECTED_RESOURCE_COUNT == 7
    assert handoff.EXPECTED_TRAINING_COUNT == 12
    assert handoff.EXPECTED_EVALUATION_COUNT == 54
    assert handoff.EXPECTED_H_REFERENCE_COUNT == 18
