"""A18 confirmatory-trigger evaluation (scripts/evaluate_trigger.py): the
frozen two-route rule applied mechanically to synthetic artifact JSONs —
architecture route, representation route, the no-trigger case, the gate
precondition, and the dev-split-only guard."""

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _script():
    spec = importlib.util.spec_from_file_location(
        "evaluate_trigger", REPO / "scripts" / "evaluate_trigger.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write(path: Path, record: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record))
    return str(path)


def metrics(nmse_local, nmse_agg=1.0, split="dev"):
    return {"split": split, "local_micro_nmse": nmse_local, "aggregate_nmse": nmse_agg}


def fidelity(ce_delta):
    return {"mode": "matched", "overall": {"ce_delta": ce_delta}}


def probes(adv_point, ci_low, ci_high, key="feature_identity"):
    return {
        "advantages_vs_router": {
            key: {"point": adv_point, "ci_low": ci_low, "ci_high": ci_high}
        }
    }


def gate(passed=True):
    return {"gate": "A_vs_C_local_micro_nmse", "passed": passed}


@pytest.fixture()
def artifacts(tmp_path):
    """Baseline: A beats C by >10% in all 3 seeds; B artifacts neutral."""
    paths = {"gate": write(tmp_path / "gate.json", gate(True))}
    # A: 0.08 vs C: 0.10 → ratio 0.8 ≤ 0.9 in every seed
    paths["a_metrics"] = [
        write(tmp_path / f"a{s}.json", metrics(0.08, nmse_agg=0.5)) for s in range(3)
    ]
    paths["c_metrics"] = [
        write(tmp_path / f"c{s}.json", metrics(0.10, nmse_agg=0.9)) for s in range(3)
    ]
    return tmp_path, paths


def run(mod, tmp_path, paths, **extra):
    argv = [
        "--gate", paths["gate"],
        "--a-metrics", *paths["a_metrics"],
        "--c-metrics", *paths["c_metrics"],
        "--out", str(tmp_path / "reports" / "confirmatory_trigger.json"),
    ]
    for flag, items in extra.items():
        argv += [f"--{flag}", *items]
    return mod.main(argv)


def test_architecture_route_triggers(artifacts):
    mod = _script()
    tmp_path, paths = artifacts
    # A ≤ B on aggregate NMSE in 2 of 3 seeds
    b = [
        write(tmp_path / "b0.json", metrics(0.2, nmse_agg=0.6)),  # A 0.5 ≤ 0.6 ✓
        write(tmp_path / "b1.json", metrics(0.2, nmse_agg=0.4)),  # A 0.5 > 0.4 ✗
        write(tmp_path / "b2.json", metrics(0.2, nmse_agg=0.5)),  # tie counts ✓
    ]
    rec = run(mod, tmp_path, paths, **{"b-metrics": b})
    assert rec["nmse_condition"]["holds"] and rec["nmse_condition"]["n_seeds_passing"] == 3
    assert rec["architecture_route"]["triggers"] is True
    assert rec["representation_route"]["evaluated"] is False
    assert rec["confirmatory_run_proceeds"] is True
    assert "TRIGGERED" in rec["verdict"]
    saved = json.loads((tmp_path / "reports" / "confirmatory_trigger.json").read_text())
    assert saved["confirmatory_run_proceeds"] is True
    assert saved["inputs"]["gate"] == paths["gate"]  # inputs recorded (A18)


def test_architecture_route_via_fidelity(artifacts):
    mod = _script()
    tmp_path, paths = artifacts
    # B wins aggregate NMSE everywhere, but A wins matched fidelity 2/3
    b = [write(tmp_path / f"b{s}.json", metrics(0.2, nmse_agg=0.1)) for s in range(3)]
    a_fid = [
        write(tmp_path / f"af{s}.json", fidelity(d)) for s, d in enumerate([0.1, 0.1, 0.9])
    ]
    b_fid = [
        write(tmp_path / f"bf{s}.json", fidelity(d)) for s, d in enumerate([0.2, 0.2, 0.2])
    ]
    rec = run(
        mod, tmp_path, paths,
        **{"b-metrics": b, "a-fidelity": a_fid, "b-fidelity": b_fid},
    )
    assert rec["architecture_route"]["aggregate_nmse"]["holds"] is False
    assert rec["architecture_route"]["matched_fidelity"]["n_seeds_passing"] == 2
    assert rec["confirmatory_run_proceeds"] is True


def test_representation_route_triggers(artifacts):
    mod = _script()
    tmp_path, paths = artifacts
    # B beats A on aggregate NMSE (architecture route fails)...
    b = [write(tmp_path / f"b{s}.json", metrics(0.2, nmse_agg=0.1)) for s in range(3)]
    # ...but probe 3's dev advantage CI excludes zero in 2/3 seeds and beats C
    a_p = [
        write(tmp_path / "ap0.json", probes(0.10, 0.04, 0.16)),  # CI > 0, beats C ✓
        write(tmp_path / "ap1.json", probes(0.08, 0.02, 0.14)),  # ✓
        write(tmp_path / "ap2.json", probes(0.02, -0.03, 0.07)),  # CI spans 0 ✗
    ]
    c_p = [write(tmp_path / f"cp{s}.json", probes(0.01, -0.02, 0.04)) for s in range(3)]
    rec = run(mod, tmp_path, paths, **{"b-metrics": b, "a-probes": a_p, "c-probes": c_p})
    assert rec["architecture_route"]["triggers"] is False
    route = rec["representation_route"]
    assert route["evaluated"] is True
    assert route["per_probe"]["feature_identity"]["n_seeds_passing"] == 2
    assert route["triggers"] is True
    assert rec["confirmatory_run_proceeds"] is True


def test_representation_route_requires_beating_condition_c(artifacts):
    mod = _script()
    tmp_path, paths = artifacts
    b = [write(tmp_path / f"b{s}.json", metrics(0.2, nmse_agg=0.1)) for s in range(3)]
    # A's CI excludes zero everywhere, but C's point advantage is LARGER
    a_p = [write(tmp_path / f"ap{s}.json", probes(0.10, 0.04, 0.16)) for s in range(3)]
    c_p = [write(tmp_path / f"cp{s}.json", probes(0.20, 0.10, 0.30)) for s in range(3)]
    rec = run(mod, tmp_path, paths, **{"b-metrics": b, "a-probes": a_p, "c-probes": c_p})
    assert rec["representation_route"]["triggers"] is False
    assert rec["confirmatory_run_proceeds"] is False
    assert "NOT TRIGGERED" in rec["verdict"]
    assert "power clause" in rec["verdict"]


def test_no_trigger_when_nmse_condition_fails(tmp_path):
    """Neither route can trigger without A beating C by ≥10% in ≥2 seeds."""
    mod = _script()
    paths = {"gate": write(tmp_path / "gate.json", gate(True))}
    # A/C ratio 0.95 > 0.9 in all seeds
    paths["a_metrics"] = [
        write(tmp_path / f"a{s}.json", metrics(0.095, nmse_agg=0.1)) for s in range(3)
    ]
    paths["c_metrics"] = [
        write(tmp_path / f"c{s}.json", metrics(0.100, nmse_agg=0.9)) for s in range(3)
    ]
    b = [write(tmp_path / f"b{s}.json", metrics(0.2, nmse_agg=0.9)) for s in range(3)]
    rec = run(mod, tmp_path, paths, **{"b-metrics": b})
    assert rec["nmse_condition"]["holds"] is False
    # the second clause holds, but the shared premise fails → no trigger
    assert rec["architecture_route"]["second_clause_holds"] is True
    assert rec["confirmatory_run_proceeds"] is False


def test_gate_failure_blocks_trigger(artifacts):
    mod = _script()
    tmp_path, paths = artifacts
    paths["gate"] = write(tmp_path / "gate_fail.json", gate(False))
    b = [write(tmp_path / f"b{s}.json", metrics(0.2, nmse_agg=0.9)) for s in range(3)]
    rec = run(mod, tmp_path, paths, **{"b-metrics": b})
    assert rec["architecture_route"]["triggers"] is True  # routes hold...
    assert rec["gate_passed"] is False
    assert rec["confirmatory_run_proceeds"] is False  # ...but the gate gates


def test_representation_route_end_to_end_with_real_probes_json(tmp_path):
    """End-to-end schema check: probes.json PRODUCED BY run_probes on
    synthetic dumps (not a hand-built artifact) feeds evaluate_trigger and
    triggers the representation route — the producer emits exactly the
    `advantages_vs_router` schema the trigger consumes."""
    from test_analysis_probes import write_advantage_dump

    from moe_sae.analysis.mass import load_dump, token_bank_masses
    from moe_sae.analysis.probes import run_probes

    # "A" run: identity provably beats routing; "C" run: null structure.
    a_dump = load_dump(write_advantage_dump(tmp_path / "adv_a", informative=True))
    c_dump = load_dump(write_advantage_dump(tmp_path / "adv_c", informative=False, seed=1))
    a_prob = run_probes(
        a_dump, token_bank_masses(a_dump), a_dump, token_bank_masses(a_dump),
        seed=0, n_boot=200,
    )
    c_prob = run_probes(
        c_dump, token_bank_masses(c_dump), c_dump, token_bank_masses(c_dump),
        seed=0, n_boot=200,
    )
    a_probes = [write(tmp_path / f"probes_a{s}.json", a_prob) for s in range(3)]
    c_probes = [write(tmp_path / f"probes_c{s}.json", c_prob) for s in range(3)]

    mod = _script()
    paths = {"gate": write(tmp_path / "gate.json", gate(True))}
    paths["a_metrics"] = [
        write(tmp_path / f"a{s}.json", metrics(0.08, nmse_agg=0.5)) for s in range(3)
    ]
    paths["c_metrics"] = [
        write(tmp_path / f"c{s}.json", metrics(0.10, nmse_agg=0.9)) for s in range(3)
    ]
    # B beats A on aggregate NMSE: only the representation route can trigger
    b = [write(tmp_path / f"b{s}.json", metrics(0.2, nmse_agg=0.1)) for s in range(3)]

    rec = run(mod, tmp_path, paths, **{"b-metrics": b, "a-probes": a_probes,
                                       "c-probes": c_probes})
    assert rec["architecture_route"]["triggers"] is False
    route = rec["representation_route"]
    assert route["evaluated"] is True
    fi = route["per_probe"]["feature_identity"]
    assert fi["n_seeds_passing"] == 3 and fi["holds"] is True
    for seed_rec in fi["per_seed"]:
        assert seed_rec["available"] is True
        assert seed_rec["ci_excludes_zero"] and seed_rec["beats_condition_c"]
    # A20 null entries pass through as unavailable, never as errors
    assert not route["per_probe"]["sae_bank_mass"]["holds"]
    assert rec["confirmatory_run_proceeds"] is True


def test_dev_split_only_guard(tmp_path):
    """A18 is evaluated on the dev split only — eval-split artifacts abort."""
    mod = _script()
    paths = {"gate": write(tmp_path / "gate.json", gate(True))}
    paths["a_metrics"] = [
        write(tmp_path / f"a{s}.json", metrics(0.08, split="eval")) for s in range(3)
    ]
    paths["c_metrics"] = [
        write(tmp_path / f"c{s}.json", metrics(0.10)) for s in range(3)
    ]
    with pytest.raises(AssertionError, match="DEVELOPMENT split"):
        run(mod, tmp_path, paths)
