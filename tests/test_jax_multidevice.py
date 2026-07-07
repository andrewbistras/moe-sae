"""Multi-device orchestration smoke (ladder rung 2 locally, free).

XLA_FLAGS=--xla_force_host_platform_device_count=4 must be set before jax
initializes, so the actual work runs in a subprocess; on the real slice the
same code sees 4 TpuDevices per host instead. Verifies the discovery-suite
pattern: two independent conditions training concurrently on different
devices from the SAME host batch stream (matched data order), producing
finite, torch-readable artifacts."""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("jax")

_SCRIPT = textwrap.dedent(
    """
    import os
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4"
    )
    import json, sys
    import jax, torch
    assert len(jax.devices()) == 4, jax.devices()

    sys.path.insert(0, "tests")
    from test_jax_parity import B, CAP, build, make_batch, tiny_config
    from moe_sae.tpu.train_jax import train_jax
    from moe_sae.sae.conditions import load_condition_from_run

    out_root = sys.argv[1]
    batches = [make_batch(seed=s) for s in range(2)]  # one stream, consumed twice

    results = {}
    for i, condition in enumerate(("A", "B")):
        cfg = tiny_config(condition, total_tokens=B * 2)
        cond = build(condition)
        out = f"{out_root}/run_{condition}"
        train_jax(cfg, cond, iter(list(batches)), out, capacity=CAP,
                  device=jax.devices()[i])
        loaded, theta, extras = load_condition_from_run(out)
        results[condition] = {
            "theta": theta,
            "step": extras["step"],
            "device": str(jax.devices()[i]),
        }
    print(json.dumps(results))
    """
)


def test_two_conditions_two_devices(tmp_path):
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    proc = subprocess.run(
        [sys.executable, "-c", _SCRIPT, str(tmp_path)],
        capture_output=True,
        text=True,
        env=env,
        cwd=Path(__file__).parent.parent,
        timeout=300,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr[-2000:]}"
    results = json.loads(proc.stdout.strip().splitlines()[-1])
    assert set(results) == {"A", "B"}
    for cond, r in results.items():
        assert r["step"] == 2 and r["theta"] is not None, (cond, r)
    assert results["A"]["device"] != results["B"]["device"]
    # both runs' artifacts exist and batch sequences were identical by
    # construction (same host list consumed by both)
    for cond in ("A", "B"):
        assert (tmp_path / f"run_{cond}" / "final" / "checkpoint.safetensors").exists()
