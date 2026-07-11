"""Pre-flight go/no-go battery (scripts/preflight.py; playbook + spec §20).

Runs the real script via subprocess on CPU: a full pass over a synthetic tiny
capture (built with the production ShardWriter + CaptureProgress), report
structure and verdict/exit-code contract, and a sabotaged capture (empty dir,
then a count mismatch) yielding FAIL + nonzero exit. Never loads
deepseek-vl2-tiny."""

import json
import os
import subprocess
import sys
from pathlib import Path

import torch

from moe_sae.capture.driver import CaptureProgress
from moe_sae.capture.writer import ShardWriter

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "preflight.py"

CHECK_NAMES = ["env", "unit_parity", "checkpoint_drill", "throughput", "data_goodput", "burn_in"]


def run_preflight(*args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=REPO,
        env=env,
        timeout=900,
    )


def add_tokens(w: ShardWriter, n: int, d: int = 16, eid0: int = 0) -> None:
    g = torch.Generator().manual_seed(eid0)
    w.add(
        c=torch.randn(n, 6, d, generator=g).to(torch.bfloat16),
        experts=torch.randint(0, 64, (n, 6), generator=g, dtype=torch.int64),
        gates=torch.rand(n, 6, generator=g),
        example_id=torch.arange(eid0, eid0 + n, dtype=torch.int32),
        domain=torch.zeros(n, dtype=torch.uint8),
    )


def build_capture(root: Path, visual: int = 70, text: int = 40) -> None:
    """Synthetic tiny capture via the production writer + progress record."""
    shards = {}
    for pool, n in (("visual", visual), ("text", text)):
        w = ShardWriter(root, pool, shard_size=32)
        add_tokens(w, n, eid0=0 if pool == "visual" else 5000)
        w.close()
        shards[pool] = w.shard_count
    CaptureProgress(
        stream_consumed=10,
        captured=10,
        tokens={"visual": visual, "text": text},
        shards=shards,
        complete=True,
        config={},
    ).save(root)


def load_report(path: Path) -> dict:
    report = json.loads(path.read_text())
    # structure contract, independent of verdict
    assert [c["name"] for c in report["checks"]] == CHECK_NAMES
    for c in report["checks"]:
        assert set(c) == {"name", "status", "detail", "seconds"}
        assert c["status"] in ("PASS", "FAIL", "SKIP")
        assert c["seconds"] >= 0
    assert report["verdict"] in ("PASS", "FAIL")
    return report


def by_name(report: dict) -> dict:
    return {c["name"]: c for c in report["checks"]}


def test_full_battery_passes_on_synthetic_capture(tmp_path):
    """The whole playbook battery (minus --full burn-in) goes green on CPU."""
    build_capture(tmp_path / "cap")
    out = tmp_path / "report.json"
    res = run_preflight("--out", str(out), "--capture", str(tmp_path / "cap"), "--device", "cpu")
    assert res.returncode == 0, res.stdout + res.stderr
    assert "PREFLIGHT PASS" in res.stdout.strip().splitlines()[-1]

    report = load_report(out)
    checks = by_name(report)
    assert report["verdict"] == "PASS" and report["failed"] == []
    for name in ("env", "unit_parity", "checkpoint_drill", "throughput", "data_goodput"):
        assert checks[name]["status"] == "PASS", checks[name]
    assert checks["burn_in"]["status"] == "SKIP"  # no --full

    env = checks["env"]["detail"]
    assert env["torch"] == torch.__version__
    assert "git_commit" in env and "disk_gb" in env
    # throughput is informational: reports steps/s for both bench conditions
    for cond in ("A", "F"):
        assert checks["throughput"]["detail"][cond]["steps_per_s"] > 0
    goodput = checks["data_goodput"]["detail"]
    assert goodput["pools"]["visual"]["rows"] == 70
    assert goodput["pools"]["text"]["rows"] == 40
    assert goodput["pools"]["visual"]["shards"] == 3  # 70 tokens / shard_size 32


def test_sabotaged_capture_empty_dir_fails(tmp_path):
    """--capture at an empty dir: FAIL verdict, failing name on stdout,
    nonzero exit. --only keeps the fast subset (documented ops flag)."""
    empty = tmp_path / "empty"
    empty.mkdir()
    out = tmp_path / "report.json"
    res = run_preflight(
        "--out", str(out), "--capture", str(empty), "--only", "env,data_goodput", "--device", "cpu"
    )
    assert res.returncode != 0
    assert "PREFLIGHT FAIL" in res.stdout and "data_goodput" in res.stdout

    report = load_report(out)
    checks = by_name(report)
    assert report["verdict"] == "FAIL" and report["failed"] == ["data_goodput"]
    assert checks["data_goodput"]["status"] == "FAIL"
    assert checks["env"]["status"] == "PASS"
    # --only marks the rest SKIP, keeping the report shape stable
    assert checks["unit_parity"]["status"] == "SKIP"
    assert checks["burn_in"]["status"] == "SKIP"


def test_capture_count_mismatch_fails(tmp_path):
    """Shards parse but progress.json disagrees → goodput FAILs loudly."""
    cap = tmp_path / "cap"
    build_capture(cap, visual=70, text=40)
    progress = json.loads((cap / "progress.json").read_text())
    progress["tokens"]["text"] = 41  # recorded ≠ on disk
    (cap / "progress.json").write_text(json.dumps(progress))

    out = tmp_path / "report.json"
    res = run_preflight("--out", str(out), "--capture", str(cap), "--only", "data_goodput")
    assert res.returncode != 0
    checks = by_name(load_report(out))
    assert checks["data_goodput"]["status"] == "FAIL"
    assert any("41" in e for e in checks["data_goodput"]["detail"]["errors"])


def test_checkpoint_drill_alone_passes_and_skips_capture(tmp_path):
    """No --capture: data_goodput SKIPs and the verdict still PASSes."""
    out = tmp_path / "report.json"
    res = run_preflight("--out", str(out), "--only", "checkpoint_drill", "--device", "cpu")
    assert res.returncode == 0, res.stdout + res.stderr
    report = load_report(out)
    checks = by_name(report)
    assert report["verdict"] == "PASS"
    assert checks["checkpoint_drill"]["status"] == "PASS"
    assert checks["data_goodput"]["status"] == "SKIP"
    detail = checks["checkpoint_drill"]["detail"]
    assert "W_enc" in detail["bit_equal"] and "act_value" in detail["inference_parity_fields"]
