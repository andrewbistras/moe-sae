#!/usr/bin/env python
"""Pre-flight go/no-go battery (playbook; spec §20 execution order).

One command, run on the target host before committing GPU/TPU-days:

    uv run python scripts/preflight.py --out reports/preflight_<host>.json \
        [--capture captures/<name>] [--full]

Checks (each reported as {name, status: PASS/FAIL/SKIP, detail, seconds}):

1. env             — fingerprint: python/torch/jax versions, git commit +
                     dirty flag, device inventory, RAM/disk free. Recorded
                     for the run provenance trail; FAIL only on gather error.
2. unit_parity     — loss-matching spot check: runs the torch↔jax
                     training-step metric parity test and the capture-hook
                     bit-identity suite via pytest in a subprocess.
3. checkpoint_drill— save-restore drill: tiny SAECondition, 2 optimizer
                     steps, save_checkpoint → load_condition_from_run,
                     bit-equal params + bit-identical θ*-inference.
4. throughput      — micro-bench: 10 tiny training_step+SpecAdam steps for
                     conditions A and F on the best device. Informational
                     (steps/s, tokens/s); FAIL only on crash.
5. data_goodput    — only with --capture: reads every shard once, reports
                     GB/s and rows/s; PASS iff all shards parse and pool
                     token counts match the capture's progress.json.
6. burn_in         — only with --full: ~60 s of repeated step-4 loops
                     watching every logged scalar for NaN/inf or crashes.

Output: JSON report at --out plus a one-line verdict on stdout
('PREFLIGHT PASS (n checks)' or 'PREFLIGHT FAIL: <names>'); the exit code
reflects the verdict. Progress goes to stderr.

Decisions:
- This is OPS tooling only. The drill/bench conditions are throwaway
  tiny-dim instances built through the identical `build_condition` /
  `training_step` / `SpecAdam` code path as scientific runs (mirroring
  tests/test_sae_conditions.py::tiny_config); nothing here reads or writes
  runs/, and no frozen constant is redefined.
- Beyond the playbook command, two engineering flags: `--device` (pin the
  bench/burn-in device, used by CPU tests) and `--only` (comma-list to
  re-run a subset after a failure); excluded checks are reported as SKIP so
  the report shape is stable.
- Bench dims: real d=1280 and E=64 with f=64, k̄=32, B=512 — big enough
  that steps/s is indicative, small enough to finish in seconds. Compute
  dtype is the production BF16 on cuda/mps but FP32 on CPU: torch has no
  fast BF16 CPU kernels, so a BF16 CPU number would measure the fallback
  path (~60× slow), not the host. Numbers are informational; no threshold
  is asserted (hosts differ).
- Checkpoint drill runs on CPU regardless of --device: it asserts
  bit-equality, and CPU is the deterministic common denominator.
- data_goodput count source: progress.json is canonical (the capture
  driver and merge_captures.py both write one); meta.json tokens are the
  fallback for legacy captures, noted in the detail. Field validation
  checks the writer's key set and ndims, not d==1280, so tiny synthetic
  captures exercise the same path.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import torch

from moe_sae.capture.writer import POOLS, _FIELD_SPECS, iter_shards, shard_paths
from moe_sae.model_loading import pick_device
from moe_sae.sae import SpecAdam, TrainBatch, load_condition_from_run
from moe_sae.sae.conditions import build_condition
from moe_sae.train.config import RunConfig

REPO = Path(__file__).resolve().parent.parent

CHECK_NAMES = ("env", "unit_parity", "checkpoint_drill", "throughput", "data_goodput", "burn_in")

PARITY_TARGETS = (
    "tests/test_jax_parity.py::test_training_step_metric_parity",
    "tests/test_capture_hooks.py",
)

# Drill dims mirror tests/test_sae_conditions.py (E=64 is the model's expert
# count; everything else tiny).
DRILL = dict(E=64, f=8, d=16, B=24, k_bar=4)
# Bench dims: real d/E, small f/B (see Decisions).
BENCH = dict(E=64, f=64, d=1280, B=512, k_bar=32)
BENCH_STEPS = 10
BURN_SECONDS = 60.0


# --------------------------------------------------------------------------- #
# Tiny-condition helpers (ops-only instances through the production code path) #
# --------------------------------------------------------------------------- #


def _tiny_config(condition: str, dims: dict, compute_dtype: str) -> RunConfig:
    """Small-dim config; validate() deliberately NOT called (frozen-constant
    validation belongs to scripts/run_train.py — same decision as the tests)."""
    return RunConfig(
        run_name=f"preflight_{condition}",
        condition=condition,
        seed=1,
        f=dims["f"],
        k_bar=dims["k_bar"],
        total_tokens=dims["B"] * 3,
        capture_root="unused",
        stats_path="unused",
        pipeline=True,
        batch_tokens=dims["B"],
        compute_dtype=compute_dtype,
    )


def _make_means(dims: dict, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    mu_e = torch.randn(dims["E"], dims["d"], generator=g) * 0.1
    mu_route = torch.randn(dims["d"], generator=g) * 0.1
    return mu_e, mu_route


def _make_batch(dims: dict, seed: int = 0) -> TrainBatch:
    g = torch.Generator().manual_seed(seed)
    n, E, d = dims["B"], dims["E"], dims["d"]
    # 6 distinct experts per token (routed top-k semantics).
    experts = torch.argsort(torch.rand(n, E, generator=g), dim=1)[:, :6]
    token_type = torch.zeros(n, dtype=torch.uint8)
    token_type[: n // 2] = 1
    return TrainBatch(
        c=torch.randn(n, 6, d, generator=g).to(torch.bfloat16),
        experts=experts.to(torch.int64),
        gates=torch.rand(n, 6, generator=g),
        token_type=token_type,
        example_id=torch.arange(n, dtype=torch.int32) // 4,
        domain=torch.zeros(n, dtype=torch.uint8),
    )


def _build(condition: str, dims: dict, compute_dtype: str, device: torch.device | None = None):
    """Condition + optimizer in the training-loop order: build, move to the
    device, THEN construct SpecAdam so its FP32 moments live with the params."""
    torch.manual_seed(1)
    mu_e, mu_route = _make_means(dims)
    cond = build_condition(condition, _tiny_config(condition, dims, compute_dtype), mu_e, mu_route, seed=1)
    if device is not None:
        cond = cond.to(device)
    return cond, SpecAdam(cond.optim_groups())


def _one_step(cond, opt, batch: TrainBatch, step: int):
    """training_step → backward → SpecAdam → tracker updates (loop order)."""
    out = cond.training_step(batch, step)
    out.loss.backward()
    grad_norm = opt.step(3e-4)
    opt.zero_grad()
    cond.theta_ema.update(out.theta_b)
    cond.dead_tracker.update(out.dict_route_counts, out.fired_mask)
    return out, grad_norm


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


# --------------------------------------------------------------------------- #
# Checks                                                                       #
# --------------------------------------------------------------------------- #


def check_env(args) -> tuple[str, dict]:
    detail: dict = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": platform.node(),
        "torch": torch.__version__,
    }
    devices = {"cpu": True, "cuda": [], "mps": bool(torch.backends.mps.is_available())}
    if torch.cuda.is_available():
        devices["cuda"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    detail["torch_devices"] = devices
    try:
        import jax

        detail["jax"] = jax.__version__
        detail["jax_devices"] = [str(dv) for dv in jax.devices()]
    except Exception as e:  # jax optional on torch-only hosts
        detail["jax"] = f"unavailable: {e!r}"
    for key, cmd in (
        ("git_commit", ["git", "rev-parse", "HEAD"]),
        ("git_dirty", ["git", "status", "--porcelain"]),
    ):
        try:
            out = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=30)
            detail[key] = out.stdout.strip() if key == "git_commit" else bool(out.stdout.strip())
        except Exception as e:
            detail[key] = f"unavailable: {e!r}"
    try:
        import psutil

        vm = psutil.virtual_memory()
        detail["ram_gb"] = {"total": round(vm.total / 2**30, 2), "available": round(vm.available / 2**30, 2)}
    except Exception:
        try:
            total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
            detail["ram_gb"] = {"total": round(total / 2**30, 2)}
        except Exception as e:
            detail["ram_gb"] = f"unavailable: {e!r}"
    du = shutil.disk_usage(REPO)
    detail["disk_gb"] = {"total": round(du.total / 2**30, 2), "free": round(du.free / 2**30, 2)}
    if args.capture:
        duc = shutil.disk_usage(args.capture) if Path(args.capture).exists() else None
        detail["capture_disk_free_gb"] = round(duc.free / 2**30, 2) if duc else "capture path missing"
    return "PASS", detail


def check_unit_parity(args) -> tuple[str, dict]:
    cmd = [sys.executable, "-m", "pytest", *PARITY_TARGETS, "-q", "-p", "no:cacheprovider"]
    res = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=900)
    tail = [ln for ln in res.stdout.strip().splitlines() if ln.strip()][-3:]
    detail = {"targets": list(PARITY_TARGETS), "pytest_tail": tail, "returncode": res.returncode}
    if res.returncode != 0:
        detail["stderr_tail"] = res.stderr.strip().splitlines()[-5:]
    return ("PASS" if res.returncode == 0 else "FAIL"), detail


def check_checkpoint_drill(args) -> tuple[str, dict]:
    cond, opt = _build("A", DRILL, "float32")  # CPU, fp32: bit-determinism
    for step in range(2):
        _one_step(cond, opt, _make_batch(DRILL, seed=step), step)
    with tempfile.TemporaryDirectory(prefix="preflight_ckpt_") as tmp:
        run_dir = Path(tmp) / "run"
        cond.save_checkpoint(
            run_dir / "final" / "checkpoint.safetensors", cond.theta_ema, cond.dead_tracker, step=2
        )
        cond2, theta_star, extras = load_condition_from_run(run_dir, device="cpu")
    assert extras["step"] == 2, extras
    assert theta_star is not None and abs(theta_star - cond.theta_ema.value) < 1e-6, (
        theta_star,
        cond.theta_ema.value,
    )
    pairs = [
        ("W_enc", cond.sae.W_enc.detach(), cond2.sae.W_enc.detach()),
        ("b_enc", cond.sae.b_enc.detach(), cond2.sae.b_enc.detach()),
        ("D", cond.sae.D.detach(), cond2.sae.D.detach()),
        ("mu", cond.sae.mu, cond2.sae.mu),
        ("tokens_seen", cond.dead_tracker.tokens_seen, cond2.dead_tracker.tokens_seen),
        ("last_fire", cond.dead_tracker.last_fire, cond2.dead_tracker.last_fire),
    ]
    for name, a, b in pairs:
        assert torch.equal(a, b), f"checkpoint round trip not bit-equal: {name}"
    batch = _make_batch(DRILL, seed=99)
    enc_a = cond.encode_thresholded(batch.c, batch.experts, batch.example_id, 0.2)
    enc_b = cond2.encode_thresholded(batch.c, batch.experts, batch.example_id, 0.2)
    fields = ("act_token", "act_slot", "act_bank", "act_latent", "act_value", "l0", "recon_slots", "mu_slots")
    for k in fields:
        assert torch.equal(getattr(enc_a, k), getattr(enc_b, k)), f"inference parity broke: {k}"
    return "PASS", {
        "condition": "A",
        "steps": 2,
        "theta_star": theta_star,
        "bit_equal": [p[0] for p in pairs],
        "inference_parity_fields": list(fields),
    }


def _bench_dtype(device: torch.device) -> str:
    """Production BF16 on accelerators; FP32 on CPU (no fast BF16 CPU kernels
    — see Decisions)."""
    return "float32" if device.type == "cpu" else "bfloat16"


def _bench_condition(condition: str, device: torch.device) -> dict:
    cond, opt = _build(condition, BENCH, _bench_dtype(device), device)
    batch = _make_batch(BENCH, seed=0)
    for step in range(2):  # warmup (lazy init, kernel autotune)
        _one_step(cond, opt, batch, step)
    _sync(device)
    t0 = time.perf_counter()
    for step in range(BENCH_STEPS):
        _one_step(cond, opt, batch, step + 2)
    _sync(device)
    dt = time.perf_counter() - t0
    return {
        "steps": BENCH_STEPS,
        "batch_tokens": BENCH["B"],
        "seconds": round(dt, 4),
        "steps_per_s": round(BENCH_STEPS / dt, 3),
        "tokens_per_s": round(BENCH_STEPS * BENCH["B"] / dt, 1),
    }


def check_throughput(args) -> tuple[str, dict]:
    device = pick_device(args.device)
    detail = {
        "device": str(device),
        "compute_dtype": _bench_dtype(device),
        "dims": BENCH,
        "note": "informational — no threshold asserted",
    }
    for condition in ("A", "F"):
        detail[condition] = _bench_condition(condition, device)
    return "PASS", detail


def check_data_goodput(args) -> tuple[str, dict]:
    root = Path(args.capture)
    if not root.exists():
        return "FAIL", {"error": f"capture root does not exist: {root}"}
    progress_path = root / "progress.json"
    meta_path = root / "meta.json"
    if progress_path.exists():
        expected = json.loads(progress_path.read_text()).get("tokens", {})
        source = "progress.json"
    elif meta_path.exists():
        expected = json.loads(meta_path.read_text()).get("tokens", {})
        source = "meta.json (no progress.json — legacy capture)"
    else:
        return "FAIL", {"error": f"no progress.json or meta.json under {root}"}

    detail: dict = {"counts_source": source, "pools": {}}
    total_rows, total_bytes = 0, 0
    t0 = time.perf_counter()
    errors: list[str] = []
    for pool in POOLS:
        paths = shard_paths(root, pool)
        rows = 0
        for shard in iter_shards(root, pool):
            if set(shard) != set(_FIELD_SPECS):
                errors.append(f"{pool}: shard keys {sorted(shard)} != writer contract")
                break
            for k, (dtype, ndim) in _FIELD_SPECS.items():
                if shard[k].dtype != dtype or shard[k].ndim != ndim:
                    errors.append(f"{pool}/{k}: dtype/ndim {shard[k].dtype}/{shard[k].ndim}")
            rows += shard["domain"].shape[0]
        nbytes = sum(p.stat().st_size for p in paths)
        want = int(expected.get(pool, 0))
        detail["pools"][pool] = {"shards": len(paths), "rows": rows, "expected_rows": want, "gb": round(nbytes / 1e9, 4)}
        if rows != want:
            errors.append(f"{pool}: {rows} rows on disk != {want} recorded")
        total_rows += rows
        total_bytes += nbytes
    dt = time.perf_counter() - t0
    detail.update(
        seconds=round(dt, 3),
        gb_per_s=round(total_bytes / 1e9 / dt, 3) if dt > 0 else None,
        rows_per_s=round(total_rows / dt, 1) if dt > 0 else None,
    )
    if total_rows == 0:
        errors.append("capture contains zero token rows")
    if errors:
        detail["errors"] = errors
        return "FAIL", detail
    return "PASS", detail


def check_burn_in(args) -> tuple[str, dict]:
    device = pick_device(args.device)
    detail: dict = {
        "device": str(device),
        "compute_dtype": _bench_dtype(device),
        "budget_seconds": BURN_SECONDS,
    }
    per_cond = BURN_SECONDS / 2
    for condition in ("A", "F"):
        cond, opt = _build(condition, BENCH, _bench_dtype(device), device)
        batch = _make_batch(BENCH, seed=0)
        steps, deadline = 0, time.perf_counter() + per_cond
        while time.perf_counter() < deadline:
            out, grad_norm = _one_step(cond, opt, batch, steps)
            scalars = {**out.scalars(), "grad_norm": grad_norm}
            bad = {k: v for k, v in scalars.items() if not math.isfinite(float(v))}
            if bad:
                detail[condition] = {"steps": steps, "nonfinite": bad}
                return "FAIL", detail
            steps += 1
        _sync(device)
        detail[condition] = {"steps": steps, "steps_per_s": round(steps / per_cond, 3)}
    return "PASS", detail


# --------------------------------------------------------------------------- #
# Battery driver                                                               #
# --------------------------------------------------------------------------- #

CHECKS = {
    "env": check_env,
    "unit_parity": check_unit_parity,
    "checkpoint_drill": check_checkpoint_drill,
    "throughput": check_throughput,
    "data_goodput": check_data_goodput,
    "burn_in": check_burn_in,
}


def _skip_reason(name: str, args, only: set[str] | None) -> str | None:
    if only is not None and name not in only:
        return "excluded by --only"
    if name == "data_goodput" and not args.capture:
        return "no --capture given"
    if name == "burn_in" and not args.full:
        return "requires --full"
    return None


def run_battery(args) -> dict:
    only = set(args.only.split(",")) if args.only else None
    if only is not None and (unknown := only - set(CHECK_NAMES)):
        raise SystemExit(f"unknown --only checks: {sorted(unknown)} (valid: {CHECK_NAMES})")
    checks = []
    for name in CHECK_NAMES:
        reason = _skip_reason(name, args, only)
        t0 = time.perf_counter()
        if reason is not None:
            status, detail = "SKIP", reason
        else:
            print(f"[preflight] {name} ...", file=sys.stderr, flush=True)
            try:
                status, detail = CHECKS[name](args)
            except Exception:
                status = "FAIL"
                detail = {"exception": traceback.format_exc().strip().splitlines()[-6:]}
        seconds = round(time.perf_counter() - t0, 3)
        print(f"[preflight] {name}: {status} ({seconds}s)", file=sys.stderr, flush=True)
        checks.append({"name": name, "status": status, "detail": detail, "seconds": seconds})

    failed = [c["name"] for c in checks if c["status"] == "FAIL"]
    return {
        "kind": "preflight",
        "spec_version": "v0.1.1",
        "host": platform.node(),
        "generated_unix": time.time(),
        "argv": sys.argv[1:],
        "capture": args.capture,
        "full": args.full,
        "checks": checks,
        "n_pass": sum(c["status"] == "PASS" for c in checks),
        "n_fail": len(failed),
        "n_skip": sum(c["status"] == "SKIP" for c in checks),
        "failed": failed,
        "verdict": "FAIL" if failed else "PASS",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--out",
        default=None,
        help="JSON report path (default reports/preflight_<host>.json)",
    )
    ap.add_argument("--capture", default=None, help="capture root for the data-goodput check")
    ap.add_argument("--full", action="store_true", help="include the ~60 s burn-in")
    ap.add_argument(
        "--device", default=None, choices=["cpu", "mps", "cuda"], help="pin bench/burn-in device (default: best available)"
    )
    ap.add_argument("--only", default=None, help="comma-list of checks to run; the rest are SKIPped")
    args = ap.parse_args()

    out = Path(args.out) if args.out else REPO / "reports" / f"preflight_{platform.node().split('.')[0]}.json"
    report = run_battery(args)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"[preflight] report: {out}", file=sys.stderr)

    if report["verdict"] == "PASS":
        extra = f", {report['n_skip']} skipped" if report["n_skip"] else ""
        print(f"PREFLIGHT PASS ({report['n_pass']} checks{extra})")
        raise SystemExit(0)
    print(f"PREFLIGHT FAIL: {', '.join(report['failed'])}")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
