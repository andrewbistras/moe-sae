"""Tiny end-to-end training runs (3 steps, CPU) for every condition A..F:
synthetic capture shards → local batch composition → moe_sae.train.loop.train
→ artifact checks.

Decision (documented): the frozen batch_tokens=4096 (spec §9) is far too
large for a seconds-scale CPU test, so the RunConfig here is built directly
with batch_tokens=32 and `RunConfig.validate()` is NOT called — validation of
frozen constants belongs to scripts/run_train.py, and `train()` deliberately
does not re-validate. The batch composition below duck-types the
`moe_sae.data.batches.BatchComposer` contract (that module is owned by a
parallel workstream) on top of the real capture writer/reader."""

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from moe_sae.capture.writer import ShardWriter, iter_shards
from moe_sae.sae import TrainBatch
from moe_sae.sae.conditions import build_condition
from moe_sae.train.config import CONDITIONS, RunConfig
from moe_sae.train.loop import lr_at_step, train

E, F_BANK, D_MODEL, B, K_BAR, N_STEPS = 64, 8, 16, 32, 4, 3


def write_capture(root: Path, n_per_pool: int = 64, seed: int = 0) -> None:
    g = torch.Generator().manual_seed(seed)
    for pool in ("visual", "text"):
        w = ShardWriter(root, pool, shard_size=48)
        experts = torch.stack(
            [torch.randperm(E, generator=g)[:6] for _ in range(n_per_pool)]
        )
        w.add(
            c=torch.randn(n_per_pool, 6, D_MODEL, generator=g),
            experts=experts,
            gates=torch.rand(n_per_pool, 6, generator=g),
            example_id=torch.arange(n_per_pool, dtype=torch.int32),
            domain=torch.zeros(n_per_pool, dtype=torch.uint8),
        )
        w.close()


def batch_iter_from_capture(root: Path, batch_tokens: int, seed: int):
    """Minimal stand-in honoring the BatchComposer contract: B/2 visual +
    B/2 text per batch, halves shuffled independently, sampling with
    replacement across the pool."""
    pools = {}
    for pool in ("visual", "text"):
        shards = list(iter_shards(root, pool))
        pools[pool] = {
            k: torch.cat([s[k] for s in shards], dim=0) for k in shards[0]
        }
    rng = torch.Generator().manual_seed(seed)
    half = batch_tokens // 2
    while True:
        parts = {"token_type": []}
        for k in ("c", "experts", "gates", "example_id", "domain"):
            parts[k] = []
        for pool, flag in (("visual", 1), ("text", 0)):
            data = pools[pool]
            n = data["domain"].shape[0]
            idx = torch.randint(0, n, (half,), generator=rng)
            for k in ("c", "experts", "gates", "example_id", "domain"):
                parts[k].append(data[k][idx])
            parts["token_type"].append(torch.full((half,), flag, dtype=torch.uint8))
        yield TrainBatch(
            c=torch.cat(parts["c"]),
            experts=torch.cat(parts["experts"]).to(torch.int64),
            gates=torch.cat(parts["gates"]),
            token_type=torch.cat(parts["token_type"]),
            example_id=torch.cat(parts["example_id"]),
            domain=torch.cat(parts["domain"]),
        )


def tiny_config(condition: str, tmp_path: Path) -> RunConfig:
    return RunConfig(
        run_name=f"loop_{condition}",
        condition=condition,
        seed=1,
        f=F_BANK,
        k_bar=K_BAR,
        total_tokens=B * N_STEPS,
        capture_root=str(tmp_path / "captures"),
        stats_path="unused-in-test",
        out_root=str(tmp_path / "runs"),
        pipeline=True,
        batch_tokens=B,  # NOT frozen-valid; validate() intentionally skipped
        compute_dtype="float32",
        log_every=1,
        ckpt_every_steps=2,
    )


@pytest.mark.parametrize("condition", CONDITIONS)
def test_three_step_run(condition, tmp_path):
    cfg = tiny_config(condition, tmp_path)
    capture_root = Path(cfg.capture_root)
    write_capture(capture_root)

    g = torch.Generator().manual_seed(7)
    mu_e = torch.randn(E, D_MODEL, generator=g) * 0.1
    mu_route = torch.randn(D_MODEL, generator=g) * 0.1
    cond = build_condition(condition, cfg, mu_e, mu_route, cfg.seed)
    batches = batch_iter_from_capture(capture_root, cfg.batch_tokens, cfg.data_seed)

    out_dir = cfg.out_dir
    train(cfg, cond, batches, out_dir)

    # metrics.jsonl: one line per step (log_every=1), finite losses, exact K
    lines = [json.loads(x) for x in (out_dir / "metrics.jsonl").read_text().splitlines()]
    assert [ln["step"] for ln in lines] == list(range(N_STEPS))
    for ln in lines:
        for key in ("loss", "l_local", "l_aggregate", "l_aux", "theta_b", "lr",
                    "l0_mean", "l0_visual", "l0_text", "wall_time", "theta_ema"):
            assert math.isfinite(ln[key]), (key, ln)
        assert ln["k_selected"] == K_BAR * B  # exact BatchTopK budget
        assert ln["lr"] == pytest.approx(cfg.lr * (ln["step"] + 1) / cfg.warmup_steps)

    # decoder columns ≈ unit norm after optimizer steps
    D = cond.decoder_columns().detach()
    axis = 1 if D.ndim == 3 else 0
    norms = D.norm(dim=axis)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    # θ EMA finite and initialized from step-0 θ_b then decayed
    assert cond.theta_ema.value is not None and math.isfinite(cond.theta_ema.value)
    theta_traj = np.load(out_dir / "theta_trajectory.npy")
    assert theta_traj.shape == (N_STEPS,) and np.isfinite(theta_traj).all()
    ema = theta_traj[0]
    for t in theta_traj[1:]:
        ema = 0.999 * ema + 0.001 * t
    assert cond.theta_ema.value == pytest.approx(ema)

    # final artifacts
    theta_star = json.loads((out_dir / "final" / "theta_star.json").read_text())
    assert theta_star["theta_star"] == pytest.approx(cond.theta_ema.value)
    loads = np.load(out_dir / "final" / "bank_loads.npy")
    assert loads.shape == (E,) and loads.sum() == 6 * B * N_STEPS
    dead = np.load(out_dir / "final" / "dead_final.npy")
    assert dead.shape == (cond.n_dict_banks, cond.f_dict)
    assert not dead.any()  # 96 tokens << 250k window: deadness undefined
    assert (out_dir / "final" / "checkpoint.safetensors").exists()
    assert (out_dir / "checkpoints" / "2.safetensors").exists()  # ckpt_every=2
    assert (out_dir / "config.yaml").exists()

    # dead tracker clock advanced by the routed loads
    expected_clock = 6 * B * N_STEPS if not cond.is_global else B * N_STEPS
    assert int(cond.dead_tracker.tokens_seen.sum()) == expected_clock


def test_lr_schedule_shape():
    cfg = RunConfig(
        run_name="lr", condition="A", seed=1, f=F_BANK, k_bar=K_BAR,
        total_tokens=4096 * 2000, capture_root="x", stats_path="x",
        pipeline=True, batch_tokens=4096,
    )
    n = cfg.n_steps  # 2000
    assert lr_at_step(0, cfg) == pytest.approx(cfg.lr / cfg.warmup_steps)
    assert lr_at_step(cfg.warmup_steps - 1, cfg) == pytest.approx(cfg.lr)
    mid = cfg.warmup_steps + (n - cfg.warmup_steps) // 2
    assert lr_at_step(mid, cfg) == pytest.approx(0.5 * (cfg.lr + cfg.lr_min))
    assert lr_at_step(n - 1, cfg) >= cfg.lr_min
    assert lr_at_step(10**9, cfg) == pytest.approx(cfg.lr_min)  # clamped tail
    # monotone: warmup up, cosine down
    lrs = [lr_at_step(s, cfg) for s in range(0, n, 25)]
    peak = max(range(len(lrs)), key=lambda i: lrs[i])
    assert all(lrs[i] <= lrs[i + 1] + 1e-12 for i in range(peak))
    assert all(lrs[i] >= lrs[i + 1] - 1e-12 for i in range(peak, len(lrs) - 1))


def test_deterministic_given_seeds(tmp_path):
    """Same config + same data stream → identical parameters (determinism of
    the full step: selection, losses, SpecAdam)."""
    results = []
    for run in range(2):
        cfg = tiny_config("A", tmp_path / f"r{run}")
        write_capture(Path(cfg.capture_root), seed=0)
        g = torch.Generator().manual_seed(7)
        mu_e = torch.randn(E, D_MODEL, generator=g) * 0.1
        mu_route = torch.randn(D_MODEL, generator=g) * 0.1
        cond = build_condition("A", cfg, mu_e, mu_route, cfg.seed)
        batches = batch_iter_from_capture(
            Path(cfg.capture_root), cfg.batch_tokens, cfg.data_seed
        )
        train(cfg, cond, batches, cfg.out_dir)
        results.append(cond)
    assert torch.equal(results[0].sae.D.detach(), results[1].sae.D.detach())
    assert torch.equal(results[0].sae.W_enc.detach(), results[1].sae.W_enc.detach())
    assert results[0].theta_ema.value == results[1].theta_ema.value
