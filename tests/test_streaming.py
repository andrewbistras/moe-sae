"""Streaming discovery correctness (bucketless lane).

The decisive property: rotating-window co-training with state swapping is
BIT-IDENTICAL to running moe_sae.train.loop.train per condition on the same
batch sequence. Everything else here guards the components that property
rests on (composer determinism, window roundtrip, Adam-memmap durability)."""

import json

import numpy as np
import pytest
import torch

from moe_sae.sae.conditions import SAECondition, build_condition
from moe_sae.sae.optim import SpecAdam
from moe_sae.train.config import RunConfig
from moe_sae.train.loop import train
from moe_sae.train.stream_discovery import stream_discovery
from moe_sae.train.streaming import (
    CollectingWriter,
    StateSwapper,
    StreamComposer,
    WindowStore,
)

E, F_BANK, D_MODEL, B = 64, 8, 16, 24


def tiny_config(condition, seed, n_batches, run_name=None):
    return RunConfig(
        run_name=run_name or f"stream_{condition}_s{seed}",
        condition=condition,
        seed=seed,
        f=F_BANK,
        k_bar=4,
        total_tokens=B * n_batches,
        capture_root="unused",
        stats_path="unused",
        pipeline=True,
        batch_tokens=B,
        compute_dtype="float32",
        log_every=1,
        ckpt_every_steps=10_000,
    )


def make_means():
    g = torch.Generator().manual_seed(0)
    return torch.randn(E, D_MODEL, generator=g) * 0.1, torch.randn(D_MODEL, generator=g) * 0.1


def synthetic_source(n_examples=200, vis_per=6, txt_per=3, seed=5):
    """Per-example {pool: rows} dicts, deterministic given seed."""
    g = torch.Generator().manual_seed(seed)
    for i in range(n_examples):
        ex_id = i  # bucket 0-99 irrelevant here
        out = {}
        for pool, n in (("visual", vis_per), ("text", txt_per)):
            experts = torch.stack([torch.randperm(E, generator=g)[:6] for _ in range(n)])
            out[pool] = {
                "c": torch.randn(n, 6, D_MODEL, generator=g).to(torch.bfloat16),
                "experts": experts.to(torch.uint8),
                "gates": torch.rand(n, 6, generator=g),
                "example_id": torch.full((n,), ex_id, dtype=torch.int32),
                "domain": torch.zeros(n, dtype=torch.uint8),
            }
        yield out


def test_collecting_writer_roundtrip():
    w = CollectingWriter()
    w.add(
        c=torch.randn(3, 6, D_MODEL),
        experts=torch.zeros(3, 6),
        gates=torch.zeros(3, 6),
        example_id=torch.arange(3),
        domain=torch.zeros(3),
    )
    rows = w.drain()
    assert len(rows) == 1 and rows[0]["c"].dtype == torch.bfloat16
    assert w.drain() == [] and w.total_written == 3


def test_composer_halves_and_determinism():
    def batches(seed):
        comp = StreamComposer(
            synthetic_source(seed=9), batch_tokens=B, min_visual_buffer=30
        )
        return [comp.next_batch() for _ in range(5)]

    a, b = batches(1), batches(2)  # stream RNG is fixed, not seed-dependent
    for ba, bb in zip(a, b):
        assert torch.equal(ba.c, bb.c) and torch.equal(ba.experts, bb.experts)
        assert int((ba.token_type == 1).sum()) == B // 2
        assert int((ba.token_type == 0).sum()) == B // 2


def test_composer_visual_rows_unique_across_batches():
    comp = StreamComposer(synthetic_source(seed=9), batch_tokens=B, min_visual_buffer=30)
    seen = set()
    for _ in range(5):
        batch = comp.next_batch()
        vis = batch.c[batch.token_type == 1].float()
        for row in vis.reshape(vis.shape[0], -1):
            key = row.numpy().tobytes()
            assert key not in seen  # visual rows drawn once, ever
            seen.add(key)


def test_window_store_roundtrip(tmp_path):
    comp = StreamComposer(synthetic_source(), batch_tokens=B, min_visual_buffer=30)
    batches = [comp.next_batch() for _ in range(3)]
    store = WindowStore(tmp_path)
    store.write(0, batches)
    back = store.read(0)
    for x, y in zip(batches, back):
        assert torch.equal(x.c, y.c) and torch.equal(x.example_id, y.example_id)
    store.delete(0)
    assert not store.exists(0)


def test_state_swapper_durability(tmp_path):
    mu_e, mu_route = make_means()
    cfg = tiny_config("A", 1, 2)
    cond = build_condition("A", cfg, mu_e, mu_route, 1)
    opt = SpecAdam(cond.optim_groups())
    comp = StreamComposer(synthetic_source(), batch_tokens=B, min_visual_buffer=30)
    out = cond.training_step(comp.next_batch(), 0)
    out.loss.backward()
    opt.step(3e-4)

    ref = {
        i: (st["exp_avg"].clone(), st["exp_avg_sq"].clone())
        for i, st in enumerate(opt.state.values())
    }
    sw = StateSwapper(tmp_path)
    sw.swap_out("A1", cond, opt)
    # fresh swapper instance (fresh process semantics) reads the same moments
    sw2 = StateSwapper(tmp_path)
    sw2.swap_in("A1", cond, opt, "cpu")
    for i, st in enumerate(opt.state.values()):
        assert torch.equal(st["exp_avg"], ref[i][0])
        assert torch.equal(st["exp_avg_sq"], ref[i][1])


@pytest.mark.parametrize("pipelined", [False, True])
def test_windowed_equals_sequential(tmp_path, pipelined):
    """THE invariant: stream_discovery == loop.train per condition, bit-equal
    — in both serial and pipelined (producer-thread) modes."""
    mu_e, mu_route = make_means()
    n_batches = 6
    specs = [("A", 1), ("B", 2), ("C", 3)]

    # Baseline: capture the exact batch sequence, then plain loop.train each.
    comp = StreamComposer(synthetic_source(seed=9), batch_tokens=B, min_visual_buffer=30)
    batches = [comp.next_batch() for _ in range(n_batches)]
    base_params = {}
    for cond_name, seed in specs:
        cfg = tiny_config(cond_name, seed, n_batches, run_name=f"base_{cond_name}{seed}")
        condition = build_condition(cond_name, cfg, mu_e, mu_route, seed)
        train(cfg, condition, iter(list(batches)), tmp_path / cfg.run_name)
        base_params[(cond_name, seed)] = {
            "W_enc": condition.sae.W_enc.detach().clone(),
            "D": condition.sae.D.detach().clone(),
            "theta": condition.theta_ema.value,
        }

    # Streamed: same synthetic source ⇒ same composed batches; windows of 2.
    cfgs = [
        tiny_config(c, s, n_batches, run_name=f"stream_{c}{s}") for c, s in specs
    ]
    stream_discovery(
        cfgs,
        synthetic_source(seed=9),
        mu_e,
        mu_route,
        out_root=tmp_path / "runs",
        window_batches=2,
        device="cpu",
        work_dir=tmp_path / "work",
        ckpt_every_windows=100,
        min_visual_buffer=30,  # must match the reference composer above
        pipelined=pipelined,
    )

    for cond_name, seed in specs:
        run_dir = tmp_path / "runs" / f"stream_{cond_name}{seed}"
        loaded, theta, extras = SAECondition.load_checkpoint(
            run_dir / "final" / "checkpoint.safetensors"
        )
        ref = base_params[(cond_name, seed)]
        assert torch.equal(loaded.sae.W_enc.detach(), ref["W_enc"]), (cond_name, seed)
        assert torch.equal(loaded.sae.D.detach(), ref["D"]), (cond_name, seed)
        assert theta == pytest.approx(ref["theta"], rel=0, abs=0)
        # artifact layout identical to loop.train
        assert (run_dir / "final" / "bank_loads.npy").exists()
        lines = (run_dir / "metrics.jsonl").read_text().splitlines()
        assert json.loads(lines[-1])["step"] == n_batches - 1
    # windows cleaned up
    assert not list((tmp_path / "work" / "windows").glob("window_*.pt"))
